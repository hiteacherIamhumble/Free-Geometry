#!/usr/bin/env python3
"""
Compare internal tokens between:
  - Teacher 8v->4v (encode 8 views, slice even-indexed 4 views)
  - Teacher 4v (direct 4 views)
  - Student LoRA 4v

Computes MSE (lower is better) and cosine similarity (higher is better) against the
Teacher 8v->4v reference, for selected backbone layers.

Default token types:
  - camera_token: 1536-dim camera token (second half of cam_token_raw, like distillation code)
  - global_mean: mean-pooled global features over patches (1536-dim)

This script re-runs backbone forward passes (tokens are not saved by the benchmark).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.distillation.models import StudentModel


EVEN_INDICES = [0, 2, 4, 6]


def sample_8_frames(image_files: Sequence[str], seed: int) -> List[int]:
    n = len(image_files)
    if n <= 8:
        return list(range(n))
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    return sorted(idx[:8])


def _scene_slug(scene: str) -> str:
    return "-".join(scene.split("/")[-3:])


def _find_gt_meta_npz(work_dir: str, scene: str) -> Optional[str]:
    """
    Find a saved gt_meta.npz for this scene under any experiment folder.
    """
    for exp in ("teacher", "teacher_4v", "student"):
        p = os.path.join(
            work_dir,
            exp,
            "model_results",
            "hiroom",
            scene,
            "unposed",
            "exports",
            "gt_meta.npz",
        )
        if os.path.exists(p):
            return p
    return None


def _load_saved_frames(
    *,
    work_dir: str,
    image_dir: str,
    scene: str,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Load the exact frames used by the benchmark if available:
      - 8-view list from `<image_dir>/<scene_slug>/*.jpg`
      - 4-view list from `<work_dir>/*/model_results/.../gt_meta.npz`

    Returns: (images_8, images_4), each can be None if not available.
    """
    images_8 = None
    d = os.path.join(image_dir, _scene_slug(scene))
    if os.path.isdir(d):
        imgs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".jpg")]
        if imgs:
            images_8 = imgs

    images_4 = None
    meta = _find_gt_meta_npz(work_dir, scene)
    if meta is not None:
        data = np.load(meta, allow_pickle=True)
        image_files = list(data.get("image_files", []))
        image_files = [str(x) for x in image_files]
        if image_files:
            images_4 = image_files

    return images_8, images_4


def _autocast_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class TokenPack:
    # layer -> (S, D) float32 cpu
    camera_token: Dict[int, torch.Tensor]
    global_mean: Dict[int, torch.Tensor]


def _extract_tokens_from_feats(feats, out_layers: Sequence[int], view_indices: Optional[List[int]] = None) -> TokenPack:
    """
    Args:
      feats: list of feat tuples returned by forward_backbone_only
      out_layers: corresponding layer indices for feats
      view_indices: optionally slice views dimension
    """
    cam: Dict[int, torch.Tensor] = {}
    glob: Dict[int, torch.Tensor] = {}

    for feat_tuple, layer in zip(feats, out_layers):
        features = feat_tuple[0]  # [B,S,P,3072]
        cam_token_raw = feat_tuple[1]  # [B,S,3072] (local_cls + camera_token)
        if cam_token_raw is None or features is None:
            continue

        # Slice views if requested
        if view_indices is not None:
            features = features[:, view_indices, ...]
            cam_token_raw = cam_token_raw[:, view_indices, ...]

        # 1536-dim camera token (second half)
        cam_token = cam_token_raw[..., 1536:]  # [B,S,1536]

        # Mean-pooled global features over patches (second half of 3072)
        global_feat = features[..., 1536:]  # [B,S,P,1536]
        global_mean = global_feat.mean(dim=2)  # [B,S,1536]

        cam[int(layer)] = cam_token.squeeze(0).detach().float().cpu()
        glob[int(layer)] = global_mean.squeeze(0).detach().float().cpu()

    return TokenPack(camera_token=cam, global_mean=glob)


def extract_tokens_da3(
    da3: DepthAnything3,
    image_paths: List[str],
    *,
    ref_view_strategy: str = "first",
    view_indices: Optional[List[int]] = None,
) -> Tuple[TokenPack, List[int]]:
    """
    Run backbone-only forward and return token packs.
    """
    imgs_cpu, _ex, _in = da3._preprocess_inputs(image_paths, None, None)
    imgs, _ex_t, _in_t = da3._prepare_model_inputs(imgs_cpu, None, None)

    with torch.no_grad():
        with torch.autocast(device_type=imgs.device.type, dtype=_autocast_dtype()):
            feats, _aux, _H, _W = da3.model.forward_backbone_only(
                imgs, extrinsics=None, intrinsics=None, ref_view_strategy=ref_view_strategy
            )

    out_layers = list(getattr(da3.model.backbone, "out_layers", list(range(len(feats)))))
    return _extract_tokens_from_feats(feats, out_layers, view_indices=view_indices), out_layers


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().item())


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: (S,D)
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


@dataclass
class CompareRow:
    scene: str
    token_type: str
    layer: int
    mse_t4: float
    cos_t4: float
    mse_lora: float
    cos_lora: float


def compare_scene(
    scene: str,
    image_files: List[str],
    *,
    seed: int,
    teacher_da3: DepthAnything3,
    student_da3: DepthAnything3,
    work_dir: str,
    image_dir: str,
    prefer_saved_frames: bool = True,
    layers: Optional[Sequence[int]] = None,
    token_types: Sequence[str] = ("camera_token", "global_mean"),
) -> List[CompareRow]:
    images_8 = None
    images_4 = None
    if prefer_saved_frames:
        images_8, images_4 = _load_saved_frames(work_dir=work_dir, image_dir=image_dir, scene=scene)

    # Fallback: deterministic sampling from the dataset list.
    if images_8 is None or images_4 is None:
        idx8 = sample_8_frames(image_files, seed=seed)
        images_8 = [image_files[i] for i in idx8]
        idx4 = [idx8[i] for i in EVEN_INDICES]
        images_4 = [image_files[i] for i in idx4]

    # Reference: teacher 8v, then slice even-indexed views (0,2,4,6) in that 8-view ordering.
    t8_pack, t8_out_layers = extract_tokens_da3(
        teacher_da3, images_8, ref_view_strategy="first", view_indices=EVEN_INDICES
    )
    # Baseline: same teacher model, but only 4 views.
    t4_pack, t4_out_layers = extract_tokens_da3(teacher_da3, images_4, ref_view_strategy="first")
    # Student: LoRA merged weights, 4 views.
    s_pack, s_out_layers = extract_tokens_da3(student_da3, images_4, ref_view_strategy="first")

    common_layers = set(t8_out_layers) & set(t4_out_layers) & set(s_out_layers)
    if layers is not None:
        common_layers &= set(int(x) for x in layers)
    common_layers = sorted(common_layers)

    rows: List[CompareRow] = []
    for layer in common_layers:
        for token_type in token_types:
            ref = getattr(t8_pack, token_type).get(layer)
            t4 = getattr(t4_pack, token_type).get(layer)
            st = getattr(s_pack, token_type).get(layer)
            if ref is None or t4 is None or st is None:
                continue
            rows.append(
                CompareRow(
                    scene=scene,
                    token_type=token_type,
                    layer=layer,
                    mse_t4=mse(t4, ref),
                    cos_t4=cos_sim(t4, ref),
                    mse_lora=mse(st, ref),
                    cos_lora=cos_sim(st, ref),
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare internal tokens (MSE/cosine) vs teacher 8v->4v reference")
    parser.add_argument("--work_dir", type=str, default="./results/hiroom_teacher_student")
    parser.add_argument("--image_dir", type=str, default="./results/images")
    parser.add_argument("--all_scenes", action="store_true", help="Run on all HiRoom scenes from selected list")
    parser.add_argument("--scene", type=str, default=None, help="Single scene to run (e.g. 20241230/.../cam_...)")
    parser.add_argument("--max_scenes", type=int, default=0, help="Optional cap for quick runs (0 = no cap)")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--teacher_model", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--student_base_model", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--lora_path", type=str, default="checkpoints/da3_tuned_lora_dist/hiroom/epoch_2_lora.pt")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional backbone layer indices to compare (default: backbone.out_layers)",
    )
    parser.add_argument(
        "--token_types",
        type=str,
        nargs="*",
        default=["camera_token", "global_mean"],
        choices=["camera_token", "global_mean"],
    )
    parser.add_argument(
        "--prefer_saved_frames",
        action="store_true",
        help="Prefer using saved 8-view frames under --image_dir and saved gt_meta.npz under --work_dir",
    )
    parser.add_argument("--out_json", type=str, default="", help="Optional path to save detailed per-scene results JSON")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if not args.all_scenes and not args.scene:
        raise SystemExit("Pass --scene <scene> or --all_scenes")

    dataset = MV_REGISTRY.get("hiroom")()
    scenes: List[str]
    if args.all_scenes:
        scenes = list(dataset.SCENES)
    else:
        scenes = [args.scene]

    if args.max_scenes and args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading teacher model: {args.teacher_model}")
    teacher_da3 = DepthAnything3.from_pretrained(args.teacher_model).to(device)

    print(f"Loading student base model: {args.student_base_model}")
    print(f"Loading LoRA weights: {args.lora_path}")

    # Mirror benchmark wrapper: auto-detect rank/alpha from adapter_config.json if present.
    lora_rank = None
    lora_alpha = None
    peft_dir = args.lora_path.replace(".pt", "_peft")
    cfg_path = os.path.join(peft_dir, "adapter_config.json")
    if os.path.isdir(peft_dir) and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            lora_rank = cfg.get("r")
            lora_alpha = cfg.get("lora_alpha")
            print(f"Auto-detected from checkpoint: rank={lora_rank}, alpha={lora_alpha}")
        except Exception:
            pass
    if lora_rank is None:
        lora_rank = 16
    if lora_alpha is None:
        lora_alpha = float(lora_rank)

    # Use StudentModel so we can load adapter + merge for inference.
    student = StudentModel(
        model_name=args.student_base_model,
        lora_rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
        patch_swiglu_mlp_for_lora=False,
    )
    student.load_lora_weights(args.lora_path)
    peft_model = student.da3.model.backbone.pretrained
    student.da3.model.backbone.pretrained = peft_model.merge_and_unload()
    student.eval()
    student = student.to(device)
    student_da3 = student.da3

    rows_all: List[CompareRow] = []
    for i, scene in enumerate(scenes):
        scene_data = dataset.get_data(scene)
        if not scene_data.image_files:
            print(f"[WARN] {scene}: no images")
            continue
        print(f"[{i+1}/{len(scenes)}] {scene}: comparing tokens...")
        try:
            rows = compare_scene(
                scene,
                list(scene_data.image_files),
                seed=args.seed,
                teacher_da3=teacher_da3,
                student_da3=student_da3,
                work_dir=args.work_dir,
                image_dir=args.image_dir,
                prefer_saved_frames=args.prefer_saved_frames,
                layers=args.layers,
                token_types=args.token_types,
            )
            rows_all.extend(rows)
        except Exception as e:
            print(f"[ERROR] scene failed: {scene}: {e}")
            continue

    if not rows_all:
        raise SystemExit("No comparison rows computed (check layers/token_types).")

    # Aggregate: token_type + layer
    summary = {}
    for r in rows_all:
        key = (r.token_type, r.layer)
        summary.setdefault(key, {"mse_t4": [], "cos_t4": [], "mse_lora": [], "cos_lora": []})
        summary[key]["mse_t4"].append(r.mse_t4)
        summary[key]["cos_t4"].append(r.cos_t4)
        summary[key]["mse_lora"].append(r.mse_lora)
        summary[key]["cos_lora"].append(r.cos_lora)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs).item()) if xs else float("nan")

    print("\nResults (mean over scenes; ref = teacher 8v->4v):")
    for (token_type, layer), d in sorted(summary.items(), key=lambda x: (x[0][0], x[0][1])):
        mse_t4 = _mean(d["mse_t4"])
        cos_t4 = _mean(d["cos_t4"])
        mse_l = _mean(d["mse_lora"])
        cos_l = _mean(d["cos_lora"])
        print(
            f"  {token_type:11s} layer {layer:>2d} | "
            f"T4 mse {mse_t4:.6g} cos {cos_t4:.4f} || "
            f"LoRA mse {mse_l:.6g} cos {cos_l:.4f}"
        )

    # How often LoRA is closer than T4
    closer_counts = {}
    total_counts = {}
    for r in rows_all:
        key = (r.token_type, r.layer)
        total_counts[key] = total_counts.get(key, 0) + 1
        closer = (r.mse_lora < r.mse_t4) and (r.cos_lora > r.cos_t4)
        closer_counts[key] = closer_counts.get(key, 0) + (1 if closer else 0)

    print("\nLoRA closer than Teacher4v? (requires: mse lower AND cosine higher)")
    for key in sorted(total_counts.keys(), key=lambda x: (x[0], x[1])):
        n = total_counts[key]
        c = closer_counts.get(key, 0)
        pct = 100.0 * c / max(n, 1)
        print(f"  {key[0]:11s} layer {key[1]:>2d}: {pct:5.1f}% ({c}/{n})")

    if args.out_json:
        out = {
            "args": vars(args),
            "rows": [r.__dict__ for r in rows_all],
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote: {args.out_json}")


if __name__ == "__main__":
    main()

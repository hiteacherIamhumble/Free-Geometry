#!/usr/bin/env python3
"""
Compare feature similarity on ETH3D for DA3 and VGGT.

For each scene:
  1) Sample up to 8 frames (same random logic as benchmark scripts)
  2) Build shared 4-view subset from even positions [0, 2, 4, 6]
  3) Extract backbone features from:
       - teacher 8v (then slice to shared 4 views)
       - teacher 4v (direct 4 views)
       - student 4v (direct 4 views)
  4) Compute MSE and cosine similarity with reference = teacher(8v)->shared4

Comparisons:
  (1) teacher(8v->shared4) vs teacher(4v)
  (2) teacher(8v->shared4) vs student(4v)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))

if TYPE_CHECKING:
    from depth_anything_3.api import DepthAnything3


EVEN_INDICES = [0, 2, 4, 6]
FAMILIES = ["da3", "vggt"]


@dataclass
class Row:
    family: str
    dataset: str
    scene: str
    layer: int
    mse_teacher4v: float
    cos_teacher4v: float
    mse_student4v: float
    cos_student4v: float


def sample_8_indices(image_files: Sequence[str], seed: int) -> List[int]:
    n = len(image_files)
    if n <= 8:
        return list(range(n))
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    return sorted(idx[:8])


def detect_lora_rank_alpha(lora_path: str, default_rank: int, default_alpha: float) -> tuple[int, float]:
    rank = default_rank
    alpha = default_alpha
    peft_dir = lora_path.replace(".pt", "_peft")
    cfg_path = os.path.join(peft_dir, "adapter_config.json")
    if os.path.isdir(peft_dir) and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            rank = int(cfg.get("r", rank))
            alpha = float(cfg.get("lora_alpha", alpha))
            print(f"  Auto-detected LoRA config from {cfg_path}: rank={rank}, alpha={alpha}")
        except Exception as e:
            print(f"  [WARN] Failed to parse {cfg_path}: {e}")
    return rank, alpha


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _mse_cos(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    # a,b: [B, S, P, D]
    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    mse = float(((a2 - b2) ** 2).mean().item())
    cos = float(F.cosine_similarity(a2, b2, dim=-1).mean().item())
    return mse, cos


def _resolve_scenes(dataset, scene: Optional[str], scenes: Optional[List[str]], max_scenes: int) -> List[str]:
    available = list(getattr(dataset, "SCENES", []))
    available_set = set(available)

    if scenes:
        selected = [s for s in scenes if s in available_set]
        missing = [s for s in scenes if s not in available_set]
        if missing:
            print(f"[WARN] skipping unknown scenes: {missing}")
    elif scene:
        selected = [scene] if scene in available_set else []
        if not selected:
            print(f"[WARN] scene not found: {scene}")
    else:
        selected = available

    if max_scenes > 0:
        selected = selected[:max_scenes]
    return selected


def _print_summary(rows: List[Row], family: str) -> None:
    family_rows = [r for r in rows if r.family == family]
    if not family_rows:
        print(f"\n[{family}] No rows to summarize.")
        return

    by_layer: Dict[int, List[Row]] = {}
    for r in family_rows:
        by_layer.setdefault(r.layer, []).append(r)

    print(f"\n[{family}] Mean over scenes (ref = teacher 8v extracted shared 4v):")
    print(
        f"{'layer':>6} | {'t4_mse':>10} {'t4_cos':>9} || {'s4_mse':>10} {'s4_cos':>9} | {'student closer':>14}"
    )
    print("-" * 74)
    for layer in sorted(by_layer.keys()):
        bucket = by_layer[layer]
        t4_mse = float(np.mean([x.mse_teacher4v for x in bucket]))
        t4_cos = float(np.mean([x.cos_teacher4v for x in bucket]))
        s4_mse = float(np.mean([x.mse_student4v for x in bucket]))
        s4_cos = float(np.mean([x.cos_student4v for x in bucket]))
        closer = [
            (x.mse_student4v < x.mse_teacher4v) and (x.cos_student4v > x.cos_teacher4v) for x in bucket
        ]
        closer_pct = 100.0 * float(np.mean(closer))
        print(
            f"{layer:6d} | {t4_mse:10.6g} {t4_cos:9.4f} || {s4_mse:10.6g} {s4_cos:9.4f} | {closer_pct:12.1f}%"
        )


def _load_vggt_images(image_files: List[str], image_size: int) -> torch.Tensor:
    import cv2

    patch_size = 14
    images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        scale = image_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        new_h = max(patch_size, round(new_h / patch_size) * patch_size)
        new_w = max(patch_size, round(new_w / patch_size) * patch_size)

        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        images.append(img.astype(np.float32) / 255.0)

    images_np = np.stack(images, axis=0)
    return torch.from_numpy(images_np).permute(0, 3, 1, 2).float().unsqueeze(0)  # [1, S, 3, H, W]


def _extract_da3_layer_features(da3: "DepthAnything3", image_paths: List[str]) -> Dict[int, torch.Tensor]:
    imgs_cpu, _, _ = da3._preprocess_inputs(image_paths, None, None)
    imgs, _, _ = da3._prepare_model_inputs(imgs_cpu, None, None)
    with torch.no_grad():
        with _autocast_ctx(imgs.device):
            feats, _aux, _h, _w = da3.model.forward_backbone_only(
                imgs, extrinsics=None, intrinsics=None, ref_view_strategy="first"
            )
    out_layers = list(getattr(da3.model.backbone, "out_layers", list(range(len(feats)))))
    return {int(layer): feat_tuple[0].detach().float().cpu() for feat_tuple, layer in zip(feats, out_layers)}


def run_da3(args, device: torch.device, scenes: List[str], dataset) -> List[Row]:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.distillation.models import StudentModel

    print("\n" + "=" * 72)
    print("Running DA3 feature similarity")
    print("=" * 72)

    print(f"Loading DA3 teacher: {args.da3_teacher_model}")
    teacher = DepthAnything3.from_pretrained(args.da3_teacher_model).to(device)
    teacher.eval()

    print(f"Loading DA3 student base: {args.da3_student_base_model}")
    print(f"Loading DA3 LoRA weights: {args.da3_lora_path}")
    if not os.path.exists(args.da3_lora_path):
        raise FileNotFoundError(f"DA3 LoRA checkpoint not found: {args.da3_lora_path}")
    rank, alpha = detect_lora_rank_alpha(
        args.da3_lora_path, default_rank=args.da3_lora_rank, default_alpha=args.da3_lora_alpha
    )
    student = StudentModel(
        model_name=args.da3_student_base_model,
        output_layers=args.da3_layers,
        lora_rank=rank,
        lora_alpha=alpha,
        patch_swiglu_mlp_for_lora=False,
    )
    student.load_lora_weights(args.da3_lora_path)
    peft_model = student.da3.model.backbone.pretrained
    student.da3.model.backbone.pretrained = peft_model.merge_and_unload()
    student.eval()
    student = student.to(device)

    rows: List[Row] = []
    for i, scene in enumerate(scenes, start=1):
        scene_data = dataset.get_data(scene)
        if not scene_data.image_files:
            print(f"[DA3][{i}/{len(scenes)}] {scene}: no images, skip")
            continue

        idx8 = sample_8_indices(scene_data.image_files, seed=args.seed)
        if len(idx8) < 7:
            print(f"[DA3][{i}/{len(scenes)}] {scene}: need >= 7 sampled frames, got {len(idx8)}")
            continue
        idx4 = [idx8[j] for j in EVEN_INDICES]

        images8 = [scene_data.image_files[j] for j in idx8]
        images4 = [scene_data.image_files[j] for j in idx4]
        print(f"[DA3][{i}/{len(scenes)}] {scene}")

        try:
            t8 = _extract_da3_layer_features(teacher, images8)
            t4 = _extract_da3_layer_features(teacher, images4)
            s4 = _extract_da3_layer_features(student.da3, images4)

            common_layers = sorted(set(t8.keys()) & set(t4.keys()) & set(s4.keys()) & set(args.da3_layers))
            for layer in common_layers:
                ref = t8[layer][:, EVEN_INDICES, :, :]
                teacher4 = t4[layer]
                student4 = s4[layer]
                mse_t4, cos_t4 = _mse_cos(ref, teacher4)
                mse_s4, cos_s4 = _mse_cos(ref, student4)
                rows.append(
                    Row(
                        family="da3",
                        dataset=args.dataset,
                        scene=scene,
                        layer=layer,
                        mse_teacher4v=mse_t4,
                        cos_teacher4v=cos_t4,
                        mse_student4v=mse_s4,
                        cos_student4v=cos_s4,
                    )
                )
        except Exception as e:
            print(f"[DA3][ERROR] {scene}: {e}")
            continue

    del teacher
    del student
    torch.cuda.empty_cache()
    return rows


def _extract_vggt_layer_features(model: Any, images: torch.Tensor) -> Dict[int, torch.Tensor]:
    with torch.no_grad():
        out = model(images)
    return {int(layer): feat.detach().float().cpu() for layer, feat in out.layer_features.items()}


def run_vggt(args, device: torch.device, scenes: List[str], dataset) -> List[Row]:
    from vggt.vggt.distillation.models import VGGTStudentModel, VGGTTeacherModel

    print("\n" + "=" * 72)
    print("Running VGGT feature similarity")
    print("=" * 72)

    print(f"Loading VGGT teacher: {args.vggt_teacher_model}")
    teacher = VGGTTeacherModel(
        model_name=args.vggt_teacher_model,
        output_layers=args.vggt_layers,
    ).to(device)
    teacher.eval()

    print(f"Loading VGGT student base: {args.vggt_student_base_model}")
    print(f"Loading VGGT LoRA weights: {args.vggt_lora_path}")
    if not os.path.exists(args.vggt_lora_path):
        raise FileNotFoundError(f"VGGT LoRA checkpoint not found: {args.vggt_lora_path}")
    rank, alpha = detect_lora_rank_alpha(
        args.vggt_lora_path, default_rank=args.vggt_lora_rank, default_alpha=args.vggt_lora_alpha
    )
    student = VGGTStudentModel(
        model_name=args.vggt_student_base_model,
        output_layers=args.vggt_layers,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_layers=list(range(args.vggt_lora_layers_start, 24)),
    ).to(device)
    student.load_lora_weights(args.vggt_lora_path)
    student.eval()

    rows: List[Row] = []
    for i, scene in enumerate(scenes, start=1):
        scene_data = dataset.get_data(scene)
        if not scene_data.image_files:
            print(f"[VGGT][{i}/{len(scenes)}] {scene}: no images, skip")
            continue

        idx8 = sample_8_indices(scene_data.image_files, seed=args.seed)
        if len(idx8) < 7:
            print(f"[VGGT][{i}/{len(scenes)}] {scene}: need >= 7 sampled frames, got {len(idx8)}")
            continue
        idx4 = [idx8[j] for j in EVEN_INDICES]

        images8 = [scene_data.image_files[j] for j in idx8]
        images4 = [scene_data.image_files[j] for j in idx4]
        print(f"[VGGT][{i}/{len(scenes)}] {scene}")

        try:
            imgs8 = _load_vggt_images(images8, image_size=args.vggt_image_size).to(device)
            imgs4 = _load_vggt_images(images4, image_size=args.vggt_image_size).to(device)

            t8 = _extract_vggt_layer_features(teacher, imgs8)
            t4 = _extract_vggt_layer_features(teacher, imgs4)
            s4 = _extract_vggt_layer_features(student, imgs4)

            common_layers = sorted(set(t8.keys()) & set(t4.keys()) & set(s4.keys()) & set(args.vggt_layers))
            for layer in common_layers:
                ref = t8[layer][:, EVEN_INDICES, :, :]
                teacher4 = t4[layer]
                student4 = s4[layer]
                mse_t4, cos_t4 = _mse_cos(ref, teacher4)
                mse_s4, cos_s4 = _mse_cos(ref, student4)
                rows.append(
                    Row(
                        family="vggt",
                        dataset=args.dataset,
                        scene=scene,
                        layer=layer,
                        mse_teacher4v=mse_t4,
                        cos_teacher4v=cos_t4,
                        mse_student4v=mse_s4,
                        cos_student4v=cos_s4,
                    )
                )
        except Exception as e:
            print(f"[VGGT][ERROR] {scene}: {e}")
            continue

        del imgs8
        del imgs4
        torch.cuda.empty_cache()

    del teacher
    del student
    torch.cuda.empty_cache()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute feature MSE/cosine similarity on ETH3D for DA3 and VGGT teacher/student."
    )
    parser.add_argument("--dataset", type=str, default="eth3d")
    parser.add_argument("--scene", type=str, default=None, help="Single scene")
    parser.add_argument(
        "--scenes",
        action="append",
        default=None,
        help="Repeatable scene filter: --scenes scene_a --scenes scene_b",
    )
    parser.add_argument("--max_scenes", type=int, default=0, help="0 means all selected scenes")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--families",
        nargs="+",
        default=FAMILIES,
        choices=FAMILIES,
        help="Model families to run",
    )
    parser.add_argument("--out_json", type=str, default="", help="Optional JSON output path")

    # DA3 options
    parser.add_argument("--da3_teacher_model", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--da3_student_base_model", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--da3_lora_path", type=str, default="checkpoints/da3_lora_final/eth3d/lora.pt")
    parser.add_argument("--da3_lora_rank", type=int, default=16)
    parser.add_argument("--da3_lora_alpha", type=float, default=16.0)
    parser.add_argument(
        "--da3_layers",
        nargs="+",
        type=int,
        default=[19, 27, 33, 39],
        help="DA3 backbone layers to compare",
    )

    # VGGT options
    parser.add_argument("--vggt_teacher_model", type=str, default="facebook/vggt-1b")
    parser.add_argument("--vggt_student_base_model", type=str, default="facebook/vggt-1b")
    parser.add_argument("--vggt_lora_path", type=str, default="checkpoints/vggt_lora_final/eth3d/lora.pt")
    parser.add_argument("--vggt_lora_rank", type=int, default=16)
    parser.add_argument("--vggt_lora_alpha", type=float, default=16.0)
    parser.add_argument("--vggt_lora_layers_start", type=int, default=12)
    parser.add_argument("--vggt_image_size", type=int, default=504)
    parser.add_argument(
        "--vggt_layers",
        nargs="+",
        type=int,
        default=[19, 23],
        help="VGGT aggregator layers to compare",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from depth_anything_3.bench.registries import MV_REGISTRY

    if not MV_REGISTRY.has(args.dataset):
        raise SystemExit(f"Unknown dataset: {args.dataset}")
    dataset = MV_REGISTRY.get(args.dataset)()
    scenes = _resolve_scenes(dataset, args.scene, args.scenes, args.max_scenes)
    if not scenes:
        raise SystemExit("No scenes selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} ({len(scenes)} scenes)")

    rows: List[Row] = []

    if "da3" in args.families:
        rows.extend(run_da3(args, device, scenes, dataset))
        _print_summary(rows, "da3")

    if "vggt" in args.families:
        rows.extend(run_vggt(args, device, scenes, dataset))
        _print_summary(rows, "vggt")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "args": vars(args),
            "rows": [asdict(r) for r in rows],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote: {args.out_json}")


if __name__ == "__main__":
    main()

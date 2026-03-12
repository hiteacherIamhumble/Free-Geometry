#!/usr/bin/env python3
"""
Benchmark: Teacher (DA3-GIANT 8v encode -> extract 4v decode) vs Student (LoRA 4v)
on multiple datasets.

Datasets supported by default:
  - scannetpp
  - hiroom
  - 7scenes
  - eth3d

For each scene, this script:
  1) Samples up to 8 frames (same seed logic as Evaluator)
  2) Builds a shared 4-view subset from even positions [0, 2, 4, 6]
  3) Runs:
       - teacher        : 8v encode -> extract 4v decode
       - teacher_4v     : direct 4v baseline
       - student        : LoRA 4v
  4) Saves mini_npz + GT meta + visualizations
  5) Runs pose + recon_unposed evaluation

Usage:
    python scripts/benchmark_teacher_student_all_datasets.py
    python scripts/benchmark_teacher_student_all_datasets.py --datasets hiroom
    python scripts/benchmark_teacher_student_all_datasets.py --all_scenes
    python scripts/benchmark_teacher_student_all_datasets.py --eval_only
    python scripts/benchmark_teacher_student_all_datasets.py --report_only
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATASETS = ["scannetpp", "hiroom", "7scenes", "eth3d"]
DEFAULT_SCENE = "20241230/828805/cam_sampled_06"  # used when dataset=hiroom
EVAL_MODES = ["pose", "recon_unposed"]
EVEN_INDICES = [0, 2, 4, 6]


# ---------------------------------------------------------------------------
# LoRA model wrapper (adapted from scripts/benchmark_lora.py)
# ---------------------------------------------------------------------------
class LoRADepthAnything3:
    """Wrapper to make LoRA StudentModel compatible with the Evaluator API."""

    def __init__(self, base_model="depth-anything/DA3-GIANT", lora_path=None, device="cuda"):
        from depth_anything_3.distillation.models import StudentModel

        print(f"Loading base model: {base_model}")
        print(f"Loading LoRA weights: {lora_path}")

        lora_rank = None
        lora_alpha = None

        # Auto-detect rank/alpha from PEFT adapter_config.json
        peft_dir = lora_path.replace(".pt", "_peft") if lora_path else None
        if peft_dir and os.path.isdir(peft_dir):
            config_path = os.path.join(peft_dir, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    peft_cfg = json.load(f)
                lora_rank = peft_cfg.get("r")
                lora_alpha = peft_cfg.get("lora_alpha")
                print(f"Auto-detected from checkpoint: rank={lora_rank}, alpha={lora_alpha}")

        if lora_rank is None:
            lora_rank = 16
        if lora_alpha is None:
            lora_alpha = float(lora_rank)

        self.student = StudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            patch_swiglu_mlp_for_lora=False,
        )

        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        else:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        peft_model = self.student.da3.model.backbone.pretrained
        self.student.da3.model.backbone.pretrained = peft_model.merge_and_unload()
        print("Merged LoRA weights into base model for inference")

        self.student.eval()
        self.device = device

    def to(self, device):
        self.device = device
        self.student = self.student.to(device)
        return self

    def inference(
        self,
        image,
        export_dir=None,
        export_format="mini_npz",
        ref_view_strategy="first",
        **kwargs,
    ):
        return self.student.da3.inference(
            image=image,
            export_dir=export_dir,
            export_format=export_format,
            ref_view_strategy=ref_view_strategy,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def scene_slug(scene: str) -> str:
    return "-".join(scene.split("/")[-3:])


def sample_8_frames(scene_data, seed=43):
    """Sample up to 8 frames using the same logic as Evaluator._sample_frames."""
    num_frames = len(scene_data.image_files)
    if num_frames <= 8:
        return list(range(num_frames))
    random.seed(seed)
    indices = list(range(num_frames))
    random.shuffle(indices)
    return sorted(indices[:8])


def select_shared_4_indices(sampled_8_indices: List[int]) -> List[int]:
    """Select shared 4-view indices from even positions [0,2,4,6]."""
    if len(sampled_8_indices) < 7:
        return []
    return [sampled_8_indices[i] for i in EVEN_INDICES]


def save_results_npz(export_dir, depth, extrinsics, intrinsics, conf=None):
    """Save prediction results in the format expected by Evaluator.eval()."""
    output_file = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_dict = {"depth": np.round(depth, 8), "extrinsics": extrinsics, "intrinsics": intrinsics}
    if conf is not None:
        save_dict["conf"] = np.round(conf, 2)
    np.savez_compressed(output_file, **save_dict)


def save_gt_meta(export_dir, extrinsics, intrinsics, image_files):
    """Save GT metadata for evaluation and visualizers."""
    meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    np.savez_compressed(
        meta_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_files=np.array(image_files, dtype=object),
    )


def save_images(image_files, output_dir, prefix="frame"):
    """Save input images to output directory."""
    try:
        import imageio
    except Exception as e:
        raise RuntimeError(f"imageio is required to save sampled images: {e}")

    os.makedirs(output_dir, exist_ok=True)
    for i, img_path in enumerate(image_files):
        img = imageio.imread(img_path)
        out_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
        imageio.imwrite(out_path, img, quality=95)
        print(f"  Saved: {out_path}")


def _results_npz_path(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> str:
    return os.path.join(
        work_dir,
        exp_key,
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "mini_npz",
        "results.npz",
    )


def _exp_visualization_dir(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> str:
    # Include dataset_name for multi-dataset safety; keep old fallback in the viewer.
    return os.path.join(work_dir, exp_key, "visualizations", dataset_name, scene_slug(scene))


# ---------------------------------------------------------------------------
# Teacher inference: 8v encode -> extract 4v -> decode
# ---------------------------------------------------------------------------
def run_teacher_inference(api, images_8, images_4, ext_4, int_4, work_dir, dataset_name, scene):
    """
    Teacher: encode all 8 frames through backbone, extract even-indexed 4
    frame features, decode to get predictions for those 4 views.
    """
    print(f"\n{'=' * 60}")
    print("Teacher: 8v encode -> extract 4v [0,2,4,6] -> decode")
    print(f"{'=' * 60}")
    from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
    from depth_anything_3.utils.export.glb import export_to_glb

    # Preprocess all 8 images
    imgs_cpu, _, _ = api._preprocess_inputs(images_8, None, None)
    imgs, _, _ = api._prepare_model_inputs(imgs_cpu, None, None)

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        # Encode all 8 frames (cross-frame attention)
        with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
            feats, _aux_feats, H, W = api.model.forward_backbone_only(
                imgs, extrinsics=None, intrinsics=None, ref_view_strategy="first"
            )

        # Extract even-indexed 4 frame features
        reduced_feats = []
        for feat_tuple in feats:
            features, cam_tokens = feat_tuple
            reduced_feats.append(
                (
                    features[:, EVEN_INDICES, :, :],
                    cam_tokens[:, EVEN_INDICES, :] if cam_tokens is not None else None,
                )
            )

        # Decode with 4-frame features
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            output = api.model.forward_head_only(reduced_feats, H, W)

    prediction = api._convert_to_prediction(output)

    # Add processed images for visualization (even-indexed 4 from the 8)
    imgs_4v_cpu = imgs_cpu[EVEN_INDICES]  # (4, 3, H, W)
    prediction = api._add_processed_images(prediction, imgs_4v_cpu)

    # Save benchmark results
    export_dir = os.path.join(work_dir, "teacher", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, prediction.depth, prediction.extrinsics, prediction.intrinsics, prediction.conf)
    save_gt_meta(export_dir, ext_4, int_4, images_4)
    print(f"  Benchmark results saved to: {export_dir}")

    # Save visualizations
    vis_dir = _exp_visualization_dir(work_dir, "teacher", dataset_name, scene)
    os.makedirs(vis_dir, exist_ok=True)
    export_to_glb(prediction, vis_dir)
    export_to_depth_vis(prediction, vis_dir)
    print(f"  GLB point cloud saved to: {vis_dir}/scene.glb")
    print(f"  Depth visualizations saved to: {vis_dir}/depth_vis/")

    print("  [teacher] Inference complete")
    return prediction


# ---------------------------------------------------------------------------
# Teacher 4v inference: direct 4v (no 8v cross-attention)
# ---------------------------------------------------------------------------
def run_teacher_4v_inference(api, images_4, ext_4, int_4, work_dir, dataset_name, scene):
    """Teacher with only 4 shared views as input (baseline without 8v attention)."""
    print(f"\n{'=' * 60}")
    print("Teacher 4v: direct 4-view inference (no 8v cross-attention)")
    print(f"{'=' * 60}")
    from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
    from depth_anything_3.utils.export.glb import export_to_glb

    prediction = api.inference(images_4, ref_view_strategy="first")

    # Save benchmark results
    export_dir = os.path.join(work_dir, "teacher_4v", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, prediction.depth, prediction.extrinsics, prediction.intrinsics, prediction.conf)
    save_gt_meta(export_dir, ext_4, int_4, images_4)
    print(f"  Benchmark results saved to: {export_dir}")

    # Save visualizations
    vis_dir = _exp_visualization_dir(work_dir, "teacher_4v", dataset_name, scene)
    os.makedirs(vis_dir, exist_ok=True)
    export_to_glb(prediction, vis_dir)
    export_to_depth_vis(prediction, vis_dir)
    print(f"  GLB point cloud saved to: {vis_dir}/scene.glb")
    print(f"  Depth visualizations saved to: {vis_dir}/depth_vis/")

    print("  [teacher_4v] Inference complete")
    return prediction


# ---------------------------------------------------------------------------
# Student inference: direct 4v with LoRA
# ---------------------------------------------------------------------------
def run_student_inference(student_api, images_4, ext_4, int_4, work_dir, dataset_name, scene):
    """Student: direct 4-view inference with LoRA model."""
    print(f"\n{'=' * 60}")
    print("Student: LoRA 4v direct inference")
    print(f"{'=' * 60}")
    from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
    from depth_anything_3.utils.export.glb import export_to_glb

    prediction = student_api.inference(images_4, ref_view_strategy="first")

    # Save benchmark results
    export_dir = os.path.join(work_dir, "student", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, prediction.depth, prediction.extrinsics, prediction.intrinsics, prediction.conf)
    save_gt_meta(export_dir, ext_4, int_4, images_4)
    print(f"  Benchmark results saved to: {export_dir}")

    # Save visualizations
    vis_dir = _exp_visualization_dir(work_dir, "student", dataset_name, scene)
    os.makedirs(vis_dir, exist_ok=True)
    export_to_glb(prediction, vis_dir)
    export_to_depth_vis(prediction, vis_dir)
    print(f"  GLB point cloud saved to: {vis_dir}/scene.glb")
    print(f"  Depth visualizations saved to: {vis_dir}/depth_vis/")

    print("  [student] Inference complete")
    return prediction


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(work_dir: str, scenes_by_dataset: Dict[str, List[str]]) -> None:
    """Run pose + recon_unposed evaluation for teacher/teacher_4v/student."""
    from depth_anything_3.bench.evaluator import Evaluator

    for exp_name in ["teacher", "teacher_4v", "student"]:
        exp_work_dir = os.path.join(work_dir, exp_name)
        if not os.path.exists(os.path.join(exp_work_dir, "model_results")):
            print(f"  Skipping {exp_name} (no model_results found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'=' * 60}")

        for dataset_name, scenes in scenes_by_dataset.items():
            if not scenes:
                continue

            # Only evaluate scenes that have inference outputs to avoid fusion crashes.
            scenes_to_eval = []
            for scene in scenes:
                result_npz = _results_npz_path(work_dir, exp_name, dataset_name, scene)
                if os.path.exists(result_npz):
                    scenes_to_eval.append(scene)

            if not scenes_to_eval:
                print(f"  Skipping {exp_name}/{dataset_name} (no scenes with results.npz found)")
                continue

            print(f"\n  Dataset: {dataset_name} ({len(scenes_to_eval)} scenes)")
            evaluator = Evaluator(
                work_dir=exp_work_dir,
                datas=[dataset_name],
                modes=EVAL_MODES,
                max_frames=-1,
                scenes=scenes_to_eval,
            )
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)


def _load_metric_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _load_scene_metrics(work_dir: str, exp_name: str, dataset_name: str, mode: str) -> Dict[str, dict]:
    path = os.path.join(work_dir, exp_name, "metric_results", f"{dataset_name}_{mode}.json")
    data = _load_metric_json(path)
    return {k: v for k, v in data.items() if k != "mean" and isinstance(v, dict)}


def _choose_metric_key(
    scene_maps: List[Dict[str, dict]],
    preferred_keys: List[str],
    fallback_keys: List[str],
) -> Tuple[Optional[str], bool]:
    """
    Pick a metric key available in any scene map.
    Returns (key, higher_is_better).
    """
    all_entries = []
    for scene_map in scene_maps:
        all_entries.extend(scene_map.values())

    for key in preferred_keys:
        if any(key in entry for entry in all_entries):
            return key, True
    for key in fallback_keys:
        if any(key in entry for entry in all_entries):
            return key, False
    return None, True


def _metric_value(scene_metrics: Dict[str, dict], scene: str, key: Optional[str]) -> Optional[float]:
    if key is None:
        return None
    m = scene_metrics.get(scene)
    if not isinstance(m, dict):
        return None
    if key not in m:
        return None
    try:
        return float(m[key])
    except Exception:
        return None


def _delta(v: float, baseline: float, higher_is_better: bool) -> float:
    return (v - baseline) if higher_is_better else (baseline - v)


def _rel_delta(v: float, baseline: float, higher_is_better: bool) -> float:
    denom = abs(baseline) + 1e-9
    return _delta(v, baseline, higher_is_better) / denom


def summarize_scene_rankings(work_dir: str, scenes_by_dataset: Dict[str, List[str]]) -> None:
    """
    Print scenes where student (LoRA 4v) is better than teacher (8v->4v)
    on both pose and recon_unposed, sorted by combined relative improvement.
    """
    print(f"\n{'=' * 72}")
    print("Scene Ranking Report")
    print(f"{'=' * 72}")

    for dataset_name, scenes in scenes_by_dataset.items():
        if not scenes:
            continue

        teacher_pose = _load_scene_metrics(work_dir, "teacher", dataset_name, "pose")
        teacher_recon = _load_scene_metrics(work_dir, "teacher", dataset_name, "recon_unposed")
        teacher4_pose = _load_scene_metrics(work_dir, "teacher_4v", dataset_name, "pose")
        teacher4_recon = _load_scene_metrics(work_dir, "teacher_4v", dataset_name, "recon_unposed")
        student_pose = _load_scene_metrics(work_dir, "student", dataset_name, "pose")
        student_recon = _load_scene_metrics(work_dir, "student", dataset_name, "recon_unposed")

        # Use AUC@3 as requested for pose ranking.
        pose_key, pose_higher_is_better = _choose_metric_key(
            [teacher_pose, teacher4_pose, student_pose],
            preferred_keys=["auc03", "auc_03", "auc3", "auc_3"],
            fallback_keys=[],
        )
        recon_key, recon_higher_is_better = _choose_metric_key(
            [teacher_recon, teacher4_recon, student_recon],
            preferred_keys=["fscore", "f_score", "f-score", "precision", "recall"],
            fallback_keys=["overall", "acc", "comp"],
        )

        if pose_key is None or recon_key is None:
            print(f"\n[{dataset_name}] Missing usable metric keys for ranking, skipping.")
            continue

        rows = []
        for scene in scenes:
            t_pose = _metric_value(teacher_pose, scene, pose_key)
            t_recon = _metric_value(teacher_recon, scene, recon_key)
            t4_pose = _metric_value(teacher4_pose, scene, pose_key)
            t4_recon = _metric_value(teacher4_recon, scene, recon_key)
            s_pose = _metric_value(student_pose, scene, pose_key)
            s_recon = _metric_value(student_recon, scene, recon_key)

            if None in (t_pose, t_recon, t4_pose, t4_recon, s_pose, s_recon):
                continue

            # Keep only scenes where student outperforms teacher on both metrics.
            cond_student_vs_teacher = (
                _delta(s_pose, t_pose, pose_higher_is_better) > 0
                and _delta(s_recon, t_recon, recon_higher_is_better) > 0
            )

            if not cond_student_vs_teacher:
                continue

            # Ranking score: how much student improves over teacher.
            s_over_t_pose_rel = _rel_delta(s_pose, t_pose, pose_higher_is_better)
            s_over_t_recon_rel = _rel_delta(s_recon, t_recon, recon_higher_is_better)
            score = s_over_t_pose_rel + s_over_t_recon_rel

            rows.append(
                {
                    "scene": scene,
                    "score": score,
                    "s_over_t_pose_rel": s_over_t_pose_rel,
                    "s_over_t_recon_rel": s_over_t_recon_rel,
                    "student_pose": s_pose,
                    "teacher_pose": t_pose,
                    "student_recon": s_recon,
                    "teacher_recon": t_recon,
                }
            )

        rows.sort(key=lambda x: x["score"], reverse=True)

        print(f"\n[{dataset_name}]")
        print(f"Pose metric key: {pose_key} ({'higher' if pose_higher_is_better else 'lower'} is better)")
        print(f"Recon metric key: {recon_key} ({'higher' if recon_higher_is_better else 'lower'} is better)")

        if not rows:
            print("No scenes where student is better than teacher on both metrics.")
            continue

        print("Scenes where student > teacher on pose+recon (highest -> lowest improvement):")
        for i, row in enumerate(rows, 1):
            print(
                f"  {i:2d}. {row['scene']} | score={row['score']:.4f} | "
                f"student_over_teacher_rel(pose={row['s_over_t_pose_rel']:.4f}, recon={row['s_over_t_recon_rel']:.4f}) | "
                f"pose(s={row['student_pose']:.4f}, t={row['teacher_pose']:.4f}) | "
                f"recon(s={row['student_recon']:.4f}, t={row['teacher_recon']:.4f})"
            )


def _scenes_from_existing_metrics(work_dir: str, dataset_name: str) -> List[str]:
    """
    Best-effort scene discovery from saved metrics without loading datasets.
    """
    candidates = [
        os.path.join(work_dir, "teacher_4v", "metric_results", f"{dataset_name}_pose.json"),
        os.path.join(work_dir, "teacher", "metric_results", f"{dataset_name}_pose.json"),
        os.path.join(work_dir, "student", "metric_results", f"{dataset_name}_pose.json"),
    ]
    for path in candidates:
        data = _load_metric_json(path)
        if data:
            scenes = [k for k, v in data.items() if k != "mean" and isinstance(v, dict)]
            if scenes:
                return sorted(scenes)
    return []


def _resolve_scenes(dataset_name: str, dataset, args) -> List[str]:
    available = list(getattr(dataset, "SCENES", []))
    if not available:
        print(f"[WARN] Dataset {dataset_name} has no registered scenes")
        return []

    available_set = set(available)

    if args.all_scenes:
        return available

    if args.scenes:
        matched = [s for s in args.scenes if s in available_set]
        missing = [s for s in args.scenes if s not in available_set]
        if missing:
            print(f"[WARN] {dataset_name}: skipping unknown scenes: {missing}")
        return matched

    if args.scene:
        if args.scene in available_set:
            return [args.scene]
        print(f"[WARN] {dataset_name}: scene not found: {args.scene}")
        return []

    # Default behavior: benchmark all scenes for the selected dataset(s).
    if dataset_name == "hiroom" and DEFAULT_SCENE in available_set:
        # Keep default scene as first element for consistency in logs, but still run all.
        return [DEFAULT_SCENE] + [s for s in available if s != DEFAULT_SCENE]
    return available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DA3: Teacher (8v->4v extract) vs Student (LoRA 4v) on multiple datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to run (default: {' '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--all_scenes",
        action="store_true",
        help="Run on all scenes for each dataset",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Single scene to run (applied per selected dataset)",
    )
    parser.add_argument(
        "--scenes",
        action="append",
        default=None,
        help="Repeatable scene filter. Example: --scenes scene_a --scenes scene_b",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--work_dir", type=str, default="./results/teacher_student_4datasets")
    parser.add_argument("--image_dir", type=str, default="./results/images")
    parser.add_argument("--teacher_model", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--student_base_model", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/da3_tuned_lora_dist/hiroom/epoch_2_lora.pt",
    )
    parser.add_argument("--eval_only", action="store_true", help="Skip inference, only evaluate")
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only print the scene-ranking report from existing metric JSONs",
    )
    parser.add_argument("--teacher_only", action="store_true", help="Run teacher + teacher_4v only")
    parser.add_argument("--student_only", action="store_true", help="Run student only")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip inference for scene/experiment if results.npz already exists",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if args.report_only:
        scenes_by_dataset = {}
        for dataset_name in args.datasets:
            scenes = _scenes_from_existing_metrics(args.work_dir, dataset_name)
            if scenes:
                scenes_by_dataset[dataset_name] = scenes
        if not scenes_by_dataset:
            raise RuntimeError(
                "No scenes discovered from existing metric files. "
                "Run benchmark/evaluation first or pass the correct --work_dir/--datasets."
            )
        summarize_scene_rankings(args.work_dir, scenes_by_dataset=scenes_by_dataset)
        return

    from depth_anything_3.bench.registries import MV_REGISTRY

    # Resolve selected datasets and scenes.
    scenes_by_dataset: Dict[str, List[str]] = {}
    datasets = []
    for dataset_name in args.datasets:
        if not MV_REGISTRY.has(dataset_name):
            print(f"[WARN] Unknown dataset in MV_REGISTRY: {dataset_name}")
            continue
        dataset = MV_REGISTRY.get(dataset_name)()
        scenes = _resolve_scenes(dataset_name, dataset, args)
        if not scenes:
            print(f"[WARN] No scenes selected for dataset={dataset_name}; skipping")
            continue
        datasets.append((dataset_name, dataset))
        scenes_by_dataset[dataset_name] = scenes

    if not datasets:
        raise RuntimeError("No valid dataset/scene combination selected.")

    if args.eval_only:
        run_evaluation(args.work_dir, scenes_by_dataset=scenes_by_dataset)
        summarize_scene_rankings(args.work_dir, scenes_by_dataset=scenes_by_dataset)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models once
    teacher_api = None
    student_api = None

    if not args.student_only:
        from depth_anything_3.api import DepthAnything3

        print(f"\nLoading teacher model: {args.teacher_model}")
        teacher_api = DepthAnything3.from_pretrained(args.teacher_model).to(device)

    if not args.teacher_only:
        print(f"\nLoading student LoRA model: {args.lora_path}")
        student_api = LoRADepthAnything3(
            base_model=args.student_base_model,
            lora_path=args.lora_path,
        ).to(device)

    include_dataset_in_image_dir = len(datasets) > 1

    # Run scenes
    for dataset_name, dataset in datasets:
        scenes = scenes_by_dataset[dataset_name]
        print(f"\n{'#' * 72}")
        print(f"Dataset: {dataset_name} ({len(scenes)} scenes)")
        print(f"{'#' * 72}")

        for scene in scenes:
            try:
                scene_data = dataset.get_data(scene)
                num_frames = len(scene_data.image_files)
                print(f"\nScene {scene}: {num_frames} total frames")

                sampled_8_indices = sample_8_frames(scene_data, seed=args.seed)
                print(f"Sampled 8 frame indices (seed={args.seed}): {sampled_8_indices}")

                shared_4_indices = select_shared_4_indices(sampled_8_indices)
                if len(shared_4_indices) < 4:
                    print(
                        f"  [SKIP] {dataset_name}/{scene}: need >=7 sampled frames "
                        f"for [0,2,4,6], got {len(sampled_8_indices)}"
                    )
                    continue
                print(f"Shared 4 view indices (even from 8): {shared_4_indices}")

                # Save sampled input images
                all_8_images = [scene_data.image_files[i] for i in sampled_8_indices]
                if include_dataset_in_image_dir:
                    out_img_dir = os.path.join(args.image_dir, dataset_name, scene_slug(scene))
                else:
                    out_img_dir = os.path.join(args.image_dir, scene_slug(scene))
                print(f"\nSaving sampled images to {out_img_dir}")
                save_images(all_8_images, out_img_dir)

                # Shared 4-view data
                images_4 = [scene_data.image_files[i] for i in shared_4_indices]
                ext_4 = scene_data.extrinsics[shared_4_indices]
                int_4 = scene_data.intrinsics[shared_4_indices]

                # Teacher inference
                if teacher_api is not None:
                    teacher_npz = _results_npz_path(args.work_dir, "teacher", dataset_name, scene)
                    teacher4_npz = _results_npz_path(args.work_dir, "teacher_4v", dataset_name, scene)

                    if not (args.skip_existing and os.path.exists(teacher_npz)):
                        run_teacher_inference(
                            teacher_api,
                            all_8_images,
                            images_4,
                            ext_4,
                            int_4,
                            args.work_dir,
                            dataset_name,
                            scene,
                        )
                    else:
                        print(f"  [teacher] Skipping (exists): {teacher_npz}")

                    if not (args.skip_existing and os.path.exists(teacher4_npz)):
                        run_teacher_4v_inference(
                            teacher_api,
                            images_4,
                            ext_4,
                            int_4,
                            args.work_dir,
                            dataset_name,
                            scene,
                        )
                    else:
                        print(f"  [teacher_4v] Skipping (exists): {teacher4_npz}")

                # Student inference
                if student_api is not None:
                    student_npz = _results_npz_path(args.work_dir, "student", dataset_name, scene)
                    if not (args.skip_existing and os.path.exists(student_npz)):
                        run_student_inference(
                            student_api,
                            images_4,
                            ext_4,
                            int_4,
                            args.work_dir,
                            dataset_name,
                            scene,
                        )
                    else:
                        print(f"  [student] Skipping (exists): {student_npz}")

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\n[ERROR] {dataset_name}/{scene} failed:\n{e}")
                continue

    # Clean up models
    if teacher_api is not None:
        del teacher_api
    if student_api is not None:
        del student_api
    torch.cuda.empty_cache()

    # Evaluation
    run_evaluation(args.work_dir, scenes_by_dataset=scenes_by_dataset)
    summarize_scene_rankings(args.work_dir, scenes_by_dataset=scenes_by_dataset)


if __name__ == "__main__":
    main()

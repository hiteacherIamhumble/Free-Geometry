#!/usr/bin/env python3
"""
Benchmark: Teacher (VGGT-1B 8v) vs Student (LoRA 4v) on multiple datasets.

Datasets supported by default:
  - scannetpp
  - hiroom
  - 7scenes
  - eth3d

For each scene, this script:
  1) Samples up to 8 frames (same seed logic as Evaluator)
  2) Builds a shared 4-view subset from even positions [0, 2, 4, 6]
  3) Runs:
       - teacher        : VGGT 8v (all frames)
       - teacher_4v     : VGGT 4v baseline (shared subset)
       - student        : VGGT LoRA 4v (shared subset)
  4) Saves predictions + GT meta + visualizations
  5) Runs pose + recon_unposed evaluation

Usage:
    python scripts/benchmark_teacher_student_all_datasets_vggt.py
    python scripts/benchmark_teacher_student_all_datasets_vggt.py --datasets hiroom
    python scripts/benchmark_teacher_student_all_datasets_vggt.py --all_scenes
    python scripts/benchmark_teacher_student_all_datasets_vggt.py --eval_only
    python scripts/benchmark_teacher_student_all_datasets_vggt.py --report_only
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATASETS = ["scannetpp", "hiroom", "7scenes", "eth3d"]
DEFAULT_SCENE = "20241230/828805/cam_sampled_06"  # used when dataset=hiroom
EVAL_MODES = ["pose", "recon_unposed"]
EVEN_INDICES = [0, 2, 4, 6]


# ---------------------------------------------------------------------------
# VGGT LoRA model wrapper
# ---------------------------------------------------------------------------
class LoRAVGGT:
    """Wrapper to make VGGT LoRA StudentModel compatible with the benchmark API."""

    def __init__(
        self,
        base_model="facebook/vggt-1b",
        lora_path=None,
        lora_rank=16,
        lora_alpha=16.0,
        lora_layers_start=12,
        device="cuda",
        image_size=504,
    ):
        from vggt.vggt.distillation.models import VGGTStudentModel
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        print(f"Loading VGGT base model: {base_model}")
        print(f"Loading LoRA weights: {lora_path}")

        self.image_size = image_size
        self.device = device
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri

        # Create student model with LoRA
        lora_layers = list(range(lora_layers_start, 24))
        self.student = VGGTStudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers,
        )

        # Load LoRA weights
        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        elif lora_path:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        self.student.eval()

    def to(self, device):
        self.device = device
        self.student = self.student.to(device)
        return self

    def _load_images(self, image_files):
        """Load and preprocess images for VGGT."""
        PATCH_SIZE = 14
        images = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            scale = self.image_size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            new_h = max(PATCH_SIZE, round(new_h / PATCH_SIZE) * PATCH_SIZE)
            new_w = max(PATCH_SIZE, round(new_w / PATCH_SIZE) * PATCH_SIZE)

            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

            img = img.astype(np.float32) / 255.0
            images.append(img)

        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)  # [1, S, 3, H, W]

    def inference(self, image_files, **kwargs):
        """Run VGGT inference."""
        images = self._load_images(image_files).to(self.device)

        # Access underlying VGGT model
        if hasattr(self.student.vggt, 'base_model'):
            vggt_model = self.student.vggt.base_model.model
        else:
            vggt_model = self.student.vggt

        with torch.no_grad():
            predictions = vggt_model(images)

        # Convert pose_enc to extrinsics/intrinsics
        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)

            _, _, _, H, W = images.shape
            ext, intr = self.pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(H, W),
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)
            predictions["intrinsics"] = intr.squeeze(0)

        return predictions


class BaseVGGT:
    """Wrapper for base VGGT model (without LoRA)."""

    def __init__(
        self,
        model_name="facebook/vggt-1b",
        device="cuda",
        image_size=504,
    ):
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        print(f"Loading base VGGT model: {model_name}")

        self.image_size = image_size
        self.device = device
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri

        self.vggt = VGGT.from_pretrained(model_name)
        self.vggt.eval()

    def to(self, device):
        self.device = device
        self.vggt = self.vggt.to(device)
        return self

    def _load_images(self, image_files):
        """Load and preprocess images for VGGT."""
        PATCH_SIZE = 14
        images = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            scale = self.image_size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            new_h = max(PATCH_SIZE, round(new_h / PATCH_SIZE) * PATCH_SIZE)
            new_w = max(PATCH_SIZE, round(new_w / PATCH_SIZE) * PATCH_SIZE)

            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

            img = img.astype(np.float32) / 255.0
            images.append(img)

        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)

    def inference(self, image_files, **kwargs):
        """Run VGGT inference."""
        images = self._load_images(image_files).to(self.device)

        with torch.no_grad():
            predictions = self.vggt(images)

        # Convert pose_enc to extrinsics/intrinsics
        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)

            _, _, _, H, W = images.shape
            ext, intr = self.pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(H, W),
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)
            predictions["intrinsics"] = intr.squeeze(0)

        return predictions


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


def _export_vggt_visualizations(depth, extrinsics, intrinsics, image_files, vis_dir):
    """
    Export VGGT predictions to GLB and depth_vis formats compatible with the viewer.

    Args:
        depth: [S, H, W] depth maps
        extrinsics: [S, 3, 4] or [S, 4, 4] camera extrinsics
        intrinsics: [S, 3, 3] camera intrinsics
        image_files: List of image file paths
        vis_dir: Output directory for visualizations
    """
    from depth_anything_3.specs import Prediction
    from depth_anything_3.utils.export.depth_vis import export_to_depth_vis
    from depth_anything_3.utils.export.glb import export_to_glb

    # Load and preprocess images
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    images = []
    for img_path in image_files:
        img = imageio.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        images.append(img)

    images = np.stack(images, axis=0)  # [S, H, W, 3]

    # Ensure depth has correct shape [S, H, W]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)  # [S, H, W, 1] -> [S, H, W]

    # Resize images to match depth resolution if needed
    if images.shape[1:3] != depth.shape[1:3]:
        import cv2
        resized_images = []
        for img in images:
            resized = cv2.resize(img, (depth.shape[2], depth.shape[1]), interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized)
        images = np.stack(resized_images, axis=0)

    # Convert extrinsics to [S, 4, 4] if needed
    if extrinsics.shape[-1] == 4 and extrinsics.shape[-2] == 3:
        # [S, 3, 4] -> [S, 4, 4]
        ext_4x4 = np.zeros((extrinsics.shape[0], 4, 4), dtype=extrinsics.dtype)
        ext_4x4[:, :3, :] = extrinsics
        ext_4x4[:, 3, 3] = 1.0
        extrinsics = ext_4x4

    # Create fake confidence (VGGT doesn't output confidence)
    conf = np.ones_like(depth) * 1.5  # Above threshold to keep all points

    # Create Prediction object
    prediction = Prediction(
        depth=depth,
        is_metric=1,
        conf=conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        processed_images=images.astype(np.uint8),
    )

    # Export GLB and depth_vis
    os.makedirs(vis_dir, exist_ok=True)
    export_to_glb(prediction, vis_dir)
    export_to_depth_vis(prediction, vis_dir)


# ---------------------------------------------------------------------------
# VGGT Teacher inference: 8v full scene
# ---------------------------------------------------------------------------
def run_teacher_inference(api, images_8, images_4, ext_4, int_4, work_dir, dataset_name, scene):
    """
    VGGT Teacher: run inference on all 8 frames.
    """
    print(f"\n{'=' * 60}")
    print("VGGT Teacher: 8v full scene inference")
    print(f"{'=' * 60}")

    # Run VGGT on all 8 frames
    predictions = api.inference(images_8)

    # Extract predictions for the 4 shared views (even indices)
    # VGGT returns [B, S, H, W] for depth, [B, S, 3, 4] for extrinsics, [B, S, 3, 3] for intrinsics
    depth_8v = predictions["depth"].squeeze(0).cpu().numpy()  # [S, H, W]
    ext_8v = predictions["extrinsics"].squeeze(0).cpu().numpy()  # [S, 3, 4]
    intr_8v = predictions["intrinsics"].squeeze(0).cpu().numpy()  # [S, 3, 3]

    # Select even-indexed frames
    depth_4v = depth_8v[EVEN_INDICES]
    ext_4v = ext_8v[EVEN_INDICES]
    intr_4v = intr_8v[EVEN_INDICES]

    # Save benchmark results (4-view subset for fair comparison)
    export_dir = os.path.join(work_dir, "teacher", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, depth_4v, ext_4v, intr_4v, conf=None)
    save_gt_meta(export_dir, ext_4, int_4, images_4)
    print(f"  Benchmark results saved to: {export_dir}")

    # Save visualizations (4-view subset)
    vis_dir = _exp_visualization_dir(work_dir, "teacher", dataset_name, scene)
    os.makedirs(vis_dir, exist_ok=True)
    _export_vggt_visualizations(depth_4v, ext_4v, intr_4v, images_4, vis_dir)
    print(f"  GLB point cloud saved to: {vis_dir}/scene.glb")
    print(f"  Depth visualizations saved to: {vis_dir}/depth_vis/")

    print("  [teacher] Inference complete")
    return predictions


# ---------------------------------------------------------------------------
# VGGT Teacher 4v inference: direct 4v baseline
# ---------------------------------------------------------------------------
def run_teacher_4v_inference(api, images_4, ext_4, int_4, work_dir, dataset_name, scene):
    """VGGT Teacher with only 4 shared views as input (baseline)."""
    print(f"\n{'=' * 60}")
    print("VGGT Teacher 4v: direct 4-view inference")
    print(f"{'=' * 60}")

    predictions = api.inference(images_4)

    depth = predictions["depth"].squeeze(0).cpu().numpy()
    ext = predictions["extrinsics"].squeeze(0).cpu().numpy()
    intr = predictions["intrinsics"].squeeze(0).cpu().numpy()

    # Save benchmark results
    export_dir = os.path.join(work_dir, "teacher_4v", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, depth, ext, intr, conf=None)
    save_gt_meta(export_dir, ext_4, int_4, images_4)
    print(f"  Benchmark results saved to: {export_dir}")

    # Save visualizations
    vis_dir = _exp_visualization_dir(work_dir, "teacher_4v", dataset_name, scene)
    os.makedirs(vis_dir, exist_ok=True)
    _export_vggt_visualizations(depth, ext, intr, images_4, vis_dir)
    print(f"  GLB point cloud saved to: {vis_dir}/scene.glb")
    print(f"  Depth visualizations saved to: {vis_dir}/depth_vis/")

    print("  [teacher_4v] Inference complete")
    return predictions


# ---------------------------------------------------------------------------
# VGGT Student inference: direct 4v with LoRA
# ---------------------------------------------------------------------------
def run_student_inference(student_api, images_4, ext_4, int_4, work_dir, dataset_name, scene):
    """VGGT Student: direct 4-view inference with LoRA model."""
    print(f"\n{'=' * 60}")
    print("VGGT Student: LoRA 4v direct inference")
    print(f"{'=' * 60}")

    predictions = student_api.inference(images_4)

    depth = predictions["depth"].squeeze(0).cpu().numpy()
    ext = predictions["extrinsics"].squeeze(0).cpu().numpy()
    intr = predictions["intrinsics"].squeeze(0).cpu().numpy()

    # Save benchmark results
    export_dir = os.path.join(work_dir, "student", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, depth, ext, intr, conf=None)
    save_gt_meta(export_dir, ext_4, int_4, images_4)
    print(f"  Benchmark results saved to: {export_dir}")

    # Save visualizations
    vis_dir = _exp_visualization_dir(work_dir, "student", dataset_name, scene)
    os.makedirs(vis_dir, exist_ok=True)
    _export_vggt_visualizations(depth, ext, intr, images_4, vis_dir)
    print(f"  GLB point cloud saved to: {vis_dir}/scene.glb")
    print(f"  Depth visualizations saved to: {vis_dir}/depth_vis/")

    print("  [student] Inference complete")
    return predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(work_dir: str, scenes_by_dataset: Dict[str, List[str]]) -> None:
    """Run pose + recon_unposed evaluation for teacher/teacher_4v/student using VGGT evaluator."""
    from vggt.vggt.bench.evaluator import VGGTEvaluator

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
            evaluator = VGGTEvaluator(
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
        description="VGGT: Teacher (8v) vs Student (LoRA 4v) on multiple datasets"
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
    parser.add_argument("--work_dir", type=str, default="./results/vggt_teacher_student_4datasets")
    parser.add_argument("--image_dir", type=str, default="./results/images")
    parser.add_argument("--teacher_model", type=str, default="facebook/vggt-1b")
    parser.add_argument("--student_base_model", type=str, default="facebook/vggt-1b")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/vggt_distill/epoch_1_lora.pt",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (must match training)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha (must match training)",
    )
    parser.add_argument(
        "--lora_layers_start",
        type=int,
        default=12,
        help="First layer with LoRA (must match training)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=504,
        help="VGGT input image size (longest side)",
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

    from vggt.vggt.bench.registries import VGGT_MV_REGISTRY as MV_REGISTRY

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
        print(f"\nLoading VGGT teacher model: {args.teacher_model}")
        teacher_api = BaseVGGT(
            model_name=args.teacher_model,
            image_size=args.image_size,
        ).to(device)

    if not args.teacher_only:
        print(f"\nLoading VGGT student LoRA model: {args.lora_path}")
        student_api = LoRAVGGT(
            base_model=args.student_base_model,
            lora_path=args.lora_path,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_layers_start=args.lora_layers_start,
            image_size=args.image_size,
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
                import traceback
                traceback.print_exc()
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

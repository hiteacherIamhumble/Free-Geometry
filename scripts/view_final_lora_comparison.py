#!/usr/bin/env python3
"""
Viewer for fixed-view baseline vs LoRA comparison results.

This targets the regenerated fixed-input 8v/16v/32v benchmark layout:
  - baseline:
      results/multiview_da3_hiroom_scannetpp_base_fixed/{8v,16v,32v}/
      results/multiview_vggt_hiroom_scannetpp_base_fixed/{8v,16v,32v}/
  - LoRA:
      results/multiview_da3_hiroom_scannetpp_lora_fixed/{8v,16v,32v}/
      results/multiview_vggt_hiroom_scannetpp_lora_fixed/{8v,16v,32v}/

It compares:
  - original model reconstruction
  - LoRA ("Free Geometry") reconstruction
  - GT threshold overlays and depth-threshold maps

Unlike scripts/view_pointclouds.py, HiRoom threshold visualization here uses the
original evaluation threshold and does NOT double it.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gradio as gr
except ImportError:
    gr = None

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import view_pointclouds as vp


EXPERIMENTS: Dict[str, str] = {
    "baseline": "Original Model",
    "lora": "LoRA (Free Geometry)",
}
NO_SCENE_SENTINEL = "__NO_SCENE_AVAILABLE__"
DEFAULT_WORK_ROOTS = {
    "da3": "./results/multiview_da3_hiroom_scannetpp_lora_fixed",
    "vggt": "./results/multiview_vggt_hiroom_scannetpp_lora_fixed",
}
DEFAULT_VIEWER_MAX_POINTS = vp.DEFAULT_VIEWER_MAX_POINTS


def _seed_tag(seed: str | int) -> str:
    s = str(seed)
    return s if s.startswith("seed") else f"seed{s}"


def _comparison_roots(model_family: str, frame_count: int) -> Dict[str, str]:
    frame_count = int(frame_count)
    if frame_count not in (8, 16, 32):
        raise ValueError(f"Unsupported frame_count={frame_count}. This viewer supports only 8, 16, 32.")
    baseline_root = os.path.join(PROJECT_ROOT, "results", f"multiview_{model_family}_hiroom_scannetpp_base_fixed")
    lora_root = os.path.join(PROJECT_ROOT, "results", f"multiview_{model_family}_hiroom_scannetpp_lora_fixed")
    return {"baseline": baseline_root, "lora": lora_root}


def _run_dir(model_family: str, exp_key: str, frame_count: int, dataset_name: str, seed: str | int) -> str:
    frame_count = int(frame_count)
    roots = _comparison_roots(model_family, frame_count)
    root = roots[exp_key]
    return os.path.join(root, f"{frame_count}v")


def _metric_json_path(
    model_family: str,
    exp_key: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    mode: str,
) -> str:
    return os.path.join(_run_dir(model_family, exp_key, frame_count, dataset_name, seed), "metric_results", f"{dataset_name}_{mode}.json")


def _results_npz_path(model_family: str, exp_key: str, frame_count: int, dataset_name: str, seed: str | int, scene: str) -> str:
    return os.path.join(
        _run_dir(model_family, exp_key, frame_count, dataset_name, seed),
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "mini_npz",
        "results.npz",
    )


def _gt_meta_path(model_family: str, exp_key: str, frame_count: int, dataset_name: str, seed: str | int, scene: str) -> str:
    return os.path.join(
        _run_dir(model_family, exp_key, frame_count, dataset_name, seed),
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "gt_meta.npz",
    )


def _fused_ply_path(model_family: str, exp_key: str, frame_count: int, dataset_name: str, seed: str | int, scene: str) -> str:
    return os.path.join(
        _run_dir(model_family, exp_key, frame_count, dataset_name, seed),
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "fuse",
        "pcd.ply",
    )


def _viewer_cache_dir(
    work_root: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    viewer_max_points: int = DEFAULT_VIEWER_MAX_POINTS,
) -> str:
    return os.path.join(
        work_root,
        "_viewer_cache_v2",
        vp._cache_tag_for_max_points(viewer_max_points),
        f"frames_{int(frame_count)}",
        dataset_name,
        _seed_tag(seed),
    )


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_gt_meta(model_family: str, frame_count: int, dataset_name: str, seed: str | int, scene: str) -> Optional[dict]:
    for exp_key in ["baseline", "lora"]:
        meta = _load_gt_meta_for_exp(model_family, exp_key, frame_count, dataset_name, seed, scene)
        if meta is not None:
            return meta
    return None


def _load_gt_meta_for_exp(
    model_family: str,
    exp_key: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    scene: str,
) -> Optional[dict]:
    path = _gt_meta_path(model_family, exp_key, frame_count, dataset_name, seed, scene)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return {
            "meta_path": path,
            "meta_source_exp": exp_key,
            "image_files": [str(x) for x in list(data.get("image_files", []))],
            "extrinsics": np.asarray(data["extrinsics"]),
            "intrinsics": np.asarray(data["intrinsics"]),
        }
    except Exception:
        return None


def _list_scenes(model_family: str, frame_count: int, dataset_name: str, seed: str | int) -> List[str]:
    for exp_key in EXPERIMENTS.keys():
        data = _load_json(_metric_json_path(model_family, exp_key, frame_count, dataset_name, seed, "pose"))
        scenes = sorted(k for k, v in data.items() if k != "mean" and isinstance(v, dict))
        if scenes:
            return scenes
    return []


def _metric_value(metric_json: dict, scene: str, key: str) -> Optional[float]:
    entry = metric_json.get(scene) or metric_json.get("mean") or {}
    if not isinstance(entry, dict):
        return None
    value = entry.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _format_metrics_md(model_family: str, frame_count: int, dataset_name: str, seed: str | int, scene: str) -> str:
    cols = [EXPERIMENTS["baseline"], EXPERIMENTS["lora"]]
    pose_jsons = {
        exp_key: _load_json(_metric_json_path(model_family, exp_key, frame_count, dataset_name, seed, "pose"))
        for exp_key in EXPERIMENTS.keys()
    }
    recon_jsons = {
        exp_key: _load_json(_metric_json_path(model_family, exp_key, frame_count, dataset_name, seed, "recon_unposed"))
        for exp_key in EXPERIMENTS.keys()
    }

    def fmt(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:.4f}"

    rows = []
    for label, src, key in [
        ("AUC03", pose_jsons, "auc03"),
        ("AUC30", pose_jsons, "auc30"),
        ("F-score", recon_jsons, "fscore"),
        ("Overall", recon_jsons, "overall"),
    ]:
        row = [label]
        for exp_key in ["baseline", "lora"]:
            row.append(fmt(_metric_value(src[exp_key], scene, key)))
        rows.append(row)

    header = "| Metric | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header, sep, body])


def _ensure_gt_glb(
    model_family: str,
    work_root: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    scene: str,
    *,
    viewer_max_points: int = DEFAULT_VIEWER_MAX_POINTS,
) -> Optional[str]:
    gt_meta = _load_gt_meta(model_family, frame_count, dataset_name, seed, scene)
    if gt_meta is None:
        return None

    gt_ply = vp._ensure_gt_fuse_ply_4v(work_root, dataset_name, scene, gt_meta)
    if gt_ply is None or not os.path.exists(gt_ply):
        return None

    out_glb = os.path.join(
        _viewer_cache_dir(work_root, frame_count, dataset_name, seed, viewer_max_points),
        "gt",
        vp._scene_slug(scene),
        "scene.glb",
    )
    return vp._ensure_glb_from_ply_with_cameras(
        ply_path=gt_ply,
        out_glb_path=out_glb,
        extrinsics_w2c=gt_meta["extrinsics"],
        intrinsics=gt_meta["intrinsics"],
        image_sizes=vp._read_image_sizes(gt_meta["image_files"]),
        num_max_points=viewer_max_points,
    )


def _ensure_exp_fused_glb(
    model_family: str,
    work_root: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    scene: str,
    exp_key: str,
    *,
    viewer_max_points: int = DEFAULT_VIEWER_MAX_POINTS,
) -> Optional[str]:
    gt_meta = _load_gt_meta(model_family, frame_count, dataset_name, seed, scene)
    if gt_meta is None:
        return None

    fuse_ply = _fused_ply_path(model_family, exp_key, frame_count, dataset_name, seed, scene)
    out_glb = os.path.join(
        _viewer_cache_dir(work_root, frame_count, dataset_name, seed, viewer_max_points),
        exp_key,
        "visualizations_fuse",
        vp._scene_slug(scene),
        "scene.glb",
    )
    return vp._ensure_glb_from_ply_with_cameras(
        ply_path=fuse_ply,
        out_glb_path=out_glb,
        extrinsics_w2c=gt_meta["extrinsics"],
        intrinsics=gt_meta["intrinsics"],
        image_sizes=vp._read_image_sizes(gt_meta["image_files"]),
        num_max_points=viewer_max_points,
    )


def _comparison_glb_path(
    work_root: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    name: str,
    scene: str,
    *,
    viewer_max_points: int = DEFAULT_VIEWER_MAX_POINTS,
) -> str:
    return os.path.join(
        _viewer_cache_dir(work_root, frame_count, dataset_name, seed, viewer_max_points),
        "comparisons",
        name,
        vp._scene_slug(scene),
        "scene.glb",
    )


def _ensure_exp_vs_gt_threshold_overlay_glb(
    model_family: str,
    work_root: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    scene: str,
    *,
    exp_key: str,
    out_name: str,
    viewer_max_points: int = DEFAULT_VIEWER_MAX_POINTS,
) -> Optional[str]:
    threshold, down_sample = vp._dataset_recon_eval_params(dataset_name)
    if threshold is None:
        return None

    gt_meta = _load_gt_meta(model_family, frame_count, dataset_name, seed, scene)
    if gt_meta is None:
        return None

    pred_ply = _fused_ply_path(model_family, exp_key, frame_count, dataset_name, seed, scene)
    if not os.path.exists(pred_ply):
        return None

    out_glb = _comparison_glb_path(
        work_root,
        frame_count,
        dataset_name,
        seed,
        out_name,
        scene,
        viewer_max_points=viewer_max_points,
    )
    if os.path.exists(out_glb):
        return out_glb

    try:
        import open3d as o3d
        import trimesh
        from depth_anything_3.bench.utils import nn_correspondance
        from depth_anything_3.utils.export.glb import (
            _add_cameras_to_scene,
            _compute_alignment_transform_first_cam_glTF_center_by_points,
            _estimate_scene_scale,
        )
    except Exception:
        return None

    gt_pcd = vp._load_scene_gt_pointcloud_for_overlay(work_root, dataset_name, scene, gt_meta=gt_meta)
    if gt_pcd is None:
        return None

    pred_pcd = o3d.io.read_point_cloud(pred_ply)
    if down_sample is not None and down_sample > 0:
        pred_pcd = pred_pcd.voxel_down_sample(down_sample)
        gt_pcd = gt_pcd.voxel_down_sample(down_sample)

    pred_points = np.asarray(pred_pcd.points, dtype=np.float32)
    gt_points = np.asarray(gt_pcd.points, dtype=np.float32)
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return None

    rng = np.random.default_rng(seed=42)
    if pred_points.shape[0] > 350_000:
        pred_points = pred_points[rng.choice(pred_points.shape[0], size=350_000, replace=False)]
    if gt_points.shape[0] > 350_000:
        gt_points = gt_points[rng.choice(gt_points.shape[0], size=350_000, replace=False)]
    pred_points, _ = vp._cap_point_cloud_arrays(pred_points, None, num_max_points=viewer_max_points)
    gt_points, _ = vp._cap_point_cloud_arrays(gt_points, None, num_max_points=viewer_max_points)

    dist_pred_to_gt = nn_correspondance(gt_points, pred_points)
    dist_gt_to_pred = nn_correspondance(pred_points, gt_points)
    pred_good = dist_pred_to_gt < float(threshold)
    gt_good = dist_gt_to_pred < float(threshold)

    pred_colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (pred_points.shape[0], 1))
    pred_colors[pred_good] = np.array([175, 175, 175], dtype=np.uint8)
    gt_colors = np.tile(np.array([160, 0, 0], dtype=np.uint8), (gt_points.shape[0], 1))
    gt_colors[gt_good] = np.array([120, 120, 120], dtype=np.uint8)

    A = _compute_alignment_transform_first_cam_glTF_center_by_points(gt_meta["extrinsics"][0], gt_points)
    pred_points_t = trimesh.transform_points(pred_points, A)
    gt_points_t = trimesh.transform_points(gt_points, A)

    scene_obj = trimesh.Scene()
    if scene_obj.metadata is None:
        scene_obj.metadata = {}
    scene_obj.metadata["hf_alignment"] = A
    scene_obj.add_geometry(trimesh.points.PointCloud(vertices=gt_points_t, colors=gt_colors))
    scene_obj.add_geometry(trimesh.points.PointCloud(vertices=pred_points_t, colors=pred_colors))

    all_points = np.concatenate([gt_points_t, pred_points_t], axis=0)
    scene_scale = _estimate_scene_scale(all_points, fallback=1.0)
    _add_cameras_to_scene(
        scene=scene_obj,
        K=gt_meta["intrinsics"],
        ext_w2c=gt_meta["extrinsics"],
        image_sizes=vp._read_image_sizes(gt_meta["image_files"]),
        scale=scene_scale * 0.03,
    )

    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    scene_obj.export(out_glb)
    return out_glb


def _ensure_lora_vs_baseline_delta_overlay_glb(
    model_family: str,
    work_root: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    scene: str,
    *,
    viewer_max_points: int = DEFAULT_VIEWER_MAX_POINTS,
) -> Optional[str]:
    threshold, down_sample = vp._dataset_recon_eval_params(dataset_name)
    if threshold is None:
        return None

    gt_meta = _load_gt_meta(model_family, frame_count, dataset_name, seed, scene)
    if gt_meta is None:
        return None

    baseline_ply = _fused_ply_path(model_family, "baseline", frame_count, dataset_name, seed, scene)
    lora_ply = _fused_ply_path(model_family, "lora", frame_count, dataset_name, seed, scene)
    if not os.path.exists(baseline_ply) or not os.path.exists(lora_ply):
        return None

    out_glb = _comparison_glb_path(
        work_root,
        frame_count,
        dataset_name,
        seed,
        "lora_vs_baseline_delta_threshold",
        scene,
        viewer_max_points=viewer_max_points,
    )
    if os.path.exists(out_glb):
        return out_glb

    try:
        import open3d as o3d
        import trimesh
        from scipy.spatial import cKDTree
        from depth_anything_3.utils.export.glb import (
            _add_cameras_to_scene,
            _compute_alignment_transform_first_cam_glTF_center_by_points,
            _estimate_scene_scale,
        )
    except Exception:
        return None

    gt_pcd = vp._load_scene_gt_pointcloud_for_overlay(work_root, dataset_name, scene, gt_meta=gt_meta)
    if gt_pcd is None:
        return None

    baseline_pcd = o3d.io.read_point_cloud(baseline_ply)
    lora_pcd = o3d.io.read_point_cloud(lora_ply)
    if down_sample is not None and down_sample > 0:
        baseline_pcd = baseline_pcd.voxel_down_sample(down_sample)
        lora_pcd = lora_pcd.voxel_down_sample(down_sample)
        gt_pcd = gt_pcd.voxel_down_sample(down_sample)

    baseline_points = np.asarray(baseline_pcd.points, dtype=np.float32)
    lora_points = np.asarray(lora_pcd.points, dtype=np.float32)
    gt_points = np.asarray(gt_pcd.points, dtype=np.float32)
    baseline_points, _ = vp._cap_point_cloud_arrays(baseline_points, None, num_max_points=viewer_max_points)
    lora_points, _ = vp._cap_point_cloud_arrays(lora_points, None, num_max_points=viewer_max_points)
    gt_points, _ = vp._cap_point_cloud_arrays(gt_points, None, num_max_points=viewer_max_points)
    if baseline_points.shape[0] == 0 or lora_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return None

    gt_tree = cKDTree(gt_points)
    lora_to_gt, _ = gt_tree.query(lora_points, k=1, workers=-1)
    base_to_gt, _ = gt_tree.query(baseline_points, k=1, workers=-1)

    base_tree = cKDTree(baseline_points)
    _, lora_to_base_idx = base_tree.query(lora_points, k=1, workers=-1)

    colors = np.zeros((lora_points.shape[0], 3), dtype=np.uint8)
    thr = float(threshold)
    for i in range(lora_points.shape[0]):
        s_err = float(lora_to_gt[i])
        t_err = float(base_to_gt[lora_to_base_idx[i]])
        s_in = s_err <= thr
        t_in = t_err <= thr
        if s_in and t_in:
            colors[i] = np.array([170, 170, 170], dtype=np.uint8)
        elif s_in and not t_in:
            colors[i] = np.array([0, 100, 255], dtype=np.uint8)
        elif (not s_in) and t_in:
            colors[i] = np.array([255, 0, 0], dtype=np.uint8)
        elif s_err < t_err:
            colors[i] = np.array([0, 100, 255], dtype=np.uint8)
        else:
            colors[i] = np.array([255, 0, 0], dtype=np.uint8)

    A = _compute_alignment_transform_first_cam_glTF_center_by_points(gt_meta["extrinsics"][0], gt_points)
    lora_points_t = trimesh.transform_points(lora_points, A)
    gt_points_t = trimesh.transform_points(gt_points, A)

    scene_obj = trimesh.Scene()
    if scene_obj.metadata is None:
        scene_obj.metadata = {}
    scene_obj.metadata["hf_alignment"] = A
    scene_obj.add_geometry(trimesh.points.PointCloud(vertices=lora_points_t, colors=colors))

    scene_scale = _estimate_scene_scale(np.concatenate([lora_points_t, gt_points_t], axis=0), fallback=1.0)
    _add_cameras_to_scene(
        scene=scene_obj,
        K=gt_meta["intrinsics"],
        ext_w2c=gt_meta["extrinsics"],
        image_sizes=vp._read_image_sizes(gt_meta["image_files"]),
        scale=scene_scale * 0.03,
    )

    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    scene_obj.export(out_glb)
    return out_glb


def _load_prediction_depths(model_family: str, exp_key: str, frame_count: int, dataset_name: str, seed: str | int, scene: str):
    path = _results_npz_path(model_family, exp_key, frame_count, dataset_name, seed, scene)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        depth = data.get("depth")
        return np.asarray(depth, dtype=np.float32) if depth is not None else None
    except Exception:
        return None


def _prepare_pred_depth_gallery(model_family: str, work_root: str, exp_key: str, frame_count: int, dataset_name: str, seed: str | int, scene: str):
    gt_meta = _load_gt_meta_for_exp(model_family, exp_key, frame_count, dataset_name, seed, scene)
    if gt_meta is None:
        gt_meta = _load_gt_meta(model_family, frame_count, dataset_name, seed, scene)
    if gt_meta is None:
        return []

    try:
        import cv2
    except Exception:
        return []

    pred_depths = _load_prediction_depths(model_family, exp_key, frame_count, dataset_name, seed, scene)
    if pred_depths is None:
        return []

    out = []
    for idx, img_path in enumerate(gt_meta["image_files"]):
        if idx >= len(pred_depths):
            break
        rgb = vp._read_rgb_image(img_path)
        if rgb is None:
            continue
        rgb = rgb.astype(np.uint8)
        depth = np.asarray(pred_depths[idx], dtype=np.float32)
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            dmin = float(np.percentile(depth[valid], 5))
            dmax = float(np.percentile(depth[valid], 95))
        else:
            dmin, dmax = 0.0, 1.0

        depth_rgb = vp._colorize_scalar_map(depth, valid, vmin=dmin, vmax=dmax, cmap_name="turbo")
        out.append(np.concatenate([rgb, depth_rgb], axis=1))
    return out


def _prepare_depth_error_galleries(model_family: str, work_root: str, frame_count: int, dataset_name: str, seed: str | int, scene: str):
    try:
        import cv2
    except Exception as e:
        return [], [], [], f"Missing deps for depth error maps: {e}"

    gt_metas = {
        "baseline": _load_gt_meta_for_exp(model_family, "baseline", frame_count, dataset_name, seed, scene),
        "lora": _load_gt_meta_for_exp(model_family, "lora", frame_count, dataset_name, seed, scene),
    }
    if gt_metas["baseline"] is None and gt_metas["lora"] is None:
        return [], [], [], "No GT meta found for this scene."

    pred_depths = {
        "baseline": _load_prediction_depths(model_family, "baseline", frame_count, dataset_name, seed, scene),
        "lora": _load_prediction_depths(model_family, "lora", frame_count, dataset_name, seed, scene),
    }
    threshold, _ = vp._dataset_recon_eval_params(dataset_name)
    if threshold is None:
        threshold = 0.05

    galleries = {"baseline": [], "lora": [], "lora_minus_baseline": []}
    mae_stats = {"baseline": [], "lora": []}
    inlier_stats = {"baseline": [], "lora": []}
    err_by_image: Dict[str, Dict[str, np.ndarray]] = {"baseline": {}, "lora": {}}
    rgb_by_image: Dict[str, np.ndarray] = {}

    for exp_key in ["baseline", "lora"]:
        gt_meta = gt_metas[exp_key]
        depths = pred_depths[exp_key]
        if gt_meta is None or depths is None:
            continue

        for idx, img_path in enumerate(gt_meta["image_files"]):
            if idx >= len(depths):
                continue

            rgb = vp._read_rgb_image(img_path)
            if rgb is None:
                continue
            rgb = rgb.astype(np.uint8)
            gt_depth, gt_valid = vp._load_gt_depth_map(dataset_name, img_path)
            if gt_depth is None or gt_valid is None:
                continue

            if gt_depth.shape[:2] != rgb.shape[:2]:
                gt_depth = cv2.resize(gt_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt_valid = cv2.resize(gt_valid.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

            pred = np.asarray(depths[idx], dtype=np.float32)
            if pred.shape[:2] != gt_depth.shape[:2]:
                pred = cv2.resize(pred, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_NEAREST)

            valid = gt_valid & np.isfinite(pred) & (pred > 0)
            if valid.sum() < 20:
                continue

            pred_med = float(np.median(pred[valid]))
            gt_med = float(np.median(gt_depth[valid]))
            scale = gt_med / max(pred_med, 1e-6)
            pred_scaled = pred * scale

            err = np.full_like(gt_depth, np.nan, dtype=np.float32)
            err[valid] = np.abs(pred_scaled[valid] - gt_depth[valid])
            err_by_image[exp_key][img_path] = err
            rgb_by_image[img_path] = rgb
            mae_stats[exp_key].append(float(np.nanmean(err)))
            inlier_stats[exp_key].append(float(np.mean((err[valid] <= float(threshold)).astype(np.float32))))

            err_rgb = np.zeros((err.shape[0], err.shape[1], 3), dtype=np.uint8)
            good = err <= float(threshold)
            err_rgb[np.isfinite(err)] = np.array([255, 0, 0], dtype=np.uint8)
            err_rgb[np.isfinite(err) & good] = np.array([170, 170, 170], dtype=np.uint8)
            galleries[exp_key].append(np.concatenate([rgb, err_rgb], axis=1))

    baseline_imgs = set(err_by_image["baseline"].keys())
    lora_imgs = set(err_by_image["lora"].keys())
    common_imgs = []
    baseline_meta = gt_metas["baseline"]
    if baseline_meta is not None:
        common_imgs = [img for img in baseline_meta["image_files"] if img in baseline_imgs and img in lora_imgs]
    if not common_imgs:
        lora_meta = gt_metas["lora"]
        if lora_meta is not None:
            common_imgs = [img for img in lora_meta["image_files"] if img in baseline_imgs and img in lora_imgs]

    for img_path in common_imgs:
        rgb = rgb_by_image.get(img_path)
        b_err = err_by_image["baseline"][img_path]
        l_err = err_by_image["lora"][img_path]
        valid = np.isfinite(b_err) & np.isfinite(l_err)
        if rgb is None or not valid.any():
            continue
        thr = float(threshold)
        b_in = b_err <= thr
        l_in = l_err <= thr
        delta_rgb = np.zeros((b_err.shape[0], b_err.shape[1], 3), dtype=np.uint8)
        delta_rgb[valid & b_in & l_in] = np.array([170, 170, 170], dtype=np.uint8)
        both_out = valid & (~b_in) & (~l_in)
        tied = both_out & np.isclose(l_err, b_err, rtol=1e-4, atol=1e-6)
        delta_rgb[tied] = np.array([170, 170, 170], dtype=np.uint8)
        delta_rgb[both_out & (~tied) & (l_err < b_err)] = np.array([0, 100, 255], dtype=np.uint8)
        delta_rgb[both_out & (~tied) & (l_err > b_err)] = np.array([255, 0, 0], dtype=np.uint8)
        delta_rgb[valid & l_in & (~b_in)] = np.array([0, 100, 255], dtype=np.uint8)
        delta_rgb[valid & (~l_in) & b_in] = np.array([255, 0, 0], dtype=np.uint8)
        galleries["lora_minus_baseline"].append(np.concatenate([rgb, delta_rgb], axis=1))

    def _mean_or_nan(vals):
        return float(np.mean(vals)) if vals else float("nan")

    summary = (
        f"Depth threshold coloring uses the original benchmark threshold: `{float(threshold):.4f} m`.\n\n"
        f"- Original model MAE: `{_mean_or_nan(mae_stats['baseline']):.4f}`\n"
        f"- LoRA MAE: `{_mean_or_nan(mae_stats['lora']):.4f}`\n"
        f"- Original inlier-rate (<=thr): `{_mean_or_nan(inlier_stats['baseline']):.4f}`\n"
        f"- LoRA inlier-rate (<=thr): `{_mean_or_nan(inlier_stats['lora']):.4f}`\n"
        f"- Common frames used in delta tab: `{len(common_imgs)}`\n"
        "- Delta panel: gray = tie or both within threshold, blue = LoRA better, red = original better."
    )
    return galleries["baseline"], galleries["lora"], galleries["lora_minus_baseline"], summary


def build_app(
    *,
    model_family: str,
    work_root: str,
    frame_count: int,
    dataset_name: str,
    seed: str | int,
    default_scene: Optional[str] = None,
    viewer_max_points: int = DEFAULT_VIEWER_MAX_POINTS,
):
    scenes = _list_scenes(model_family, frame_count, dataset_name, seed)
    if scenes:
        scene_choices = scenes
        init_scene = default_scene if default_scene in scenes else scenes[0]
    else:
        scene_choices = [NO_SCENE_SENTINEL]
        init_scene = NO_SCENE_SENTINEL

    threshold_cache: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], str]] = {}
    depth_error_cache: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], str]] = {}

    with gr.Blocks(title="Fixed-View Baseline vs LoRA Viewer") as demo:
        gr.Markdown(
            f"# Fixed-View Baseline vs LoRA Viewer\n"
            f"`{model_family}` | dataset=`{dataset_name}` | frames=`{frame_count}` | seed=`{_seed_tag(seed)}`"
        )

        scene_dd = gr.Dropdown(choices=scene_choices, value=init_scene, label="Scene", interactive=bool(scenes))
        info_md = gr.Markdown("")

        gr.Markdown("### Input Frames")
        load_inputs_btn = gr.Button("Load Input Frames")
        input_status_md = gr.Markdown("Input frames are lazy-loaded.")
        input_gallery = gr.Gallery(value=[], columns=5, height=220, label="Input frames")

        gr.Markdown("### 3D Point Cloud Comparison")
        load_models_btn = gr.Button("Load 3D Point Clouds")
        model_status_md = gr.Markdown(
            f"3D models are lazy-loaded and capped at `{int(viewer_max_points):,}` points per scene to keep remote sessions stable."
        )
        with gr.Row():
            gt_model = gr.Model3D(value=None, label="GT Fused", height=480, clear_color=(1.0, 1.0, 1.0, 1.0))
            baseline_model = gr.Model3D(value=None, label=EXPERIMENTS["baseline"], height=480, clear_color=(1.0, 1.0, 1.0, 1.0))
            lora_model = gr.Model3D(value=None, label=EXPERIMENTS["lora"], height=480, clear_color=(1.0, 1.0, 1.0, 1.0))

        gr.Markdown("### GT Threshold Overlay")
        gr.Markdown(
            "Threshold coloring matches benchmark semantics: gray = within threshold, red = outside threshold. "
            "HiRoom uses the original evaluation threshold here."
        )
        render_thresh_btn = gr.Button("Render Threshold Overlays")
        with gr.Row():
            baseline_gt_thresh_model = gr.Model3D(value=None, label="Original vs GT threshold overlay", height=520, clear_color=(1.0, 1.0, 1.0, 1.0))
            lora_gt_thresh_model = gr.Model3D(value=None, label="LoRA vs GT threshold overlay", height=520, clear_color=(1.0, 1.0, 1.0, 1.0))
            delta_thresh_model = gr.Model3D(value=None, label="LoRA vs Original delta overlay", height=520, clear_color=(1.0, 1.0, 1.0, 1.0))
        threshold_md = gr.Markdown("Click 'Render Threshold Overlays'.")

        gr.Markdown("### Predicted Depth Maps")
        load_depths_btn = gr.Button("Load Predicted Depth Maps")
        pred_depth_status_md = gr.Markdown("Predicted depth maps are lazy-loaded.")
        with gr.Tab(EXPERIMENTS["baseline"]):
            baseline_depth_gallery = gr.Gallery(value=[], columns=2, height=260, label="RGB | predicted depth")
        with gr.Tab(EXPERIMENTS["lora"]):
            lora_depth_gallery = gr.Gallery(value=[], columns=2, height=260, label="RGB | predicted depth")

        gr.Markdown("### Thresholded Depth Error vs GT")
        render_depth_btn = gr.Button("Render Depth Threshold Maps")
        with gr.Tab(EXPERIMENTS["baseline"]):
            baseline_err_gallery = gr.Gallery(value=[], columns=2, height=260, label="RGB | thresholded depth error")
        with gr.Tab(EXPERIMENTS["lora"]):
            lora_err_gallery = gr.Gallery(value=[], columns=2, height=260, label="RGB | thresholded depth error")
        with gr.Tab("LoRA - Original"):
            delta_err_gallery = gr.Gallery(value=[], columns=2, height=260, label="RGB | delta threshold map")
        depth_err_md = gr.Markdown("Click 'Render Depth Threshold Maps'.")

        gr.Markdown("### Metrics")
        metrics_md = gr.Markdown("")

        def _update(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return (
                    "No scenes found for the resolved baseline/LoRA result roots.",
                    [],
                    None,
                    None,
                    None,
                    "No valid scene selected.",
                    [],
                    [],
                    "No valid scene selected.",
                    "No valid scene selected.",
                    "",
                )

            gt_meta = _load_gt_meta(model_family, frame_count, dataset_name, seed, scene)
            cache_root = _viewer_cache_dir(work_root, frame_count, dataset_name, seed, viewer_max_points)
            info = [
                f"viewer_cache_root: `{os.path.abspath(cache_root)}`",
                f"dataset: `{dataset_name}`",
                f"scene: `{scene}`",
                f"frames: `{frame_count}`",
                f"seed: `{_seed_tag(seed)}`",
            ]
            roots = _comparison_roots(model_family, frame_count)
            info.append(f"baseline_root: `{roots['baseline']}`")
            info.append(f"lora_root: `{roots['lora']}`")
            if gt_meta:
                info.append(f"gt_meta: `{gt_meta['meta_path']}`")
                info.append(f"GT source: `{gt_meta['meta_source_exp']}`")

            metrics = _format_metrics_md(model_family, frame_count, dataset_name, seed, scene)

            return (
                "  \n".join(info),
                [],
                None,
                None,
                None,
                f"Ready to load input frames for `{scene}`. Click `Load Input Frames`.",
                [],
                [],
                f"Ready to load predicted depth maps for `{scene}`. Click `Load Predicted Depth Maps`.",
                f"Ready to load 3D models for `{scene}`. Click `Load 3D Point Clouds`.",
                metrics,
            )

        def _load_inputs(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return [], "No valid scene selected."

            gt_meta = _load_gt_meta(model_family, frame_count, dataset_name, seed, scene)
            input_imgs = gt_meta["image_files"] if gt_meta else []
            if not input_imgs:
                return [], f"No input frames found for `{scene}`."
            return input_imgs, f"Loaded `{len(input_imgs)}` input frames for `{scene}`."

        def _load_pred_depths(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return [], [], "No valid scene selected."

            baseline_depth = _prepare_pred_depth_gallery(
                model_family,
                work_root,
                "baseline",
                frame_count,
                dataset_name,
                seed,
                scene,
            )
            lora_depth = _prepare_pred_depth_gallery(
                model_family,
                work_root,
                "lora",
                frame_count,
                dataset_name,
                seed,
                scene,
            )
            status = (
                f"Loaded predicted depth maps for `{scene}`: "
                f"`{len(baseline_depth)}` original panels, `{len(lora_depth)}` LoRA panels."
            )
            return baseline_depth, lora_depth, status

        def _load_models(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return None, None, None, "No valid scene selected."

            gt_glb = _ensure_gt_glb(
                model_family,
                work_root,
                frame_count,
                dataset_name,
                seed,
                scene,
                viewer_max_points=viewer_max_points,
            )
            baseline_glb = _ensure_exp_fused_glb(
                model_family,
                work_root,
                frame_count,
                dataset_name,
                seed,
                scene,
                "baseline",
                viewer_max_points=viewer_max_points,
            )
            lora_glb = _ensure_exp_fused_glb(
                model_family,
                work_root,
                frame_count,
                dataset_name,
                seed,
                scene,
                "lora",
                viewer_max_points=viewer_max_points,
            )
            status = (
                f"Loaded 3D models for `{scene}` from cache root "
                f"`{os.path.abspath(_viewer_cache_dir(work_root, frame_count, dataset_name, seed, viewer_max_points))}` "
                f"with a `{int(viewer_max_points):,}`-point cap."
            )
            return gt_glb, baseline_glb, lora_glb, status

        def _load_threshold_overlays(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return None, None, None, "No valid scene selected."
            if scene in threshold_cache:
                return threshold_cache[scene]

            missing_deps = vp._check_threshold_overlay_deps()
            if missing_deps:
                out = (None, None, None, f"Missing dependencies: `{missing_deps}`.")
                threshold_cache[scene] = out
                return out

            baseline_glb = _ensure_exp_vs_gt_threshold_overlay_glb(
                model_family,
                work_root,
                frame_count,
                dataset_name,
                seed,
                scene,
                exp_key="baseline",
                out_name="baseline_vs_gt_threshold",
                viewer_max_points=viewer_max_points,
            )
            lora_glb = _ensure_exp_vs_gt_threshold_overlay_glb(
                model_family,
                work_root,
                frame_count,
                dataset_name,
                seed,
                scene,
                exp_key="lora",
                out_name="lora_vs_gt_threshold",
                viewer_max_points=viewer_max_points,
            )
            delta_glb = _ensure_lora_vs_baseline_delta_overlay_glb(
                model_family,
                work_root,
                frame_count,
                dataset_name,
                seed,
                scene,
                viewer_max_points=viewer_max_points,
            )
            out = (baseline_glb, lora_glb, delta_glb, "Rendered threshold overlays using original benchmark thresholds.")
            threshold_cache[scene] = out
            return out

        def _load_depth_errors(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return [], [], [], "No valid scene selected."
            if scene in depth_error_cache:
                return depth_error_cache[scene]
            out = _prepare_depth_error_galleries(model_family, work_root, frame_count, dataset_name, seed, scene)
            depth_error_cache[scene] = out
            return out

        scene_dd.change(
            fn=_update,
            inputs=[scene_dd],
            outputs=[
                info_md,
                input_gallery,
                gt_model,
                baseline_model,
                lora_model,
                input_status_md,
                baseline_depth_gallery,
                lora_depth_gallery,
                pred_depth_status_md,
                model_status_md,
                metrics_md,
            ],
            queue=False,
        )
        demo.load(
            fn=_update,
            inputs=[scene_dd],
            outputs=[
                info_md,
                input_gallery,
                gt_model,
                baseline_model,
                lora_model,
                input_status_md,
                baseline_depth_gallery,
                lora_depth_gallery,
                pred_depth_status_md,
                model_status_md,
                metrics_md,
            ],
        )
        load_inputs_btn.click(
            fn=_load_inputs,
            inputs=[scene_dd],
            outputs=[input_gallery, input_status_md],
        )
        load_depths_btn.click(
            fn=_load_pred_depths,
            inputs=[scene_dd],
            outputs=[baseline_depth_gallery, lora_depth_gallery, pred_depth_status_md],
        )
        load_models_btn.click(
            fn=_load_models,
            inputs=[scene_dd],
            outputs=[gt_model, baseline_model, lora_model, model_status_md],
        )
        render_thresh_btn.click(
            fn=_load_threshold_overlays,
            inputs=[scene_dd],
            outputs=[baseline_gt_thresh_model, lora_gt_thresh_model, delta_thresh_model, threshold_md],
        )
        render_depth_btn.click(
            fn=_load_depth_errors,
            inputs=[scene_dd],
            outputs=[baseline_err_gallery, lora_err_gallery, delta_err_gallery, depth_err_md],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Viewer for 8v/16v/32v baseline vs LoRA model comparisons")
    parser.add_argument("--model_family", choices=["da3", "vggt"], required=True)
    parser.add_argument("--work_root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="hiroom")
    parser.add_argument("--frames", type=int, choices=[8, 16, 32], default=8)
    parser.add_argument("--seed", type=str, default="43")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--viewer_max_points", type=int, default=DEFAULT_VIEWER_MAX_POINTS)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if gr is None:
        raise SystemExit("Missing dependency: gradio. Install with `pip install gradio`.")

    work_root = args.work_root or DEFAULT_WORK_ROOTS[args.model_family]
    demo = build_app(
        model_family=args.model_family,
        work_root=work_root,
        frame_count=args.frames,
        dataset_name=args.dataset,
        seed=args.seed,
        default_scene=args.scene,
        viewer_max_points=max(1_000, int(args.viewer_max_points)),
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

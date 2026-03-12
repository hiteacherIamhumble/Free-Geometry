#!/usr/bin/env python3
"""
Gradio viewer for comparing point cloud / camera pose visualizations from
`scripts/benchmark_hiroom_teacher_student.py`.

Usage:
    python scripts/view_pointclouds.py
    python scripts/view_pointclouds.py --dataset hiroom --work_dir ./results/hiroom_teacher_student
    python scripts/view_pointclouds.py --dataset 7scenes --work_dir ./results/teacher_student_7scenes_all
    python scripts/view_pointclouds.py --port 7860 --share

Notes:
  - This app can optionally generate Ground Truth (GT) visualizations (GLB + depth_vis)
    under `<work_dir>/gt/visualizations/` by reading HiRoom GT data.
  - For each experiment, you can toggle between:
      - Raw: `<work_dir>/<exp>/visualizations/scene.glb` (direct model export)
      - Fused: `<work_dir>/<exp>/model_results/.../exports/fuse/pcd.ply` converted to GLB
        with GT camera poses (aligned-to-GT recon_unposed output).
"""

import argparse
import glob

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import gradio as gr
except ImportError:
    gr = None


_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# These names match the output folder layout created by benchmark_hiroom_teacher_student.py
EXPERIMENTS: Dict[str, str] = {
    "teacher": "Teacher 8v→4v extract",
    "teacher_4v": "Teacher 4v direct",
    "student": "Student LoRA 4v",
}
GT_KEY = "gt"
NO_SCENE_SENTINEL = "__NO_SCENE_AVAILABLE__"


def _scene_slug(scene: str) -> str:
    return "-".join(scene.split("/")[-3:])


def _find_raw_glb(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> Optional[str]:
    # Newest layout from benchmark_teacher_student_all_datasets.py
    path = os.path.join(work_dir, exp_key, "visualizations", dataset_name, _scene_slug(scene), "scene.glb")
    if os.path.exists(path):
        return path
    # New (per-scene) layout:
    path = os.path.join(work_dir, exp_key, "visualizations", _scene_slug(scene), "scene.glb")
    if os.path.exists(path):
        return path
    # Backward-compat (single-scene) layout:
    legacy = os.path.join(work_dir, exp_key, "visualizations", "scene.glb")
    return legacy if os.path.exists(legacy) else None


def _find_depth_vis_dir(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> str:
    # Newest layout from benchmark_teacher_student_all_datasets.py
    d = os.path.join(work_dir, exp_key, "visualizations", dataset_name, _scene_slug(scene), "depth_vis")
    if os.path.isdir(d):
        return d
    # Prefer new per-scene layout; keep legacy fallback for old runs.
    d = os.path.join(work_dir, exp_key, "visualizations", _scene_slug(scene), "depth_vis")
    if os.path.isdir(d):
        return d
    return os.path.join(work_dir, exp_key, "visualizations", "depth_vis")


def _find_depth_vis_images(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> List[str]:
    d = _find_depth_vis_dir(work_dir, exp_key, dataset_name, scene)
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*.jpg")))


def _find_input_images(image_dir: str, dataset_name: str, scene: str) -> List[str]:
    if not image_dir:
        return []
    d = os.path.join(image_dir, dataset_name, _scene_slug(scene))
    if os.path.isdir(d):
        return sorted(glob.glob(os.path.join(d, "*.jpg")))
    d = os.path.join(image_dir, _scene_slug(scene))
    if not os.path.isdir(d):
        # Backward-compat: older benchmark saved directly under image_dir
        if os.path.isdir(image_dir):
            return sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        return []
    return sorted(glob.glob(os.path.join(d, "*.jpg")))


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _has_experiment_layout(work_dir: str) -> bool:
    for exp_key in EXPERIMENTS.keys():
        if os.path.isdir(os.path.join(work_dir, exp_key)):
            return True
    return False


def _resolve_work_dir_for_dataset(work_dir: str, dataset_name: str) -> str:
    """
    Accept both:
      - <root>/<teacher|teacher_4v|student>/...
      - <root>/<dataset>/<teacher|teacher_4v|student>/...
    """
    if _has_experiment_layout(work_dir):
        return work_dir
    ds_dir = os.path.join(work_dir, dataset_name)
    if _has_experiment_layout(ds_dir):
        return ds_dir
    return work_dir


def _list_scenes(work_dir: str, dataset_name: str) -> List[str]:
    """
    Derive available scenes from metric JSONs (preferred) or from model_results folders.
    """
    # 1) Metric JSONs (most robust)
    for exp_key in EXPERIMENTS.keys():
        pose_json = os.path.join(work_dir, exp_key, "metric_results", f"{dataset_name}_pose.json")
        data = _load_json(pose_json)
        if isinstance(data, dict):
            scenes = sorted([k for k in data.keys() if k != "mean"])
            if scenes:
                return scenes

    # 2) Fallback: model_results directory walk
    for exp_key in EXPERIMENTS.keys():
        base = os.path.join(work_dir, exp_key, "model_results", dataset_name)
        if not os.path.isdir(base):
            continue
        # Expect: .../model_results/<dataset>/<scene>/unposed
        scenes = []
        for root, dirs, _files in os.walk(base):
            if root.endswith(os.path.join("unposed")):
                scene = os.path.relpath(os.path.dirname(root), base)
                scenes.append(scene)
        scenes = sorted(set(scenes))
        if scenes:
            return scenes

    return []


def _gt_meta_path_for_scene(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> str:
    return os.path.join(
        work_dir,
        exp_key,
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "gt_meta.npz",
    )


def _load_gt_meta(work_dir: str, dataset_name: str, scene: str) -> Optional[dict]:
    """
    Load gt_meta saved by benchmark script (4 shared views).

    GT assets in this viewer should use teacher_4v shared views only.
    """
    import numpy as np

    meta_path = None
    meta_source_exp = None
    # Prioritize teacher_4v so GT visualizations always match the original 4-view baseline.
    search_order = ["teacher_4v", "student", "teacher"]
    for exp_key in search_order:
        p = _gt_meta_path_for_scene(work_dir, exp_key, dataset_name, scene)
        if os.path.exists(p):
            meta_path = p
            meta_source_exp = exp_key
            break
    if meta_path is None:
        return None

    data = np.load(meta_path, allow_pickle=True)
    image_files = [str(x) for x in list(data.get("image_files", []))]
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]

    # Enforce the 4-view setup used by teacher_4v.
    n = min(4, len(image_files), len(extrinsics), len(intrinsics))
    if n <= 0:
        return None

    return {
        "meta_path": meta_path,
        "meta_source_exp": meta_source_exp,
        "image_files": image_files[:n],
        "extrinsics": extrinsics[:n],
        "intrinsics": intrinsics[:n],
    }


def _hiroom_gt_ply_for_scene(scene: str) -> str:
    from depth_anything_3.utils.constants import HIROOM_GT_ROOT_PATH

    scene_name = "-".join(scene.split("/")[-3:])
    return os.path.join(HIROOM_GT_ROOT_PATH, f"{scene_name}.ply")


def _read_hiroom_gt_depth_and_mask(scene_image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    HiRoom layout:
      .../<scene>/image/<frame>.jpg
      .../<scene>/depth/<frame>.png
      .../<scene>/aliasing_mask/<frame>.png
    """
    img_dir = os.path.dirname(scene_image_path)
    scene_dir = os.path.dirname(img_dir)
    frame = os.path.splitext(os.path.basename(scene_image_path))[0]
    depth_path = os.path.join(scene_dir, "depth", f"{frame}.png")
    mask_path = os.path.join(scene_dir, "aliasing_mask", f"{frame}.png")
    return (depth_path if os.path.exists(depth_path) else None), (mask_path if os.path.exists(mask_path) else None)


def _read_7scenes_gt_depth(scene_image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    7Scenes layout:
      .../frame-XXXXXX.color.png
      .../frame-XXXXXX.depth.png
    """
    depth_path = scene_image_path.replace(".color.", ".depth.")
    return (depth_path if os.path.exists(depth_path) else None), None


def _read_scannetpp_gt_depth(scene_image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    ScanNet++ layout:
      .../<scene>/iphone/rgb/<frame>.JPG
      .../<scene>/merge_dslr_iphone/render_depth/<frame>.png
    """
    # Extract scene root and frame name
    parts = scene_image_path.split(os.sep)
    try:
        # Find scene directory (contains 'iphone' or 'dslr')
        scene_idx = -1
        for i, part in enumerate(parts):
            if part in ["iphone", "dslr"]:
                scene_idx = i - 1
                break
        if scene_idx < 0:
            return None, None

        scene_dir = os.sep.join(parts[:scene_idx + 1])
        frame_name = os.path.splitext(os.path.basename(scene_image_path))[0]
        depth_path = os.path.join(scene_dir, "merge_dslr_iphone", "render_depth", f"{frame_name}.png")
        return (depth_path if os.path.exists(depth_path) else None), None
    except Exception:
        return None, None


def _read_eth3d_gt_depth_and_mask(scene_image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    ETH3D layout (actual):
      .../<scene>/images/dslr_images/<frame>.JPG
      .../<scene>/ground_truth_depth/dslr_images/<frame>.JPG   (binary float32)
      .../<scene>/masks_for_images/dslr_images/<frame>.png
    """
    parts = scene_image_path.split(os.sep)
    try:
        # Find 'dslr_images' and walk up to the scene root.
        # Path may be <scene>/dslr_images/... or <scene>/images/dslr_images/...
        dslr_idx = -1
        for i, part in enumerate(parts):
            if part == "dslr_images":
                dslr_idx = i
                break
        if dslr_idx < 0:
            return None, None

        # Scene root is the ancestor that contains ground_truth_depth/.
        # Try one level up, then two levels up (for the images/ subdirectory case).
        frame_name = os.path.basename(scene_image_path)
        frame_name_no_ext = os.path.splitext(frame_name)[0]

        for levels_up in [1, 2]:
            candidate_idx = dslr_idx - levels_up
            if candidate_idx < 0:
                continue
            scene_dir = os.sep.join(parts[:candidate_idx + 1])

            # Try with and without extension — some ETH3D copies keep .JPG
            for ext in ["", ".JPG", ".jpg"]:
                depth_path = os.path.join(scene_dir, "ground_truth_depth", "dslr_images", frame_name_no_ext + ext)
                if os.path.exists(depth_path):
                    mask_path = os.path.join(scene_dir, "masks_for_images", "dslr_images", f"{frame_name_no_ext}.png")
                    return depth_path, (mask_path if os.path.exists(mask_path) else None)

        return None, None
    except Exception:
        return None, None


def _read_gt_depth_and_mask(dataset_name: str, scene_image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get GT depth and mask paths using dataset-specific logic.

    This function uses the dataset loaders' knowledge of file structure
    instead of hardcoding path patterns.
    """
    try:
        # Import dataset registries
        from depth_anything_3.bench.registries import MV_REGISTRY

        # Get dataset loader
        if not MV_REGISTRY.has(dataset_name):
            return None, None

        dataset = MV_REGISTRY.get(dataset_name)()

        # Extract scene name from image path
        # For ETH3D: .../eth3d/courtyard/images/...
        # For HiRoom: .../hiroom/20241230/828805/cam_sampled_06/...
        # For 7Scenes: .../7Scenes/chess/...
        # For ScanNet++: .../scannetpp/scene_id/...

        parts = scene_image_path.split(os.sep)

        # Find dataset name in path and extract scene
        dataset_idx = -1
        for i, part in enumerate(parts):
            if dataset_name in part.lower() or part == dataset_name:
                dataset_idx = i
                break

        if dataset_idx < 0:
            # Fallback: use old path-based logic
            return _read_gt_depth_and_mask_fallback(dataset_name, scene_image_path)

        # Scene is the path components between dataset root and final file
        # For ETH3D: courtyard
        # For HiRoom: 20241230/828805/cam_sampled_06
        # For 7Scenes: chess
        # For ScanNet++: scene_id

        if dataset_name == "eth3d":
            # .../eth3d/courtyard/images/dslr_images/DSC_0307.JPG
            scene = parts[dataset_idx + 1]
            image_name = os.path.basename(scene_image_path)

            depth_path = os.path.join(
                dataset.data_root, scene, "ground_truth_depth", "dslr_images", image_name
            )
            mask_path = os.path.join(
                dataset.data_root, scene, "masks_for_images", "dslr_images",
                image_name.replace(".JPG", ".png").replace(".jpg", ".png")
            )

        elif dataset_name == "hiroom":
            # .../hiroom/.../20241230/828805/cam_sampled_06/image/frame.jpg
            # scene_dir is two levels up from the image file (past image/)
            img_dir = os.path.dirname(scene_image_path)   # .../cam_sampled_06/image
            scene_dir = os.path.dirname(img_dir)           # .../cam_sampled_06
            frame = os.path.splitext(os.path.basename(scene_image_path))[0]

            depth_path = os.path.join(scene_dir, "depth", f"{frame}.png")
            mask_path = os.path.join(scene_dir, "aliasing_mask", f"{frame}.png")

        elif dataset_name == "7scenes":
            # .../7Scenes/chess/seq-01/frame-000000.color.png
            scene = parts[dataset_idx + 1]
            depth_path = scene_image_path.replace(".color.", ".depth.")
            mask_path = None

        elif dataset_name == "scannetpp":
            # .../scannetpp/scene_id/iphone/rgb/frame.JPG
            scene = parts[dataset_idx + 1]
            scene_dir = os.path.join(dataset.data_root, scene)
            frame_name = os.path.splitext(os.path.basename(scene_image_path))[0]

            depth_path = os.path.join(scene_dir, "merge_dslr_iphone", "render_depth", f"{frame_name}.png")
            mask_path = None

        else:
            return None, None

        return (
            depth_path if os.path.exists(depth_path) else None,
            mask_path if mask_path and os.path.exists(mask_path) else None
        )

    except Exception as e:
        # Fallback to old logic if dataset loader fails
        return _read_gt_depth_and_mask_fallback(dataset_name, scene_image_path)


def _read_gt_depth_and_mask_fallback(dataset_name: str, scene_image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Fallback path-based GT depth/mask resolution (old logic)."""
    if dataset_name == "hiroom":
        return _read_hiroom_gt_depth_and_mask(scene_image_path)
    if dataset_name == "7scenes":
        return _read_7scenes_gt_depth(scene_image_path)
    if dataset_name == "scannetpp":
        return _read_scannetpp_gt_depth(scene_image_path)
    if dataset_name == "eth3d":
        return _read_eth3d_gt_depth_and_mask(scene_image_path)
    return None, None


def _load_raw_gt_depth(dataset_name: str, depth_path: str, image_hw: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Load GT depth from file and convert to meters.
    ``image_hw`` is (height, width) — required for ETH3D (raw float32 binary).
    Returns None if loading fails.
    """
    try:
        import numpy as np

        if dataset_name == "hiroom":
            import cv2

            depth_raw = cv2.imread(depth_path, -1)
            if depth_raw is None:
                return None
            return depth_raw.astype(np.float32) / 65535.0 * 100.0

        elif dataset_name == "7scenes":
            import cv2

            depth_raw = cv2.imread(depth_path, -1)
            if depth_raw is None:
                return None
            depth_raw[depth_raw == 65535] = 0
            return depth_raw.astype(np.float32) / 1000.0

        elif dataset_name == "scannetpp":
            import cv2

            depth_raw = cv2.imread(depth_path, -1)
            if depth_raw is None:
                return None
            return depth_raw.astype(np.float32) / 1000.0

        elif dataset_name == "eth3d":
            if not os.path.exists(depth_path):
                return None
            if image_hw is None:
                return None
            h, w = image_hw
            depth_m = np.fromfile(depth_path, dtype=np.float32).reshape(h, w)
            depth_m[~np.isfinite(depth_m)] = 0.0
            return depth_m

        return None
    except Exception:
        return None


def _ensure_gt_fuse_ply_4v(work_dir: str, dataset_name: str, scene: str, gt_meta: dict) -> Optional[str]:
    """
    Fuse a GT point cloud from the *same 4 views* used by the benchmark (gt_meta.npz),
    using the GT depth maps + GT camera intrinsics/extrinsics.

    Output: `<work_dir>/gt/fuse_4v/<dataset>/<scene_slug>/pcd.ply`
    """
    _SUPPORTED_GT_FUSION = {"hiroom", "7scenes", "scannetpp", "eth3d"}
    if dataset_name not in _SUPPORTED_GT_FUSION:
        return None

    fuse_dir = os.path.join(work_dir, GT_KEY, "fuse_4v", dataset_name, _scene_slug(scene))
    out_ply = os.path.join(fuse_dir, "pcd.ply")
    if os.path.exists(out_ply):
        return out_ply

    try:
        import cv2
        import numpy as np
        import open3d as o3d

        from depth_anything_3.bench.utils import create_tsdf_volume, fuse_depth_to_tsdf, sample_points_from_mesh
        from depth_anything_3.utils.constants import (
            ETH3D_MAX_DEPTH,
            ETH3D_SAMPLING_NUMBER,
            ETH3D_SDF_TRUNC,
            ETH3D_VOXEL_LENGTH,
            HIROOM_MAX_DEPTH,
            HIROOM_SAMPLING_NUMBER,
            HIROOM_SDF_TRUNC,
            HIROOM_VOXEL_LENGTH,
            SCANNETPP_MAX_DEPTH,
            SCANNETPP_SAMPLING_NUMBER,
            SCANNETPP_SDF_TRUNC,
            SCANNETPP_VOXEL_LENGTH,
            SEVENSCENES_MAX_DEPTH,
            SEVENSCENES_SAMPLING_NUMBER,
            SEVENSCENES_SDF_TRUNC,
            SEVENSCENES_VOXEL_LENGTH,
        )
    except Exception:
        return None

    _FUSION_PARAMS = {
        "hiroom": (HIROOM_MAX_DEPTH, HIROOM_SAMPLING_NUMBER, HIROOM_SDF_TRUNC, HIROOM_VOXEL_LENGTH),
        "7scenes": (SEVENSCENES_MAX_DEPTH, SEVENSCENES_SAMPLING_NUMBER, SEVENSCENES_SDF_TRUNC, SEVENSCENES_VOXEL_LENGTH),
        "scannetpp": (SCANNETPP_MAX_DEPTH, SCANNETPP_SAMPLING_NUMBER, SCANNETPP_SDF_TRUNC, SCANNETPP_VOXEL_LENGTH),
        "eth3d": (ETH3D_MAX_DEPTH, ETH3D_SAMPLING_NUMBER, ETH3D_SDF_TRUNC, ETH3D_VOXEL_LENGTH),
    }
    max_depth_value, sampling_number, sdf_trunc_value, voxel_length_value = (
        float(v) for v in _FUSION_PARAMS[dataset_name]
    )
    sampling_number = int(sampling_number)

    image_files = gt_meta.get("image_files") or []
    if not image_files:
        return None

    images = []
    depths = []
    used_indices = []
    for idx, img_path in enumerate(image_files):
        rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_path, mask_path = _read_gt_depth_and_mask(dataset_name, img_path)
        if depth_path is None:
            continue

        depth_m = _load_raw_gt_depth(dataset_name, depth_path, image_hw=(rgb.shape[0], rgb.shape[1]))
        if depth_m is None:
            continue

        valid = np.isfinite(depth_m) & (depth_m > 0)
        if mask_path is not None:
            mask_img = _read_mask_image(mask_path)
            if mask_img is not None:
                if dataset_name == "eth3d":
                    valid = _eth3d_valid_mask(depth_m, mask_img)
                else:
                    valid = valid & (~(mask_img > 0))
        depth_m = depth_m * valid.astype(np.float32)

        if depth_m.shape[0] != rgb.shape[0] or depth_m.shape[1] != rgb.shape[1]:
            depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        images.append(rgb.astype(np.uint8))
        depths.append(depth_m.astype(np.float32))
        used_indices.append(idx)

    if not images or not depths or not used_indices:
        return None

    images = np.stack(images, axis=0)
    depths = np.stack(depths, axis=0)
    intrinsics = gt_meta["intrinsics"][used_indices]
    extrinsics = gt_meta["extrinsics"][used_indices]

    # TSDF fusion in world coordinates using GT cameras.
    volume = create_tsdf_volume(
        voxel_length=voxel_length_value,
        sdf_trunc=sdf_trunc_value,
        color_type="RGB8",
    )
    mesh = fuse_depth_to_tsdf(
        volume,
        depths,
        images,
        intrinsics,
        extrinsics,
        max_depth=max_depth_value,
    )
    # Sample points from fused mesh - use same sampling as reconstruction eval
    pcd = sample_points_from_mesh(mesh, num_points=sampling_number)

    os.makedirs(fuse_dir, exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pcd)
    return out_ply


def _ensure_glb_from_ply_with_cameras(
    *,
    ply_path: str,
    out_glb_path: str,
    extrinsics_w2c,
    intrinsics,
    image_sizes: List[Tuple[int, int]],
    camera_size: float = 0.03,
) -> Optional[str]:
    """
    Convert a point cloud PLY into a GLB and add camera frustums.
    """
    if os.path.exists(out_glb_path):
        return out_glb_path
    if not os.path.exists(ply_path):
        return None

    # Heavy deps are local to keep import time low.
    try:
        import numpy as np
        import open3d as o3d
        import trimesh

        from depth_anything_3.utils.export.glb import (
            _add_cameras_to_scene,
            _compute_alignment_transform_first_cam_glTF_center_by_points,
            _estimate_scene_scale,
        )
    except Exception:
        return None

    pcd = o3d.io.read_point_cloud(ply_path)
    points_world = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    if colors is not None and colors.size > 0:
        colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors_u8 = None

    A = _compute_alignment_transform_first_cam_glTF_center_by_points(extrinsics_w2c[0], points_world)
    points = trimesh.transform_points(points_world, A) if points_world.shape[0] > 0 else points_world

    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    scene.metadata["hf_alignment"] = A

    if points.shape[0] > 0:
        if colors_u8 is not None and colors_u8.shape[0] == points.shape[0]:
            pc = trimesh.points.PointCloud(vertices=points, colors=colors_u8)
        else:
            pc = trimesh.points.PointCloud(vertices=points)
        scene.add_geometry(pc)

    scene_scale = _estimate_scene_scale(points, fallback=1.0)
    _add_cameras_to_scene(
        scene=scene,
        K=intrinsics,
        ext_w2c=extrinsics_w2c,
        image_sizes=image_sizes,
        scale=scene_scale * camera_size,
    )

    os.makedirs(os.path.dirname(out_glb_path), exist_ok=True)
    scene.export(out_glb_path)
    return out_glb_path


def _fused_ply_path(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> str:
    return os.path.join(
        work_dir,
        exp_key,
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "fuse",
        "pcd.ply",
    )


def _read_rgb_image(img_path: str) -> Optional[np.ndarray]:
    """
    Read image as RGB uint8.
    Prefer OpenCV to match benchmark fusion behavior.
    """
    try:
        import cv2

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    try:
        import imageio.v2 as imageio

        img = imageio.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.uint8)
    except Exception:
        return None


def _read_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """Read a mask as a 2D array."""
    try:
        import cv2

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    except Exception:
        mask = None
    if mask is None:
        try:
            import imageio.v2 as imageio

            mask = imageio.imread(mask_path)
        except Exception:
            return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.asarray(mask)


def _eth3d_valid_mask(depth_m: np.ndarray, mask_img: Optional[np.ndarray]) -> np.ndarray:
    """
    ETH3D-valid mask logic aligned with benchmark ETH3D._load_gt_mask:
      - mask value 1 is invalid/occluded
      - zero/inf depth is invalid
    """
    if mask_img is None:
        mask_img = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    if mask_img.ndim == 3:
        mask_img = mask_img[..., 0]
    if mask_img.shape[:2] != depth_m.shape[:2]:
        try:
            import cv2

            mask_img = cv2.resize(
                mask_img.astype(np.uint8),
                (depth_m.shape[1], depth_m.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        except Exception:
            return np.isfinite(depth_m) & (depth_m > 0)

    invalid_from_mask = mask_img == 1
    depth_copy = depth_m.copy()
    depth_copy[invalid_from_mask] = 0
    invalid_from_depth = (
        np.logical_or(depth_copy == 0, depth_copy == np.inf) | (~np.isfinite(depth_copy))
    )
    return np.logical_and(~invalid_from_mask, ~invalid_from_depth)


def _read_image_sizes(image_files: List[str]) -> List[Tuple[int, int]]:
    image_sizes: List[Tuple[int, int]] = []
    for img_path in image_files:
        img = _read_rgb_image(img_path)
        if img is None:
            image_sizes.append((512, 512))
        else:
            image_sizes.append((int(img.shape[0]), int(img.shape[1])))
    return image_sizes


def _ensure_two_exp_overlay_glb(
    work_dir: str,
    dataset_name: str,
    scene: str,
    *,
    exp_a: str,
    exp_b: str,
    out_name: str,
    color_a: Tuple[int, int, int],
    color_b: Tuple[int, int, int],
) -> Optional[str]:
    """
    Build a fused-point overlay GLB for two experiments using teacher_4v GT cameras.
    """
    gt_meta = _load_gt_meta(work_dir, dataset_name, scene)
    if gt_meta is None:
        return None

    exp_a_ply = _fused_ply_path(work_dir, exp_a, dataset_name, scene)
    exp_b_ply = _fused_ply_path(work_dir, exp_b, dataset_name, scene)
    if not os.path.exists(exp_a_ply) or not os.path.exists(exp_b_ply):
        return None

    out_glb = os.path.join(
        work_dir,
        "comparisons",
        dataset_name,
        out_name,
        _scene_slug(scene),
        "scene.glb",
    )
    if os.path.exists(out_glb):
        return out_glb

    try:
        import numpy as np
        import open3d as o3d
        import trimesh

        from depth_anything_3.utils.export.glb import (
            _add_cameras_to_scene,
            _compute_alignment_transform_first_cam_glTF_center_by_points,
            _estimate_scene_scale,
        )
    except Exception:
        return None

    def _refine_alignment_icp(ref_points: np.ndarray, mov_points: np.ndarray) -> np.ndarray:
        """
        Refine moving point cloud alignment to reference with a lightweight ICP pass.
        This fixes small residual offsets between independently fused reconstructions.
        """
        if ref_points.shape[0] < 128 or mov_points.shape[0] < 128:
            return mov_points

        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_points.astype(np.float64))
        mov_pcd = o3d.geometry.PointCloud()
        mov_pcd.points = o3d.utility.Vector3dVector(mov_points.astype(np.float64))

        diag = float(np.linalg.norm(ref_pcd.get_axis_aligned_bounding_box().get_extent()))
        if not np.isfinite(diag) or diag <= 1e-6:
            return mov_points

        voxel = max(diag / 220.0, 1e-4)
        ref_ds = ref_pcd.voxel_down_sample(voxel)
        mov_ds = mov_pcd.voxel_down_sample(voxel)
        if len(ref_ds.points) < 64 or len(mov_ds.points) < 64:
            return mov_points

        ref_ctr = np.asarray(ref_ds.get_center())
        mov_ctr = np.asarray(mov_ds.get_center())
        init = np.eye(4, dtype=np.float64)
        init[:3, 3] = ref_ctr - mov_ctr

        threshold = max(diag * 0.08, voxel * 4.0)
        reg = o3d.pipelines.registration.registration_icp(
            mov_ds,
            ref_ds,
            threshold,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
        )

        # Keep original if ICP failed to find a meaningful overlap.
        if float(reg.fitness) < 0.05:
            return mov_points

        return trimesh.transform_points(mov_points, reg.transformation)

    exp_a_pcd = o3d.io.read_point_cloud(exp_a_ply)
    exp_b_pcd = o3d.io.read_point_cloud(exp_b_ply)
    exp_a_points_world = np.asarray(exp_a_pcd.points, dtype=np.float32)
    exp_b_points_world = np.asarray(exp_b_pcd.points, dtype=np.float32)

    if exp_a_points_world.shape[0] == 0 and exp_b_points_world.shape[0] == 0:
        return None

    reference_points = exp_a_points_world
    if reference_points.shape[0] == 0:
        reference_points = exp_b_points_world

    A = _compute_alignment_transform_first_cam_glTF_center_by_points(gt_meta["extrinsics"][0], reference_points)
    exp_a_points = (
        trimesh.transform_points(exp_a_points_world, A)
        if exp_a_points_world.shape[0] > 0
        else exp_a_points_world
    )
    exp_b_points = (
        trimesh.transform_points(exp_b_points_world, A)
        if exp_b_points_world.shape[0] > 0
        else exp_b_points_world
    )

    # Align exp_b to exp_a for clear visual differencing.
    if exp_a_points.shape[0] > 0 and exp_b_points.shape[0] > 0:
        exp_b_points = _refine_alignment_icp(exp_a_points, exp_b_points)

    scene_obj = trimesh.Scene()
    if scene_obj.metadata is None:
        scene_obj.metadata = {}
    scene_obj.metadata["hf_alignment"] = A

    if exp_a_points.shape[0] > 0:
        exp_a_colors = np.tile(np.array(color_a, dtype=np.uint8), (exp_a_points.shape[0], 1))
        exp_a_pc = trimesh.points.PointCloud(vertices=exp_a_points, colors=exp_a_colors)
        scene_obj.add_geometry(exp_a_pc)

    if exp_b_points.shape[0] > 0:
        exp_b_colors = np.tile(np.array(color_b, dtype=np.uint8), (exp_b_points.shape[0], 1))
        exp_b_pc = trimesh.points.PointCloud(vertices=exp_b_points, colors=exp_b_colors)
        scene_obj.add_geometry(exp_b_pc)

    if exp_a_points.shape[0] > 0 and exp_b_points.shape[0] > 0:
        all_points = np.concatenate([exp_a_points, exp_b_points], axis=0)
    elif exp_a_points.shape[0] > 0:
        all_points = exp_a_points
    else:
        all_points = exp_b_points

    scene_scale = _estimate_scene_scale(all_points, fallback=1.0)
    _add_cameras_to_scene(
        scene=scene_obj,
        K=gt_meta["intrinsics"],
        ext_w2c=gt_meta["extrinsics"],
        image_sizes=_read_image_sizes(gt_meta["image_files"]),
        scale=scene_scale * 0.03,
    )

    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    scene_obj.export(out_glb)
    return out_glb


def _ensure_teacher4v_student_overlay_glb(work_dir: str, dataset_name: str, scene: str) -> Optional[str]:
    """
    teacher_4v (gray) + student LoRA (red)
    """
    return _ensure_two_exp_overlay_glb(
        work_dir,
        dataset_name,
        scene,
        exp_a="teacher_4v",
        exp_b="student",
        out_name="teacher4v_student_overlay_icp",
        color_a=(170, 170, 170),
        color_b=(255, 0, 0),
    )


def _ensure_teacher_teacher4v_overlay_glb(work_dir: str, dataset_name: str, scene: str) -> Optional[str]:
    """
    teacher_4v (gray) + teacher 8v->4v (blue)
    """
    return _ensure_two_exp_overlay_glb(
        work_dir,
        dataset_name,
        scene,
        exp_a="teacher_4v",
        exp_b="teacher",
        out_name="teacher4v_teacher8v4v_overlay_icp",
        color_a=(170, 170, 170),
        color_b=(64, 160, 255),
    )


def _dataset_recon_eval_params(dataset_name: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        from depth_anything_3.utils.constants import (
            ETH3D_DOWN_SAMPLE,
            ETH3D_EVAL_THRESHOLD,
            HIROOM_DOWN_SAMPLE,
            HIROOM_EVAL_THRESHOLD,
            SCANNETPP_DOWN_SAMPLE,
            SCANNETPP_EVAL_THRESHOLD,
            SEVENSCENES_DOWN_SAMPLE,
            SEVENSCENES_EVAL_THRESHOLD,
        )
    except Exception:
        return None, None

    _PARAMS = {
        "hiroom": (HIROOM_EVAL_THRESHOLD, HIROOM_DOWN_SAMPLE),
        "7scenes": (SEVENSCENES_EVAL_THRESHOLD, SEVENSCENES_DOWN_SAMPLE),
        "scannetpp": (SCANNETPP_EVAL_THRESHOLD, SCANNETPP_DOWN_SAMPLE),
        "eth3d": (ETH3D_EVAL_THRESHOLD, ETH3D_DOWN_SAMPLE),
    }
    if dataset_name in _PARAMS:
        return float(_PARAMS[dataset_name][0]), float(_PARAMS[dataset_name][1])
    return None, None


def _load_dataset_gt_pointcloud(dataset_name: str, scene: str):
    try:
        import open3d as o3d
    except Exception:
        return None

    if dataset_name == "hiroom":
        gt_ply = _hiroom_gt_ply_for_scene(scene)
        if not os.path.exists(gt_ply):
            return None
        return o3d.io.read_point_cloud(gt_ply)

    if dataset_name == "7scenes":
        try:
            from depth_anything_3.bench.utils import sample_points_from_mesh
            from depth_anything_3.utils.constants import SEVENSCENES_EVAL_DATA_ROOT, SEVENSCENES_SAMPLING_NUMBER
        except Exception:
            return None
        mesh_path = os.path.join(SEVENSCENES_EVAL_DATA_ROOT, "7Scenes", "meshes", f"{scene}.ply")
        if not os.path.exists(mesh_path):
            return None
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        return sample_points_from_mesh(mesh, num_points=int(min(SEVENSCENES_SAMPLING_NUMBER, 400_000)))

    if dataset_name == "scannetpp":
        try:
            from depth_anything_3.bench.utils import sample_points_from_mesh
            from depth_anything_3.utils.constants import SCANNETPP_EVAL_DATA_ROOT, SCANNETPP_SAMPLING_NUMBER
        except Exception:
            return None
        mesh_path = os.path.join(SCANNETPP_EVAL_DATA_ROOT, scene, "scans", "mesh_aligned_0.05.ply")
        if not os.path.exists(mesh_path):
            return None
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        return sample_points_from_mesh(mesh, num_points=int(min(SCANNETPP_SAMPLING_NUMBER, 400_000)))

    if dataset_name == "eth3d":
        try:
            from depth_anything_3.bench.utils import sample_points_from_mesh
            from depth_anything_3.utils.constants import ETH3D_EVAL_DATA_ROOT, ETH3D_SAMPLING_NUMBER
        except Exception:
            return None
        mesh_path = os.path.join(ETH3D_EVAL_DATA_ROOT, scene, "combined_mesh.ply")
        if not os.path.exists(mesh_path):
            return None
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        return sample_points_from_mesh(mesh, num_points=int(min(ETH3D_SAMPLING_NUMBER, 400_000)))

    return None


def _load_scene_gt_pointcloud_for_overlay(
    work_dir: str,
    dataset_name: str,
    scene: str,
    *,
    gt_meta: Optional[dict] = None,
):
    """
    Load GT point cloud for overlays/metrics.

    Priority:
      1) GT fused from the same 4 benchmark views (teacher_4v gt_meta)
      2) Dataset full GT fallback (mesh/ply)
    """
    try:
        import open3d as o3d
    except Exception:
        return None

    if gt_meta is None:
        gt_meta = _load_gt_meta(work_dir, dataset_name, scene)

    if gt_meta is not None:
        gt_4v_ply = _ensure_gt_fuse_ply_4v(work_dir, dataset_name, scene, gt_meta)
        if gt_4v_ply and os.path.exists(gt_4v_ply):
            return o3d.io.read_point_cloud(gt_4v_ply)

    return _load_dataset_gt_pointcloud(dataset_name, scene)


def _ensure_exp_vs_gt_threshold_overlay_glb(
    work_dir: str,
    dataset_name: str,
    scene: str,
    *,
    exp_key: str,
    out_name: str,
) -> Optional[str]:
    """
    Build prediction-vs-GT overlay where both clouds are threshold-colored:
      - gray: under threshold
      - red: over threshold
    using pred->GT and GT->pred distances (F1-style view).
    """
    threshold, down_sample = _dataset_recon_eval_params(dataset_name)
    if threshold is None:
        return None

    # Double threshold for hiroom visualization
    if dataset_name == "hiroom":
        threshold = threshold * 2.0

    gt_meta = _load_gt_meta(work_dir, dataset_name, scene)
    if gt_meta is None:
        return None

    pred_ply = _fused_ply_path(work_dir, exp_key, dataset_name, scene)
    if not os.path.exists(pred_ply):
        return None

    out_glb = os.path.join(
        work_dir,
        "comparisons",
        dataset_name,
        out_name,
        _scene_slug(scene),
        "scene.glb",
    )
    if os.path.exists(out_glb):
        return out_glb

    try:
        import numpy as np
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

    gt_pcd = _load_scene_gt_pointcloud_for_overlay(
        work_dir,
        dataset_name,
        scene,
        gt_meta=gt_meta,
    )
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
        idx = rng.choice(pred_points.shape[0], size=350_000, replace=False)
        pred_points = pred_points[idx]
    if gt_points.shape[0] > 350_000:
        idx = rng.choice(gt_points.shape[0], size=350_000, replace=False)
        gt_points = gt_points[idx]

    dist_pred_to_gt = nn_correspondance(gt_points, pred_points)
    dist_gt_to_pred = nn_correspondance(pred_points, gt_points)
    pred_good = dist_pred_to_gt < float(threshold)
    gt_good = dist_gt_to_pred < float(threshold)

    # Same semantics (gray/red) for both sets, with different intensity to keep them distinguishable.
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

    pred_pc = trimesh.points.PointCloud(vertices=pred_points_t, colors=pred_colors)
    gt_pc = trimesh.points.PointCloud(vertices=gt_points_t, colors=gt_colors)
    scene_obj.add_geometry(gt_pc)
    scene_obj.add_geometry(pred_pc)

    all_points = np.concatenate([gt_points_t, pred_points_t], axis=0)
    scene_scale = _estimate_scene_scale(all_points, fallback=1.0)
    _add_cameras_to_scene(
        scene=scene_obj,
        K=gt_meta["intrinsics"],
        ext_w2c=gt_meta["extrinsics"],
        image_sizes=_read_image_sizes(gt_meta["image_files"]),
        scale=scene_scale * 0.03,
    )

    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    scene_obj.export(out_glb)
    return out_glb


def _exp_vs_gt_threshold_overlay_glb_path(work_dir: str, dataset_name: str, scene: str, out_name: str) -> str:
    return os.path.join(
        work_dir,
        "comparisons",
        dataset_name,
        out_name,
        _scene_slug(scene),
        "scene.glb",
    )


def _ensure_student_vs_teacher4v_delta_overlay_glb(
    work_dir: str,
    dataset_name: str,
    scene: str,
) -> Optional[str]:
    """
    Build a point cloud overlay comparing student vs teacher_4v fused reconstructions.
    Per-point coloring based on distance-to-GT difference:
      - gray: |student_dist - teacher4v_dist| <= threshold (similar quality)
      - blue: student closer to GT by > threshold (student better)
      - red: student farther from GT by > threshold (student worse)
    Only shows the student point cloud colored by the delta.
    """
    threshold, down_sample = _dataset_recon_eval_params(dataset_name)
    if threshold is None:
        return None

    gt_meta = _load_gt_meta(work_dir, dataset_name, scene)
    if gt_meta is None:
        return None

    student_ply = _fused_ply_path(work_dir, "student", dataset_name, scene)
    teacher4v_ply = _fused_ply_path(work_dir, "teacher_4v", dataset_name, scene)
    if not os.path.exists(student_ply) or not os.path.exists(teacher4v_ply):
        return None

    out_glb = _exp_vs_gt_threshold_overlay_glb_path(
        work_dir, dataset_name, scene,
        "student_vs_teacher4v_delta_threshold",
    )
    if os.path.exists(out_glb):
        return out_glb

    try:
        import numpy as np
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

    gt_pcd = _load_scene_gt_pointcloud_for_overlay(
        work_dir, dataset_name, scene, gt_meta=gt_meta,
    )
    if gt_pcd is None:
        return None

    student_pcd = o3d.io.read_point_cloud(student_ply)
    teacher4v_pcd = o3d.io.read_point_cloud(teacher4v_ply)
    if down_sample is not None and down_sample > 0:
        student_pcd = student_pcd.voxel_down_sample(down_sample)
        teacher4v_pcd = teacher4v_pcd.voxel_down_sample(down_sample)
        gt_pcd = gt_pcd.voxel_down_sample(down_sample)

    student_pts = np.asarray(student_pcd.points, dtype=np.float32)
    teacher4v_pts = np.asarray(teacher4v_pcd.points, dtype=np.float32)
    gt_pts = np.asarray(gt_pcd.points, dtype=np.float32)
    if student_pts.shape[0] == 0 or teacher4v_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return None

    rng = np.random.default_rng(seed=42)
    max_pts = 350_000
    if student_pts.shape[0] > max_pts:
        idx = rng.choice(student_pts.shape[0], size=max_pts, replace=False)
        student_pts = student_pts[idx]
    if teacher4v_pts.shape[0] > max_pts:
        idx = rng.choice(teacher4v_pts.shape[0], size=max_pts, replace=False)
        teacher4v_pts = teacher4v_pts[idx]
    if gt_pts.shape[0] > max_pts:
        idx = rng.choice(gt_pts.shape[0], size=max_pts, replace=False)
        gt_pts = gt_pts[idx]

    # Distance from each point cloud to GT
    tree_gt = cKDTree(gt_pts)
    s_dist_to_gt, _ = tree_gt.query(student_pts, k=1)
    t4v_dist_to_gt, _ = tree_gt.query(teacher4v_pts, k=1)

    thr = float(threshold)

    # Color student points: gray if within threshold of GT, else blue/red by comparison
    # For student points outside threshold, find nearest teacher4v point to compare
    tree_t4v = cKDTree(teacher4v_pts)
    _, t4v_nn_for_student = tree_t4v.query(student_pts, k=1)
    t4v_paired_dist = t4v_dist_to_gt[t4v_nn_for_student]

    s_colors = np.tile(np.array([170, 170, 170], dtype=np.uint8), (student_pts.shape[0], 1))
    s_outside = s_dist_to_gt > thr
    # Outside threshold: blue if student closer to GT than paired teacher4v, red otherwise
    s_colors[s_outside & (s_dist_to_gt < t4v_paired_dist)] = np.array([0, 100, 255], dtype=np.uint8)
    s_colors[s_outside & (s_dist_to_gt >= t4v_paired_dist)] = np.array([255, 0, 0], dtype=np.uint8)

    # Color teacher4v points: gray if within threshold of GT, else blue/red by comparison
    tree_s = cKDTree(student_pts)
    _, s_nn_for_t4v = tree_s.query(teacher4v_pts, k=1)
    s_paired_dist = s_dist_to_gt[s_nn_for_t4v]

    t4v_colors = np.tile(np.array([120, 120, 120], dtype=np.uint8), (teacher4v_pts.shape[0], 1))
    t4v_outside = t4v_dist_to_gt > thr
    # Outside threshold: blue if student (paired) closer to GT, red if teacher4v closer
    t4v_colors[t4v_outside & (s_paired_dist < t4v_dist_to_gt)] = np.array([0, 70, 200], dtype=np.uint8)
    t4v_colors[t4v_outside & (s_paired_dist >= t4v_dist_to_gt)] = np.array([200, 0, 0], dtype=np.uint8)

    A = _compute_alignment_transform_first_cam_glTF_center_by_points(gt_meta["extrinsics"][0], gt_pts)
    student_pts_t = trimesh.transform_points(student_pts, A)
    teacher4v_pts_t = trimesh.transform_points(teacher4v_pts, A)

    scene_obj = trimesh.Scene()
    if scene_obj.metadata is None:
        scene_obj.metadata = {}
    scene_obj.metadata["hf_alignment"] = A

    s_pc = trimesh.points.PointCloud(vertices=student_pts_t, colors=s_colors)
    t4v_pc = trimesh.points.PointCloud(vertices=teacher4v_pts_t, colors=t4v_colors)
    scene_obj.add_geometry(t4v_pc)
    scene_obj.add_geometry(s_pc)

    all_points = np.concatenate([student_pts_t, teacher4v_pts_t], axis=0)
    scene_scale = _estimate_scene_scale(all_points, fallback=1.0)
    _add_cameras_to_scene(
        scene=scene_obj,
        K=gt_meta["intrinsics"],
        ext_w2c=gt_meta["extrinsics"],
        image_sizes=_read_image_sizes(gt_meta["image_files"]),
        scale=scene_scale * 0.03,
    )

    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    scene_obj.export(out_glb)
    return out_glb


def _compute_exp_vs_gt_f1_metrics(
    work_dir: str,
    dataset_name: str,
    scene: str,
    *,
    exp_key: str,
) -> Optional[dict]:
    """
    Compute precision/recall/fscore with the same threshold/downsample convention as benchmark eval.
    """
    threshold, down_sample = _dataset_recon_eval_params(dataset_name)
    if threshold is None:
        return None

    pred_ply = _fused_ply_path(work_dir, exp_key, dataset_name, scene)
    if not os.path.exists(pred_ply):
        return None

    try:
        import open3d as o3d
        from depth_anything_3.bench.utils import evaluate_3d_reconstruction
    except Exception:
        return None

    gt_meta = _load_gt_meta(work_dir, dataset_name, scene)
    gt_pcd = _load_scene_gt_pointcloud_for_overlay(
        work_dir,
        dataset_name,
        scene,
        gt_meta=gt_meta,
    )
    if gt_pcd is None:
        return None

    pred_pcd = o3d.io.read_point_cloud(pred_ply)
    metrics = evaluate_3d_reconstruction(
        pred_pcd,
        gt_pcd,
        threshold=float(threshold),
        down_sample=float(down_sample) if down_sample is not None else None,
    )
    return {
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "fscore": float(metrics.get("fscore", 0.0)),
        "threshold": float(threshold),
    }


def _check_threshold_overlay_deps() -> Optional[str]:
    missing: List[str] = []
    try:
        import open3d  # noqa: F401
    except Exception:
        missing.append("open3d")
    try:
        import trimesh  # noqa: F401
    except Exception:
        missing.append("trimesh")
    if missing:
        return ", ".join(missing)
    return None


def _ensure_gt_assets(work_dir: str, dataset_name: str, scene: str) -> Tuple[Optional[str], List[str]]:
    """
    Ensure GT (GLB + depth_vis JPGs) exist under
    `<work_dir>/gt/visualizations/<dataset>/<scene_slug>/`.
    Returns: (gt_glb_path, gt_depth_vis_images)
    """
    gt_vis_dir = os.path.join(work_dir, GT_KEY, "visualizations", dataset_name, _scene_slug(scene))
    gt_glb = os.path.join(gt_vis_dir, "scene.glb")
    gt_depth_dir = os.path.join(gt_vis_dir, "depth_vis")

    _SUPPORTED_GT_VIS = {"hiroom", "7scenes", "scannetpp", "eth3d"}
    if dataset_name not in _SUPPORTED_GT_VIS:
        return None, []

    gt_meta = _load_gt_meta(work_dir, dataset_name, scene)
    if gt_meta is None:
        return None, []

    # Image sizes for camera frustums.
    image_sizes = _read_image_sizes(gt_meta["image_files"])

    # 1) GT point cloud fused from the same 4 views -> GLB with GT cameras
    gt_ply = _ensure_gt_fuse_ply_4v(work_dir, dataset_name, scene, gt_meta)
    if gt_ply is None:
        return None, []
    gt_glb_out = _ensure_glb_from_ply_with_cameras(
        ply_path=gt_ply,
        out_glb_path=gt_glb,
        extrinsics_w2c=gt_meta["extrinsics"],
        intrinsics=gt_meta["intrinsics"],
        image_sizes=image_sizes,
    )

    # 2) GT depth_vis (RGB + depth)
    gt_depth_imgs: List[str] = []
    if os.path.isdir(gt_depth_dir):
        gt_depth_imgs = sorted(glob.glob(os.path.join(gt_depth_dir, "*.jpg")))

    if not gt_depth_imgs:
        try:
            import numpy as np

            from depth_anything_3.specs import Prediction
            from depth_anything_3.utils.export.depth_vis import export_to_depth_vis

            images = []
            depths = []
            for img_path in gt_meta["image_files"]:
                rgb = _read_rgb_image(img_path)
                if rgb is None:
                    continue

                depth_path, mask_path = _read_gt_depth_and_mask(dataset_name, img_path)
                if depth_path is None:
                    continue

                depth_m = _load_raw_gt_depth(dataset_name, depth_path, image_hw=(rgb.shape[0], rgb.shape[1]))
                if depth_m is None:
                    continue

                valid = np.isfinite(depth_m) & (depth_m > 0)
                if mask_path is not None:
                    mask_img = _read_mask_image(mask_path)
                    if mask_img is not None:
                        if dataset_name == "eth3d":
                            valid = _eth3d_valid_mask(depth_m, mask_img)
                        else:
                            valid = valid & (~(mask_img > 0))
                depth_m = depth_m * valid.astype(np.float32)

                # Ensure RGB and depth match shapes; if not, resize depth to RGB.
                if depth_m.shape[0] != rgb.shape[0] or depth_m.shape[1] != rgb.shape[1]:
                    import cv2
                    depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

                images.append(rgb)
                depths.append(depth_m)

            if images and depths:
                # Normalize GT depth to relative scale (like predictions) for consistent visualization
                # Use median-ratio normalization to match predicted depth scale
                depths_normalized = []
                for depth_m in depths:
                    valid = (depth_m > 0) & np.isfinite(depth_m)
                    if valid.sum() > 100:
                        # Normalize to ~1.0 median (typical relative depth scale)
                        med = np.median(depth_m[valid])
                        depth_normalized = depth_m / med
                    else:
                        depth_normalized = depth_m
                    depths_normalized.append(depth_normalized)

                pred = Prediction(
                    depth=np.stack(depths_normalized, axis=0),
                    is_metric=0,  # Use relative scale visualization
                    processed_images=np.stack(images, axis=0),
                )
                os.makedirs(gt_vis_dir, exist_ok=True)
                export_to_depth_vis(pred, gt_vis_dir)
                gt_depth_imgs = sorted(glob.glob(os.path.join(gt_depth_dir, "*.jpg")))
        except Exception:
            gt_depth_imgs = []

    return gt_glb_out, gt_depth_imgs


def _ensure_exp_fused_glb(work_dir: str, exp_key: str, dataset_name: str, scene: str) -> Optional[str]:
    """
    Convert recon_unposed fused point cloud to GLB (aligned to GT), with GT camera poses.
    """
    if exp_key not in EXPERIMENTS:
        return None

    gt_meta = _load_gt_meta(work_dir, dataset_name, scene)
    if gt_meta is None:
        return None

    fuse_ply = os.path.join(
        work_dir,
        exp_key,
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "fuse",
        "pcd.ply",
    )
    out_glb = os.path.join(work_dir, exp_key, "visualizations_fuse", dataset_name, _scene_slug(scene), "scene.glb")

    # Image sizes for camera frustums.
    image_sizes = _read_image_sizes(gt_meta["image_files"])

    return _ensure_glb_from_ply_with_cameras(
        ply_path=fuse_ply,
        out_glb_path=out_glb,
        extrinsics_w2c=gt_meta["extrinsics"],
        intrinsics=gt_meta["intrinsics"],
        image_sizes=image_sizes,
    )


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


def _load_prediction_depths(work_dir: str, exp_key: str, dataset_name: str, scene: str):
    path = _results_npz_path(work_dir, exp_key, dataset_name, scene)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        depth = data.get("depth")
        if depth is None:
            return None
        return np.asarray(depth, dtype=np.float32)
    except Exception:
        return None


def _load_gt_depth_map(dataset_name: str, img_path: str):
    depth_path, mask_path = _read_gt_depth_and_mask(dataset_name, img_path)
    if depth_path is None:
        return None, None

    # Read RGB to get image dimensions (needed for ETH3D binary format)
    rgb = _read_rgb_image(img_path)
    image_hw = (rgb.shape[0], rgb.shape[1]) if rgb is not None else None

    depth_m = _load_raw_gt_depth(dataset_name, depth_path, image_hw=image_hw)
    if depth_m is None:
        return None, None

    valid = np.isfinite(depth_m) & (depth_m > 0)
    if mask_path is not None:
        mask_img = _read_mask_image(mask_path)
        if mask_img is not None:
            if dataset_name == "eth3d":
                valid = _eth3d_valid_mask(depth_m, mask_img)
            else:
                valid = valid & (~(mask_img > 0))
    return depth_m.astype(np.float32), valid


def _colorize_scalar_map(values: np.ndarray, valid: np.ndarray, *, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    import matplotlib

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    norm = ((values - vmin) / (vmax - vmin)).clip(0.0, 1.0)
    cm = matplotlib.colormaps[cmap_name]
    rgb = (cm(norm, bytes=False)[..., :3] * 255.0).astype(np.uint8)
    out = np.zeros_like(rgb, dtype=np.uint8)
    out[valid] = rgb[valid]
    return out


def _prepare_depth_error_galleries(work_dir: str, dataset_name: str, scene: str):
    """
    Build per-view error maps against GT for teacher/teacher_4v/student.
    Returns 4 galleries + markdown summary.
    """
    _SUPPORTED_DEPTH_ERROR = {"hiroom", "7scenes", "scannetpp", "eth3d"}
    if dataset_name not in _SUPPORTED_DEPTH_ERROR:
        return [], [], [], [], f"Depth error maps are not available for {dataset_name}."

    gt_meta = _load_gt_meta(work_dir, dataset_name, scene)
    if gt_meta is None:
        return [], [], [], [], "No GT meta found for this scene."

    try:
        import cv2
    except Exception as e:
        return [], [], [], [], f"Missing deps for depth error maps: {e}"

    pred_depths = {
        "teacher": _load_prediction_depths(work_dir, "teacher", dataset_name, scene),
        "teacher_4v": _load_prediction_depths(work_dir, "teacher_4v", dataset_name, scene),
        "student": _load_prediction_depths(work_dir, "student", dataset_name, scene),
    }
    threshold, _ = _dataset_recon_eval_params(dataset_name)
    if threshold is None:
        # Conservative fallback in meters.
        threshold = 0.05

    galleries = {
        "teacher": [],
        "teacher_4v": [],
        "student": [],
        "student_minus_teacher4v": [],
    }
    mae_stats = {"teacher": [], "teacher_4v": [], "student": []}
    inlier_stats = {"teacher": [], "teacher_4v": [], "student": []}

    image_files = gt_meta.get("image_files") or []
    for idx, img_path in enumerate(image_files):
        rgb = _read_rgb_image(img_path)
        if rgb is None:
            continue
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb.astype(np.uint8)

        gt_depth, gt_valid = _load_gt_depth_map(dataset_name, img_path)
        if gt_depth is None or gt_valid is None:
            continue

        # Keep all comparison images at GT size.
        if gt_depth.shape[0] != rgb.shape[0] or gt_depth.shape[1] != rgb.shape[1]:
            gt_depth = cv2.resize(gt_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_valid = cv2.resize(gt_valid.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        err_maps = {}
        for exp_key in ["teacher", "teacher_4v", "student"]:
            depths = pred_depths.get(exp_key)
            if depths is None or idx >= len(depths):
                continue
            pred = np.asarray(depths[idx], dtype=np.float32)
            if pred.shape[:2] != gt_depth.shape[:2]:
                pred = cv2.resize(pred, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_NEAREST)

            valid = gt_valid & np.isfinite(pred) & (pred > 0)
            if valid.sum() < 20:
                continue

            # Per-view robust scale alignment to GT to expose geometric differences.
            pred_med = float(np.median(pred[valid]))
            gt_med = float(np.median(gt_depth[valid]))
            scale = gt_med / max(pred_med, 1e-6)
            pred_scaled = pred * scale

            err = np.full_like(gt_depth, np.nan, dtype=np.float32)
            err[valid] = np.abs(pred_scaled[valid] - gt_depth[valid])
            err_maps[exp_key] = err
            mae_stats[exp_key].append(float(np.nanmean(err)))
            inlier_stats[exp_key].append(float(np.mean((err[valid] <= float(threshold)).astype(np.float32))))

        if not err_maps:
            continue

        for exp_key in ["teacher", "teacher_4v", "student"]:
            err = err_maps.get(exp_key)
            if err is None:
                continue
            valid = np.isfinite(err)
            err_rgb = np.zeros((err.shape[0], err.shape[1], 3), dtype=np.uint8)
            # Threshold-style coloring (same as point-cloud F-score view):
            # gray <= threshold, red > threshold.
            good = err <= float(threshold)
            err_rgb[valid] = np.array([255, 0, 0], dtype=np.uint8)
            err_rgb[valid & good] = np.array([170, 170, 170], dtype=np.uint8)
            panel = np.concatenate([rgb, err_rgb], axis=1)
            galleries[exp_key].append(panel)

        # Delta panel: compare student vs teacher_4v per-pixel errors against GT.
        if "student" in err_maps and "teacher_4v" in err_maps:
            s_err = err_maps["student"]
            t_err = err_maps["teacher_4v"]
            valid = np.isfinite(s_err) & np.isfinite(t_err)
            if valid.any():
                thr = float(threshold)
                s_in = s_err <= thr
                t_in = t_err <= thr

                delta_rgb = np.zeros((s_err.shape[0], s_err.shape[1], 3), dtype=np.uint8)
                # Both within threshold -> gray
                delta_rgb[valid & s_in & t_in] = np.array([170, 170, 170], dtype=np.uint8)
                # Both outside threshold -> blue if student closer, red otherwise
                both_out = valid & (~s_in) & (~t_in)
                delta_rgb[both_out & (s_err < t_err)] = np.array([0, 100, 255], dtype=np.uint8)
                delta_rgb[both_out & (s_err >= t_err)] = np.array([255, 0, 0], dtype=np.uint8)
                # Only student within threshold -> blue (student better)
                delta_rgb[valid & s_in & (~t_in)] = np.array([0, 100, 255], dtype=np.uint8)
                # Only teacher within threshold -> red (teacher better)
                delta_rgb[valid & (~s_in) & t_in] = np.array([255, 0, 0], dtype=np.uint8)

                delta_panel = np.concatenate([rgb, delta_rgb], axis=1)
                galleries["student_minus_teacher4v"].append(delta_panel)

    def _mean_or_nan(vals):
        return float(np.mean(vals)) if vals else float("nan")

    summary = (
        "Depth error maps use GT from the same 4 shared views (teacher_4v meta), "
        f"with per-view robust scale alignment and threshold coloring at `{float(threshold):.4f} m`.\n\n"
        f"- Teacher MAE: `{_mean_or_nan(mae_stats['teacher']):.4f}`\n"
        f"- Teacher 4v MAE: `{_mean_or_nan(mae_stats['teacher_4v']):.4f}`\n"
        f"- Student MAE: `{_mean_or_nan(mae_stats['student']):.4f}`\n"
        f"- Teacher inlier-rate (<=thr): `{_mean_or_nan(inlier_stats['teacher']):.4f}`\n"
        f"- Teacher 4v inlier-rate (<=thr): `{_mean_or_nan(inlier_stats['teacher_4v']):.4f}`\n"
        f"- Student inlier-rate (<=thr): `{_mean_or_nan(inlier_stats['student']):.4f}`\n"
        "- Delta panel (`Student vs Teacher4v`): gray = both within threshold, blue = student better (closer to GT or within threshold), red = teacher4v better."
    )

    return (
        galleries["teacher"],
        galleries["teacher_4v"],
        galleries["student"],
        galleries["student_minus_teacher4v"],
        summary,
    )


def _format_metrics_md(work_dir: str, dataset_name: str, scene: str) -> str:
    """
    Build a compact markdown table from saved JSON metrics.
    """
    cols = ["GT", *[EXPERIMENTS[k] for k in EXPERIMENTS.keys()]]
    rows = []

    def get_pose(exp_key: str) -> Optional[dict]:
        p = os.path.join(work_dir, exp_key, "metric_results", f"{dataset_name}_pose.json")
        data = _load_json(p) or {}
        return data.get(scene) or data.get("mean")

    def get_recon(exp_key: str) -> Optional[dict]:
        p = os.path.join(work_dir, exp_key, "metric_results", f"{dataset_name}_recon_unposed.json")
        data = _load_json(p) or {}
        return data.get(scene) or data.get("mean")

    def fmt(x: Optional[float]) -> str:
        if x is None:
            return "—"
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "—"

    # Pose AUC03/AUC30 + Recon F-score/Overall (matching the old static table).
    auc03 = ["AUC03"]
    auc30 = ["AUC30"]
    fscore = ["F-score"]
    overall = ["Overall"]

    auc03.append("—")
    auc30.append("—")
    fscore.append("—")
    overall.append("—")

    for exp_key in EXPERIMENTS.keys():
        pose = get_pose(exp_key) or {}
        recon = get_recon(exp_key) or {}
        auc03.append(fmt(pose.get("auc03")))
        auc30.append(fmt(pose.get("auc30")))
        fscore.append(fmt(recon.get("fscore")))
        overall.append(fmt(recon.get("overall")))

    rows = [auc03, auc30, fscore, overall]

    # Build markdown table.
    header = "| Metric | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = "\n".join(["| " + " | ".join(r) + " |" for r in rows])
    return "\n".join([header, sep, body])


def build_app(*, work_dir: str, image_dir: str, dataset_name: str, default_scene: Optional[str] = None):
    scenes = _list_scenes(work_dir, dataset_name)
    if scenes:
        scene_choices = scenes
        if default_scene and default_scene in scenes:
            init_scene = default_scene
        else:
            init_scene = scenes[0]
    else:
        scene_choices = [NO_SCENE_SENTINEL]
        init_scene = NO_SCENE_SENTINEL

    dark_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.gray,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
    ).set(
        body_background_fill="#111111",
        block_background_fill="#1a1a1a",
        block_label_background_fill="#222222",
        block_label_text_color="#e0e0e0",
        block_title_text_color="#e0e0e0",
        panel_background_fill="#1a1a1a",
        input_background_fill="#222222",
        button_primary_background_fill="#333333",
        button_primary_text_color="#e0e0e0",
        button_secondary_background_fill="#282828",
        button_secondary_text_color="#e0e0e0",
        border_color_primary="#444444",
        background_fill_primary="#1a1a1a",
        background_fill_secondary="#151515",
    )
    blocks_kwargs = {
        "title": "Benchmark Point Cloud Comparison",
        "theme": dark_theme,
        "css": """
            * { color-scheme: dark !important; }
            body, .gradio-container { background-color: #111111 !important; color: #e0e0e0 !important; }
            label, span, p, h1, h2, h3, h4, h5, h6, a, td, th,
            .gr-markdown, .gr-markdown * { color: #e0e0e0 !important; }
        """,
    }

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(f"# Benchmark Viewer — {dataset_name} Teacher/Student (+ GT)")

        with gr.Row():
            scene_dd = gr.Dropdown(
                choices=scene_choices,
                value=init_scene,
                label="Scene",
                interactive=bool(scenes),
            )
            view_mode = gr.Radio(
                choices=["Raw (scene.glb)", "Fused (pcd.ply aligned)"],
                value="Fused (pcd.ply aligned)",
                label="3D View Mode",
                interactive=True,
            )

        info_md = gr.Markdown("")

        # Input images
        gr.Markdown("### Input Frames")
        input_gallery = gr.Gallery(value=[], columns=4, height=200, label="Input frames")

        # 2x2 grid comparison with GT fused point cloud
        gr.Markdown("### 3D Point Cloud Comparison (2x2)")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**GT Fused 4v**")
                gt_fused_model = gr.Model3D(
                    value=None,
                    label="GT Fused 4v",
                    height=480,
                    clear_color=(1.0, 1.0, 1.0, 1.0),
                    elem_id="cmp_gt_fused4v",
                )
            with gr.Column():
                gr.Markdown("**Teacher 8v→4v (shared 4 views)**")
                teacher_gt_model = gr.Model3D(
                    value=None,
                    label="Teacher 8v→4v",
                    height=480,
                    clear_color=(1.0, 1.0, 1.0, 1.0),
                    elem_id="cmp_gt_teacher8v4v",
                )
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"**{EXPERIMENTS['teacher_4v']}**")
                teacher4v_model = gr.Model3D(
                    value=None,
                    label=EXPERIMENTS["teacher_4v"],
                    height=480,
                    clear_color=(1.0, 1.0, 1.0, 1.0),
                    elem_id="cmp_teacher4v",
                )
            with gr.Column():
                gr.Markdown(f"**{EXPERIMENTS['student']}**")
                student_model = gr.Model3D(
                    value=None,
                    label=EXPERIMENTS["student"],
                    height=480,
                    clear_color=(1.0, 1.0, 1.0, 1.0),
                    elem_id="cmp_student4v",
                )

        gr.Markdown("### GT Threshold Overlay (F-score style)")
        gr.Markdown(
            "Per-point threshold coloring (same logic as recon F-score): "
            "gray=within threshold, red=outside threshold. "
            "Pred cloud uses brighter gray/red; GT cloud uses darker gray/red."
        )
        render_thresh_btn = gr.Button("Render GT Threshold Overlays (fast, cached)")
        build_thresh_btn = gr.Button("Build Missing GT Threshold Overlays (slow)")
        with gr.Row():
            teacher_gt_thresh_model = gr.Model3D(
                value=None,
                label="Teacher 8v→4v vs GT threshold overlay",
                height=520,
                clear_color=(1.0, 1.0, 1.0, 1.0),
                elem_id="thr_teacher8v4v",
            )
            student_gt_thresh_model = gr.Model3D(
                value=None,
                label="Student (LoRA) vs GT threshold overlay",
                height=520,
                clear_color=(1.0, 1.0, 1.0, 1.0),
                elem_id="thr_student4v",
            )
            teacher4v_gt_thresh_model = gr.Model3D(
                value=None,
                label="Teacher 4v vs GT threshold overlay",
                height=520,
                clear_color=(1.0, 1.0, 1.0, 1.0),
                elem_id="thr_teacher4v",
            )
        f1_overlay_md = gr.Markdown("")

        gr.Markdown("### Student vs Teacher 4v Delta Overlay")
        gr.Markdown(
            "Student point cloud colored by comparison to teacher_4v (both vs GT): "
            "gray = both within threshold of GT, "
            "blue = student closer to GT (better), "
            "red = teacher_4v closer to GT (student worse)."
        )
        render_delta_btn = gr.Button("Render Student vs Teacher4v Delta (fast, cached)")
        build_delta_btn = gr.Button("Build Student vs Teacher4v Delta (slow)")
        student_t4v_delta_model = gr.Model3D(
            value=None,
            label="Student vs Teacher4v delta threshold overlay",
            height=520,
            clear_color=(1.0, 1.0, 1.0, 1.0),
            elem_id="delta_student_t4v",
        )
        delta_overlay_md = gr.Markdown("")

        # Depth visualizations
        gr.Markdown("### Depth Visualizations (RGB + Depth per view)")
        with gr.Tab("Ground Truth"):
            gt_depth_gallery = gr.Gallery(value=[], columns=4, height=250, label="GT depth_vis")
        exp_depth_galleries = {}
        for exp_key, exp_label in EXPERIMENTS.items():
            with gr.Tab(exp_label):
                exp_depth_galleries[exp_key] = gr.Gallery(
                    value=[],
                    columns=4,
                    height=250,
                    label=f"{exp_label} depth_vis (from inference export)",
                )

        gr.Markdown("### Depth Error vs GT (RGB | Threshold Map)")
        depth_error_md = gr.Markdown("")
        with gr.Tab("Teacher 8v→4v error") as teacher_err_tab:
            teacher_err_gallery = gr.Gallery(value=[], columns=2, height=260, label="Teacher error vs GT")
        with gr.Tab("Teacher 4v error") as teacher4v_err_tab:
            teacher4v_err_gallery = gr.Gallery(value=[], columns=2, height=260, label="Teacher 4v error vs GT")
        with gr.Tab("Student 4v error") as student_err_tab:
            student_err_gallery = gr.Gallery(value=[], columns=2, height=260, label="Student error vs GT")
        with gr.Tab("Student - Teacher4v") as delta_err_tab:
            delta_err_gallery = gr.Gallery(value=[], columns=2, height=260, label="Signed error delta")

        # Metrics summary
        gr.Markdown("### Benchmark Results")
        metrics_md = gr.Markdown("")

        threshold_cache: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], str]] = {}
        depth_error_cache: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], str]] = {}

        def _update(scene: str, mode: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return (
                    "No scenes found under work_dir.",
                    [],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "",
                    [],
                    [],
                    [],
                    [],
                    "No depth error maps.",
                    [],
                    [],
                    [],
                    [],
                    "",
                )

            # Info
            meta = _load_gt_meta(work_dir, dataset_name, scene)
            meta_path = meta["meta_path"] if meta else None
            meta_source_exp = meta.get("meta_source_exp") if meta else None
            info = (
                f"work_dir: `{os.path.abspath(work_dir)}`"
                f"  \ndataset: `{dataset_name}`"
                f"  \nscene: `{scene}`"
            )
            if meta_path:
                info += f"  \nGT meta: `{meta_path}`"
            if meta_source_exp:
                info += f"  \nGT source views: `{meta_source_exp}` (4-view only)"

            # Input images: prefer `image_dir` (8 sampled frames), fallback to GT meta image_files (4 views)
            input_imgs = _find_input_images(image_dir, dataset_name, scene)
            if not input_imgs and meta:
                input_imgs = meta["image_files"]

            # Ensure GT assets (GLB + depth_vis) are generated from the 4 shared views.
            gt_glb, gt_depth_imgs_raw = _ensure_gt_assets(work_dir, dataset_name, scene)
            gt_depth_imgs = _find_depth_vis_images(work_dir, GT_KEY, dataset_name, scene)
            teacher_gt_thresh_glb = None
            student_gt_thresh_glb = None
            teacher4v_gt_thresh_glb = None
            f1_overlay_text = "Click 'Render GT Threshold Overlays (fast, cached)' for instant load."
            teacher_err_imgs, teacher4v_err_imgs, student_err_imgs, delta_err_imgs = [], [], [], []
            depth_err_summary = "Switch to a Depth Error tab to render this scene."

            # Experiment model paths
            exp_paths = {}
            if mode.startswith("Fused"):
                for exp_key in EXPERIMENTS.keys():
                    exp_paths[exp_key] = _ensure_exp_fused_glb(work_dir, exp_key, dataset_name, scene)
            else:
                for exp_key in EXPERIMENTS.keys():
                    exp_paths[exp_key] = _find_raw_glb(work_dir, exp_key, dataset_name, scene)

            # Experiment depth_vis (per scene)
            exp_depth = {
                exp_key: _find_depth_vis_images(work_dir, exp_key, dataset_name, scene)
                for exp_key in EXPERIMENTS.keys()
            }

            md = _format_metrics_md(work_dir, dataset_name, scene)

            return (
                info,
                input_imgs,
                gt_glb,
                exp_paths.get("teacher"),
                exp_paths.get("teacher_4v"),
                exp_paths.get("student"),
                teacher_gt_thresh_glb,
                student_gt_thresh_glb,
                teacher4v_gt_thresh_glb,
                f1_overlay_text,
                teacher_err_imgs,
                teacher4v_err_imgs,
                student_err_imgs,
                delta_err_imgs,
                depth_err_summary,
                gt_depth_imgs,
                exp_depth.get("teacher") or [],
                exp_depth.get("teacher_4v") or [],
                exp_depth.get("student") or [],
                md,
            )

        def _load_threshold_cached(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return None, None, None, "No valid scene selected."

            if scene in threshold_cache:
                return threshold_cache[scene]

            teacher_path = _exp_vs_gt_threshold_overlay_glb_path(
                work_dir,
                dataset_name,
                scene,
                "teacher8v4v_vs_gt_threshold_f1style_v3",
            )
            student_path = _exp_vs_gt_threshold_overlay_glb_path(
                work_dir,
                dataset_name,
                scene,
                "student_vs_gt_threshold_f1style_v3",
            )
            teacher4v_path = _exp_vs_gt_threshold_overlay_glb_path(
                work_dir,
                dataset_name,
                scene,
                "teacher4v_vs_gt_threshold_f1style_v3",
            )
            teacher_gt_thresh_glb = teacher_path if os.path.exists(teacher_path) else None
            student_gt_thresh_glb = student_path if os.path.exists(student_path) else None
            teacher4v_gt_thresh_glb = teacher4v_path if os.path.exists(teacher4v_path) else None

            missing = []
            if teacher_gt_thresh_glb is None:
                missing.append("teacher 8v->4v")
            if student_gt_thresh_glb is None:
                missing.append("student")
            if teacher4v_gt_thresh_glb is None:
                missing.append("teacher 4v")

            if missing:
                f1_overlay_text = (
                    "Loaded cached overlays; missing: "
                    + ", ".join(missing)
                    + ". Click 'Build Missing GT Threshold Overlays (slow)' if you need them."
                )
            else:
                f1_overlay_text = "Loaded cached GT threshold overlays."

            out = (teacher_gt_thresh_glb, student_gt_thresh_glb, teacher4v_gt_thresh_glb, f1_overlay_text)
            threshold_cache[scene] = out
            return out

        def _build_threshold_for_scene(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return None, None, None, "No valid scene selected."

            missing_deps = _check_threshold_overlay_deps()
            if missing_deps:
                return (
                    None,
                    None,
                    None,
                    "Cannot build overlays: missing dependencies `"
                    + missing_deps
                    + "`. Install them in your runtime environment first.",
                )

            teacher_gt_thresh_glb = _ensure_exp_vs_gt_threshold_overlay_glb(
                work_dir,
                dataset_name,
                scene,
                exp_key="teacher",
                out_name="teacher8v4v_vs_gt_threshold_f1style_v3",
            )
            student_gt_thresh_glb = _ensure_exp_vs_gt_threshold_overlay_glb(
                work_dir,
                dataset_name,
                scene,
                exp_key="student",
                out_name="student_vs_gt_threshold_f1style_v3",
            )
            teacher4v_gt_thresh_glb = _ensure_exp_vs_gt_threshold_overlay_glb(
                work_dir,
                dataset_name,
                scene,
                exp_key="teacher_4v",
                out_name="teacher4v_vs_gt_threshold_f1style_v3",
            )

            if teacher_gt_thresh_glb or student_gt_thresh_glb or teacher4v_gt_thresh_glb:
                f1_overlay_text = "Finished building available GT threshold overlays."
            else:
                f1_overlay_text = (
                    "Failed to build GT threshold overlays for this scene. "
                    "Please check GT data/dependencies (open3d, trimesh) and scene assets."
                )

            out = (teacher_gt_thresh_glb, student_gt_thresh_glb, teacher4v_gt_thresh_glb, f1_overlay_text)
            threshold_cache[scene] = out
            return out

        def _load_depth_errors_for_scene(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return [], [], [], [], "No valid scene selected."

            if scene in depth_error_cache:
                return depth_error_cache[scene]

            out = _prepare_depth_error_galleries(work_dir, dataset_name, scene)
            depth_error_cache[scene] = out
            return out

        delta_overlay_cache = {}

        def _load_delta_overlay_cached(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return None, "No valid scene selected."
            if scene in delta_overlay_cache:
                return delta_overlay_cache[scene]
            path = _exp_vs_gt_threshold_overlay_glb_path(
                work_dir, dataset_name, scene,
                "student_vs_teacher4v_delta_threshold",
            )
            glb = path if os.path.exists(path) else None
            if glb:
                text = "Loaded cached student vs teacher4v delta overlay."
            else:
                text = "No cached delta overlay. Click 'Build' to generate."
            out = (glb, text)
            delta_overlay_cache[scene] = out
            return out

        def _build_delta_overlay(scene: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                return None, "No valid scene selected."
            missing_deps = _check_threshold_overlay_deps()
            if missing_deps:
                return None, f"Missing dependencies: `{missing_deps}`."
            glb = _ensure_student_vs_teacher4v_delta_overlay_glb(
                work_dir, dataset_name, scene,
            )
            if glb:
                text = "Built student vs teacher4v delta overlay."
            else:
                text = "Failed to build delta overlay. Check GT data/dependencies."
            out = (glb, text)
            delta_overlay_cache[scene] = out
            return out

        scene_dd.change(
            fn=_update,
            inputs=[scene_dd, view_mode],
            outputs=[
                info_md,
                input_gallery,
                gt_fused_model,
                teacher_gt_model,
                teacher4v_model,
                student_model,
                teacher_gt_thresh_model,
                student_gt_thresh_model,
                teacher4v_gt_thresh_model,
                f1_overlay_md,
                teacher_err_gallery,
                teacher4v_err_gallery,
                student_err_gallery,
                delta_err_gallery,
                depth_error_md,
                gt_depth_gallery,
                exp_depth_galleries["teacher"],
                exp_depth_galleries["teacher_4v"],
                exp_depth_galleries["student"],
                metrics_md,
            ],
            queue=False,
        )
        view_mode.change(
            fn=_update,
            inputs=[scene_dd, view_mode],
            outputs=[
                info_md,
                input_gallery,
                gt_fused_model,
                teacher_gt_model,
                teacher4v_model,
                student_model,
                teacher_gt_thresh_model,
                student_gt_thresh_model,
                teacher4v_gt_thresh_model,
                f1_overlay_md,
                teacher_err_gallery,
                teacher4v_err_gallery,
                student_err_gallery,
                delta_err_gallery,
                depth_error_md,
                gt_depth_gallery,
                exp_depth_galleries["teacher"],
                exp_depth_galleries["teacher_4v"],
                exp_depth_galleries["student"],
                metrics_md,
            ],
            queue=False,
        )

        # Initial fill
        demo.load(
            fn=_update,
            inputs=[scene_dd, view_mode],
            outputs=[
                info_md,
                input_gallery,
                gt_fused_model,
                teacher_gt_model,
                teacher4v_model,
                student_model,
                teacher_gt_thresh_model,
                student_gt_thresh_model,
                teacher4v_gt_thresh_model,
                f1_overlay_md,
                teacher_err_gallery,
                teacher4v_err_gallery,
                student_err_gallery,
                delta_err_gallery,
                depth_error_md,
                gt_depth_gallery,
                exp_depth_galleries["teacher"],
                exp_depth_galleries["teacher_4v"],
                exp_depth_galleries["student"],
                metrics_md,
            ],
        )
        render_thresh_btn.click(
            fn=_load_threshold_cached,
            inputs=[scene_dd],
            outputs=[teacher_gt_thresh_model, student_gt_thresh_model, teacher4v_gt_thresh_model, f1_overlay_md],
            queue=False,
        )
        build_thresh_btn.click(
            fn=_build_threshold_for_scene,
            inputs=[scene_dd],
            outputs=[teacher_gt_thresh_model, student_gt_thresh_model, teacher4v_gt_thresh_model, f1_overlay_md],
        )
        teacher_err_tab.select(
            fn=_load_depth_errors_for_scene,
            inputs=[scene_dd],
            outputs=[teacher_err_gallery, teacher4v_err_gallery, student_err_gallery, delta_err_gallery, depth_error_md],
        )
        teacher4v_err_tab.select(
            fn=_load_depth_errors_for_scene,
            inputs=[scene_dd],
            outputs=[teacher_err_gallery, teacher4v_err_gallery, student_err_gallery, delta_err_gallery, depth_error_md],
        )
        student_err_tab.select(
            fn=_load_depth_errors_for_scene,
            inputs=[scene_dd],
            outputs=[teacher_err_gallery, teacher4v_err_gallery, student_err_gallery, delta_err_gallery, depth_error_md],
        )
        delta_err_tab.select(
            fn=_load_depth_errors_for_scene,
            inputs=[scene_dd],
            outputs=[teacher_err_gallery, teacher4v_err_gallery, student_err_gallery, delta_err_gallery, depth_error_md],
        )
        render_delta_btn.click(
            fn=_load_delta_overlay_cached,
            inputs=[scene_dd],
            outputs=[student_t4v_delta_model, delta_overlay_md],
            queue=False,
        )
        build_delta_btn.click(
            fn=_build_delta_overlay,
            inputs=[scene_dd],
            outputs=[student_t4v_delta_model, delta_overlay_md],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio point cloud viewer")
    parser.add_argument("--dataset", type=str, default="hiroom", help="Dataset name (e.g., hiroom, 7scenes)")
    parser.add_argument("--work_dir", type=str, default="./results/hiroom_teacher_student")
    parser.add_argument("--image_dir", type=str, default="./results/images")
    parser.add_argument("--scene", type=str, default=None, help="Optional scene to preselect")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if gr is None:
        raise SystemExit("Missing dependency: gradio. Install with `pip install gradio`.")

    resolved_work_dir = _resolve_work_dir_for_dataset(args.work_dir, args.dataset)
    if os.path.abspath(resolved_work_dir) != os.path.abspath(args.work_dir):
        print(f"[view_pointclouds] Resolved work_dir: {resolved_work_dir}")

    demo = build_app(
        work_dir=resolved_work_dir,
        image_dir=args.image_dir,
        dataset_name=args.dataset,
        default_scene=args.scene,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

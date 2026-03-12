#!/usr/bin/env python3
"""
Quick visualization test for ETH3D VGGT benchmark results.

This script provides a simple way to visualize the point clouds and camera poses
without launching the full Gradio viewer.

Usage:
    python scripts/quick_viz_test.py --exp student --scene courtyard
    python scripts/quick_viz_test.py --exp teacher --scene courtyard --show_cameras
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. Install with: pip install open3d")
    sys.exit(1)


def load_point_cloud(pcd_path):
    """Load point cloud from PLY file."""
    if not os.path.exists(pcd_path):
        print(f"ERROR: Point cloud not found: {pcd_path}")
        return None

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded point cloud: {len(pcd.points)} points")
    return pcd


def load_gt_meta(gt_meta_path):
    """Load GT camera poses and intrinsics."""
    if not os.path.exists(gt_meta_path):
        print(f"ERROR: GT metadata not found: {gt_meta_path}")
        return None, None, None

    data = np.load(gt_meta_path, allow_pickle=True)
    extrinsics = data['extrinsics']  # [S, 4, 4]
    intrinsics = data['intrinsics']  # [S, 3, 3]
    image_files = data['image_files']  # [S]

    print(f"Loaded GT metadata: {len(extrinsics)} cameras")
    return extrinsics, intrinsics, image_files


def create_camera_frustum(extrinsic, intrinsic, scale=0.5, color=[1, 0, 0]):
    """Create camera frustum visualization."""
    # Camera center in world coordinates
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    camera_center = -R.T @ t

    # Create frustum lines
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Image corners in camera coordinates
    w, h = 640, 480  # Approximate image size
    corners_cam = np.array([
        [(0 - cx) / fx, (0 - cy) / fy, 1],
        [(w - cx) / fx, (0 - cy) / fy, 1],
        [(w - cx) / fx, (h - cy) / fy, 1],
        [(0 - cx) / fx, (h - cy) / fy, 1],
    ]) * scale

    # Transform to world coordinates
    corners_world = (R.T @ corners_cam.T).T + camera_center

    # Create line set
    points = np.vstack([camera_center.reshape(1, 3), corners_world])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Camera center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Frustum edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

    return line_set


def visualize_scene(work_dir, exp_name, dataset, scene, show_cameras=False):
    """Visualize point cloud and optionally camera poses."""
    print(f"\n{'='*60}")
    print(f"Visualizing: {exp_name} / {dataset} / {scene}")
    print(f"{'='*60}\n")

    # Paths
    pcd_path = os.path.join(
        work_dir, exp_name, "model_results", dataset, scene,
        "unposed", "exports", "fuse", "pcd.ply"
    )
    gt_meta_path = os.path.join(
        work_dir, exp_name, "model_results", dataset, scene,
        "unposed", "exports", "gt_meta.npz"
    )

    # Load point cloud
    pcd = load_point_cloud(pcd_path)
    if pcd is None:
        return

    # Visualize
    geometries = [pcd]

    if show_cameras:
        # Load GT camera poses
        extrinsics, intrinsics, image_files = load_gt_meta(gt_meta_path)
        if extrinsics is not None:
            print(f"\nAdding {len(extrinsics)} camera frustums...")
            for i, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
                frustum = create_camera_frustum(ext, intr, scale=0.3, color=[1, 0, 0])
                geometries.append(frustum)

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)

    print("\nLaunching Open3D viewer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Ctrl+Mouse: Pan")
    print("  - Q: Quit")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{exp_name} - {scene}",
        width=1280,
        height=720,
    )


def main():
    parser = argparse.ArgumentParser(description="Quick visualization test for VGGT benchmark")
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./results/eth3d_vggt_benchmark",
        help="Work directory with benchmark results",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="student",
        choices=["teacher", "teacher_4v", "student"],
        help="Experiment to visualize",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eth3d",
        help="Dataset name",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="courtyard",
        help="Scene name",
    )
    parser.add_argument(
        "--show_cameras",
        action="store_true",
        help="Show camera frustums",
    )
    args = parser.parse_args()

    visualize_scene(
        args.work_dir,
        args.exp,
        args.dataset,
        args.scene,
        args.show_cameras,
    )


if __name__ == "__main__":
    main()

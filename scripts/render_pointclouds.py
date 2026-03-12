#!/usr/bin/env python3
"""
Render point cloud visualizations from PLY files to PNG images using matplotlib.
Generates top-down and front views for each experiment, plus a comparison grid.

Usage:
    python scripts/render_pointclouds.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None

WORK_DIR = "./results/hiroom_teacher_student"
OUTPUT_DIR = "./results/pointcloud_renders"
EXPERIMENTS = {
    "teacher": "Teacher 8v→4v",
    "teacher_4v": "Teacher 4v",
    "student": "Student LoRA 4v",
}
PLY_SUBPATH = "model_results/hiroom/20241230/828805/cam_sampled_06/unposed/exports/fuse/pcd.ply"


def load_and_subsample(ply_path, max_points=80000):
    """Load PLY and subsample for fast rendering."""
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) == 0:
        raise ValueError(f"Point cloud is empty: {ply_path}")

    # Some PLY files may not carry per-point color information.
    if len(cols) != len(pts):
        cols = np.full((len(pts), 3), 0.65, dtype=np.float32)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


def render_view(ax, pts, cols, elev, azim, title, point_size=0.3):
    """Render a single 3D scatter view."""
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=point_size, marker='.', linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')


def render_single_experiment(exp_key, exp_label, pts, cols, output_dir):
    """Render multiple views for a single experiment."""
    views = [
        ("top", -90, 0),
        ("front", 0, 0),
        ("side", 0, 90),
        ("oblique", -30, 45),
    ]
    for view_name, elev, azim in views:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        render_view(ax, pts, cols, elev, azim, f"{exp_label} — {view_name}")
        out_path = os.path.join(output_dir, f"{exp_key}_{view_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def render_comparison_grid(all_data, output_dir):
    """Render a side-by-side comparison grid: 3 experiments x 2 views."""
    views = [("front", 0, 0), ("oblique", -30, 45)]
    rows = len(all_data)
    fig = plt.figure(figsize=(6 * len(views), 5 * rows))

    for row, (exp_key, (exp_label, pts, cols)) in enumerate(all_data.items()):
        for col, (view_name, elev, azim) in enumerate(views):
            idx = row * len(views) + col + 1
            ax = fig.add_subplot(rows, len(views), idx, projection='3d')
            render_view(ax, pts, cols, elev, azim, f"{exp_label} — {view_name}", point_size=0.15)

    fig.suptitle("Point Cloud Comparison", fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "comparison_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Comparison grid saved: {out_path}")


def main():
    if o3d is None:
        raise SystemExit("Missing dependency: open3d. Install with `pip install open3d`.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    all_data = {}
    for exp_key, exp_label in EXPERIMENTS.items():
        ply_path = os.path.join(WORK_DIR, exp_key, PLY_SUBPATH)
        if not os.path.exists(ply_path):
            print(f"  Skipping {exp_key}: {ply_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Rendering: {exp_label}")
        print(f"{'='*50}")

        pts, cols = load_and_subsample(ply_path)
        all_data[exp_key] = (exp_label, pts, cols)
        render_single_experiment(exp_key, exp_label, pts, cols, OUTPUT_DIR)

    if len(all_data) > 1:
        print(f"\n{'='*50}")
        print("Rendering comparison grid")
        print(f"{'='*50}")
        render_comparison_grid(all_data, OUTPUT_DIR)

    # Also copy depth vis images into output dir for easy access
    print(f"\n{'='*50}")
    print("Copying depth visualizations")
    print(f"{'='*50}")
    for exp_key, exp_label in EXPERIMENTS.items():
        depth_dir = os.path.join(WORK_DIR, exp_key, "visualizations", "depth_vis")
        if not os.path.exists(depth_dir):
            continue
        out_dir = os.path.join(OUTPUT_DIR, f"{exp_key}_depth_vis")
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.listdir(depth_dir)):
            src = os.path.join(depth_dir, f)
            dst = os.path.join(out_dir, f)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
        print(f"  {exp_label}: {out_dir}")

    print(f"\nAll renders saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

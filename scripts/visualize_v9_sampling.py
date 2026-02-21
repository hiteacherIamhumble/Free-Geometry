#!/usr/bin/env python3
"""
Visualize the V9 sampling strategy for VGGT distillation.

Strategy: For each scene, randomly select 10 anchor frames (using seed),
then for each anchor pick 8 views at stride 2 from the sorted file list:
  anchor, anchor+2, anchor+4, anchor+6, anchor+8, anchor+10, anchor+12, anchor+14

Produces a compact grid image: 10 rows (one per anchor) x 8 columns (the 8 views).

Usage:
    python scripts/visualize_v9_sampling.py
    python scripts/visualize_v9_sampling.py --scenes 09c1414f1b --seed 42
    python scripts/visualize_v9_sampling.py --all_scenes --seed 42
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np


DATA_BASE = "/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset/scannetpp"
OUTPUT_DIR = "/home/22097845d/Depth-Anything-3/workspace/v9_sampling_preview"


def get_scene_image_files(scene_dir):
    """Get sorted list of iphone image paths for a scene."""
    iphone_dir = os.path.join(scene_dir, "merge_dslr_iphone", "images", "iphone")
    if not os.path.isdir(iphone_dir):
        return []
    files = sorted(os.listdir(iphone_dir))
    return [os.path.join(iphone_dir, f) for f in files if f.endswith(".jpg")]


def sample_anchors_and_views(num_files, num_anchors, num_views, stride, seed):
    """
    Randomly select anchor indices, then build 8-view windows at given stride.

    Args:
        num_files: total number of files in the scene
        num_anchors: how many anchor frames to pick (e.g. 10)
        num_views: views per anchor (e.g. 8)
        stride: file-list stride (e.g. 2)
        seed: random seed

    Returns:
        List of (anchor_idx, [view_indices]) tuples
    """
    # Maximum valid anchor index: need anchor + (num_views-1)*stride < num_files
    max_anchor = num_files - 1 - (num_views - 1) * stride
    if max_anchor < 0:
        print(f"  WARNING: scene has only {num_files} files, need at least {1 + (num_views-1)*stride}. Skipping.")
        return []

    rng = random.Random(seed)
    valid_anchors = list(range(0, max_anchor + 1))

    # Sample anchors (without replacement if possible)
    k = min(num_anchors, len(valid_anchors))
    anchors = sorted(rng.sample(valid_anchors, k))

    results = []
    for anchor in anchors:
        view_indices = [anchor + i * stride for i in range(num_views)]
        results.append((anchor, view_indices))

    return results


def load_thumbnail(img_path, thumb_h=120):
    """Load image and resize to thumbnail height, preserving aspect ratio."""
    img = cv2.imread(img_path)
    if img is None:
        # Return a placeholder
        return np.zeros((thumb_h, int(thumb_h * 4 / 3), 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = thumb_h / h
    thumb_w = int(w * scale)
    img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    return img


def create_grid(image_files, anchor_views, scene_name, thumb_h=120, seed=None):
    """
    Create a compact grid image.

    Rows = anchors, Columns = 8 views per anchor.
    Each cell shows the thumbnail with the file-list index overlaid.
    """
    if not anchor_views:
        return None

    num_rows = len(anchor_views)
    num_cols = len(anchor_views[0][1])

    # Load one image to get thumbnail width
    sample_thumb = load_thumbnail(image_files[anchor_views[0][1][0]], thumb_h)
    thumb_w = sample_thumb.shape[1]

    # Label area
    label_w = 90
    padding = 2

    grid_h = num_rows * (thumb_h + padding) + padding + 30  # 30 for title
    grid_w = label_w + num_cols * (thumb_w + padding) + padding

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark background

    # Title
    title = f"{scene_name}  |  seed={seed}  |  {num_rows} anchors x {num_cols} views (stride=2)"
    cv2.putText(grid, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for row_idx, (anchor, view_indices) in enumerate(anchor_views):
        y0 = 30 + row_idx * (thumb_h + padding) + padding

        # Row label
        fname = os.path.basename(image_files[anchor])
        label = f"#{anchor}"
        cv2.putText(grid, label, (5, y0 + thumb_h // 2 - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(grid, fname[:16], (5, y0 + thumb_h // 2 + 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)

        for col_idx, file_idx in enumerate(view_indices):
            x0 = label_w + col_idx * (thumb_w + padding) + padding

            thumb = load_thumbnail(image_files[file_idx], thumb_h)
            # Ensure thumbnail fits
            th, tw = thumb.shape[:2]
            grid[y0:y0 + th, x0:x0 + tw] = thumb

            # Overlay file index and filename
            idx_text = f"[{file_idx}]"
            cv2.putText(grid, idx_text, (x0 + 3, y0 + 14),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Highlight anchor (first column) with a colored border
            if col_idx == 0:
                cv2.rectangle(grid, (x0, y0), (x0 + tw - 1, y0 + th - 1), (0, 200, 255), 2)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize V9 sampling strategy")
    parser.add_argument("--data_base", type=str, default=DATA_BASE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Specific scene IDs to visualize (default: first 3)")
    parser.add_argument("--all_scenes", action="store_true",
                        help="Visualize all scenes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for anchor selection")
    parser.add_argument("--num_anchors", type=int, default=10,
                        help="Number of anchor frames per scene")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views per anchor")
    parser.add_argument("--stride", type=int, default=2,
                        help="Stride in file list for consecutive views")
    parser.add_argument("--thumb_h", type=int, default=120,
                        help="Thumbnail height in pixels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get scenes
    all_scenes = sorted(os.listdir(args.data_base))
    all_scenes = [s for s in all_scenes if os.path.isdir(os.path.join(args.data_base, s))]

    if args.scenes:
        scenes = [s for s in args.scenes if s in all_scenes]
    elif args.all_scenes:
        scenes = all_scenes
    else:
        scenes = all_scenes[:3]

    print(f"Visualizing {len(scenes)} scenes with V9 sampling strategy:")
    print(f"  {args.num_anchors} anchors x {args.num_views} views, stride={args.stride}, seed={args.seed}")
    print()

    for scene in scenes:
        scene_dir = os.path.join(args.data_base, scene)
        image_files = get_scene_image_files(scene_dir)
        num_files = len(image_files)

        print(f"Scene: {scene} ({num_files} images)")

        anchor_views = sample_anchors_and_views(
            num_files, args.num_anchors, args.num_views, args.stride, args.seed
        )

        if not anchor_views:
            continue

        # Print sampling info
        for anchor, views in anchor_views:
            fnames = [os.path.basename(image_files[v]) for v in views]
            print(f"  Anchor #{anchor}: {fnames[0]} -> {', '.join(fnames)}")

        grid = create_grid(image_files, anchor_views, scene, args.thumb_h, args.seed)
        if grid is not None:
            out_path = os.path.join(args.output_dir, f"v9_sampling_{scene}_seed{args.seed}.png")
            cv2.imwrite(out_path, grid)
            print(f"  Saved: {out_path}")
        print()

    # Also create a combined summary for all visualized scenes
    if len(scenes) > 1:
        print(f"Individual scene previews saved to: {args.output_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()

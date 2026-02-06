#!/usr/bin/env python3
"""
Visualize sampled frames for each scene using consecutive window sampling.
Creates a compact image showing the 8 sampled frames for manual inspection.
"""

import argparse
import os
import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from pathlib import Path

from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
from depth_anything_3.bench.datasets.sevenscenes import SevenScenes


def get_subset_indices(num_available: int, subset_ratio: float, num_views: int = 8):
    """Get uniformly sampled subset indices."""
    subset_size = max(num_views, int(num_available * subset_ratio))
    if subset_size >= num_available:
        return list(range(num_available))
    step = num_available / subset_size
    return [int(i * step) for i in range(subset_size)]


def get_consecutive_window(subset_indices, sample_idx: int, num_views: int = 8):
    """Get consecutive window from subset based on sample_idx."""
    subset_size = len(subset_indices)

    if subset_size <= num_views:
        # Use all with repetition
        indices = list(subset_indices)
        while len(indices) < num_views:
            indices.append(subset_indices[0])
        return indices

    num_full_windows = subset_size // num_views

    if sample_idx < num_full_windows:
        start = sample_idx * num_views
        return list(subset_indices[start:start + num_views])
    else:
        offset = num_views // 2
        wrap_idx = sample_idx - num_full_windows
        max_start = subset_size - num_views
        start = (offset + wrap_idx * num_views) % (max_start + 1)
        return list(subset_indices[start:start + num_views])


def create_scene_visualization(image_files, frame_indices, scene_name, output_path, thumb_size=200):
    """Create a compact visualization of sampled frames."""
    images = []
    for idx in frame_indices:
        img = cv2.imread(image_files[idx])
        if img is None:
            img = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

        # Resize to thumbnail
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

        # Pad to square
        pad_h = (thumb_size - new_h) // 2
        pad_w = (thumb_size - new_w) // 2
        img_padded = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
        img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img

        # Add frame index label
        cv2.putText(img_padded, f"#{idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        images.append(img_padded)

    # Arrange in 2x4 grid
    row1 = np.hstack(images[:4])
    row2 = np.hstack(images[4:])
    grid = np.vstack([row1, row2])

    # Add scene name header
    header = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"{scene_name} - frames: {frame_indices}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    final = np.vstack([header, grid])
    cv2.imwrite(str(output_path), final)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize sampled frames for each scene")
    parser.add_argument("--dataset", type=str, required=True, choices=["scannetpp", "7scenes"])
    parser.add_argument("--subset_ratio", type=float, default=0.05)
    parser.add_argument("--sample_idx", type=int, default=3, help="Which sample window to visualize (seed 43 = sample_idx 3)")
    parser.add_argument("--output_dir", type=str, default="./temp/sampled_frames")
    parser.add_argument("--thumb_size", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.dataset == "scannetpp":
        dataset = ScanNetPP()
    else:
        dataset = SevenScenes()

    print(f"Dataset: {args.dataset}")
    print(f"Subset ratio: {args.subset_ratio * 100:.0f}%")
    print(f"Sample index: {args.sample_idx}")
    print(f"Scenes: {len(dataset.SCENES)}")
    print()

    for scene in dataset.SCENES:
        data = dataset.get_data(scene)
        image_files = data['image_files']
        num_frames = len(image_files)

        # Get subset and consecutive window
        subset_indices = get_subset_indices(num_frames, args.subset_ratio)
        frame_indices = get_consecutive_window(subset_indices, args.sample_idx)

        print(f"{scene}: {num_frames} frames, subset={len(subset_indices)}, window={frame_indices}")

        # Create visualization
        output_path = output_dir / f"{scene}_sample{args.sample_idx}.jpg"
        create_scene_visualization(image_files, frame_indices, scene, output_path, args.thumb_size)

    print()
    print(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

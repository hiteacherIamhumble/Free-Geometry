#!/usr/bin/env python3
"""
Visualize 8v training samples from ScanNet++ and 7scenes datasets.

This script shows example 8-view samples that would be used for training,
using the subset sampling with 20% ratio for ScanNet++ and 5% for 7scenes.
"""

import os
import sys
import random
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
from depth_anything_3.bench.datasets.sevenscenes import SevenScenes


def get_subset_indices(num_available: int, subset_ratio: float, num_views: int = 8) -> list:
    """Get uniformly sampled subset indices."""
    subset_size = max(num_views, int(num_available * subset_ratio))
    if subset_size >= num_available:
        return list(range(num_available))
    step = num_available / subset_size
    return [int(i * step) for i in range(subset_size)]


def sample_consecutive_window(subset_indices: list, num_views: int, sample_idx: int) -> list:
    """Sample a consecutive window from subset indices."""
    subset_size = len(subset_indices)

    if subset_size <= num_views:
        return list(subset_indices)

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


def create_image_grid(images: list, titles: list = None, max_size: int = 200) -> np.ndarray:
    """Create a grid of images (2 rows x 4 cols for 8 images)."""
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized.append(cv2.resize(img, (new_w, new_h)))

    # Pad to same size
    max_h = max(img.shape[0] for img in resized)
    max_w = max(img.shape[1] for img in resized)

    padded = []
    for i, img in enumerate(resized):
        h, w = img.shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w
        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Add frame index text
        if titles:
            cv2.putText(padded_img, titles[i], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        padded.append(padded_img)

    # Create 2x4 grid
    row1 = np.hstack(padded[:4])
    row2 = np.hstack(padded[4:])
    grid = np.vstack([row1, row2])

    return grid


def visualize_dataset_samples(dataset, dataset_name: str, subset_ratio: float, output_dir: str, num_samples: int = 2):
    """Visualize training samples for a dataset."""
    scenes = list(dataset.SCENES)[:3]  # Use first 3 scenes

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Subset ratio: {subset_ratio*100:.0f}%")
    print(f"Scenes: {scenes}")
    print(f"{'='*60}")

    for scene in scenes:
        scene_data = dataset.get_data(scene)
        image_files = scene_data['image_files']
        num_frames = len(image_files)

        # Get subset indices
        subset_indices = get_subset_indices(num_frames, subset_ratio, num_views=8)

        print(f"\nScene: {scene}")
        print(f"  Total frames: {num_frames}")
        print(f"  Subset size: {len(subset_indices)} ({len(subset_indices)/num_frames*100:.1f}%)")
        print(f"  Subset indices (first 20): {subset_indices[:20]}...")

        for sample_idx in range(num_samples):
            # Sample consecutive window
            view_indices = sample_consecutive_window(subset_indices, num_views=8, sample_idx=sample_idx)

            print(f"  Sample {sample_idx}: window indices in subset = {view_indices}")
            print(f"    Actual frame indices: {[subset_indices[i] if i < len(subset_indices) else view_indices[i] for i in range(len(view_indices))]}")

            # Load images
            images = []
            titles = []
            for i, idx in enumerate(view_indices):
                actual_idx = subset_indices[idx] if idx < len(subset_indices) else idx
                img_path = image_files[actual_idx]
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    titles.append(f"F{actual_idx}")

            if len(images) == 8:
                # Create grid
                grid = create_image_grid(images, titles)

                # Save
                output_path = os.path.join(output_dir, f"{dataset_name}_{scene}_sample{sample_idx}.png")
                cv2.imwrite(output_path, grid)
                print(f"    Saved: {output_path}")
            else:
                print(f"    Warning: Only loaded {len(images)} images")


def main():
    output_dir = "./temp"
    os.makedirs(output_dir, exist_ok=True)

    print("Visualizing 8v training samples with subset sampling")
    print(f"Output directory: {output_dir}")

    # ScanNet++ with 20% subset ratio
    try:
        scannetpp = ScanNetPP()
        visualize_dataset_samples(scannetpp, "scannetpp", subset_ratio=0.20, output_dir=output_dir, num_samples=2)
    except Exception as e:
        print(f"Error with ScanNet++: {e}")

    # 7scenes with 5% subset ratio
    try:
        sevenscenes = SevenScenes()
        visualize_dataset_samples(sevenscenes, "7scenes", subset_ratio=0.05, output_dir=output_dir, num_samples=2)
    except Exception as e:
        print(f"Error with 7scenes: {e}")

    print(f"\n{'='*60}")
    print(f"Done! Check {output_dir} for visualization images.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

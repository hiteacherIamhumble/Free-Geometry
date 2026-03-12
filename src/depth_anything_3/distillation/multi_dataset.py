"""
Multi-Dataset Loader for Distillation Training.

This module combines ScanNet++ training data with all benchmark datasets
for expanded distillation training.
"""

import os
from typing import List, Optional

import torch
from torch.utils.data import ConcatDataset

from depth_anything_3.distillation.dataset import ScanNetPPDistillDataset
from depth_anything_3.distillation.benchmark_dataset import BenchmarkDistillDataset


def create_multi_dataset(
    scannetpp_data_root: str,
    benchmark_data_root: str = 'workspace/benchmark_dataset',
    num_views: int = 8,
    student_indices: List[int] = None,
    augment: bool = True,
    samples_per_scene: int = 4,
    seed: int = 42,
    image_size: tuple = (518, 518),
    per_dataset_samples: Optional[dict] = None,
) -> ConcatDataset:
    """
    Create a combined dataset from ScanNet++ train split and all benchmark datasets.

    Args:
        scannetpp_data_root: Path to ScanNet++ data directory
        benchmark_data_root: Path to benchmark datasets directory
        num_views: Number of views for teacher (default: 8)
        student_indices: Which teacher views to use for student (default: [0,2,4,6])
        augment: Whether to apply data augmentation
        samples_per_scene: Number of samples per scene for benchmark datasets (default: 4)
        seed: Random seed for sampling
        image_size: Target image size (H, W)
        per_dataset_samples: Optional dict mapping dataset names to samples_per_scene
                           (e.g., {'eth3d': 8, '7scenes': 8}). Overrides samples_per_scene.

    Returns:
        ConcatDataset combining all datasets
    """
    if student_indices is None:
        student_indices = [0, 2, 4, 6]

    datasets = []

    # 1. ScanNet++ train split (1 sample per scene - existing behavior)
    print("=" * 60)
    print("Loading ScanNet++ training split...")
    print("=" * 60)
    scannetpp_train = ScanNetPPDistillDataset(
        data_root=scannetpp_data_root,
        split='train',
        num_views=num_views,
        image_size=image_size,
        student_indices=student_indices,
        augment=augment,
        load_cameras=False,
        cache_metadata=True,
    )
    datasets.append(scannetpp_train)
    print(f"ScanNet++ train: {len(scannetpp_train)} samples")

    # 2. Benchmark datasets (configurable samples per scene)
    benchmark_names = ['eth3d', '7scenes', 'scannetpp', 'hiroom', 'dtu']

    print("\n" + "=" * 60)
    print("Loading benchmark datasets...")
    print("=" * 60)

    for dataset_name in benchmark_names:
        print(f"\nLoading {dataset_name}...")
        try:
            # Use per-dataset samples if specified, otherwise use default
            dataset_samples = samples_per_scene
            if per_dataset_samples and dataset_name in per_dataset_samples:
                dataset_samples = per_dataset_samples[dataset_name]

            ds = BenchmarkDistillDataset(
                dataset_name=dataset_name,
                num_views=num_views,
                image_size=image_size,
                student_indices=student_indices,
                augment=augment,
                samples_per_scene=dataset_samples,
                seed=seed,
            )
            datasets.append(ds)
            print(f"{dataset_name}: {len(ds)} samples ({dataset_samples} samples/scene)")
        except Exception as e:
            print(f"Warning: Failed to load {dataset_name}: {e}")
            print(f"Skipping {dataset_name}...")

    # 3. Combine all datasets
    combined_dataset = ConcatDataset(datasets)

    print("\n" + "=" * 60)
    print("Multi-Dataset Summary")
    print("=" * 60)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total samples: {len(combined_dataset)}")
    print(f"Breakdown:")
    for i, ds in enumerate(datasets):
        name = ds.split if hasattr(ds, 'split') else ds.dataset_name if hasattr(ds, 'dataset_name') else f"dataset_{i}"
        print(f"  - {name}: {len(ds)} samples")
    print("=" * 60)

    return combined_dataset


def create_multi_dataloader(
    scannetpp_data_root: str,
    benchmark_data_root: str = 'workspace/benchmark_dataset',
    batch_size: int = 2,
    num_workers: int = 4,
    num_views: int = 8,
    student_indices: List[int] = None,
    augment: bool = True,
    samples_per_scene: int = 4,
    seed: int = 42,
    image_size: tuple = (518, 518),
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the multi-dataset.

    Args:
        scannetpp_data_root: Path to ScanNet++ data directory
        benchmark_data_root: Path to benchmark datasets directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_views: Number of views for teacher
        student_indices: Which teacher views to use for student
        augment: Whether to apply data augmentation
        samples_per_scene: Number of samples per scene for benchmark datasets
        seed: Random seed for sampling
        image_size: Target image size (H, W)

    Returns:
        DataLoader instance
    """
    dataset = create_multi_dataset(
        scannetpp_data_root=scannetpp_data_root,
        benchmark_data_root=benchmark_data_root,
        num_views=num_views,
        student_indices=student_indices,
        augment=augment,
        samples_per_scene=samples_per_scene,
        seed=seed,
        image_size=image_size,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

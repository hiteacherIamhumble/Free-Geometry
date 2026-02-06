#!/usr/bin/env python3
"""
Compare VGGT original vs distilled model results across multiple seeds.

Generates tables with mean ± std for pose and reconstruction metrics.
Each table has 4 subtables (one per dataset).

Usage:
    python scripts/compare_vggt_results.py \
        --base_root workspace/vggt_full_experiment/base \
        --lora_root workspace/vggt_full_experiment/lora \
        --datasets eth3d scannetpp hiroom 7scenes \
        --seeds 42 43 44
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Metrics to compare
POSE_METRICS = ["auc03", "auc05", "auc30"]
RECON_METRICS = ["acc", "comp", "fscore"]


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_metrics(
    root_dir: Path,
    dataset: str,
    seeds: List[int],
    mode: str,
    flat_structure: bool = False,
) -> Dict[str, List[float]]:
    """
    Collect metrics across multiple seeds for a dataset.

    Args:
        root_dir: Root directory containing seed subdirectories
        dataset: Dataset name
        seeds: List of seeds to collect
        mode: Evaluation mode ('pose' or 'recon_unposed')
        flat_structure: If True, expect root_dir/seed{N}/metric_results/{dataset}_{mode}.json
                       If False, expect root_dir/{dataset}/seed{N}/metric_results/{dataset}_{mode}.json

    Returns:
        Dict mapping metric names to lists of values across seeds
    """
    metrics_list = POSE_METRICS if mode == "pose" else RECON_METRICS
    metrics = {m: [] for m in metrics_list}

    for seed in seeds:
        if flat_structure:
            # Flat: root_dir/seed{N}/metric_results/{dataset}_{mode}.json
            metric_file = root_dir / f"seed{seed}" / "metric_results" / f"{dataset}_{mode}.json"
        else:
            # Nested: root_dir/{dataset}/seed{N}/metric_results/{dataset}_{mode}.json
            metric_file = root_dir / dataset / f"seed{seed}" / "metric_results" / f"{dataset}_{mode}.json"

        if not metric_file.exists():
            print(f"  Warning: Missing {metric_file}")
            continue

        try:
            data = load_json(metric_file)
            if "mean" in data:
                for metric in metrics_list:
                    if metric in data["mean"]:
                        metrics[metric].append(data["mean"][metric])
        except Exception as e:
            print(f"  Error loading {metric_file}: {e}")

    return metrics


def compute_stats(values: List[float]) -> Tuple[float, float]:
    """Compute mean and std from a list of values."""
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def format_metric_name(metric: str) -> str:
    """Format metric name for display."""
    if metric.startswith("auc"):
        return f"AUC@{metric[3:]}"
    if metric == "acc":
        return "Accuracy"
    if metric == "comp":
        return "Completeness"
    return metric.capitalize()


def format_value(mean: float, std: float, is_percentage: bool = False) -> str:
    """Format value with mean ± std."""
    if is_percentage:
        return f"{mean*100:.2f} ± {std*100:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def compute_improvement(base_mean: float, lora_mean: float) -> str:
    """Compute percentage improvement."""
    if abs(base_mean) < 1e-8:
        return ""
    diff_pct = ((lora_mean - base_mean) / abs(base_mean)) * 100
    return f"({diff_pct:+.1f}%)"


def print_table(
    title: str,
    datasets: List[str],
    base_stats: Dict[str, Dict[str, Tuple[float, float]]],
    lora_stats: Dict[str, Dict[str, Tuple[float, float]]],
    metrics: List[str],
    is_pose: bool = True,
    seeds: List[int] = None,
) -> str:
    """
    Print comparison table and return as string.

    Args:
        title: Table title
        datasets: List of dataset names
        base_stats: Base model stats {dataset: {metric: (mean, std)}}
        lora_stats: LoRA model stats {dataset: {metric: (mean, std)}}
        metrics: List of metrics to display
        is_pose: Whether these are pose metrics (affects formatting)
        seeds: List of seeds used

    Returns:
        Formatted table as string
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"{title:^80}")
    lines.append("=" * 80)

    seeds_str = ", ".join(map(str, seeds)) if seeds else "N/A"

    for dataset in datasets:
        lines.append("")
        lines.append(f"{dataset.upper()} (seeds: {seeds_str})")
        lines.append("┌" + "─" * 13 + "┬" + "─" * 24 + "┬" + "─" * 30 + "┐")
        lines.append(f"│ {'Metric':<11} │ {'Original VGGT':<22} │ {'Distilled VGGT':<28} │")
        lines.append("├" + "─" * 13 + "┼" + "─" * 24 + "┼" + "─" * 30 + "┤")

        for metric in metrics:
            base_mean, base_std = base_stats.get(dataset, {}).get(metric, (0.0, 0.0))
            lora_mean, lora_std = lora_stats.get(dataset, {}).get(metric, (0.0, 0.0))

            name = format_metric_name(metric)
            is_pct = is_pose  # Pose metrics are percentages

            base_str = format_value(base_mean, base_std, is_pct)
            lora_str = format_value(lora_mean, lora_std, is_pct)
            improvement = compute_improvement(base_mean, lora_mean)

            if improvement:
                lora_str = f"{lora_str} {improvement}"

            lines.append(f"│ {name:<11} │ {base_str:<22} │ {lora_str:<28} │")

        lines.append("└" + "─" * 13 + "┴" + "─" * 24 + "┴" + "─" * 30 + "┘")

    output = "\n".join(lines)
    print(output)
    return output


def save_results_json(
    output_dir: Path,
    datasets: List[str],
    seeds: List[int],
    base_pose_stats: Dict,
    lora_pose_stats: Dict,
    base_recon_stats: Dict,
    lora_recon_stats: Dict,
) -> None:
    """Save results to JSON file."""
    results = {
        "config": {
            "datasets": datasets,
            "seeds": seeds,
        },
        "pose": {},
        "recon_unposed": {},
    }

    for dataset in datasets:
        results["pose"][dataset] = {
            "base": {m: {"mean": s[0], "std": s[1]}
                     for m, s in base_pose_stats.get(dataset, {}).items()},
            "lora": {m: {"mean": s[0], "std": s[1]}
                     for m, s in lora_pose_stats.get(dataset, {}).items()},
        }
        results["recon_unposed"][dataset] = {
            "base": {m: {"mean": s[0], "std": s[1]}
                     for m, s in base_recon_stats.get(dataset, {}).items()},
            "lora": {m: {"mean": s[0], "std": s[1]}
                     for m, s in lora_recon_stats.get(dataset, {}).items()},
        }

    output_file = output_dir / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare VGGT original vs distilled model results"
    )
    parser.add_argument(
        "--base_root",
        type=str,
        required=True,
        help="Root directory for base model results",
    )
    parser.add_argument(
        "--lora_root",
        type=str,
        required=True,
        help="Root directory for LoRA model results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eth3d", "scannetpp", "hiroom", "7scenes"],
        help="Datasets to compare",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seeds used for benchmarking",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="Dataset used for training (for cross-dataset experiments)",
    )
    parser.add_argument(
        "--flat_structure",
        action="store_true",
        help="Use flat directory structure: root/seed{N}/metric_results/ instead of root/{dataset}/seed{N}/metric_results/",
    )
    args = parser.parse_args()

    base_root = Path(args.base_root)
    lora_root = Path(args.lora_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VGGT Model Comparison: Original vs Distilled")
    print("=" * 80)
    print(f"Base results:  {base_root}")
    print(f"LoRA results:  {lora_root}")
    if args.train_dataset:
        print(f"LoRA trained on: {args.train_dataset}")
    print(f"Datasets:      {', '.join(args.datasets)}")
    print(f"Seeds:         {', '.join(map(str, args.seeds))}")
    print("=" * 80)

    # Collect pose metrics
    print("\nCollecting pose metrics...")
    base_pose_stats = {}
    lora_pose_stats = {}

    for dataset in args.datasets:
        print(f"  {dataset}...")
        base_metrics = collect_metrics(base_root, dataset, args.seeds, "pose", args.flat_structure)
        lora_metrics = collect_metrics(lora_root, dataset, args.seeds, "pose", args.flat_structure)

        base_pose_stats[dataset] = {m: compute_stats(v) for m, v in base_metrics.items()}
        lora_pose_stats[dataset] = {m: compute_stats(v) for m, v in lora_metrics.items()}

    # Collect reconstruction metrics
    print("\nCollecting reconstruction metrics...")
    base_recon_stats = {}
    lora_recon_stats = {}

    for dataset in args.datasets:
        print(f"  {dataset}...")
        base_metrics = collect_metrics(base_root, dataset, args.seeds, "recon_unposed", args.flat_structure)
        lora_metrics = collect_metrics(lora_root, dataset, args.seeds, "recon_unposed", args.flat_structure)

        base_recon_stats[dataset] = {m: compute_stats(v) for m, v in base_metrics.items()}
        lora_recon_stats[dataset] = {m: compute_stats(v) for m, v in lora_metrics.items()}

    # Print tables
    pose_table = print_table(
        "VGGT Pose Evaluation Results",
        args.datasets,
        base_pose_stats,
        lora_pose_stats,
        POSE_METRICS,
        is_pose=True,
        seeds=args.seeds,
    )

    recon_table = print_table(
        "VGGT Reconstruction (Unposed) Results",
        args.datasets,
        base_recon_stats,
        lora_recon_stats,
        RECON_METRICS,
        is_pose=False,
        seeds=args.seeds,
    )

    # Save tables to text files
    pose_file = output_dir / "pose_table.txt"
    with open(pose_file, 'w', encoding='utf-8') as f:
        f.write(pose_table)
    print(f"\nSaved pose table to: {pose_file}")

    recon_file = output_dir / "recon_table.txt"
    with open(recon_file, 'w', encoding='utf-8') as f:
        f.write(recon_table)
    print(f"Saved recon table to: {recon_file}")

    # Save JSON results
    save_results_json(
        output_dir,
        args.datasets,
        args.seeds,
        base_pose_stats,
        lora_pose_stats,
        base_recon_stats,
        lora_recon_stats,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for dataset in args.datasets:
        print(f"\n{dataset.upper()}:")

        # Pose summary
        for metric in POSE_METRICS:
            base_mean, _ = base_pose_stats.get(dataset, {}).get(metric, (0, 0))
            lora_mean, _ = lora_pose_stats.get(dataset, {}).get(metric, (0, 0))
            if base_mean > 0:
                diff = ((lora_mean - base_mean) / base_mean) * 100
                status = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  {format_metric_name(metric)}: {status} {diff:+.1f}%")

        # Recon summary (F-score only)
        base_mean, _ = base_recon_stats.get(dataset, {}).get("fscore", (0, 0))
        lora_mean, _ = lora_recon_stats.get(dataset, {}).get("fscore", (0, 0))
        if base_mean > 0:
            diff = ((lora_mean - base_mean) / base_mean) * 100
            status = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  F-score: {status} {diff:+.1f}%")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare VGGT baseline vs LoRA for V3 experiment.

Generates 2 consolidated tables (one for 4v, one for 8v) with all 4 datasets,
showing pose metrics (AUC@03, AUC@05, AUC@30) and F1 recon metric.

Usage:
    python scripts/compare_vggt_v3.py \
        --benchmark_root workspace/vggt_experiment_v3 \
        --output_dir workspace/vggt_experiment_v3/comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# Metrics to compare
POSE_METRICS = ["auc03", "auc05", "auc30"]
RECON_METRIC = "fscore"

DATASETS = ["eth3d", "scannetpp", "hiroom", "7scenes"]


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_metric(
    root_dir: Path,
    dataset: str,
    seeds: List[int],
    mode: str,
    metric: str,
) -> List[float]:
    """
    Collect a single metric across multiple seeds for a dataset.

    Args:
        root_dir: Directory containing seed subdirectories (e.g., base_4v/eth3d/)
        dataset: Dataset name
        seeds: List of seeds to collect
        mode: Evaluation mode ('pose' or 'recon_unposed')
        metric: Metric name to collect

    Returns:
        List of metric values across seeds
    """
    values = []

    for seed in seeds:
        metric_file = root_dir / f"seed{seed}" / "metric_results" / f"{dataset}_{mode}.json"

        if not metric_file.exists():
            print(f"  Warning: Missing {metric_file}")
            continue

        try:
            data = load_json(metric_file)
            if "mean" in data and metric in data["mean"]:
                values.append(data["mean"][metric])
        except Exception as e:
            print(f"  Error loading {metric_file}: {e}")

    return values


def compute_stats(values: List[float]) -> Tuple[float, float]:
    """Compute mean and std from a list of values."""
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def format_pct(mean: float, std: float) -> str:
    """Format percentage value with mean ± std."""
    if std > 0:
        return f"{mean*100:.2f}±{std*100:.2f}"
    return f"{mean*100:.2f}"


def format_float(mean: float, std: float) -> str:
    """Format float value with mean ± std."""
    if std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def compute_diff(base_mean: float, lora_mean: float) -> str:
    """Compute difference indicator."""
    if abs(base_mean) < 1e-8:
        return ""
    diff_pct = ((lora_mean - base_mean) / abs(base_mean)) * 100
    if diff_pct > 0:
        return f"↑{diff_pct:.1f}%"
    elif diff_pct < 0:
        return f"↓{abs(diff_pct):.1f}%"
    return "="


def generate_table(
    setting: str,
    benchmark_root: Path,
    datasets: List[str],
    seeds: List[int],
) -> str:
    """
    Generate a consolidated comparison table for a setting (4v or 8v).

    Args:
        setting: '4v' or '8v'
        benchmark_root: Root directory of benchmark results
        datasets: List of dataset names
        seeds: List of seeds

    Returns:
        Formatted table as string
    """
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 120)
    lines.append(f"VGGT Experiment V3 - {setting.upper()} Comparison (Baseline vs LoRA)")
    lines.append(f"Seeds: {', '.join(map(str, seeds))}")
    lines.append("=" * 120)
    lines.append("")

    # Table header
    lines.append("┌" + "─" * 12 + "┬" + "─" * 18 + "┬" + "─" * 18 + "┬" + "─" * 18 + "┬" + "─" * 18 + "┬" + "─" * 18 + "┬" + "─" * 10 + "┐")
    lines.append(f"│ {'Dataset':<10} │ {'Model':<16} │ {'AUC@03':<16} │ {'AUC@05':<16} │ {'AUC@30':<16} │ {'F-score':<16} │ {'Δ F1':<8} │")
    lines.append("├" + "─" * 12 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 10 + "┤")

    for dataset in datasets:
        base_dir = benchmark_root / f"base_{setting}" / dataset
        lora_dir = benchmark_root / f"lora_{setting}" / dataset

        # Collect baseline metrics
        base_pose = {}
        for metric in POSE_METRICS:
            values = collect_metric(base_dir, dataset, seeds, "pose", metric)
            base_pose[metric] = compute_stats(values)

        base_fscore_values = collect_metric(base_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
        base_fscore = compute_stats(base_fscore_values)

        # Collect LoRA metrics
        lora_pose = {}
        for metric in POSE_METRICS:
            values = collect_metric(lora_dir, dataset, seeds, "pose", metric)
            lora_pose[metric] = compute_stats(values)

        lora_fscore_values = collect_metric(lora_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
        lora_fscore = compute_stats(lora_fscore_values)

        # Format baseline row
        base_auc03 = format_pct(*base_pose.get("auc03", (0, 0)))
        base_auc05 = format_pct(*base_pose.get("auc05", (0, 0)))
        base_auc30 = format_pct(*base_pose.get("auc30", (0, 0)))
        base_f1 = format_float(*base_fscore)

        lines.append(f"│ {dataset:<10} │ {'Baseline':<16} │ {base_auc03:<16} │ {base_auc05:<16} │ {base_auc30:<16} │ {base_f1:<16} │ {'':<8} │")

        # Format LoRA row
        lora_auc03 = format_pct(*lora_pose.get("auc03", (0, 0)))
        lora_auc05 = format_pct(*lora_pose.get("auc05", (0, 0)))
        lora_auc30 = format_pct(*lora_pose.get("auc30", (0, 0)))
        lora_f1 = format_float(*lora_fscore)
        f1_diff = compute_diff(base_fscore[0], lora_fscore[0])

        lines.append(f"│ {'':<10} │ {'LoRA':<16} │ {lora_auc03:<16} │ {lora_auc05:<16} │ {lora_auc30:<16} │ {lora_f1:<16} │ {f1_diff:<8} │")

        # Separator between datasets
        if dataset != datasets[-1]:
            lines.append("├" + "─" * 12 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 18 + "┼" + "─" * 10 + "┤")

    lines.append("└" + "─" * 12 + "┴" + "─" * 18 + "┴" + "─" * 18 + "┴" + "─" * 18 + "┴" + "─" * 18 + "┴" + "─" * 18 + "┴" + "─" * 10 + "┘")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare VGGT baseline vs LoRA for V3 experiment"
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        default="./workspace/vggt_experiment_v3",
        help="Root directory of benchmark results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: benchmark_root/comparison)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds used for benchmarking",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to compare",
    )
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    output_dir = Path(args.output_dir) if args.output_dir else benchmark_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VGGT Experiment V3 Comparison")
    print("=" * 80)
    print(f"Benchmark root: {benchmark_root}")
    print(f"Output dir:     {output_dir}")
    print(f"Datasets:       {', '.join(args.datasets)}")
    print(f"Seeds:          {', '.join(map(str, args.seeds))}")
    print("=" * 80)

    # Generate 4v table
    print("\nGenerating 4v comparison table...")
    table_4v = generate_table("4v", benchmark_root, args.datasets, args.seeds)
    print(table_4v)

    table_4v_file = output_dir / "comparison_4v.txt"
    with open(table_4v_file, 'w', encoding='utf-8') as f:
        f.write(table_4v)
    print(f"Saved to: {table_4v_file}")

    # Generate 8v table
    print("\nGenerating 8v comparison table...")
    table_8v = generate_table("8v", benchmark_root, args.datasets, args.seeds)
    print(table_8v)

    table_8v_file = output_dir / "comparison_8v.txt"
    with open(table_8v_file, 'w', encoding='utf-8') as f:
        f.write(table_8v)
    print(f"Saved to: {table_8v_file}")

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

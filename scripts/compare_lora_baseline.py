#!/usr/bin/env python3
"""
Compare LoRA model results with baseline across all datasets.
Creates a visualization with color-coded tables showing improvements (green) and regressions (red).
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import numpy as np

# Configuration
REPO_ROOT = Path("/home/22097845d/Depth-Anything-3")
BASELINE_DIR = REPO_ROOT / "checkpoints/baseline/20260117_143036"
LORA_ROOT = REPO_ROOT / "workspace"
# Number of views used for comparison
NUM_VIEWS = 4
# Per-dataset LoRA eval folders follow: eval_{dataset}_local_softmax_seed42_1epoch_1e-4/metric_results/{dataset}_{mode}.json
LORA_PATTERN = "eval_{dataset}_local_softmax_seed42_1epoch_1e-4"
# Optional per-dataset overrides
LORA_OVERRIDE = {
    "scannetpp": REPO_ROOT / "workspace/eval_scannetpp_local_softmax_seed42_1epoch_1e-3_v1",
    "7scenes": REPO_ROOT / "workspace/eval_7scenes_local_softmax_seed42_1epoch_1e-4_8v",
    "eth3d": REPO_ROOT / "workspace/eval_eth3d_local_softmax_seed42",
}
OUTPUT_DIR = REPO_ROOT / "comparison_results_eth3d_5epoch"

# Datasets to compare (excluding DTU as requested)
DATASETS = ["eth3d"]

# Metrics to display
POSE_METRICS = ["auc03", "auc05", "auc30"]  # pose@3, pose@5, pose@30
RECON_METRICS = ["fscore"]  # only report F1 score as requested


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_baseline_path(dataset, mode, frames=NUM_VIEWS):
    """Get baseline metric file path."""
    return (
        BASELINE_DIR
        / dataset
        / f"benchmark_results_{frames}views"
        / "metric_results"
        / f"{dataset}_{mode}.json"
    )


def get_lora_path(dataset, mode):
    """Get LoRA metric file path."""
    base_dir = LORA_OVERRIDE.get(dataset, LORA_ROOT / LORA_PATTERN.format(dataset=dataset))
    lora_dir = base_dir / "metric_results"
    return lora_dir / f"{dataset}_{mode}.json"


def compare_metrics(baseline_data, lora_data, metrics):
    """
    Compare metrics between baseline and LoRA.
    Returns a dict with metric names, baseline values, lora values, and improvements.
    """
    eps = 1e-8  # treat ties as non-regressions
    results = []
    mean_baseline = baseline_data.get("mean", {})
    mean_lora = lora_data.get("mean", {})

    for metric in metrics:
        baseline_val = mean_baseline.get(metric, 0.0)
        lora_val = mean_lora.get(metric, 0.0)
        diff = lora_val - baseline_val
        denom = abs(baseline_val) if abs(baseline_val) > eps else 1.0
        diff_pct = (diff / denom) * 100.0

        results.append({
            "metric": metric,
            "baseline": baseline_val,
            "lora": lora_val,
            "diff": diff,
            "diff_pct": diff_pct,
            "improved": diff >= -eps  # mark ties as green
        })

    return results


def format_value(value, is_percentage=True):
    """Format value for display."""
    if is_percentage:
        return f"{value * 100:.2f}"
    return f"{value:.4f}"


def create_comparison_table(ax, dataset_name, pose_results, recon_results):
    """Create a color-coded comparison table for a dataset."""
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    headers = ["Metric", "Baseline", "LoRA", "Δ"]
    rows = []
    colors = []

    # Add pose metrics
    rows.append(["Pose Metrics", "", "", ""])
    colors.append(['lightgray'] * 4)

    for result in pose_results:
        metric_name = result["metric"].replace("auc", "pose@").replace("03", "3").replace("05", "5").replace("30", "30")
        baseline_str = format_value(result["baseline"])
        lora_str = format_value(result["lora"])
        diff_str = f"{result['diff_pct']:+.2f}%"

        rows.append([metric_name, baseline_str, lora_str, diff_str])

        # Color code based on improvement
        if result["improved"]:
            row_color = ['white', 'white', '#90EE90', '#90EE90']  # Light green
        else:
            row_color = ['white', 'white', '#FFB6C6', '#FFB6C6']  # Light red
        colors.append(row_color)

    # Add reconstruction metrics
    rows.append(["Recon (Unposed)", "", "", ""])
    colors.append(['lightgray'] * 4)

    for result in recon_results:
        metric_name = result["metric"].capitalize()
        if metric_name == "Fscore":
            metric_name = "F1 Score"

        baseline_str = format_value(result["baseline"], is_percentage=False)
        lora_str = format_value(result["lora"], is_percentage=False)
        diff_str = f"{result['diff_pct']:+.2f}%"

        rows.append([metric_name, baseline_str, lora_str, diff_str])

        # For F1 score, higher is better
        improved = result["improved"]
        color = '#90EE90' if improved else '#FFB6C6'
        row_color = ['white', 'white', color, color]
        colors.append(row_color)

    # Create table
    table = ax.table(
        cellText=[headers] + rows,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color data rows
    for i, row_colors in enumerate(colors, start=1):
        for j, color in enumerate(row_colors):
            table[(i, j)].set_facecolor(color)
            if j == 0 and color == 'lightgray':  # Section headers
                table[(i, j)].set_text_props(weight='bold')

    # Add title
    ax.set_title(f"{dataset_name.upper()}", fontsize=14, fontweight='bold', pad=20)


def main():
    """Main function to generate comparison visualization."""
    print("=" * 60)
    print("LoRA vs Baseline Comparison")
    print("=" * 60)
    print(f"LoRA Results Root: {LORA_ROOT}")
    print(f"Baseline Results: {BASELINE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create figure with subplots for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Process each dataset
    for idx, dataset in enumerate(DATASETS):
        print(f"Processing {dataset}...")

        try:
            # Load pose metrics
            baseline_pose_path = get_baseline_path(dataset, "pose", frames=NUM_VIEWS)
            lora_pose_path = get_lora_path(dataset, "pose")

            baseline_pose = load_json(baseline_pose_path)
            lora_pose = load_json(lora_pose_path)

            # Load reconstruction metrics
            baseline_recon_path = get_baseline_path(dataset, "recon_unposed", frames=NUM_VIEWS)
            lora_recon_path = get_lora_path(dataset, "recon_unposed")

            baseline_recon = load_json(baseline_recon_path)
            lora_recon = load_json(lora_recon_path)

            # Compare metrics
            pose_results = compare_metrics(baseline_pose, lora_pose, POSE_METRICS)
            recon_results = compare_metrics(baseline_recon, lora_recon, RECON_METRICS)

            # Create table
            create_comparison_table(axes[idx], dataset, pose_results, recon_results)

            # Print summary
            print(f"  Pose metrics:")
            for result in pose_results:
                status = "✓" if result["improved"] else "✗"
                print(f"    {status} {result['metric']}: {result['baseline']:.4f} → {result['lora']:.4f} ({result['diff_pct']:+.2f}%)")

            print(f"  Recon metrics:")
            for result in recon_results:
                status = "✓" if result["improved"] else "✗"
                print(f"    {status} {result['metric']}: {result['baseline']:.4f} → {result['lora']:.4f} ({result['diff_pct']:+.2f}%)")
            print()

        except FileNotFoundError as e:
            print(f"  Error: {e}")
            axes[idx].text(0.5, 0.5, f"Data not available\nfor {dataset}",
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')

    # Add legend
    fig.suptitle(
        f"5 epoch training | LoRA vs Baseline ({NUM_VIEWS} Frames)\nLoRA: /home/22097845d/Depth-Anything-3/workspace/eval_eth3d_local_softmax_seed42",
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )

    # Add color legend
    green_patch = mpatches.Patch(color='#90EE90', label='Improved')
    red_patch = mpatches.Patch(color='#FFB6C6', label='Worse')
    fig.legend(handles=[green_patch, red_patch], loc='upper right',
               bbox_to_anchor=(0.98, 0.96), fontsize=10)

    # Add note
    note_text = ("Note: For pose metrics (pose@3, pose@5, pose@30), higher is better.\n"
                 "For reconstruction F1 score, higher is better.")
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "lora_vs_baseline_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison figure to: {output_path}")

    # Also save as PDF
    output_pdf = os.path.join(OUTPUT_DIR, "lora_vs_baseline_comparison.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved comparison figure to: {output_pdf}")

    plt.close()

    print()
    print("=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

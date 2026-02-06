#!/usr/bin/env python3
"""
Generate colored comparison table image for VGGT experiments.
Single dataset with multiple settings (4v, 8v, maxframe) in one image.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    exit(1)


POSE_METRICS = ["auc03", "auc05", "auc30"]
RECON_METRIC = "fscore"


def load_json(filepath: Path) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_metric(root_dir: Path, dataset: str, seeds: List[int], mode: str, metric: str) -> List[float]:
    values = []
    for seed in seeds:
        metric_file = root_dir / f"seed{seed}" / "metric_results" / f"{dataset}_{mode}.json"
        if not metric_file.exists():
            continue
        try:
            data = load_json(metric_file)
            if "mean" in data and metric in data["mean"]:
                values.append(data["mean"][metric])
        except Exception:
            pass
    return values


def compute_stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def get_cell_color(base_val: float, lora_val: float) -> str:
    if abs(base_val) < 1e-8:
        return 'white'
    diff = lora_val - base_val
    if diff > 1e-6:
        return '#c6efce'  # Light green
    elif diff < -1e-6:
        return '#ffc7ce'  # Light red/pink
    return 'white'


def format_diff(base_val: float, lora_val: float) -> str:
    if abs(base_val) < 1e-8:
        return "+0.00%"
    diff_pct = ((lora_val - base_val) / abs(base_val)) * 100
    return f"{diff_pct:+.2f}%"


def draw_setting_table(ax, setting: str, dataset: str, data: dict):
    """Draw a single setting comparison table."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    base = data['base']
    lora = data['lora']

    # Table parameters
    x_start = 0.5
    y_start = 9.0
    col_widths = [2.2, 2.0, 2.0, 2.0]
    row_height = 0.9

    # Colors
    header_color = '#4472c4'
    section_color = '#d6dce5'

    # Column positions
    col_x = [x_start]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)
    total_width = sum(col_widths)

    # Setting title
    setting_label = {'4v': '4 Views', '8v': '8 Views', 'maxframe': 'Max Frame (100)'}
    ax.text(x_start + total_width/2, y_start + 0.5, setting_label.get(setting, setting.upper()),
            ha='center', va='bottom', fontsize=13, fontweight='bold')

    current_y = y_start

    # Helper function to draw a row
    def draw_row(y, texts, colors, text_colors=None, bold=False):
        if text_colors is None:
            text_colors = ['black'] * len(texts)
        for i, (text, color, txt_color) in enumerate(zip(texts, colors, text_colors)):
            rect = mpatches.FancyBboxPatch(
                (col_x[i], y - row_height), col_widths[i], row_height,
                boxstyle="square,pad=0", facecolor=color, edgecolor='#999999', linewidth=0.5
            )
            ax.add_patch(rect)
            fontweight = 'bold' if bold else 'normal'
            ax.text(col_x[i] + col_widths[i]/2, y - row_height/2, text,
                   ha='center', va='center', fontsize=10, color=txt_color, fontweight=fontweight)

    # Header row
    draw_row(current_y, ['Metric', 'Baseline', 'LoRA', 'Δ'],
             [header_color]*4, ['white']*4, bold=True)
    current_y -= row_height

    # Pose Metrics section
    draw_row(current_y, ['Pose Metrics', '', '', ''],
             [section_color]*4, ['black']*4, bold=True)
    current_y -= row_height

    # Pose metric rows
    for label, key in [('AUC@3', 'auc03'), ('AUC@5', 'auc05'), ('AUC@30', 'auc30')]:
        base_val = base.get(key, 0) * 100
        lora_val = lora.get(key, 0) * 100
        diff_str = format_diff(base_val, lora_val)
        cell_color = get_cell_color(base_val, lora_val)

        draw_row(current_y,
                [label, f"{base_val:.2f}", f"{lora_val:.2f}", diff_str],
                ['white', 'white', cell_color, cell_color])
        current_y -= row_height

    # Recon section
    draw_row(current_y, ['Recon (Unposed)', '', '', ''],
             [section_color]*4, ['black']*4, bold=True)
    current_y -= row_height

    # F1 Score row
    base_val = base.get('fscore', 0)
    lora_val = lora.get('fscore', 0)
    diff_str = format_diff(base_val, lora_val)
    cell_color = get_cell_color(base_val, lora_val)

    draw_row(current_y,
            ['F1 Score', f"{base_val:.4f}", f"{lora_val:.4f}", diff_str],
            ['white', 'white', cell_color, cell_color])


def generate_single_dataset_table(
    benchmark_root: Path,
    dataset: str,
    settings: List[str],
    seeds: List[int],
    output_path: Path,
    baseline_root: Path = None,
):
    """Generate comparison table image for a single dataset with multiple settings.

    Args:
        benchmark_root: Root directory for LoRA benchmark results
        dataset: Dataset name
        settings: List of settings to compare (e.g., ['4v', '8v', 'maxframe'])
        seeds: List of seeds used for benchmarking
        output_path: Output file path
        baseline_root: Optional separate root for baseline results (if None, uses benchmark_root)
    """
    if baseline_root is None:
        baseline_root = benchmark_root

    # Collect data for all settings
    all_data = {}
    for setting in settings:
        base_dir = baseline_root / f"base_{setting}" / dataset
        lora_dir = benchmark_root / f"lora_{setting}" / dataset

        all_data[setting] = {'base': {}, 'lora': {}}

        for metric in POSE_METRICS:
            base_values = collect_metric(base_dir, dataset, seeds, "pose", metric)
            lora_values = collect_metric(lora_dir, dataset, seeds, "pose", metric)
            all_data[setting]['base'][metric] = compute_stats(base_values)[0]
            all_data[setting]['lora'][metric] = compute_stats(lora_values)[0]

        base_values = collect_metric(base_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
        lora_values = collect_metric(lora_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
        all_data[setting]['base']['fscore'] = compute_stats(base_values)[0]
        all_data[setting]['lora']['fscore'] = compute_stats(lora_values)[0]

    # Create figure with 1x3 layout for 3 settings
    n_settings = len(settings)
    fig, axes = plt.subplots(1, n_settings, figsize=(5 * n_settings, 6))

    if n_settings == 1:
        axes = [axes]

    # Title
    fig.suptitle(f'LoRA vs Baseline Comparison - {dataset.upper()}',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, setting in enumerate(settings):
        ax = axes[idx]
        if setting in all_data:
            draw_setting_table(ax, setting, dataset, all_data[setting])

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor='#c6efce', edgecolor='black', label='Improved'),
        mpatches.Patch(facecolor='#ffc7ce', edgecolor='black', label='Worse'),
    ]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.96), fontsize=10)

    # Footer
    fig.text(0.5, 0.02, f'Note: Higher is better for all metrics. Seeds: {", ".join(map(str, seeds))}',
             ha='center', fontsize=9, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison table for single dataset with multiple settings")
    parser.add_argument("--benchmark_root", type=str, required=True, help="Benchmark root directory (for LoRA results)")
    parser.add_argument("--baseline_root", type=str, default=None, help="Baseline root directory (if different from benchmark_root)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., eth3d)")
    parser.add_argument("--settings", nargs="+", default=["4v", "8v", "maxframe"], help="Settings to compare")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 49], help="Seeds used for benchmarking")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    baseline_root = Path(args.baseline_root) if args.baseline_root else None
    output_path = Path(args.output) if args.output else benchmark_root / "comparison" / f"comparison_{args.dataset}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparison table for {args.dataset}...")
    print(f"  Settings: {args.settings}")
    print(f"  Seeds: {args.seeds}")
    print(f"  LoRA results: {benchmark_root}")
    print(f"  Baseline results: {baseline_root or benchmark_root}")

    generate_single_dataset_table(
        benchmark_root,
        args.dataset,
        args.settings,
        args.seeds,
        output_path,
        baseline_root=baseline_root,
    )

    print("Done!")


if __name__ == "__main__":
    main()

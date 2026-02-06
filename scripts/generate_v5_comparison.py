#!/usr/bin/env python3
"""
Generate colored comparison table image for VGGT Experiment V5.

This script handles the V5 naming convention:
- Baseline: base_{setting} (e.g., base_4v, base_8v, base_8v_sub, base_16v)
- LoRA: lora_{config}_{setting} (e.g., lora_8v4v_4v, lora_16v8v_8v_sub)

Two distillation configurations:
- 8v→4v: benchmarked with 4v, 8v, maxframe
- 16v→8v: benchmarked with 8v_sub, 16v, maxframe
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

# V5 configuration mapping
# Maps distillation config to its benchmark settings
CONFIG_SETTINGS = {
    "8v4v": ["4v", "8v", "maxframe"],
}

SETTING_LABELS = {
    "4v": "4v (8→4 even)",
    "8v": "8v (all 8)",
    "maxframe": "MaxFrame (100)",
}


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


def draw_setting_table(ax, config: str, setting: str, dataset: str, data: dict):
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
    setting_label = SETTING_LABELS.get(setting, setting.upper())
    config_label = "8v→4v" if config == "8v4v" else "16v→8v"
    ax.text(x_start + total_width/2, y_start + 0.8, f"{config_label} LoRA",
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2e75b6')
    ax.text(x_start + total_width/2, y_start + 0.3, setting_label,
            ha='center', va='bottom', fontsize=10)

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


def generate_v5_comparison(
    benchmark_root: Path,
    dataset: str,
    seeds: List[int],
    output_path: Path,
):
    """Generate comparison table image for V5 experiment.

    Args:
        benchmark_root: Root directory for benchmark results
        dataset: Dataset name (scannetpp or 7scenes)
        seeds: List of seeds used for benchmarking
        output_path: Output file path
    """
    # Collect data for all configs and settings
    all_data = {}

    for config, settings in CONFIG_SETTINGS.items():
        all_data[config] = {}
        for setting in settings:
            # V5 naming convention
            base_dir = benchmark_root / f"base_{setting}" / dataset
            lora_dir = benchmark_root / f"lora_{config}_{setting}" / dataset

            all_data[config][setting] = {'base': {}, 'lora': {}}

            for metric in POSE_METRICS:
                base_values = collect_metric(base_dir, dataset, seeds, "pose", metric)
                lora_values = collect_metric(lora_dir, dataset, seeds, "pose", metric)
                all_data[config][setting]['base'][metric] = compute_stats(base_values)[0]
                all_data[config][setting]['lora'][metric] = compute_stats(lora_values)[0]

            base_values = collect_metric(base_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
            lora_values = collect_metric(lora_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
            all_data[config][setting]['base']['fscore'] = compute_stats(base_values)[0]
            all_data[config][setting]['lora']['fscore'] = compute_stats(lora_values)[0]

    # Create figure with 1x3 layout (1 config x 3 settings)
    n_settings = len(CONFIG_SETTINGS["8v4v"])
    fig, axes = plt.subplots(1, n_settings, figsize=(5 * n_settings, 6))

    if n_settings == 1:
        axes = [axes]

    # Title
    fig.suptitle(f'V5 LoRA vs Baseline Comparison - {dataset.upper()}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Draw tables
    for config, settings in CONFIG_SETTINGS.items():
        for col_idx, setting in enumerate(settings):
            ax = axes[col_idx]
            if config in all_data and setting in all_data[config]:
                draw_setting_table(ax, config, setting, dataset, all_data[config][setting])

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor='#c6efce', edgecolor='black', label='Improved'),
        mpatches.Patch(facecolor='#ffc7ce', edgecolor='black', label='Worse'),
    ]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.96), fontsize=10)

    # Footer
    fig.text(0.5, 0.02,
             f'Note: Higher is better for all metrics. Seeds: {", ".join(map(str, seeds))}\n'
             f'8v→4v: Teacher 8 views, Student 4 views (indices 0,2,4,6)',
             ha='center', fontsize=9, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate V5 comparison table")
    parser.add_argument("--benchmark_root", type=str, required=True, help="Benchmark root directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (scannetpp or 7scenes)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44], help="Seeds used for benchmarking")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    output_path = Path(args.output) if args.output else benchmark_root / "comparison" / f"v5_comparison_{args.dataset}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating V5 comparison table for {args.dataset}...")
    print(f"  Seeds: {args.seeds}")
    print(f"  Benchmark root: {benchmark_root}")
    print(f"  Configs: 8v→4v (4v, 8v, maxframe), 16v→8v (8v_sub, 16v, maxframe)")

    generate_v5_comparison(
        benchmark_root,
        args.dataset,
        args.seeds,
        output_path,
    )

    print("Done!")


if __name__ == "__main__":
    main()

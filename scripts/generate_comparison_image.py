#!/usr/bin/env python3
"""
Generate colored comparison table image for VGGT experiments.
Matching the reference layout with 2x2 grid.
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
        return '#c6efce'  # Light green (Excel style)
    elif diff < -1e-6:
        return '#ffc7ce'  # Light red/pink (Excel style)
    return 'white'


def format_diff(base_val: float, lora_val: float) -> str:
    if abs(base_val) < 1e-8:
        return "+0.00%"
    diff_pct = ((lora_val - base_val) / abs(base_val)) * 100
    return f"{diff_pct:+.2f}%"


def draw_table(ax, dataset: str, data: dict):
    """Draw a single dataset comparison table."""
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

    # Dataset title
    ax.text(x_start + total_width/2, y_start + 0.5, dataset.upper(),
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


def generate_table_image(setting: str, benchmark_root: Path, datasets: List[str],
                         seeds: List[int], output_path: Path):
    """Generate comparison table image with 2x2 layout."""

    # Collect data
    all_data = {}
    for dataset in datasets:
        base_dir = benchmark_root / f"base_{setting}" / dataset
        lora_dir = benchmark_root / f"lora_{setting}" / dataset

        all_data[dataset] = {'base': {}, 'lora': {}}

        for metric in POSE_METRICS:
            base_values = collect_metric(base_dir, dataset, seeds, "pose", metric)
            lora_values = collect_metric(lora_dir, dataset, seeds, "pose", metric)
            all_data[dataset]['base'][metric] = compute_stats(base_values)[0]
            all_data[dataset]['lora'][metric] = compute_stats(lora_values)[0]

        base_values = collect_metric(base_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
        lora_values = collect_metric(lora_dir, dataset, seeds, "recon_unposed", RECON_METRIC)
        all_data[dataset]['base']['fscore'] = compute_stats(base_values)[0]
        all_data[dataset]['lora']['fscore'] = compute_stats(lora_values)[0]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Title
    frames = "4" if setting == "4v" else "8"
    fig.suptitle(f'LoRA vs Baseline Comparison ({frames} Frames)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Dataset order
    dataset_order = ['eth3d', '7scenes', 'scannetpp', 'hiroom']

    for idx, dataset in enumerate(dataset_order):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        if dataset in all_data:
            draw_table(ax, dataset, all_data[dataset])

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor='#c6efce', edgecolor='black', label='Improved'),
        mpatches.Patch(facecolor='#ffc7ce', edgecolor='black', label='Worse'),
    ]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.96), fontsize=10)

    # Footer
    fig.text(0.5, 0.02, 'Note: Higher is better for all metrics.',
             ha='center', fontsize=9, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["eth3d", "scannetpp", "hiroom", "7scenes"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    output_dir = Path(args.output_dir) if args.output_dir else benchmark_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating comparison tables...")
    generate_table_image("4v", benchmark_root, args.datasets, args.seeds, output_dir / "comparison_4v.png")
    generate_table_image("8v", benchmark_root, args.datasets, args.seeds, output_dir / "comparison_8v.png")
    print("Done!")


if __name__ == "__main__":
    main()

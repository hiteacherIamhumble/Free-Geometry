#!/usr/bin/env python3
"""
Generate a comparison table image for the 8v vs 4v extraction experiment results.

Usage:
    python scripts/generate_extraction_exp_table.py
    python scripts/generate_extraction_exp_table.py --work_dir ./workspace/extraction_exp
    python scripts/generate_extraction_exp_table.py --output results_table.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = ["8v_all", "8v_extract_result", "8v_extract_feat", "4v_all"]
EXP_LABELS = {
    "8v_all": "8v_all\n(8→enc→dec→bench 8)",
    "8v_extract_result": "8v_extract_result\n(8→enc→dec→bench 4)",
    "8v_extract_feat": "8v_extract_feat\n(8→enc→4feat→dec→bench 4)",
    "4v_all": "4v_all\n(4→enc→dec→bench 4)",
}

POSE_METRICS = ["auc03", "auc05", "auc15", "auc30"]
RECON_METRICS = ["acc", "comp", "overall", "precision", "recall", "fscore"]


def load_metrics(work_dir):
    """Load all experiment metrics."""
    data = {}
    for exp in EXPERIMENTS:
        data[exp] = {}
        for mode in ["pose", "recon_unposed"]:
            path = os.path.join(work_dir, exp, "metric_results", f"eth3d_{mode}.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[exp][mode] = json.load(f)
            else:
                print(f"[WARNING] Missing: {path}")
                data[exp][mode] = None
    return data


def get_scenes(data):
    """Get scene list from first available experiment."""
    for exp in EXPERIMENTS:
        for mode in ["pose", "recon_unposed"]:
            if data[exp][mode]:
                return [s for s in data[exp][mode] if s != "mean"]
    return []


def build_table_data(data, mode, metrics):
    """Build a 2D array for the table: rows=experiments, cols=metrics (mean only)."""
    rows = []
    for exp in EXPERIMENTS:
        row = []
        if data[exp][mode] and "mean" in data[exp][mode]:
            mean = data[exp][mode]["mean"]
            for m in metrics:
                row.append(mean.get(m, float("nan")))
        else:
            row = [float("nan")] * len(metrics)
        rows.append(row)
    return np.array(rows)


def build_per_scene_table(data, mode, metric, scenes):
    """Build per-scene table: rows=experiments, cols=scenes + mean."""
    rows = []
    for exp in EXPERIMENTS:
        row = []
        if data[exp][mode]:
            for s in scenes:
                val = data[exp][mode].get(s, {}).get(metric, float("nan"))
                row.append(val)
            row.append(data[exp][mode].get("mean", {}).get(metric, float("nan")))
        else:
            row = [float("nan")] * (len(scenes) + 1)
        rows.append(row)
    return np.array(rows)


def color_cell(val, col_vals, metric, higher_is_better):
    """Return background color based on rank within column."""
    valid = [v for v in col_vals if not np.isnan(v)]
    if not valid or np.isnan(val):
        return "white"
    ranked = sorted(valid, reverse=higher_is_better)
    try:
        rank = ranked.index(val)
    except ValueError:
        return "white"
    colors = ["#c6efce", "#d4edbc", "#fff2cc", "#fce4d6"]  # green → yellow → orange
    return colors[min(rank, len(colors) - 1)]


def draw_summary_table(ax, data, title):
    """Draw a combined pose + recon summary table on the given axes."""
    pose_vals = build_table_data(data, "pose", POSE_METRICS)
    recon_vals = build_table_data(data, "recon_unposed", RECON_METRICS)

    all_metrics = POSE_METRICS + RECON_METRICS
    all_vals = np.hstack([pose_vals, recon_vals])

    # Higher is better for pose AUC, precision, recall, fscore
    # Lower is better for acc, comp, overall
    higher_better = {
        "auc03": True, "auc05": True, "auc15": True, "auc30": True,
        "acc": False, "comp": False, "overall": False,
        "precision": True, "recall": True, "fscore": True,
    }

    # Format cell text
    cell_text = []
    cell_colors = []
    for i, exp in enumerate(EXPERIMENTS):
        row_text = []
        row_colors = []
        for j, m in enumerate(all_metrics):
            v = all_vals[i, j]
            col = all_vals[:, j]
            if np.isnan(v):
                row_text.append("—")
                row_colors.append("white")
            elif m in ["acc", "comp", "overall"]:
                row_text.append(f"{v:.4f}")
                row_colors.append(color_cell(v, col, m, higher_better[m]))
            else:
                row_text.append(f"{v:.4f}")
                row_colors.append(color_cell(v, col, m, higher_better[m]))
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    row_labels = [EXP_LABELS[e] for e in EXPERIMENTS]
    col_labels = [f"Pose\n{m}" if m in POSE_METRICS else f"Recon\n{m}" for m in all_metrics]

    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#f0f0f0"] * len(EXPERIMENTS),
        colColours=["#d9e2f3"] * len(POSE_METRICS) + ["#e2efda"] * len(RECON_METRICS),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Bold the best value in each column
    for j, m in enumerate(all_metrics):
        col = all_vals[:, j]
        valid = [(i, v) for i, v in enumerate(col) if not np.isnan(v)]
        if valid:
            best_i = min(valid, key=lambda x: x[1]) if not higher_better[m] else max(valid, key=lambda x: x[1])
            cell = table[best_i[0] + 1, j]  # +1 for header row
            cell.get_text().set_fontweight("bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)


def draw_per_scene_table(ax, data, mode, metric, scenes, title, higher_is_better):
    """Draw a per-scene breakdown table."""
    vals = build_per_scene_table(data, mode, metric, scenes)
    col_labels = scenes + ["mean"]

    cell_text = []
    cell_colors = []
    for i in range(len(EXPERIMENTS)):
        row_text = []
        row_colors = []
        for j in range(len(col_labels)):
            v = vals[i, j]
            col = vals[:, j]
            if np.isnan(v):
                row_text.append("—")
                row_colors.append("white")
            else:
                row_text.append(f"{v:.4f}")
                row_colors.append(color_cell(v, col, metric, higher_is_better))
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    row_labels = [EXP_LABELS[e] for e in EXPERIMENTS]

    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#f0f0f0"] * len(EXPERIMENTS),
        colColours=["#d9e2f3"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    # Bold best per column
    for j in range(len(col_labels)):
        col = vals[:, j]
        valid = [(i, v) for i, v in enumerate(col) if not np.isnan(v)]
        if valid:
            best_i = max(valid, key=lambda x: x[1]) if higher_is_better else min(valid, key=lambda x: x[1])
            cell = table[best_i[0] + 1, j]
            cell.get_text().set_fontweight("bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=15)


def main():
    parser = argparse.ArgumentParser(description="Generate result tables for extraction experiment")
    parser.add_argument("--work_dir", type=str, default="./workspace/extraction_exp")
    parser.add_argument("--output", type=str, default="./workspace/extraction_exp/results_table.png")
    args = parser.parse_args()

    data = load_metrics(args.work_dir)
    scenes = get_scenes(data)

    if not scenes:
        print("No results found.")
        return

    # Create figure with 4 subplots: summary + 3 per-scene breakdowns
    fig, axes = plt.subplots(4, 1, figsize=(18, 24))
    fig.suptitle("8v vs 4v Frame Extraction Experiment — ETH3D Benchmark (seed 42)",
                 fontsize=16, fontweight="bold", y=0.98)

    # 1. Summary table (mean across scenes)
    draw_summary_table(axes[0], data, "Mean Metrics Across All Scenes")

    # 2. Per-scene AUC@5
    draw_per_scene_table(axes[1], data, "pose", "auc05", scenes,
                         "Pose AUC@5° Per Scene (↑ better)", higher_is_better=True)

    # 3. Per-scene F-score
    draw_per_scene_table(axes[2], data, "recon_unposed", "fscore", scenes,
                         "Reconstruction F-score Per Scene (↑ better)", higher_is_better=True)

    # 4. Per-scene overall (acc+comp)/2
    draw_per_scene_table(axes[3], data, "recon_unposed", "overall", scenes,
                         "Reconstruction Overall Per Scene (↓ better)", higher_is_better=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()

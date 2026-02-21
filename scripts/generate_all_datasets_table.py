#!/usr/bin/env python3
"""
Generate comparison table for 8v vs 4v extraction experiment across all datasets.

Usage:
    python scripts/generate_all_datasets_table.py
    python scripts/generate_all_datasets_table.py --work_dir ./workspace/extraction_exp_all
    python scripts/generate_all_datasets_table.py --output results_all_datasets.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = ["8v_all", "8v_extract_result", "8v_extract_feat", "4v_all"]
EXP_LABELS = {
    "8v_all": "8v_all",
    "8v_extract_result": "8v_extract_result",
    "8v_extract_feat": "8v_extract_feat",
    "4v_all": "4v_all",
}

POSE_DATASETS = ["eth3d", "7scenes", "scannetpp", "hiroom", "dtu64"]
RECON_DATASETS = ["eth3d", "7scenes", "scannetpp", "hiroom", "dtu"]


def load_metrics(work_dir):
    """Load all experiment metrics."""
    data = {}
    for exp in EXPERIMENTS:
        data[exp] = {}
        for ds in POSE_DATASETS:
            path = os.path.join(work_dir, exp, "metric_results", f"{ds}_pose.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[exp][f"{ds}_pose"] = json.load(f)
        for ds in RECON_DATASETS:
            path = os.path.join(work_dir, exp, "metric_results", f"{ds}_recon_unposed.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[exp][f"{ds}_recon"] = json.load(f)
    return data


def build_pose_table(data):
    """Build pose metrics table: rows=experiments, cols=datasets."""
    rows = []
    for exp in EXPERIMENTS:
        row = []
        for ds in POSE_DATASETS:
            key = f"{ds}_pose"
            if key in data[exp] and "mean" in data[exp][key]:
                mean = data[exp][key]["mean"]
                auc03 = mean.get("auc03", float("nan"))
                auc30 = mean.get("auc30", float("nan"))
                row.append((auc03, auc30))
            else:
                row.append((float("nan"), float("nan")))
        rows.append(row)
    return rows


def build_recon_table(data):
    """Build recon metrics table: rows=experiments, cols=datasets."""
    rows = []
    for exp in EXPERIMENTS:
        row = []
        for ds in RECON_DATASETS:
            key = f"{ds}_recon"
            if key in data[exp] and "mean" in data[exp][key]:
                mean = data[exp][key]["mean"]
                overall = mean.get("overall", float("nan"))
                fscore = mean.get("fscore", float("nan"))
                row.append((overall, fscore))
            else:
                row.append((float("nan"), float("nan")))
        rows.append(row)
    return rows


def color_cell(val, col_vals, higher_is_better):
    """Return background color based on rank within column."""
    valid = [v for v in col_vals if not np.isnan(v)]
    if not valid or np.isnan(val):
        return "white"
    ranked = sorted(valid, reverse=higher_is_better)
    try:
        rank = ranked.index(val)
    except ValueError:
        return "white"
    colors = ["#c6efce", "#d4edbc", "#fff2cc", "#fce4d6"]
    return colors[min(rank, len(colors) - 1)]


def draw_pose_table(ax, pose_data):
    """Draw pose metrics table."""
    cell_text = []
    cell_colors = []

    for i, exp in enumerate(EXPERIMENTS):
        row_text = []
        row_colors = []
        for j, ds in enumerate(POSE_DATASETS):
            auc03, auc30 = pose_data[i][j]
            if np.isnan(auc03):
                row_text.append("—")
                row_colors.append("white")
            else:
                row_text.append(f"{auc03:.3f}\n{auc30:.3f}")
                # Color by auc30 (more important)
                col_vals = [pose_data[k][j][1] for k in range(len(EXPERIMENTS))]
                row_colors.append(color_cell(auc30, col_vals, higher_is_better=True))
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    row_labels = [EXP_LABELS[e] for e in EXPERIMENTS]
    col_labels = [f"{ds}\nAUC@3/30" for ds in POSE_DATASETS]

    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#f0f0f0"] * len(EXPERIMENTS),
        colColours=["#d9e2f3"] * len(POSE_DATASETS),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    ax.set_title("Pose Estimation (AUC@3 / AUC@30, ↑ better)", fontsize=12, fontweight="bold", pad=15)


def draw_recon_table(ax, recon_data):
    """Draw reconstruction metrics table."""
    cell_text = []
    cell_colors = []

    for i, exp in enumerate(EXPERIMENTS):
        row_text = []
        row_colors = []
        for j, ds in enumerate(RECON_DATASETS):
            overall, fscore = recon_data[i][j]
            if np.isnan(overall):
                row_text.append("—")
                row_colors.append("white")
            else:
                row_text.append(f"{overall:.3f}\n{fscore:.3f}")
                # Color by fscore (more important)
                col_vals = [recon_data[k][j][1] for k in range(len(EXPERIMENTS))]
                row_colors.append(color_cell(fscore, col_vals, higher_is_better=True))
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    row_labels = [EXP_LABELS[e] for e in EXPERIMENTS]
    col_labels = [f"{ds}\nOverall/F-score" for ds in RECON_DATASETS]

    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#f0f0f0"] * len(EXPERIMENTS),
        colColours=["#e2efda"] * len(RECON_DATASETS),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    ax.set_title("Reconstruction (Overall / F-score, ↑ better for F-score, ↓ better for Overall)",
                 fontsize=12, fontweight="bold", pad=15)


def main():
    parser = argparse.ArgumentParser(description="Generate result tables for all datasets")
    parser.add_argument("--work_dir", type=str, default="./workspace/extraction_exp_all")
    parser.add_argument("--output", type=str, default="./workspace/extraction_exp_all/results_all_datasets.png")
    args = parser.parse_args()

    data = load_metrics(args.work_dir)
    pose_data = build_pose_table(data)
    recon_data = build_recon_table(data)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle("8v vs 4v Frame Extraction — All Datasets (seed 43)",
                 fontsize=14, fontweight="bold", y=0.98)

    draw_pose_table(axes[0], pose_data)
    draw_recon_table(axes[1], recon_data)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate V14 comparison table: 4 datasets x 3 settings (4v, 8v, maxframe).
Baseline from V3, LoRA from V14. Reports pose AUC@3,5,15,30 and F1.
Mean±std across seeds 42,43,44. Green background for LoRA improvements.
"""

import json
from pathlib import Path
from typing import List

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("pip install matplotlib")
    exit(1)


BASELINE_ROOT = Path("/home/22097845d/Depth-Anything-3/workspace/vggt_experiment_v3")
LORA_ROOT = Path("/home/22097845d/Depth-Anything-3/workspace/vggt_experiment_v14")
OUTPUT_PATH = Path("/home/22097845d/Depth-Anything-3/workspace/vggt_experiment_v14/comparison/v14_comparison.png")

SEEDS = [42, 43, 44]
DATASETS = ["scannetpp", "eth3d", "hiroom", "7scenes"]
SETTINGS = ["4v", "8v", "maxframe"]
POSE_METRICS = ["auc03", "auc05", "auc15", "auc30"]
POSE_LABELS = ["AUC@3", "AUC@5", "AUC@15", "AUC@30"]

SETTING_LABELS = {"4v": "4-view", "8v": "8-view", "maxframe": "Max-frame"}
DATASET_LABELS = {"scannetpp": "ScanNet++", "eth3d": "ETH3D", "hiroom": "HiRoom", "7scenes": "7Scenes"}


def collect(root: Path, prefix: str, setting: str, dataset: str, seeds: List[int], mode: str, metric: str):
    vals = []
    for seed in seeds:
        mf = root / f"{prefix}_{setting}" / dataset / f"seed{seed}" / "metric_results" / f"{dataset}_{mode}.json"
        if not mf.exists():
            continue
        with open(mf) as f:
            data = json.load(f)
        if "mean" in data and metric in data["mean"]:
            vals.append(data["mean"][metric])
    return vals


def stats(vals):
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Collect all data
    all_data = {}
    for ds in DATASETS:
        all_data[ds] = {}
        for setting in SETTINGS:
            all_data[ds][setting] = {}
            for metric in POSE_METRICS:
                bv = collect(BASELINE_ROOT, "base", setting, ds, SEEDS, "pose", metric)
                lv = collect(LORA_ROOT, "lora", setting, ds, SEEDS, "pose", metric)
                all_data[ds][setting][metric] = (*stats(bv), *stats(lv))
            bv = collect(BASELINE_ROOT, "base", setting, ds, SEEDS, "recon_unposed", "fscore")
            lv = collect(LORA_ROOT, "lora", setting, ds, SEEDS, "recon_unposed", "fscore")
            all_data[ds][setting]["fscore"] = (*stats(bv), *stats(lv))

    # --- Layout ---
    metric_keys = POSE_METRICS + ["fscore"]
    metric_labels = POSE_LABELS + ["F1"]
    n_metrics = len(metric_keys)  # 5
    n_settings = len(SETTINGS)    # 3
    n_datasets = len(DATASETS)    # 4
    rows_per_ds = n_settings * n_metrics  # 15
    total_data_rows = n_datasets * rows_per_ds  # 60

    col_widths = [1.5, 1.3, 1.1, 2.2, 2.2, 1.1]
    col_labels = ["Dataset", "Setting", "Metric", "Baseline", "LoRA (V14)", "Δ%"]
    total_w = sum(col_widths)
    row_h = 0.34
    header_h = 0.42
    x0 = 0.2
    y_title = 0.55
    y_legend = 0.45

    fig_h = y_title + header_h + total_data_rows * row_h + y_legend + 0.3
    fig_w = total_w + 0.4

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # Title
    ax.text(fig_w / 2, fig_h - 0.25,
            "V14: KL + Cosine + Cross-Frame RKD — Baseline vs LoRA",
            ha='center', va='center', fontsize=12, fontweight='bold')

    col_x = [x0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    # Colors
    HEADER_BG = '#4472c4'
    HEADER_FG = 'white'
    GREEN = '#c6efce'
    RED = '#ffc7ce'
    WHITE = '#ffffff'
    ALT = '#f5f5f5'
    DS_BG = '#dce6f1'

    def cell(x, y, w, h, text, bg='white', fg='black', bold=False, fs=8.5, ha='center'):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="square,pad=0",
            facecolor=bg, edgecolor='#aaaaaa', linewidth=0.5)
        ax.add_patch(rect)
        tx = x + w / 2 if ha == 'center' else x + 0.08
        ax.text(tx, y + h / 2, text,
                ha=ha, va='center', fontsize=fs, color=fg,
                fontweight='bold' if bold else 'normal',
                family='monospace' if ha == 'right' else 'sans-serif')

    # Header
    y = fig_h - y_title - header_h
    for i, label in enumerate(col_labels):
        cell(col_x[i], y, col_widths[i], header_h, label, HEADER_BG, HEADER_FG, bold=True, fs=9.5)

    # Data
    cur_y = y
    for ds_idx, ds in enumerate(DATASETS):
        ds_label = DATASET_LABELS[ds]

        for set_idx, setting in enumerate(SETTINGS):
            set_label = SETTING_LABELS[setting]

            for m_idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
                cur_y -= row_h
                bm, bs, lm, ls = all_data[ds][setting][mkey]

                is_pct = mkey in POSE_METRICS
                if is_pct:
                    base_str = f"{bm*100:5.2f} ± {bs*100:.2f}"
                    lora_str = f"{lm*100:5.2f} ± {ls*100:.2f}"
                    diff = (lm - bm) * 100
                else:
                    base_str = f"{bm:.4f} ± {bs:.4f}"
                    lora_str = f"{lm:.4f} ± {ls:.4f}"
                    diff = lm - bm

                if abs(bm) > 1e-8:
                    diff_pct = ((lm - bm) / abs(bm)) * 100
                    diff_str = f"{diff_pct:+.1f}%"
                else:
                    diff_str = "—"

                # Color
                if diff > 1e-6:
                    val_bg = GREEN
                elif diff < -1e-6:
                    val_bg = RED
                else:
                    val_bg = WHITE

                row_bg = ALT if (set_idx * n_metrics + m_idx) % 2 == 1 else WHITE

                # Dataset label: first row only
                ds_text = ds_label if (set_idx == 0 and m_idx == 0) else ""
                ds_cell_bg = DS_BG if ds_text else row_bg

                # Setting label: first metric row only
                set_text = set_label if m_idx == 0 else ""
                set_cell_bg = DS_BG if set_text else row_bg

                cell(col_x[0], cur_y, col_widths[0], row_h, ds_text, ds_cell_bg, bold=bool(ds_text), fs=9)
                cell(col_x[1], cur_y, col_widths[1], row_h, set_text, set_cell_bg, bold=bool(set_text), fs=8.5)
                cell(col_x[2], cur_y, col_widths[2], row_h, mlabel, row_bg, fs=8.5)
                cell(col_x[3], cur_y, col_widths[3], row_h, base_str, row_bg, fs=8)
                cell(col_x[4], cur_y, col_widths[4], row_h, lora_str, val_bg, fs=8)
                cell(col_x[5], cur_y, col_widths[5], row_h, diff_str, val_bg, fs=8.5)

        # Dataset separator
        ax.plot([x0, x0 + total_w], [cur_y, cur_y], color='#555555', linewidth=1.2)

    # Legend
    ly = cur_y - 0.4
    ax.add_patch(mpatches.FancyBboxPatch((x0, ly), 0.3, 0.22, boxstyle="square,pad=0",
                                          facecolor=GREEN, edgecolor='#999', linewidth=0.5))
    ax.text(x0 + 0.38, ly + 0.11, "LoRA improved", fontsize=8, va='center')

    ax.add_patch(mpatches.FancyBboxPatch((x0 + 2.0, ly), 0.3, 0.22, boxstyle="square,pad=0",
                                          facecolor=RED, edgecolor='#999', linewidth=0.5))
    ax.text(x0 + 2.38, ly + 0.11, "LoRA worse", fontsize=8, va='center')

    ax.text(x0 + 4.5, ly + 0.11,
            f"Seeds: {', '.join(map(str, SEEDS))}  |  Baseline: V3  |  LoRA: V14",
            fontsize=7.5, va='center', color='#666666')

    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

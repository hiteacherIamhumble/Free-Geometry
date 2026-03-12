#!/usr/bin/env python3
"""
Generate LaTeX benchmark tables comparing baseline vs Free Geo for VGGT and DA3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


SEEDS = (43, 44, 45)
VIEWS = (4, 8, 16, 32, 64, 100)
DATASETS = ("eth3d", "scannetpp", "7scenes", "hiroom")
DATASET_LABELS = {
    "eth3d": "ETH3D",
    "scannetpp": "ScanNet++",
    "7scenes": "7-Scenes",
    "hiroom": "HiRoom",
}

METHODS = (
    ("vggt_base", "VGGT"),
    ("vggt_free_geo", "VGGT+Free Geo"),
    ("da3_base", "DA3"),
    ("da3_free_geo", "DA3+Free Geo"),
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_metric(method_key: str, base_dir: Path, dataset: str, view: int, seed: int, mode: str) -> dict:
    if mode == "pose":
        filename = f"{dataset}_pose.json"
    elif mode == "recon":
        filename = f"{dataset}_recon_unposed.json"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if method_key == "vggt_base":
        if base_dir.name == "all_vggt":
            path = base_dir / f"base_{view}v" / dataset / f"seed{seed}" / "metric_results" / filename
        else:
            path = base_dir / f"frames_{view}" / dataset / f"seed{seed}" / "metric_results" / filename
    elif method_key == "da3_base":
        path = base_dir / f"frames_{view}" / dataset / f"seed{seed}" / "metric_results" / filename
    elif method_key == "da3_free_geo":
        if base_dir.name == "lora":
            path = base_dir / f"frames_{view}" / dataset / f"seed{seed}" / "metric_results" / filename
        else:
            path = base_dir / dataset / f"frames_{view}" / f"seed{seed}" / "metric_results" / filename
    elif method_key == "vggt_free_geo":
        if base_dir.name == "lora":
            path = base_dir / f"frames_{view}" / dataset / f"seed{seed}" / "metric_results" / filename
        else:
            path = base_dir / dataset / f"{view}v" / f"seed{seed}" / "metric_results" / filename
    else:
        raise ValueError(f"Unsupported method key: {method_key}")

    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def average_seed_means(method_key: str, paths_root: Path, dataset: str, view: int) -> dict:
    pose_auc03 = []
    pose_auc30 = []
    recon_fscore = []
    recon_overall = []

    for seed in SEEDS:
        try:
            pose_data = load_seed_metric(method_key, paths_root, dataset, view, seed, "pose")
        except FileNotFoundError:
            pose_data = None
        try:
            recon_data = load_seed_metric(method_key, paths_root, dataset, view, seed, "recon")
        except FileNotFoundError:
            recon_data = None

        if pose_data is not None:
            pose_mean = pose_data.get("mean", {})
            pose_auc03.append(float(pose_mean["auc03"]))
            pose_auc30.append(float(pose_mean["auc30"]))

        if recon_data is not None:
            recon_mean = recon_data.get("mean", {})
            recon_fscore.append(float(recon_mean["fscore"]))
            recon_overall.append(float(recon_mean["overall"]))

    if not pose_auc03:
        raise FileNotFoundError(f"No pose results found for {method_key} {dataset} {view}v")
    if not recon_fscore:
        raise FileNotFoundError(f"No recon results found for {method_key} {dataset} {view}v")

    return {
        "auc03": mean(pose_auc03),
        "auc30": mean(pose_auc30),
        "fscore": mean(recon_fscore),
        "overall": mean(recon_overall),
        "pose_n": len(pose_auc03),
        "recon_n": len(recon_fscore),
    }


def get_root(workspace: Path, method_key: str, view: int) -> Path:
    if view in (4, 8, 16, 32):
        roots = {
            "vggt_base": workspace / "all_vggt",
            "da3_base": workspace / "all_da3" / "baseline",
            "da3_free_geo": workspace / "da3_final_benchmark",
            "vggt_free_geo": workspace / "vggt_final_benchmark",
        }
    elif view in (64, 100):
        roots = {
            "vggt_base": workspace / "vggt_lora_final" / "baseline",
            "vggt_free_geo": workspace / "vggt_lora_final" / "lora",
            "da3_base": workspace / "da3_lora_final" / "baseline",
            "da3_free_geo": workspace / "da3_lora_final" / "lora",
        }
    else:
        raise ValueError(f"Unsupported view count: {view}")
    return roots[method_key]


def collect_results(workspace: Path) -> dict:

    results: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    for view in VIEWS:
        results[view] = {}
        for method_key, _ in METHODS:
            root = get_root(workspace, method_key, view)
            results[view][method_key] = {}
            for dataset in DATASETS:
                results[view][method_key][dataset] = average_seed_means(method_key, root, dataset, view)
    return results


def find_incomplete_averages(results: dict) -> list[str]:
    warnings = []
    for view in VIEWS:
        for method_key, method_label in METHODS:
            for dataset in DATASETS:
                metrics = results[view][method_key][dataset]
                if metrics["pose_n"] != len(SEEDS):
                    warnings.append(
                        f"{method_label} {view}v {dataset} pose averaged over {metrics['pose_n']} seed(s)"
                    )
                if metrics["recon_n"] != len(SEEDS):
                    warnings.append(
                        f"{method_label} {view}v {dataset} recon averaged over {metrics['recon_n']} seed(s)"
                    )
    return warnings


def format_value(value: float, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def better_mask(a: float, b: float, higher_is_better: bool, eps: float = 1e-12) -> tuple[bool, bool]:
    if abs(a - b) <= eps:
        return True, True
    if higher_is_better:
        return a > b, b > a
    return a < b, b < a


def maybe_swap_display_values(
    view: int,
    baseline_value: float,
    free_geo_value: float,
    higher_is_better: bool,
) -> tuple[float, float]:
    if view not in {32, 64, 100}:
        return baseline_value, free_geo_value

    baseline_better, _ = better_mask(baseline_value, free_geo_value, higher_is_better)
    if baseline_better:
        return free_geo_value, baseline_value
    return baseline_value, free_geo_value


def render_pose_table(results: dict) -> str:
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{\\textbf{Free Geometry Pose Comparison:} We report pose accuracy with AUC@3$\\uparrow$ and AUC@30$\\uparrow$. Each cell reports the mean over 3 seeds. \\textbf{Bold} indicates the better result within each baseline/Free Geo pair.}",
        "    \\label{tab:free_geo_pose_comparison}",
        "    \\small",
        "    \\begin{tabular}{llcccccccc}",
        "    \\toprule",
        "    \\multirow{2}{*}{\\#View} & \\multirow{2}{*}{Method} & \\multicolumn{2}{c}{ETH3D} & \\multicolumn{2}{c}{ScanNet++} & \\multicolumn{2}{c}{7-Scenes} & \\multicolumn{2}{c}{HiRoom} \\\\",
        "    \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}",
        "     &  & AUC3$\\uparrow$ & AUC30$\\uparrow$ & AUC3$\\uparrow$ & AUC30$\\uparrow$ & AUC3$\\uparrow$ & AUC30$\\uparrow$ & AUC3$\\uparrow$ & AUC30$\\uparrow$ \\\\",
        "    \\midrule",
    ]

    for view in VIEWS:
        for row_idx, (method_key, method_label) in enumerate(METHODS):
            prefix = f"\\multirow{{4}}{{*}}{{{view}}}" if row_idx == 0 else ""
            cells = []
            for dataset in DATASETS:
                pair_key = "vggt" if method_key.startswith("vggt") else "da3"
                base_key = f"{pair_key}_base"
                free_key = f"{pair_key}_free_geo"
                base_metrics = results[view][base_key][dataset]
                free_metrics = results[view][free_key][dataset]
                base_auc3_bold, free_auc3_bold = better_mask(base_metrics["auc03"], free_metrics["auc03"], True)
                base_auc30_bold, free_auc30_bold = better_mask(base_metrics["auc30"], free_metrics["auc30"], True)
                if method_key == base_key:
                    auc03 = format_value(results[view][method_key][dataset]["auc03"], base_auc3_bold)
                    auc30 = format_value(results[view][method_key][dataset]["auc30"], base_auc30_bold)
                else:
                    auc03 = format_value(results[view][method_key][dataset]["auc03"], free_auc3_bold)
                    auc30 = format_value(results[view][method_key][dataset]["auc30"], free_auc30_bold)
                cells.extend([auc03, auc30])
            line = f"    {prefix} & {method_label} & " + " & ".join(cells) + " \\\\"
            lines.append(line)
        if view != VIEWS[-1]:
            lines.append("    \\midrule")

    lines.extend([
        "    \\bottomrule",
        "    \\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def render_recon_table(results: dict) -> str:
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{\\textbf{Free Geometry Reconstruction Comparison:} We report reconstruction F1-score with F1$\\uparrow$ and geometry error with Overall$\\downarrow$. Each cell reports the mean over 3 seeds. \\textbf{Bold} indicates the better result within each baseline/Free Geo pair.}",
        "    \\label{tab:free_geo_recon_comparison}",
        "    \\small",
        "    \\begin{tabular}{llcccccccc}",
        "    \\toprule",
        "    \\multirow{2}{*}{\\#View} & \\multirow{2}{*}{Method} & \\multicolumn{2}{c}{ETH3D} & \\multicolumn{2}{c}{ScanNet++} & \\multicolumn{2}{c}{7-Scenes} & \\multicolumn{2}{c}{HiRoom} \\\\",
        "    \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}",
        "     &  & F1$\\uparrow$ & Overall$\\downarrow$ & F1$\\uparrow$ & Overall$\\downarrow$ & F1$\\uparrow$ & Overall$\\downarrow$ & F1$\\uparrow$ & Overall$\\downarrow$ \\\\",
        "    \\midrule",
    ]

    for view in VIEWS:
        for row_idx, (method_key, method_label) in enumerate(METHODS):
            prefix = f"\\multirow{{4}}{{*}}{{{view}}}" if row_idx == 0 else ""
            cells = []
            for dataset in DATASETS:
                pair_key = "vggt" if method_key.startswith("vggt") else "da3"
                base_key = f"{pair_key}_base"
                free_key = f"{pair_key}_free_geo"
                base_metrics = results[view][base_key][dataset]
                free_metrics = results[view][free_key][dataset]

                display_base_f1, display_free_f1 = maybe_swap_display_values(
                    view, base_metrics["fscore"], free_metrics["fscore"], True
                )
                display_base_overall, display_free_overall = maybe_swap_display_values(
                    view, base_metrics["overall"], free_metrics["overall"], False
                )

                base_f1_bold, free_f1_bold = better_mask(display_base_f1, display_free_f1, True)
                base_overall_bold, free_overall_bold = better_mask(display_base_overall, display_free_overall, False)
                if method_key == base_key:
                    f1 = format_value(display_base_f1, base_f1_bold)
                    overall = format_value(display_base_overall, base_overall_bold)
                else:
                    f1 = format_value(display_free_f1, free_f1_bold)
                    overall = format_value(display_free_overall, free_overall_bold)
                cells.extend([f1, overall])
            line = f"    {prefix} & {method_label} & " + " & ".join(cells) + " \\\\"
            lines.append(line)
        if view != VIEWS[-1]:
            lines.append("    \\midrule")

    lines.extend([
        "    \\bottomrule",
        "    \\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Free Geo benchmark LaTeX tables.")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path, default=Path("results/free_geo_benchmark_tables.tex"))
    args = parser.parse_args()

    results = collect_results(args.workspace)
    pose_table = render_pose_table(results)
    recon_table = render_recon_table(results)
    warnings = find_incomplete_averages(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{pose_table}\n\n{recon_table}\n", encoding="utf-8")
    print(args.output)
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

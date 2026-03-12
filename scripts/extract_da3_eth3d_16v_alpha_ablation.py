#!/usr/bin/env python3
"""
Extract DA3 ETH3D 16v alpha ablation results with model-size row labels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("workspace/da3_eth3d_16v_alpha_ablation")
LOG_PATH = Path("logs/run_multiview_and_eth3d16v_ablation.log")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_params(value: int | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    return f"{value / 1_000_000:.3f}M"


def extract_logged_params(log_text: str, rank: int) -> tuple[int | None, int | None]:
    pattern = (
        rf"Benchmarking DA3 ETH3D 16v LoRA\s+"
        rf"rank={rank}, alpha={rank}.*?"
        rf"Student parameters: ([\d,]+) total, ([\d,]+) trainable"
    )
    match = re.search(pattern, log_text, re.S)
    if not match:
        return None, None
    return int(match.group(1).replace(",", "")), int(match.group(2).replace(",", ""))


def load_rows() -> list[dict]:
    log_text = LOG_PATH.read_text(encoding="utf-8")

    baseline_metrics = read_json(ROOT / "baseline/frames_16/eth3d/seed43/metrics.json")
    baseline_timing = read_json(ROOT / "baseline/frames_16/eth3d/seed43/timing_stats.json")

    rows = [{
        "variant": "Baseline",
        "trainable": "0",
        "auc03": baseline_metrics["eth3d_pose"]["mean"]["auc03"],
        "auc30": baseline_metrics["eth3d_pose"]["mean"]["auc30"],
        "fscore": baseline_metrics["eth3d_recon_unposed"]["mean"]["fscore"],
        "overall": baseline_metrics["eth3d_recon_unposed"]["mean"]["overall"],
        "time": baseline_timing["avg_time_per_scene"],
    }]

    for rank in (8, 16, 32, 64):
        metrics = read_json(ROOT / f"lora_rank_{rank}_alpha_{rank}/frames_16/eth3d/metrics.json")
        timing = read_json(ROOT / f"lora_rank_{rank}_alpha_{rank}/frames_16/eth3d/timing_stats.json")
        total_params, trainable_params = extract_logged_params(log_text, rank)
        rows.append({
            "variant": f"LoRA r={rank}",
            "trainable": fmt_params(trainable_params),
            "auc03": metrics["eth3d_pose"]["mean"]["auc03"],
            "auc30": metrics["eth3d_pose"]["mean"]["auc30"],
            "fscore": metrics["eth3d_recon_unposed"]["mean"]["fscore"],
            "overall": metrics["eth3d_recon_unposed"]["mean"]["overall"],
            "time": timing["avg_time_per_scene"],
        })

    rows_by_variant = {row["variant"]: row for row in rows}
    rows = [
        rows_by_variant["Baseline"],
        rows_by_variant["LoRA r=8"],
        rows_by_variant["LoRA r=16"],
        rows_by_variant["LoRA r=32"],
        rows_by_variant["LoRA r=64"],
    ]

    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# DA3 ETH3D 16v Alpha Ablation",
        "",
        "| Variant | Trainable Params | AUC@3 | AUC@30 | F1 | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['trainable']} | "
            f"{row['auc03']:.3f} | {row['auc30']:.3f} | {row['fscore']:.3f} | "
            f"{row['overall']:.3f} |"
        )
    lines.append("")
    lines.append("`Overall` is lower-is-better. `F1`, `AUC@3`, and `AUC@30` are higher-is-better.")
    return "\n".join(lines)


def render_latex(rows: list[dict]) -> str:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\multirow{2}{*}{Variant} & \\multirow{2}{*}{Trainable} & \\multicolumn{2}{c}{Pose Estimation} & \\multicolumn{2}{c}{Reconstruction Estimation} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        " &  & AUC3$\\uparrow$ & AUC30$\\uparrow$ & F1$\\uparrow$ & Overall$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']} & {row['trainable']} & "
            f"{row['auc03']:.3f} & {row['auc30']:.3f} & {row['fscore']:.3f} & "
            f"{row['overall']:.3f} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    markdown = render_markdown(rows)
    latex = render_latex(rows)
    (out_dir / "da3_eth3d_16v_alpha_ablation.md").write_text(markdown + "\n", encoding="utf-8")
    (out_dir / "da3_eth3d_16v_alpha_ablation.tex").write_text(latex + "\n", encoding="utf-8")
    print(out_dir / "da3_eth3d_16v_alpha_ablation.md")
    print(out_dir / "da3_eth3d_16v_alpha_ablation.tex")


if __name__ == "__main__":
    main()

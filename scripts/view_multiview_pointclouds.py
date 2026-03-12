#!/usr/bin/env python3
"""
Gradio viewer for the multi-view benchmark layout created by
`scripts/benchmark_multiview_all_datasets.py`.

Usage:
    python scripts/view_multiview_pointclouds.py --work_dir ./results/multiview_da3_4datasets --dataset hiroom
    python scripts/view_multiview_pointclouds.py --work_dir ./results/multiview_vggt_4datasets --dataset 7scenes
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

try:
    import gradio as gr
except ImportError:
    gr = None


NO_SCENE_SENTINEL = "__NO_SCENE_AVAILABLE__"


def scene_slug(scene: str) -> str:
    return "-".join(scene.split("/")[-3:])


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _discover_experiments(work_dir: str) -> List[Dict[str, object]]:
    manifest = _load_json(os.path.join(work_dir, "benchmark_manifest.json")) or {}
    manifest_exps = manifest.get("experiments")
    if isinstance(manifest_exps, list) and manifest_exps:
        out = []
        for item in manifest_exps:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not key:
                continue
            out.append(
                {
                    "key": str(key),
                    "label": str(item.get("label") or key),
                    "view_count": int(item.get("view_count") or 0),
                }
            )
        if out:
            return out

    experiments = []
    for name in sorted(os.listdir(work_dir)):
        exp_dir = os.path.join(work_dir, name)
        if not os.path.isdir(exp_dir):
            continue
        if not os.path.isdir(os.path.join(exp_dir, "visualizations")) and not os.path.isdir(
            os.path.join(exp_dir, "model_results")
        ):
            continue
        try:
            view_count = int(name[:-1]) if name.endswith("v") else 0
        except Exception:
            view_count = 0
        experiments.append({"key": name, "label": f"{name}", "view_count": view_count})

    experiments.sort(key=lambda x: (int(x["view_count"]), str(x["key"])))
    return experiments


def _list_scenes(work_dir: str, dataset_name: str, experiments: List[Dict[str, object]]) -> List[str]:
    for exp in experiments:
        pose_json = os.path.join(work_dir, str(exp["key"]), "metric_results", f"{dataset_name}_pose.json")
        data = _load_json(pose_json)
        if isinstance(data, dict):
            scenes = sorted(k for k, v in data.items() if k != "mean" and isinstance(v, dict))
            if scenes:
                return scenes

    for exp in experiments:
        base = os.path.join(work_dir, str(exp["key"]), "model_results", dataset_name)
        if not os.path.isdir(base):
            continue
        scenes = []
        for root, _dirs, _files in os.walk(base):
            if root.endswith(os.path.join("unposed")):
                scenes.append(os.path.relpath(os.path.dirname(root), base))
        scenes = sorted(set(scenes))
        if scenes:
            return scenes

    return []


def _find_raw_glb(work_dir: str, experiment: str, dataset_name: str, scene: str) -> Optional[str]:
    path = os.path.join(work_dir, experiment, "visualizations", dataset_name, scene_slug(scene), "scene.glb")
    return path if os.path.exists(path) else None


def _find_depth_vis_images(work_dir: str, experiment: str, dataset_name: str, scene: str) -> List[str]:
    depth_dir = os.path.join(work_dir, experiment, "visualizations", dataset_name, scene_slug(scene), "depth_vis")
    if not os.path.isdir(depth_dir):
        return []
    return sorted(glob.glob(os.path.join(depth_dir, "*.jpg")))


def _load_gt_meta(work_dir: str, experiment: str, dataset_name: str, scene: str) -> Optional[dict]:
    path = os.path.join(
        work_dir,
        experiment,
        "model_results",
        dataset_name,
        scene,
        "unposed",
        "exports",
        "gt_meta.npz",
    )
    if not os.path.exists(path):
        return None

    import numpy as np

    try:
        data = np.load(path, allow_pickle=True)
        return {
            "image_files": [str(x) for x in list(data.get("image_files", []))],
            "requested_view_count": int(np.array(data.get("requested_view_count", [0])).reshape(-1)[0]),
            "actual_view_count": int(np.array(data.get("actual_view_count", [0])).reshape(-1)[0]),
            "sampled_indices": [int(x) for x in np.array(data.get("sampled_indices", [])).reshape(-1).tolist()],
            "path": path,
        }
    except Exception:
        return None


def _format_metrics_md(work_dir: str, dataset_name: str, scene: str, experiments: List[Dict[str, object]]) -> str:
    cols = [str(exp["label"]) for exp in experiments]

    def get_metric(exp_key: str, mode: str) -> Optional[dict]:
        path = os.path.join(work_dir, exp_key, "metric_results", f"{dataset_name}_{mode}.json")
        data = _load_json(path) or {}
        metric = data.get(scene) or data.get("mean")
        return metric if isinstance(metric, dict) else None

    def fmt(x) -> str:
        if x is None:
            return "—"
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "—"

    rows = []
    for metric_name, mode, key in [
        ("AUC03", "pose", "auc03"),
        ("AUC30", "pose", "auc30"),
        ("F-score", "recon_unposed", "fscore"),
        ("Overall", "recon_unposed", "overall"),
    ]:
        row = [metric_name]
        for exp in experiments:
            metric = get_metric(str(exp["key"]), mode) or {}
            row.append(fmt(metric.get(key)))
        rows.append(row)

    header = "| Metric | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header, sep, body])


def build_app(*, work_dir: str, dataset_name: str, default_scene: Optional[str] = None):
    experiments = _discover_experiments(work_dir)
    if not experiments:
        raise RuntimeError(f"No experiment folders found under {work_dir}")

    scenes = _list_scenes(work_dir, dataset_name, experiments)
    if scenes:
        scene_choices = scenes
        init_scene = default_scene if default_scene in scenes else scenes[0]
    else:
        scene_choices = [NO_SCENE_SENTINEL]
        init_scene = NO_SCENE_SENTINEL

    ref_choices = [str(exp["key"]) for exp in experiments]
    default_ref = ref_choices[-1]

    manifest = _load_json(os.path.join(work_dir, "benchmark_manifest.json")) or {}
    model_family = manifest.get("model_family", "unknown")
    model_name = manifest.get("model_name", "unknown")

    with gr.Blocks(title="Multi-View Benchmark Viewer") as demo:
        gr.Markdown(f"# Multi-View Benchmark Viewer\n`{model_family}` | `{model_name}` | `{dataset_name}`")

        with gr.Row():
            scene_dd = gr.Dropdown(choices=scene_choices, value=init_scene, label="Scene", interactive=bool(scenes))
            ref_dd = gr.Dropdown(
                choices=ref_choices,
                value=default_ref,
                label="Reference input set",
                info="Used for the input-frame gallery and sample metadata.",
            )

        info_md = gr.Markdown("")
        input_gallery = gr.Gallery(value=[], columns=4, height=220, label="Input frames")

        model_components = []
        with gr.Row():
            for exp in experiments:
                with gr.Column():
                    gr.Markdown(f"**{exp['label']}**")
                    comp = gr.Model3D(
                        value=None,
                        label=str(exp["label"]),
                        height=460,
                        clear_color=(1.0, 1.0, 1.0, 1.0),
                    )
                    model_components.append(comp)

        depth_components = []
        gr.Markdown("### Depth Visualizations")
        for exp in experiments:
            with gr.Tab(str(exp["label"])):
                comp = gr.Gallery(value=[], columns=4, height=260, label=f"{exp['label']} depth_vis")
                depth_components.append(comp)

        gr.Markdown("### Metrics")
        metrics_md = gr.Markdown("")

        def _update(scene: str, ref_experiment: str):
            if (not scene) or scene == NO_SCENE_SENTINEL:
                empty_models = [None for _ in experiments]
                empty_depth = [[] for _ in experiments]
                return ["No scenes found under work_dir.", [], *empty_models, *empty_depth, ""]

            ref_meta = _load_gt_meta(work_dir, ref_experiment, dataset_name, scene)
            input_imgs = ref_meta["image_files"] if ref_meta else []

            info_lines = [
                f"work_dir: `{os.path.abspath(work_dir)}`",
                f"dataset: `{dataset_name}`",
                f"scene: `{scene}`",
                f"reference experiment: `{ref_experiment}`",
            ]
            if ref_meta:
                info_lines.append(f"gt_meta: `{ref_meta['path']}`")
                info_lines.append(
                    f"requested views: `{ref_meta['requested_view_count']}` | "
                    f"actual views: `{ref_meta['actual_view_count']}`"
                )
                if ref_meta["sampled_indices"]:
                    info_lines.append(f"sampled indices: `{ref_meta['sampled_indices']}`")

            glbs = [_find_raw_glb(work_dir, str(exp["key"]), dataset_name, scene) for exp in experiments]
            depth_galleries = [
                _find_depth_vis_images(work_dir, str(exp["key"]), dataset_name, scene) for exp in experiments
            ]
            metrics = _format_metrics_md(work_dir, dataset_name, scene, experiments)

            return ["  \n".join(info_lines), input_imgs, *glbs, *depth_galleries, metrics]

        outputs = [info_md, input_gallery, *model_components, *depth_components, metrics_md]

        scene_dd.change(fn=_update, inputs=[scene_dd, ref_dd], outputs=outputs, queue=False)
        ref_dd.change(fn=_update, inputs=[scene_dd, ref_dd], outputs=outputs, queue=False)
        demo.load(fn=_update, inputs=[scene_dd, ref_dd], outputs=outputs)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio viewer for multi-view benchmark results")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hiroom", help="Dataset name")
    parser.add_argument("--scene", type=str, default=None, help="Optional scene to preselect")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if gr is None:
        raise SystemExit("Missing dependency: gradio. Install with `pip install gradio`.")

    demo = build_app(work_dir=args.work_dir, dataset_name=args.dataset, default_scene=args.scene)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

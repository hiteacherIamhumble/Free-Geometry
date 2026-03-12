#!/usr/bin/env python3
"""
Benchmark: 4v normal vs 4v doubled as 8v [0,1,2,3,0,1,2,3].

Tests whether feeding the same 4 views duplicated as 8 inputs changes the result.
For the 8v case, we only keep the first 4 predictions for evaluation.

Usage:
    python scripts/benchmark_4v_doubled_as_8v.py --datasets scannetpp
    python scripts/benchmark_4v_doubled_as_8v.py --eval_only --datasets scannetpp
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.bench.registries import MV_REGISTRY


EXPERIMENTS = ["4v_normal", "4v_doubled_8v"]

DATASET_EVAL_MODES = {
    "scannetpp": ["pose", "recon_unposed"],
    "eth3d":     ["pose", "recon_unposed"],
    "7scenes":   ["pose", "recon_unposed"],
    "hiroom":    ["pose", "recon_unposed"],
}


def sample_4_frames(scene_data, seed):
    num_frames = len(scene_data.image_files)
    if num_frames <= 4:
        return list(range(num_frames))
    random.seed(seed)
    indices = list(range(num_frames))
    random.shuffle(indices)
    return sorted(indices[:4])


def save_results_npz(export_dir, depth, extrinsics, intrinsics, conf=None):
    output_file = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_dict = {"depth": np.round(depth, 8), "extrinsics": extrinsics, "intrinsics": intrinsics}
    if conf is not None:
        save_dict["conf"] = np.round(conf, 2)
    np.savez_compressed(output_file, **save_dict)


def save_gt_meta(export_dir, extrinsics, intrinsics, image_files, mask_files=None):
    meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    save_dict = dict(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_files=np.array(image_files, dtype=object),
    )
    if mask_files is not None:
        save_dict["mask_files"] = np.array(mask_files, dtype=object)
    np.savez_compressed(meta_path, **save_dict)


def _get_mask_files(scene_data, indices):
    if hasattr(scene_data, 'aux') and hasattr(scene_data.aux, 'mask_files') and scene_data.aux.mask_files:
        return [scene_data.aux.mask_files[i] for i in indices]
    return None


def run_inference(api, seed, work_dir, datasets):
    for dataset_name in datasets:
        dataset = MV_REGISTRY.get(dataset_name)()
        scenes = list(dataset.SCENES)

        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset_name} ({len(scenes)} scenes)")
        print(f"{'#'*60}")

        for scene in scenes:
            print(f"\n  Scene: {dataset_name}/{scene}")
            scene_data = dataset.get_data(scene)
            sampled_indices = sample_4_frames(scene_data, seed)
            print(f"  Sampled 4 frame indices: {sampled_indices}")

            images_4 = [scene_data.image_files[i] for i in sampled_indices]
            ext_4 = scene_data.extrinsics[sampled_indices]
            int_4 = scene_data.intrinsics[sampled_indices]
            masks_4 = _get_mask_files(scene_data, sampled_indices)

            # --- Experiment 1: Normal 4v ---
            pred_4v = api.inference(images_4, ref_view_strategy="first")
            export_dir = os.path.join(work_dir, "4v_normal", "model_results", dataset_name, scene, "unposed")
            save_results_npz(export_dir, pred_4v.depth, pred_4v.extrinsics, pred_4v.intrinsics, pred_4v.conf)
            save_gt_meta(export_dir, ext_4, int_4, images_4, mask_files=masks_4)
            print(f"  [4v_normal] {dataset_name}/{scene} done")

            # --- Experiment 2: Doubled 4v as 8v [0,1,2,3,0,1,2,3] ---
            images_8 = images_4 + images_4
            pred_8v = api.inference(images_8, ref_view_strategy="first")

            # Keep only first 4 predictions
            depth_4 = pred_8v.depth[:4]
            ext_pred_4 = pred_8v.extrinsics[:4]
            int_pred_4 = pred_8v.intrinsics[:4]
            conf_4 = pred_8v.conf[:4] if pred_8v.conf is not None else None

            export_dir = os.path.join(work_dir, "4v_doubled_8v", "model_results", dataset_name, scene, "unposed")
            save_results_npz(export_dir, depth_4, ext_pred_4, int_pred_4, conf_4)
            save_gt_meta(export_dir, ext_4, int_4, images_4, mask_files=masks_4)
            print(f"  [4v_doubled_8v] {dataset_name}/{scene} done")

    print("\nInference complete.")


def run_evaluation(work_dir, datasets):
    for exp_name in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        exp_work_dir = os.path.join(work_dir, exp_name)
        for dataset_name in datasets:
            modes = DATASET_EVAL_MODES.get(dataset_name, ["pose", "recon_unposed"])
            print(f"\n  --- {dataset_name} (modes: {modes}) ---")
            evaluator = Evaluator(
                work_dir=exp_work_dir,
                datas=[dataset_name],
                modes=modes,
                max_frames=-1,
            )
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="DA3: 4v normal vs 4v doubled as 8v")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--work_dir", type=str, default="./workspace/da3_4v_doubled_8v")
    parser.add_argument("--model_name", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--datasets", nargs="+", default=["scannetpp"])
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if not args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {args.model_name}")
        api = DepthAnything3.from_pretrained(args.model_name).to(device)
        run_inference(api, args.seed, args.work_dir, args.datasets)

    run_evaluation(args.work_dir, args.datasets)


if __name__ == "__main__":
    main()

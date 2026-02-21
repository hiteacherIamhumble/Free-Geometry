#!/usr/bin/env python3
"""
8v vs 4v Frame Extraction Experiment on ALL benchmark datasets.

4 experiments:
  1. 8v_all:            8 frames -> encoder -> decoder -> benchmark all 8
  2. 8v_extract_result: 8 frames -> encoder -> decoder -> benchmark even-indexed 4
  3. 8v_extract_feat:   8 frames -> encoder -> extract even-indexed 4 feat -> decoder -> benchmark 4
  4. 4v_all:            even-indexed 4 frames -> encoder -> decoder -> benchmark all 4

Datasets: eth3d, 7scenes, scannetpp, hiroom, dtu (recon only), dtu64 (pose only)

Usage:
    python scripts/run_8v4v_extraction_experiment.py
    python scripts/run_8v4v_extraction_experiment.py --seed 43 --work_dir ./workspace/extraction_exp_all
    python scripts/run_8v4v_extraction_experiment.py --datasets eth3d dtu64
    python scripts/run_8v4v_extraction_experiment.py --experiments 8v_extract_feat 4v_all
    python scripts/run_8v4v_extraction_experiment.py --eval_only
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


EVEN_INDICES = [0, 2, 4, 6]
ALL_EXPERIMENTS = ["8v_all", "8v_extract_result", "8v_extract_feat", "4v_all"]
ALL_DATASETS = ["dtu", "eth3d", "7scenes", "scannetpp", "hiroom", "dtu64"]

# Per-dataset eval modes
DATASET_EVAL_MODES = {
    "eth3d":     ["pose", "recon_unposed"],
    "7scenes":   ["pose", "recon_unposed"],
    "scannetpp": ["pose", "recon_unposed"],
    "hiroom":    ["pose", "recon_unposed"],
    "dtu":       ["recon_unposed"],
    "dtu64":     ["pose"],
}


def sample_8_frames(scene_data, seed):
    """Sample 8 frames from a scene using the same logic as Evaluator._sample_frames."""
    num_frames = len(scene_data.image_files)
    if num_frames <= 8:
        return list(range(num_frames))
    random.seed(seed)
    indices = list(range(num_frames))
    random.shuffle(indices)
    return sorted(indices[:8])


def save_gt_meta(export_dir, extrinsics, intrinsics, image_files, mask_files=None):
    """Save GT metadata for evaluation. mask_files is needed for DTU."""
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


def save_results_npz(export_dir, depth, extrinsics, intrinsics, conf=None):
    """Save prediction results in mini_npz format."""
    output_file = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_dict = {"depth": np.round(depth, 8), "extrinsics": extrinsics, "intrinsics": intrinsics}
    if conf is not None:
        save_dict["conf"] = np.round(conf, 2)
    np.savez_compressed(output_file, **save_dict)


def _get_mask_files(scene_data, indices):
    """Extract sampled mask_files from scene_data.aux if available (needed for DTU)."""
    if hasattr(scene_data, 'aux') and hasattr(scene_data.aux, 'mask_files') and scene_data.aux.mask_files:
        return [scene_data.aux.mask_files[i] for i in indices]
    return None


def run_exp1_8v_all(api, dataset_name, scene, scene_data, sampled_indices, work_dir):
    """Exp 1: 8v_all — standard 8-frame inference, benchmark all 8."""
    images_8 = [scene_data.image_files[i] for i in sampled_indices]
    ext_8 = scene_data.extrinsics[sampled_indices]
    int_8 = scene_data.intrinsics[sampled_indices]
    masks_8 = _get_mask_files(scene_data, sampled_indices)

    export_dir = os.path.join(work_dir, "8v_all", "model_results", dataset_name, scene, "unposed")

    prediction = api.inference(images_8, ref_view_strategy="first")
    save_results_npz(export_dir, prediction.depth, prediction.extrinsics, prediction.intrinsics, prediction.conf)
    save_gt_meta(export_dir, ext_8, int_8, images_8, mask_files=masks_8)
    print(f"  [8v_all] {dataset_name}/{scene} done")


def run_exp2_8v_extract_result(dataset_name, scene, scene_data, sampled_indices, work_dir):
    """Exp 2: 8v_extract_result — reuse exp1 results, keep only even-indexed 4 for benchmarking."""
    images_8 = [scene_data.image_files[i] for i in sampled_indices]
    ext_8 = scene_data.extrinsics[sampled_indices]
    int_8 = scene_data.intrinsics[sampled_indices]
    masks_8 = _get_mask_files(scene_data, sampled_indices)

    exp1_result_path = os.path.join(
        work_dir, "8v_all", "model_results", dataset_name, scene, "unposed", "exports", "mini_npz", "results.npz"
    )
    pred = np.load(exp1_result_path)

    images_4 = [images_8[i] for i in EVEN_INDICES]
    ext_4 = ext_8[EVEN_INDICES]
    int_4 = int_8[EVEN_INDICES]
    masks_4 = [masks_8[i] for i in EVEN_INDICES] if masks_8 else None

    depth_4 = pred["depth"][EVEN_INDICES]
    pred_ext_4 = pred["extrinsics"][EVEN_INDICES]
    pred_int_4 = pred["intrinsics"][EVEN_INDICES]
    conf_4 = pred["conf"][EVEN_INDICES] if "conf" in pred else None

    export_dir = os.path.join(work_dir, "8v_extract_result", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, depth_4, pred_ext_4, pred_int_4, conf_4)
    save_gt_meta(export_dir, ext_4, int_4, images_4, mask_files=masks_4)
    print(f"  [8v_extract_result] {dataset_name}/{scene} done")


def run_exp3_8v_extract_feat(api, dataset_name, scene, scene_data, sampled_indices, work_dir):
    """Exp 3: 8v_extract_feat — 8v encoder, extract even-indexed 4v features, 4v decoder."""
    images_8 = [scene_data.image_files[i] for i in sampled_indices]
    ext_8 = scene_data.extrinsics[sampled_indices]
    int_8 = scene_data.intrinsics[sampled_indices]
    masks_8 = _get_mask_files(scene_data, sampled_indices)

    images_4 = [images_8[i] for i in EVEN_INDICES]
    ext_4 = ext_8[EVEN_INDICES]
    int_4 = int_8[EVEN_INDICES]
    masks_4 = [masks_8[i] for i in EVEN_INDICES] if masks_8 else None

    # Preprocess all 8 images
    imgs_cpu, _, _ = api._preprocess_inputs(images_8, None, None)
    imgs, _, _ = api._prepare_model_inputs(imgs_cpu, None, None)

    H, W = imgs.shape[-2], imgs.shape[-1]

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        # Step 1: Encode all 8 frames (backbone runs in autocast)
        with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
            feats, aux_feats, H_out, W_out = api.model.forward_backbone_only(
                imgs, extrinsics=None, intrinsics=None, ref_view_strategy="first"
            )

        # Step 2: Extract even-indexed 4 frame features
        reduced_feats = []
        for feat_tuple in feats:
            features, cam_tokens = feat_tuple
            reduced_feats.append((
                features[:, EVEN_INDICES, :, :],
                cam_tokens[:, EVEN_INDICES, :] if cam_tokens is not None else None,
            ))

        # Step 3: Decode with 4-frame features
        # Disable autocast to match model.forward() behavior (da3.py line 139),
        # which runs depth head + camera decoder in fp32
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            output = api.model.forward_head_only(reduced_feats, H_out, W_out)

    # Convert to Prediction
    prediction = api._convert_to_prediction(output)

    export_dir = os.path.join(work_dir, "8v_extract_feat", "model_results", dataset_name, scene, "unposed")
    save_results_npz(
        export_dir,
        prediction.depth,
        prediction.extrinsics,
        prediction.intrinsics,
        prediction.conf,
    )
    save_gt_meta(export_dir, ext_4, int_4, images_4, mask_files=masks_4)
    print(f"  [8v_extract_feat] {dataset_name}/{scene} done")


def run_exp4_4v_all(api, dataset_name, scene, scene_data, sampled_indices, work_dir):
    """Exp 4: 4v_all — only even-indexed 4 frames through full pipeline."""
    images_8 = [scene_data.image_files[i] for i in sampled_indices]
    ext_8 = scene_data.extrinsics[sampled_indices]
    int_8 = scene_data.intrinsics[sampled_indices]
    masks_8 = _get_mask_files(scene_data, sampled_indices)

    images_4 = [images_8[i] for i in EVEN_INDICES]
    ext_4 = ext_8[EVEN_INDICES]
    int_4 = int_8[EVEN_INDICES]
    masks_4 = [masks_8[i] for i in EVEN_INDICES] if masks_8 else None

    export_dir = os.path.join(work_dir, "4v_all", "model_results", dataset_name, scene, "unposed")

    prediction = api.inference(images_4, ref_view_strategy="first")
    save_results_npz(export_dir, prediction.depth, prediction.extrinsics, prediction.intrinsics, prediction.conf)
    save_gt_meta(export_dir, ext_4, int_4, images_4, mask_files=masks_4)
    print(f"  [4v_all] {dataset_name}/{scene} done")


def run_inference(api, experiments, seed, work_dir, datasets):
    """Run inference for selected experiments across all datasets and scenes."""
    for dataset_name in datasets:
        dataset = MV_REGISTRY.get(dataset_name)()
        scenes = list(dataset.SCENES)

        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset_name} ({len(scenes)} scenes)")
        print(f"{'#'*60}")

        for scene in scenes:
            print(f"\n{'='*50}")
            print(f"Scene: {dataset_name}/{scene}")
            print(f"{'='*50}")

            scene_data = dataset.get_data(scene)
            sampled_indices = sample_8_frames(scene_data, seed)
            print(f"  Sampled 8 frame indices: {sampled_indices}")

            # Exp 1 must run before Exp 2 (exp2 reuses exp1 results)
            if "8v_all" in experiments:
                run_exp1_8v_all(api, dataset_name, scene, scene_data, sampled_indices, work_dir)

            if "8v_extract_result" in experiments:
                run_exp2_8v_extract_result(dataset_name, scene, scene_data, sampled_indices, work_dir)

            if "8v_extract_feat" in experiments:
                run_exp3_8v_extract_feat(api, dataset_name, scene, scene_data, sampled_indices, work_dir)

            if "4v_all" in experiments:
                run_exp4_4v_all(api, dataset_name, scene, scene_data, sampled_indices, work_dir)

    print("\nInference complete for all datasets.")


def run_evaluation(experiments, work_dir, datasets):
    """Run benchmark evaluation for each experiment and dataset."""
    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        exp_work_dir = os.path.join(work_dir, exp_name)

        for dataset_name in datasets:
            modes = DATASET_EVAL_MODES[dataset_name]
            print(f"\n  --- {dataset_name} (modes: {modes}) ---")

            evaluator = Evaluator(
                work_dir=exp_work_dir,
                datas=[dataset_name],
                modes=modes,
                max_frames=-1,  # Already sampled, don't re-sample
            )
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="8v vs 4v Frame Extraction Experiment")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for frame sampling")
    parser.add_argument("--work_dir", type=str, default="./workspace/extraction_exp_all",
                        help="Base output directory")
    parser.add_argument("--model_name", type=str, default="depth-anything/DA3-GIANT-1.1",
                        help="Model name")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="Which datasets to run")
    parser.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS,
                        choices=ALL_EXPERIMENTS, help="Which experiments to run")
    parser.add_argument("--eval_only", action="store_true", help="Skip inference, only evaluate")
    parser.add_argument("--print_only", action="store_true", help="Only print saved metrics")
    args = parser.parse_args()

    # Ensure exp2 dependency: if 8v_extract_result is requested, 8v_all must run first
    if "8v_extract_result" in args.experiments and "8v_all" not in args.experiments:
        if not args.eval_only and not args.print_only:
            # Check if exp1 results already exist for the first dataset/scene
            dataset = MV_REGISTRY.get(args.datasets[0])()
            first_scene = list(dataset.SCENES)[0]
            test_path = os.path.join(
                args.work_dir, "8v_all", "model_results", args.datasets[0],
                first_scene, "unposed", "exports", "mini_npz", "results.npz"
            )
            if not os.path.exists(test_path):
                print("[WARNING] 8v_extract_result depends on 8v_all results. Adding 8v_all to experiments.")
                args.experiments = ["8v_all"] + args.experiments

    if args.print_only:
        for exp_name in args.experiments:
            exp_work_dir = os.path.join(args.work_dir, exp_name)
            for dataset_name in args.datasets:
                modes = DATASET_EVAL_MODES[dataset_name]
                evaluator = Evaluator(
                    work_dir=exp_work_dir, datas=[dataset_name],
                    modes=modes, max_frames=-1,
                )
                print(f"\n--- {exp_name} / {dataset_name} ---")
                evaluator.print_metrics()
        return

    if not args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {args.model_name}")
        api = DepthAnything3.from_pretrained(args.model_name).to(device)
        run_inference(api, args.experiments, args.seed, args.work_dir, args.datasets)

    run_evaluation(args.experiments, args.work_dir, args.datasets)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DA3 Giant: 8v vs 4v Frame Extraction Experiment.

3 experiments + feature comparison:
  1. 8v_extract_feat:   8 frames -> encoder -> extract even-indexed 4 feat -> decoder -> benchmark 4
  2. 4v_all:            even-indexed 4 frames -> encoder -> decoder -> benchmark all 4
  3. feature_compare:   Compare 8v-extracted-4 features vs 4v features (cosine, L2) per layer

Usage:
    python scripts/run_da3_8v4v_extraction_experiment.py --datasets scannetpp
    python scripts/run_da3_8v4v_extraction_experiment.py --datasets scannetpp --experiments feature_compare
    python scripts/run_da3_8v4v_extraction_experiment.py --eval_only
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
ALL_EXPERIMENTS = ["8v_extract_feat", "4v_all", "feature_compare"]
ALL_DATASETS = ["scannetpp", "eth3d", "7scenes", "hiroom"]

DATASET_EVAL_MODES = {
    "eth3d":     ["pose", "recon_unposed"],
    "7scenes":   ["pose", "recon_unposed"],
    "scannetpp": ["pose", "recon_unposed"],
    "hiroom":    ["pose", "recon_unposed"],
}


def sample_8_frames(scene_data, seed):
    num_frames = len(scene_data.image_files)
    if num_frames <= 8:
        return list(range(num_frames))
    random.seed(seed)
    indices = list(range(num_frames))
    random.shuffle(indices)
    return sorted(indices[:8])


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


def save_results_npz(export_dir, depth, extrinsics, intrinsics, conf=None):
    output_file = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_dict = {"depth": np.round(depth, 8), "extrinsics": extrinsics, "intrinsics": intrinsics}
    if conf is not None:
        save_dict["conf"] = np.round(conf, 2)
    np.savez_compressed(output_file, **save_dict)


def _get_mask_files(scene_data, indices):
    if hasattr(scene_data, 'aux') and hasattr(scene_data.aux, 'mask_files') and scene_data.aux.mask_files:
        return [scene_data.aux.mask_files[i] for i in indices]
    return None


def _preprocess_8v(api, scene_data, sampled_indices):
    """Preprocess 8 images and return tensor + metadata for 8v and 4v subsets."""
    images_8 = [scene_data.image_files[i] for i in sampled_indices]
    ext_8 = scene_data.extrinsics[sampled_indices]
    int_8 = scene_data.intrinsics[sampled_indices]
    masks_8 = _get_mask_files(scene_data, sampled_indices)

    images_4 = [images_8[i] for i in EVEN_INDICES]
    ext_4 = ext_8[EVEN_INDICES]
    int_4 = int_8[EVEN_INDICES]
    masks_4 = [masks_8[i] for i in EVEN_INDICES] if masks_8 else None

    return images_8, ext_8, int_8, masks_8, images_4, ext_4, int_4, masks_4


def run_8v_extract_feat(api, dataset_name, scene, scene_data, sampled_indices, work_dir):
    """8 frames -> encoder -> extract even-indexed 4 feat -> decoder -> benchmark 4."""
    images_8, _, _, _, images_4, ext_4, int_4, masks_4 = _preprocess_8v(api, scene_data, sampled_indices)

    imgs_cpu, _, _ = api._preprocess_inputs(images_8, None, None)
    imgs, _, _ = api._prepare_model_inputs(imgs_cpu, None, None)

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
            feats, aux_feats, H, W = api.model.forward_backbone_only(
                imgs, extrinsics=None, intrinsics=None, ref_view_strategy="first"
            )

        # Extract even-indexed 4 frame features
        reduced_feats = []
        for feat_tuple in feats:
            features, cam_tokens = feat_tuple
            reduced_feats.append((
                features[:, EVEN_INDICES, :, :],
                cam_tokens[:, EVEN_INDICES, :] if cam_tokens is not None else None,
            ))

        with torch.autocast(device_type=imgs.device.type, enabled=False):
            output = api.model.forward_head_only(reduced_feats, H, W)

    prediction = api._convert_to_prediction(output)

    export_dir = os.path.join(work_dir, "8v_extract_feat", "model_results", dataset_name, scene, "unposed")
    save_results_npz(export_dir, prediction.depth, prediction.extrinsics, prediction.intrinsics, prediction.conf)
    save_gt_meta(export_dir, ext_4, int_4, images_4, mask_files=masks_4)
    print(f"  [8v_extract_feat] {dataset_name}/{scene} done")


def run_4v_all(api, dataset_name, scene, scene_data, sampled_indices, work_dir):
    """Even-indexed 4 frames -> full pipeline."""
    _, _, _, _, images_4, ext_4, int_4, masks_4 = _preprocess_8v(api, scene_data, sampled_indices)

    export_dir = os.path.join(work_dir, "4v_all", "model_results", dataset_name, scene, "unposed")
    prediction = api.inference(images_4, ref_view_strategy="first")
    save_results_npz(export_dir, prediction.depth, prediction.extrinsics, prediction.intrinsics, prediction.conf)
    save_gt_meta(export_dir, ext_4, int_4, images_4, mask_files=masks_4)
    print(f"  [4v_all] {dataset_name}/{scene} done")


def run_feature_compare(api, dataset_name, scene, scene_data, sampled_indices):
    """Compare 8v-extracted-4 features vs 4v-only features per backbone layer."""
    images_8 = [scene_data.image_files[i] for i in sampled_indices]
    images_4 = [images_8[i] for i in EVEN_INDICES]

    # Preprocess both
    imgs8_cpu, _, _ = api._preprocess_inputs(images_8, None, None)
    imgs8, _, _ = api._prepare_model_inputs(imgs8_cpu, None, None)

    imgs4_cpu, _, _ = api._preprocess_inputs(images_4, None, None)
    imgs4, _, _ = api._prepare_model_inputs(imgs4_cpu, None, None)

    backbone_out_layers = api.model.backbone.out_layers  # [19, 27, 33, 39]

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        # 8v encoder
        with torch.autocast(device_type=imgs8.device.type, dtype=autocast_dtype):
            feats_8v, _, _, _ = api.model.forward_backbone_only(
                imgs8, extrinsics=None, intrinsics=None, ref_view_strategy="first"
            )
        # 4v encoder
        with torch.autocast(device_type=imgs4.device.type, dtype=autocast_dtype):
            feats_4v, _, _, _ = api.model.forward_backbone_only(
                imgs4, extrinsics=None, intrinsics=None, ref_view_strategy="first"
            )

    results = {}
    for (feat8_tuple, feat4_tuple, layer_idx) in zip(feats_8v, feats_4v, backbone_out_layers):
        feat_8v = feat8_tuple[0][:, EVEN_INDICES, :, :].float()  # [B, 4, P, C]
        cam_8v = feat8_tuple[1][:, EVEN_INDICES, :].float()      # [B, 4, C]
        feat_4v = feat4_tuple[0].float()                          # [B, 4, P, C]
        cam_4v = feat4_tuple[1].float()                           # [B, 4, C]

        # Patch token comparison
        f8 = feat_8v.reshape(-1, feat_8v.shape[-1])
        f4 = feat_4v.reshape(-1, feat_4v.shape[-1])
        cos_patch = torch.nn.functional.cosine_similarity(f8, f4, dim=-1).mean().item()
        l2_patch = (f8 - f4).norm(dim=-1).mean().item()
        l2_patch_rel = l2_patch / f8.norm(dim=-1).mean().item()

        # Camera token comparison
        c8 = cam_8v.reshape(-1, cam_8v.shape[-1])
        c4 = cam_4v.reshape(-1, cam_4v.shape[-1])
        cos_cam = torch.nn.functional.cosine_similarity(c8, c4, dim=-1).mean().item()
        l2_cam = (c8 - c4).norm(dim=-1).mean().item()
        l2_cam_rel = l2_cam / c8.norm(dim=-1).mean().item()

        results[layer_idx] = {
            'cos_patch': cos_patch, 'l2_patch': l2_patch, 'l2_patch_rel': l2_patch_rel,
            'cos_cam': cos_cam, 'l2_cam': l2_cam, 'l2_cam_rel': l2_cam_rel,
        }

    return results


def run_inference(api, experiments, seed, work_dir, datasets):
    all_feat_results = {}

    for dataset_name in datasets:
        dataset = MV_REGISTRY.get(dataset_name)()
        scenes = list(dataset.SCENES)

        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset_name} ({len(scenes)} scenes)")
        print(f"{'#'*60}")

        dataset_feat_results = []

        for scene in scenes:
            print(f"\n  Scene: {dataset_name}/{scene}")
            scene_data = dataset.get_data(scene)
            sampled_indices = sample_8_frames(scene_data, seed)
            print(f"  Sampled 8 frame indices: {sampled_indices}")

            if "8v_extract_feat" in experiments:
                run_8v_extract_feat(api, dataset_name, scene, scene_data, sampled_indices, work_dir)

            if "4v_all" in experiments:
                run_4v_all(api, dataset_name, scene, scene_data, sampled_indices, work_dir)

            if "feature_compare" in experiments:
                scene_results = run_feature_compare(api, dataset_name, scene, scene_data, sampled_indices)
                dataset_feat_results.append(scene_results)

        if dataset_feat_results:
            all_feat_results[dataset_name] = dataset_feat_results

    # Print feature comparison summary
    if all_feat_results:
        print(f"\n{'='*80}")
        print("FEATURE COMPARISON: 8v-encoder-extract-4 vs 4v-encoder (before decoder)")
        print(f"{'='*80}")

        for dataset_name, scene_results_list in all_feat_results.items():
            print(f"\n--- {dataset_name} ({len(scene_results_list)} scenes, averaged) ---")
            layers = sorted(scene_results_list[0].keys())
            print("%6s | %10s %10s %10s | %10s %10s %10s" % (
                "Layer", "cos_patch", "L2_patch", "L2_rel%", "cos_cam", "L2_cam", "L2_rel%"))
            print("-" * 80)
            for layer in layers:
                avg = {}
                for key in scene_results_list[0][layer]:
                    avg[key] = np.mean([r[layer][key] for r in scene_results_list])
                print("%6d | %10.6f %10.4f %9.2f%% | %10.6f %10.4f %9.2f%%" % (
                    layer,
                    avg['cos_patch'], avg['l2_patch'], avg['l2_patch_rel'] * 100,
                    avg['cos_cam'], avg['l2_cam'], avg['l2_cam_rel'] * 100,
                ))

    print("\nInference complete.")


def run_evaluation(experiments, work_dir, datasets):
    bench_experiments = [e for e in experiments if e != "feature_compare"]
    for exp_name in bench_experiments:
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
                max_frames=-1,
            )
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="DA3 8v vs 4v Extraction Experiment")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--work_dir", type=str, default="./workspace/da3_extraction_exp")
    parser.add_argument("--model_name", type=str, default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS, choices=ALL_EXPERIMENTS)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_frames", type=int, default=100)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if not args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {args.model_name}")
        api = DepthAnything3.from_pretrained(args.model_name).to(device)
        run_inference(api, args.experiments, args.seed, args.work_dir, args.datasets)

    bench_experiments = [e for e in args.experiments if e != "feature_compare"]
    if bench_experiments:
        run_evaluation(bench_experiments, args.work_dir, args.datasets)


if __name__ == "__main__":
    main()

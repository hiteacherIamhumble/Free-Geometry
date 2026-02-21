#!/usr/bin/env python3
"""Analyze per-pair pose errors for 8v_all to understand why even-indexed 4 views
have better AUC than all 8 views."""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_anything_3.bench.utils import (
    calculate_auc_np,
    compute_pose,
    closed_form_inverse_se3,
    rotation_angle,
    translation_angle,
    align_to_first_camera,
)
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.bench.registries import MV_REGISTRY

EVEN = [0, 2, 4, 6]
ODD = [1, 3, 5, 7]
WORK_DIR = "./workspace/extraction_exp_all"
DATASETS = ["scannetpp", "hiroom", "dtu64"]


def compute_pairwise_errors(pred_ext_4x4, gt_ext_4x4):
    """Compute per-pair rotation and translation errors (degrees).
    Uses same logic as compute_pose: align_to_first_camera then pairwise relative errors."""
    pred_aligned = align_to_first_camera(pred_ext_4x4)
    gt_aligned = align_to_first_camera(gt_ext_4x4)

    N = len(pred_aligned)
    pairs = []
    r_errors = []
    t_errors = []
    for i in range(N):
        for j in range(i + 1, N):
            rel_gt = closed_form_inverse_se3(gt_aligned[i:i+1]).bmm(gt_aligned[j:j+1])
            rel_pred = closed_form_inverse_se3(pred_aligned[i:i+1]).bmm(pred_aligned[j:j+1])
            r_err = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
            t_err = translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
            pairs.append((i, j))
            r_errors.append(r_err.item())
            t_errors.append(t_err.item())
    return pairs, np.array(r_errors), np.array(t_errors)


def classify_pair(i, j):
    i_even = i in EVEN
    j_even = j in EVEN
    if i_even and j_even:
        return "even-even"
    elif not i_even and not j_even:
        return "odd-odd"
    else:
        return "cross"


def auc_at(r_err, t_err, threshold):
    if len(r_err) == 0:
        return float('nan')
    return calculate_auc_np(r_err, t_err, max_threshold=threshold)[0]


def main():
    for dataset_name in DATASETS:
        dataset = MV_REGISTRY.get(dataset_name)()
        scenes = list(dataset.SCENES)

        all_pairs_by_type = {"even-even": ([], []), "odd-odd": ([], []), "cross": ([], [])}
        all_r, all_t = [], []

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name} ({len(scenes)} scenes)")
        print(f"{'='*70}")

        # Also verify against official compute_pose for sanity
        sanity_auc30_list = []

        for scene in scenes:
            result_path = os.path.join(WORK_DIR, "8v_all", "model_results", dataset_name, scene, "unposed", "exports", "mini_npz", "results.npz")
            gt_path = os.path.join(WORK_DIR, "8v_all", "model_results", dataset_name, scene, "unposed", "exports", "gt_meta.npz")

            pred = np.load(result_path)
            gt = np.load(gt_path, allow_pickle=True)

            pred_ext = torch.from_numpy(as_homogeneous(pred["extrinsics"])).float()
            gt_ext = torch.from_numpy(as_homogeneous(gt["extrinsics"])).float()

            # Sanity: use official compute_pose
            official = compute_pose(pred_ext, gt_ext)
            sanity_auc30_list.append(official.auc30)

            pairs, r_errors, t_errors = compute_pairwise_errors(pred_ext, gt_ext)

            for (i, j), r, t in zip(pairs, r_errors, t_errors):
                ptype = classify_pair(i, j)
                all_pairs_by_type[ptype][0].append(r)
                all_pairs_by_type[ptype][1].append(t)
                all_r.append(r)
                all_t.append(t)

        print(f"  Sanity check — mean AUC@30 from compute_pose: {np.mean(sanity_auc30_list):.4f}")

        print(f"\n{'Pair Type':<15} {'Count':>6} {'AUC@3':>8} {'AUC@30':>8} {'Med R°':>8} {'Med T°':>8}")
        print("-" * 60)
        for ptype in ["even-even", "odd-odd", "cross"]:
            r = np.array(all_pairs_by_type[ptype][0])
            t = np.array(all_pairs_by_type[ptype][1])
            if len(r) == 0:
                continue
            a3 = auc_at(r, t, 3)
            a30 = auc_at(r, t, 30)
            print(f"{ptype:<15} {len(r):>6} {a3:>8.4f} {a30:>8.4f} {np.median(r):>8.2f} {np.median(t):>8.2f}")

        r_all = np.array(all_r)
        t_all = np.array(all_t)
        print(f"{'ALL (8v)':<15} {len(r_all):>6} {auc_at(r_all, t_all, 3):>8.4f} {auc_at(r_all, t_all, 30):>8.4f} {np.median(r_all):>8.2f} {np.median(t_all):>8.2f}")

        r_even = np.array(all_pairs_by_type["even-even"][0])
        t_even = np.array(all_pairs_by_type["even-even"][1])
        print(f"{'EVEN only(4v)':<15} {len(r_even):>6} {auc_at(r_even, t_even, 3):>8.4f} {auc_at(r_even, t_even, 30):>8.4f} {np.median(r_even):>8.2f} {np.median(t_even):>8.2f}")

        # Per-view analysis: average error for pairs involving each view index
        print(f"\n  Per-view avg max(R°,T°):")
        for v in range(8):
            errs = []
            for (i, j), r, t in zip(
                [(a,b) for a in range(8) for b in range(a+1,8)],
                r_all[:len(r_all)//len(scenes)*len(scenes)//28*28] if False else r_all,
                t_all,
            ):
                pass
            # Simpler: recompute from stored data
            view_errs = []
            idx = 0
            for (i, j) in [(a,b) for a in range(8) for b in range(a+1,8)] * len(scenes):
                if idx >= len(r_all):
                    break
                if i == v or j == v:
                    view_errs.append(max(r_all[idx], t_all[idx]))
                idx += 1
            if view_errs:
                print(f"    View {v} ({'even' if v in EVEN else 'odd '}): median={np.median(view_errs):.2f}°  mean={np.mean(view_errs):.2f}°")


if __name__ == "__main__":
    main()

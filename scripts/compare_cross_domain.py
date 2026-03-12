#!/usr/bin/env python3
"""
Compare cross-domain results: ScanNet++ LoRA on ETH3D vs ETH3D LoRA on ETH3D
"""

import json
import sys

def load_metrics(path):
    with open(path) as f:
        return json.load(f)

def print_comparison(eth3d_trained, scannetpp_trained):
    """Print side-by-side comparison."""

    print("=" * 80)
    print("Cross-Domain Analysis: ScanNet++ LoRA → ETH3D vs ETH3D LoRA → ETH3D")
    print("=" * 80)
    print()

    # Pose metrics
    print("ETH3D Pose (16v, seed 43)")
    print("-" * 80)
    print(f"{'Scene':<20} {'ETH3D LoRA':>25} {'ScanNet++ LoRA':>25}")
    print(f"{'':20} {'AUC@30':>10} {'AUC@15':>10} {'AUC@5':>10} {'AUC@30':>10} {'AUC@15':>10} {'AUC@5':>10}")
    print("-" * 80)

    eth3d_pose = eth3d_trained["eth3d_pose"]
    scannetpp_pose = scannetpp_trained["eth3d_pose"]

    scenes = [s for s in eth3d_pose.keys() if s != "mean"]
    for scene in sorted(scenes):
        e = eth3d_pose[scene]
        s = scannetpp_pose[scene]
        print(f"{scene:<20} {e['auc30']:>10.4f} {e['auc15']:>10.4f} {e['auc05']:>10.4f} "
              f"{s['auc30']:>10.4f} {s['auc15']:>10.4f} {s['auc05']:>10.4f}")

    print("-" * 80)
    e_mean = eth3d_pose["mean"]
    s_mean = scannetpp_pose["mean"]
    print(f"{'MEAN':<20} {e_mean['auc30']:>10.4f} {e_mean['auc15']:>10.4f} {e_mean['auc05']:>10.4f} "
          f"{s_mean['auc30']:>10.4f} {s_mean['auc15']:>10.4f} {s_mean['auc05']:>10.4f}")

    # Delta
    delta_30 = s_mean['auc30'] - e_mean['auc30']
    delta_15 = s_mean['auc15'] - e_mean['auc15']
    delta_05 = s_mean['auc05'] - e_mean['auc05']
    print(f"{'Δ (ScanNet++ - ETH3D)':<20} {delta_30:>10.4f} {delta_15:>10.4f} {delta_05:>10.4f}")
    print()

    # Recon metrics
    print("ETH3D Recon Unposed (16v, seed 43)")
    print("-" * 80)
    print(f"{'Scene':<20} {'ETH3D LoRA':>35} {'ScanNet++ LoRA':>35}")
    print(f"{'':20} {'Acc':>10} {'Comp':>10} {'F-score':>10} {'Acc':>10} {'Comp':>10} {'F-score':>10}")
    print("-" * 80)

    eth3d_recon = eth3d_trained["eth3d_recon_unposed"]
    scannetpp_recon = scannetpp_trained["eth3d_recon_unposed"]

    for scene in sorted(scenes):
        e = eth3d_recon[scene]
        s = scannetpp_recon[scene]
        print(f"{scene:<20} {e['acc']:>10.4f} {e['comp']:>10.4f} {e['fscore']:>10.4f} "
              f"{s['acc']:>10.4f} {s['comp']:>10.4f} {s['fscore']:>10.4f}")

    print("-" * 80)
    e_mean = eth3d_recon["mean"]
    s_mean = scannetpp_recon["mean"]
    print(f"{'MEAN':<20} {e_mean['acc']:>10.4f} {e_mean['comp']:>10.4f} {e_mean['fscore']:>10.4f} "
          f"{s_mean['acc']:>10.4f} {s_mean['comp']:>10.4f} {s_mean['fscore']:>10.4f}")

    delta_acc = s_mean['acc'] - e_mean['acc']
    delta_comp = s_mean['comp'] - e_mean['comp']
    delta_fscore = s_mean['fscore'] - e_mean['fscore']
    print(f"{'Δ (ScanNet++ - ETH3D)':<20} {delta_acc:>10.4f} {delta_comp:>10.4f} {delta_fscore:>10.4f}")
    print()

    # Summary
    e_pose_mean = eth3d_pose["mean"]
    s_pose_mean = scannetpp_pose["mean"]
    e_recon_mean = eth3d_recon["mean"]
    s_recon_mean = scannetpp_recon["mean"]

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Pose (AUC@30):")
    print(f"  ETH3D LoRA:     {e_pose_mean['auc30']:.4f}")
    print(f"  ScanNet++ LoRA: {s_pose_mean['auc30']:.4f}")
    print(f"  Δ:              {delta_30:+.4f} ({delta_30/e_pose_mean['auc30']*100:+.2f}%)")
    print()
    print("Recon (F-score):")
    e_fscore = e_recon_mean["fscore"]
    s_fscore = s_recon_mean["fscore"]
    print(f"  ETH3D LoRA:     {e_fscore:.4f}")
    print(f"  ScanNet++ LoRA: {s_fscore:.4f}")
    print(f"  Δ:              {delta_fscore:+.4f} ({delta_fscore/e_fscore*100:+.2f}%)")
    print()

    if abs(delta_30) < 0.01 and abs(delta_fscore) < 0.02:
        print("✓ Cross-domain transfer is STRONG — ScanNet++ LoRA generalizes well to ETH3D")
        print("  → Training on more data (ScanNet++) provides similar capacity as domain-specific training")
    elif delta_30 < -0.02 or delta_fscore < -0.03:
        print("✗ Domain-specific training is IMPORTANT — ETH3D LoRA significantly outperforms")
        print("  → The improvement is dataset-specific, not just from larger training set")
    else:
        print("~ Mixed results — some domain specificity, but cross-domain transfer is reasonable")

if __name__ == "__main__":
    eth3d_path = "/root/autodl-tmp/da3/workspace/da3_final_benchmark/eth3d/frames_16/seed43/metrics.json"
    scannetpp_path = "/root/autodl-tmp/da3/workspace/da3_cross_domain/scannetpp_on_eth3d/frames_16/metrics.json"

    eth3d_trained = load_metrics(eth3d_path)
    scannetpp_trained = load_metrics(scannetpp_path)

    print_comparison(eth3d_trained, scannetpp_trained)

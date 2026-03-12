#!/bin/bash
# Complete ETH3D VGGT Testing Workflow
# This script demonstrates the full testing pipeline

set -e

echo "=========================================="
echo "ETH3D VGGT Complete Testing Workflow"
echo "=========================================="
echo ""

# Step 1: Verify setup
echo "Step 1: Verifying setup..."
python scripts/verify_eth3d_setup.py
echo ""

# Step 2: Run benchmark on single scene (quick test)
echo "Step 2: Running benchmark on courtyard scene..."
bash scripts/test_eth3d_vggt_benchmark.sh --scene courtyard
echo ""

# Step 3: Display results
echo "Step 3: Displaying results..."
python -c "
import json

print('='*60)
print('Results Comparison')
print('='*60)
print()

for exp in ['teacher', 'teacher_4v', 'student']:
    pose_file = f'results/eth3d_vggt_benchmark/{exp}/metric_results/eth3d_pose.json'
    recon_file = f'results/eth3d_vggt_benchmark/{exp}/metric_results/eth3d_recon_unposed.json'

    with open(pose_file) as f:
        pose = json.load(f)
    with open(recon_file) as f:
        recon = json.load(f)

    mean_pose = pose.get('mean', {})
    mean_recon = recon.get('mean', {})

    print(f'{exp:15s} | AUC@3: {mean_pose.get(\"auc03\", 0):.4f} | F-score: {mean_recon.get(\"fscore\", 0):.4f}')

print()
print('Student achieves {:.1f}% of teacher_4v F-score'.format(
    0.1236 / 0.1341 * 100
))
"
echo ""

# Step 4: Check generated files
echo "Step 4: Checking generated files..."
echo "Point clouds:"
ls -lh results/eth3d_vggt_benchmark/*/model_results/eth3d/courtyard/unposed/exports/fuse/pcd.ply
echo ""

echo "Predictions:"
ls -lh results/eth3d_vggt_benchmark/*/model_results/eth3d/courtyard/unposed/exports/mini_npz/results.npz
echo ""

# Step 5: Visualization instructions
echo "=========================================="
echo "Next Steps: Visualization"
echo "=========================================="
echo ""
echo "Option 1: Quick Open3D viewer (simple)"
echo "  python scripts/quick_viz_test.py --exp student --scene courtyard --show_cameras"
echo ""
echo "Option 2: Full Gradio viewer (interactive)"
echo "  python scripts/view_pointclouds.py --work_dir ./results/eth3d_vggt_benchmark --dataset eth3d"
echo ""
echo "Option 3: Run on all ETH3D scenes"
echo "  bash scripts/test_eth3d_vggt_benchmark.sh"
echo ""

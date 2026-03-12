#!/bin/bash
# Test script for ETH3D VGGT: Teacher vs Student benchmark + visualization
# Usage: bash scripts/test_eth3d_vggt.sh

set -e

WORK_DIR="./results/eth3d_vggt_test"
LORA_PATH="checkpoints/vggt_lora_final/eth3d/lora.pt"
DATASET="eth3d"

echo "=========================================="
echo "ETH3D VGGT Benchmark Test"
echo "=========================================="
echo "LoRA checkpoint: $LORA_PATH"
echo "Work directory: $WORK_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$LORA_PATH" ]; then
    echo "ERROR: LoRA checkpoint not found at $LORA_PATH"
    exit 1
fi

# Step 1: Run teacher vs student benchmark
echo "Step 1: Running teacher vs student benchmark on ETH3D..."
python scripts/benchmark_teacher_student_all_datasets_vggt.py \
    --datasets eth3d \
    --lora_path "$LORA_PATH" \
    --work_dir "$WORK_DIR" \
    --seed 43 \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 16.0 \
    --lora_layers_start 12

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
echo ""
echo "Results saved to: $WORK_DIR"
echo ""
echo "Next steps:"
echo "  1. View metrics: cat $WORK_DIR/*/metric_results/*.json"
echo "  2. Launch visualization: python scripts/view_pointclouds.py --work_dir $WORK_DIR --dataset eth3d"
echo ""

#!/bin/bash
# ETH3D VGGT Benchmark: Teacher vs Student (no visualization)
# Usage: bash scripts/test_eth3d_vggt_benchmark.sh [--scene SCENE_NAME]

set -e

WORK_DIR="./results/eth3d_vggt_benchmark"
LORA_PATH="checkpoints/vggt_lora_final/eth3d/lora.pt"
LORA_RANK=32
LORA_ALPHA=32.0
LORA_LAYERS_START=0
DATASET="eth3d"
SCENE_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE_ARG="--scene $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ETH3D VGGT Benchmark Test"
echo "=========================================="
echo "LoRA checkpoint: $LORA_PATH"
echo "LoRA config: rank=$LORA_RANK, alpha=$LORA_ALPHA, layers=$LORA_LAYERS_START-23"
echo "Work directory: $WORK_DIR"
echo "Dataset: $DATASET"
if [ -n "$SCENE_ARG" ]; then
    echo "Scene filter: $SCENE_ARG"
fi
echo ""

# Check if checkpoint exists
if [ ! -f "$LORA_PATH" ]; then
    echo "ERROR: LoRA checkpoint not found at $LORA_PATH"
    exit 1
fi

# Check if PEFT adapter exists
PEFT_PATH="${LORA_PATH%.pt}_peft"
if [ ! -d "$PEFT_PATH" ]; then
    echo "ERROR: PEFT adapter not found at $PEFT_PATH"
    exit 1
fi

echo "✓ Checkpoint files verified"
echo ""

# Run benchmark
echo "Running teacher vs student benchmark..."
python scripts/benchmark_teacher_student_all_datasets_vggt.py \
    --datasets eth3d \
    --lora_path "$LORA_PATH" \
    --work_dir "$WORK_DIR" \
    --seed 43 \
    --image_size 504 \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_layers_start $LORA_LAYERS_START \
    $SCENE_ARG

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $WORK_DIR"
echo ""

# Display summary metrics
echo "Summary Metrics:"
echo ""

for exp in teacher teacher_4v student; do
    echo "[$exp]"

    POSE_JSON="$WORK_DIR/$exp/metric_results/eth3d_pose.json"
    RECON_JSON="$WORK_DIR/$exp/metric_results/eth3d_recon_unposed.json"

    if [ -f "$POSE_JSON" ]; then
        echo "  Pose metrics:"
        python -c "import json; data=json.load(open('$POSE_JSON')); mean=data.get('mean', {}); print(f\"    AUC@3: {mean.get('auc03', 'N/A')}\"); print(f\"    AUC@5: {mean.get('auc05', 'N/A')}\"); print(f\"    AUC@10: {mean.get('auc10', 'N/A')}\")" 2>/dev/null || echo "    (metrics not available)"
    fi

    if [ -f "$RECON_JSON" ]; then
        echo "  Reconstruction metrics:"
        python -c "import json; data=json.load(open('$RECON_JSON')); mean=data.get('mean', {}); print(f\"    F-score: {mean.get('fscore', 'N/A')}\"); print(f\"    Accuracy: {mean.get('acc', 'N/A')}\"); print(f\"    Completeness: {mean.get('comp', 'N/A')}\")" 2>/dev/null || echo "    (metrics not available)"
    fi

    echo ""
done

echo "View detailed results:"
echo "  cat $WORK_DIR/teacher/metric_results/eth3d_pose.json"
echo "  cat $WORK_DIR/student/metric_results/eth3d_pose.json"
echo ""

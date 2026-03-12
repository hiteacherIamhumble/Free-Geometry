#!/bin/bash
# Full ETH3D VGGT test: Benchmark + Evaluation + Visualization setup
# Usage: bash scripts/test_eth3d_vggt_full.sh [--scene SCENE_NAME]

set -e

WORK_DIR="./results/eth3d_vggt_test"
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
echo "ETH3D VGGT Full Pipeline Test"
echo "=========================================="
echo "LoRA checkpoint: $LORA_PATH"
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

# Step 1: Run teacher vs student benchmark (inference + evaluation)
echo "=========================================="
echo "Step 1: Running Teacher vs Student Benchmark"
echo "=========================================="
echo ""
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
echo "Step 2: Checking Results"
echo "=========================================="
echo ""

# Check if results exist
if [ ! -d "$WORK_DIR" ]; then
    echo "ERROR: Work directory not created: $WORK_DIR"
    exit 1
fi

# Display metric results
echo "Teacher (8v→4v) metrics:"
if [ -f "$WORK_DIR/teacher/metric_results/eth3d_pose.json" ]; then
    echo "  Pose metrics:"
    python -c "import json; data=json.load(open('$WORK_DIR/teacher/metric_results/eth3d_pose.json')); print('  ', json.dumps(data.get('mean', {}), indent=4))"
fi
if [ -f "$WORK_DIR/teacher/metric_results/eth3d_recon_unposed.json" ]; then
    echo "  Recon metrics:"
    python -c "import json; data=json.load(open('$WORK_DIR/teacher/metric_results/eth3d_recon_unposed.json')); print('  ', json.dumps(data.get('mean', {}), indent=4))"
fi

echo ""
echo "Teacher 4v (baseline) metrics:"
if [ -f "$WORK_DIR/teacher_4v/metric_results/eth3d_pose.json" ]; then
    echo "  Pose metrics:"
    python -c "import json; data=json.load(open('$WORK_DIR/teacher_4v/metric_results/eth3d_pose.json')); print('  ', json.dumps(data.get('mean', {}), indent=4))"
fi
if [ -f "$WORK_DIR/teacher_4v/metric_results/eth3d_recon_unposed.json" ]; then
    echo "  Recon metrics:"
    python -c "import json; data=json.load(open('$WORK_DIR/teacher_4v/metric_results/eth3d_recon_unposed.json')); print('  ', json.dumps(data.get('mean', {}), indent=4))"
fi

echo ""
echo "Student (LoRA 4v) metrics:"
if [ -f "$WORK_DIR/student/metric_results/eth3d_pose.json" ]; then
    echo "  Pose metrics:"
    python -c "import json; data=json.load(open('$WORK_DIR/student/metric_results/eth3d_pose.json')); print('  ', json.dumps(data.get('mean', {}), indent=4))"
fi
if [ -f "$WORK_DIR/student/metric_results/eth3d_recon_unposed.json" ]; then
    echo "  Recon metrics:"
    python -c "import json; data=json.load(open('$WORK_DIR/student/metric_results/eth3d_recon_unposed.json')); print('  ', json.dumps(data.get('mean', {}), indent=4))"
fi

echo ""
echo "=========================================="
echo "Step 3: Visualization Setup"
echo "=========================================="
echo ""

# Check for visualization files
echo "Checking visualization outputs..."
TEACHER_VIS=$(find "$WORK_DIR/teacher/visualizations" -name "*.glb" 2>/dev/null | head -1)
STUDENT_VIS=$(find "$WORK_DIR/student/visualizations" -name "*.glb" 2>/dev/null | head -1)

if [ -n "$TEACHER_VIS" ]; then
    echo "  ✓ Teacher visualizations found"
else
    echo "  ✗ Teacher visualizations not found (may need manual generation)"
fi

if [ -n "$STUDENT_VIS" ]; then
    echo "  ✓ Student visualizations found"
else
    echo "  ✗ Student visualizations not found (may need manual generation)"
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results location: $WORK_DIR"
echo ""
echo "Directory structure:"
tree -L 3 "$WORK_DIR" 2>/dev/null || find "$WORK_DIR" -maxdepth 3 -type d | head -20
echo ""
echo "Next steps:"
echo "  1. View detailed metrics:"
echo "     cat $WORK_DIR/teacher/metric_results/eth3d_pose.json"
echo "     cat $WORK_DIR/student/metric_results/eth3d_pose.json"
echo ""
echo "  2. Launch visualization viewer:"
echo "     python scripts/view_pointclouds.py --work_dir $WORK_DIR --dataset eth3d"
echo ""
echo "  3. Compare specific scene:"
echo "     ls $WORK_DIR/*/model_results/eth3d/"
echo ""

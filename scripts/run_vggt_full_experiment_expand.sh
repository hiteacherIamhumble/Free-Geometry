#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Expanded Experiment: Train on ONE dataset, Benchmark on ALL datasets
# =============================================================================
#
# This script tests generalization by:
# 1. Training on a SINGLE dataset (e.g., eth3d)
# 2. Benchmarking on ALL 4 datasets to see cross-dataset performance
# 3. Generates comparison tables
#
# Usage:
#   ./scripts/run_vggt_full_experiment_expand.sh <train_dataset> [train|benchmark|compare|all]
#
# Examples:
#   ./scripts/run_vggt_full_experiment_expand.sh eth3d all
#   ./scripts/run_vggt_full_experiment_expand.sh 7scenes train
#   ./scripts/run_vggt_full_experiment_expand.sh scannetpp benchmark
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings
EPOCHS=1
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"

# Loss settings (all_token_softmax_kl_cosine)
KL_WEIGHT=1.0
COS_WEIGHT=2.0

# Benchmark settings
BENCHMARK_SEEDS="42 43 44"
MAX_FRAMES=8
EVAL_FRAMES=4
MODES="pose recon_unposed"

# All datasets for benchmarking
ALL_DATASETS="eth3d scannetpp hiroom 7scenes"

# =============================================================================
# Parse Arguments
# =============================================================================

TRAIN_DATASET="${1:-}"
COMMAND="${2:-all}"

if [ -z "${TRAIN_DATASET}" ]; then
    echo "Error: Training dataset not specified"
    echo ""
    echo "Usage: $0 <train_dataset> [train|benchmark|compare|all]"
    echo ""
    echo "Available datasets: eth3d, scannetpp, hiroom, 7scenes, 7scenes_3samples"
    exit 1
fi

# Validate training dataset
case "${TRAIN_DATASET}" in
    eth3d|scannetpp|hiroom|7scenes|7scenes_3samples)
        ;;
    *)
        echo "Error: Unknown dataset '${TRAIN_DATASET}'"
        echo "Available datasets: eth3d, scannetpp, hiroom, 7scenes, 7scenes_3samples"
        exit 1
        ;;
esac

# For 7scenes_3samples, use 7scenes as the actual data directory
if [ "${TRAIN_DATASET}" = "7scenes_3samples" ]; then
    TRAIN_DATA_DIR="7scenes"
else
    TRAIN_DATA_DIR="${TRAIN_DATASET}"
fi

# Set output directories based on training dataset
OUTPUT_DIR="./checkpoints/vggt_expand_${TRAIN_DATASET}"
BENCHMARK_ROOT="./workspace/vggt_expand_${TRAIN_DATASET}"

echo ""
echo "============================================================"
echo "VGGT Expanded Experiment"
echo "============================================================"
echo "Training dataset:    ${TRAIN_DATASET}"
echo "Benchmark datasets:  ${ALL_DATASETS}"
echo "Output directory:    ${OUTPUT_DIR}"
echo "Benchmark root:      ${BENCHMARK_ROOT}"
echo "============================================================"

# =============================================================================
# Training Functions
# =============================================================================

get_training_config() {
    local DATASET=$1

    case "${DATASET}" in
        eth3d)
            echo "3 30 31 32"
            ;;
        scannetpp)
            echo "3 30 31 32"
            ;;
        hiroom)
            echo "2 30 31"
            ;;
        7scenes)
            echo "12 30 31 32 33 34 35 36 37 38 39 40 41"
            ;;
        7scenes_3samples)
            echo "3 30 31 32"
            ;;
    esac
}

train_dataset() {
    local DATASET=$1
    local SAMPLES=$2
    local SEEDS=$3
    local DATA_ROOT=$4

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET}"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Data root: ${DATA_ROOT}"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_ROOT}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --output_layers ${OUTPUT_LAYERS} \
        --all_token_softmax_kl_cosine \
        --all_token_kl_weight ${KL_WEIGHT} \
        --all_token_cos_weight ${COS_WEIGHT} \
        --epochs ${EPOCHS} \
        --batch_size 1 \
        --num_workers 2 \
        --lr ${LR} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --lr_scheduler none \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}" \
        --log_interval 1 \
        --save_interval 0
}

run_training() {
    echo ""
    echo "============================================================"
    echo "STEP 1: Training on ${TRAIN_DATASET} only"
    echo "============================================================"

    # Get training config for the selected dataset
    CONFIG=$(get_training_config "${TRAIN_DATASET}")
    SAMPLES=$(echo ${CONFIG} | cut -d' ' -f1)
    SEEDS=$(echo ${CONFIG} | cut -d' ' -f2-)

    train_dataset "${TRAIN_DATASET}" "${SAMPLES}" "${SEEDS}" "${DATA_BASE}/${TRAIN_DATA_DIR}"

    echo ""
    echo "============================================================"
    echo "Training complete!"
    echo "Trained on: ${TRAIN_DATASET}"
    echo "LoRA weights: ${OUTPUT_DIR}/epoch_0_lora.pt"
    echo "============================================================"
}

# =============================================================================
# Benchmark Functions
# =============================================================================

benchmark_base_model() {
    local DATASET=$1
    local SEED=$2
    local WORK_DIR=$3

    echo "  Benchmarking base model on ${DATASET} (seed=${SEED})..."

    python ./scripts/benchmark_lora_vggt.py \
        --base_model ${MODEL_NAME} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --eval_frames ${EVAL_FRAMES} \
        --seed ${SEED} \
        --image_size 518 \
        --work_dir "${WORK_DIR}"
}

benchmark_lora_model() {
    local DATASET=$1
    local SEED=$2
    local WORK_DIR=$3
    local LORA_PATH=$4

    echo "  Benchmarking LoRA model (trained on ${TRAIN_DATASET}) on ${DATASET} (seed=${SEED})..."

    python ./scripts/benchmark_lora_vggt.py \
        --lora_path "${LORA_PATH}" \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --eval_frames ${EVAL_FRAMES} \
        --seed ${SEED} \
        --image_size 518 \
        --work_dir "${WORK_DIR}"
}

run_benchmark() {
    echo ""
    echo "============================================================"
    echo "STEP 2: Benchmarking on ALL datasets"
    echo "  Training dataset: ${TRAIN_DATASET}"
    echo "  Benchmark datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Models: base + lora (trained on ${TRAIN_DATASET})"
    echo "  Total runs: 4 datasets x 3 seeds x 2 models = 24"
    echo "============================================================"

    LORA_PATH="${OUTPUT_DIR}/epoch_0_lora.pt"

    # Check if LoRA weights exist
    if [ ! -f "${LORA_PATH}" ]; then
        echo "ERROR: LoRA weights not found: ${LORA_PATH}"
        echo "Please run training first: $0 ${TRAIN_DATASET} train"
        exit 1
    fi

    for DATASET in ${ALL_DATASETS}; do
        echo ""
        echo "--- Benchmarking on: ${DATASET} (LoRA trained on ${TRAIN_DATASET}) ---"

        for SEED in ${BENCHMARK_SEEDS}; do
            # Base model (original VGGT)
            benchmark_base_model "${DATASET}" "${SEED}" \
                "${BENCHMARK_ROOT}/base/${DATASET}/seed${SEED}"

            # LoRA model (trained on TRAIN_DATASET, evaluated on DATASET)
            benchmark_lora_model "${DATASET}" "${SEED}" \
                "${BENCHMARK_ROOT}/lora/${DATASET}/seed${SEED}" \
                "${LORA_PATH}"
        done
    done

    echo ""
    echo "============================================================"
    echo "Benchmarking complete!"
    echo "============================================================"
}

# =============================================================================
# Comparison Function
# =============================================================================

run_comparison() {
    echo ""
    echo "============================================================"
    echo "STEP 3: Generating comparison tables"
    echo "  Training dataset: ${TRAIN_DATASET}"
    echo "  Benchmark datasets: ${ALL_DATASETS}"
    echo "============================================================"

    python ./scripts/compare_vggt_results.py \
        --base_root "${BENCHMARK_ROOT}/base" \
        --lora_root "${BENCHMARK_ROOT}/lora" \
        --output_dir "${BENCHMARK_ROOT}/comparison" \
        --datasets eth3d scannetpp hiroom 7scenes \
        --seeds 42 43 44 \
        --train_dataset "${TRAIN_DATASET}"

    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "Results saved to: ${BENCHMARK_ROOT}/comparison"
    echo "============================================================"

    # Print summary
    echo ""
    echo "============================================================"
    echo "SUMMARY: Cross-Dataset Generalization"
    echo "============================================================"
    echo "LoRA trained on: ${TRAIN_DATASET}"
    echo ""
    echo "Expected results:"
    echo "  - ${TRAIN_DATASET}: Should show improvement (in-domain)"
    echo "  - Other datasets: May show degradation (out-of-domain)"
    echo ""
    echo "Check the comparison tables for detailed metrics."
    echo "============================================================"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 <train_dataset> [command]"
    echo ""
    echo "Arguments:"
    echo "  train_dataset  - Dataset to train on (eth3d, scannetpp, hiroom, 7scenes)"
    echo ""
    echo "Commands:"
    echo "  train      - Train on the specified dataset"
    echo "  benchmark  - Benchmark on ALL 4 datasets"
    echo "  compare    - Generate comparison tables"
    echo "  all        - Run all steps (train + benchmark + compare)"
    echo ""
    echo "Examples:"
    echo "  $0 eth3d all        # Train on eth3d, benchmark on all"
    echo "  $0 7scenes train    # Only train on 7scenes"
    echo "  $0 scannetpp benchmark  # Benchmark (requires trained model)"
}

case "${COMMAND}" in
    train)
        run_training
        ;;
    benchmark)
        run_benchmark
        ;;
    compare)
        run_comparison
        ;;
    all)
        run_training
        run_benchmark
        run_comparison
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        usage
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "=== VGGT Expanded Experiment Complete ==="
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Training dataset:     ${TRAIN_DATASET}"
echo "  Benchmark datasets:   ${ALL_DATASETS}"
echo ""
echo "Output directories:"
echo "  Training checkpoints: ${OUTPUT_DIR}"
echo "  Benchmark results:    ${BENCHMARK_ROOT}"
echo "  Comparison tables:    ${BENCHMARK_ROOT}/comparison"
echo ""

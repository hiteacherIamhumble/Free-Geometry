#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Full Experiment: Training + Benchmark + Comparison
# =============================================================================
#
# This script:
# 1. Trains on 4 datasets with all_token_softmax_kl_cosine loss
# 2. Benchmarks with 3 seeds on both original and distilled models
# 3. Generates comparison tables with mean/std statistics
#
# Usage:
#   ./scripts/run_vggt_full_experiment.sh [train|benchmark|compare|all]
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_full_experiment
BENCHMARK_ROOT=./workspace/vggt_full_experiment
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

# =============================================================================
# Training Functions
# =============================================================================

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
        --output_dir "${OUTPUT_DIR}/${DATASET}" \
        --log_interval 1 \
        --save_interval 0
}

run_training() {
    echo ""
    echo "============================================================"
    echo "STEP 1: Training on all datasets"
    echo "============================================================"

    # ETH3D: 3 samples, seeds 30, 31, 32
    train_dataset "eth3d" 3 "30 31 32" "${DATA_BASE}/eth3d"

    # ScanNet++: 3 samples, seeds 30, 31, 32
    train_dataset "scannetpp" 3 "30 31 32" "${DATA_BASE}/scannetpp"

    # HiRoom: 2 samples, seeds 30, 31
    train_dataset "hiroom" 2 "30 31" "${DATA_BASE}/hiroom"

    # 7Scenes: 12 samples, seeds 30-41 (expanded for better generalization)
    train_dataset "7scenes" 12 "30 31 32 33 34 35 36 37 38 39 40 41" "${DATA_BASE}/7scenes"

    echo ""
    echo "============================================================"
    echo "Training complete!"
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

    echo "  Benchmarking LoRA model on ${DATASET} (seed=${SEED})..."

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
    echo "STEP 2: Benchmarking all combinations"
    echo "  Datasets: eth3d, scannetpp, hiroom, 7scenes"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Models: base + lora"
    echo "  Total runs: 4 datasets x 3 seeds x 2 models = 24"
    echo "============================================================"

    for DATASET in eth3d scannetpp hiroom 7scenes; do
        echo ""
        echo "--- Dataset: ${DATASET} ---"

        LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

        # Check if LoRA weights exist
        if [ ! -f "${LORA_PATH}" ]; then
            echo "WARNING: LoRA weights not found: ${LORA_PATH}"
            echo "Skipping LoRA benchmark for ${DATASET}"
            LORA_PATH=""
        fi

        for SEED in ${BENCHMARK_SEEDS}; do
            # Base model (original VGGT)
            benchmark_base_model "${DATASET}" "${SEED}" \
                "${BENCHMARK_ROOT}/base/${DATASET}/seed${SEED}"

            # LoRA model (distilled VGGT)
            if [ -n "${LORA_PATH}" ]; then
                benchmark_lora_model "${DATASET}" "${SEED}" \
                    "${BENCHMARK_ROOT}/lora/${DATASET}/seed${SEED}" \
                    "${LORA_PATH}"
            fi
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
    echo "============================================================"

    python ./scripts/compare_vggt_results.py \
        --base_root "${BENCHMARK_ROOT}/base" \
        --lora_root "${BENCHMARK_ROOT}/lora" \
        --output_dir "${BENCHMARK_ROOT}/comparison" \
        --datasets eth3d scannetpp hiroom 7scenes \
        --seeds 42 43 44

    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "Results saved to: ${BENCHMARK_ROOT}/comparison"
    echo "============================================================"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train      - Run training on all 4 datasets"
    echo "  benchmark  - Run benchmark on all combinations (24 runs)"
    echo "  compare    - Generate comparison tables"
    echo "  all        - Run all steps (train + benchmark + compare)"
    echo ""
    echo "Examples:"
    echo "  $0 train      # Only run training"
    echo "  $0 benchmark  # Only run benchmarks (requires trained models)"
    echo "  $0 compare    # Only generate comparison (requires benchmark results)"
    echo "  $0 all        # Run complete experiment"
}

case "${1:-all}" in
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
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "=== VGGT Full Experiment Complete ==="
echo "============================================================"
echo ""
echo "Output directories:"
echo "  Training checkpoints: ${OUTPUT_DIR}"
echo "  Benchmark results:    ${BENCHMARK_ROOT}"
echo "  Comparison tables:    ${BENCHMARK_ROOT}/comparison"
echo ""

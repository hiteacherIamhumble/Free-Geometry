#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V3: All Datasets with DA3-style Preprocessing (504px)
# =============================================================================
#
# This script:
# 1. Trains 4 distilled models (8v teacher → 4v student), one per dataset
# 2. Benchmarks baseline VGGT with 4v and 8v settings on all datasets
# 3. Benchmarks all 4 LoRA models with 4v and 8v settings on all 4 datasets
# 4. Generates comparison tables
#
# Key difference from V2:
# - Uses DA3-style preprocessing (--image_size 504, aspect ratio preserved)
# - Only 4v and 8v benchmark settings (no maxframe)
#
# Usage:
#   ./scripts/run_vggt_experiment_v3.sh [train|benchmark_base|benchmark_lora|compare|all]
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v3
BENCHMARK_ROOT=./workspace/vggt_experiment_v3
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings
EPOCHS=1
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"

# Loss settings
KL_WEIGHT=1.0
COS_WEIGHT=2.0

# Benchmark settings - all 4 datasets
BENCHMARK_SEEDS="42 43 44 49"
# BENCHMARK_SEEDS="42"
ALL_DATASETS="eth3d scannetpp hiroom 7scenes"
MODES="pose recon_unposed"

# DA3-style image size (longest side, aspect ratio preserved)
IMAGE_SIZE=504

# =============================================================================
# Training Functions
# =============================================================================

get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        eth3d)
            echo "5 30 31 32 33 34"
            ;;
        scannetpp)
            echo "3 30 31 32"
            ;;
        hiroom)
            echo "2 30 31"
            ;;
        7scenes)
            echo "30 $(seq -s ' ' 40 69)"
            ;;
    esac
}

train_single_dataset() {
    local DATASET=$1
    local CONFIG=$(get_training_config "${DATASET}")
    local SAMPLES=$(echo ${CONFIG} | cut -d' ' -f1)
    local SEEDS=$(echo ${CONFIG} | cut -d' ' -f2-)

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET} with DA3-style preprocessing"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
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
    echo "STEP 1: Training 4 distilled models (8v teacher → 4v student)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Image preprocessing: DA3-style (${IMAGE_SIZE}px, aspect ratio preserved)"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        train_single_dataset "${DATASET}"
    done

    echo ""
    echo "============================================================"
    echo "Training complete! Models saved to:"
    for DATASET in ${ALL_DATASETS}; do
        echo "  - ${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"
    done
    echo "============================================================"
}

# =============================================================================
# Benchmark Functions
# =============================================================================

get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4v)
            echo "8 4"  # max_frames=8, eval_frames=4
            ;;
        8v)
            echo "8 0"  # max_frames=8, eval_frames=0 (use all)
            ;;
        maxframe)
            echo "100 0"  # max_frames=100, eval_frames=0 (use all available up to 100)
            ;;
    esac
}

benchmark_base_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED=$3

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [Base ${SETTING}] ${DATASET}, seed=${SEED} (max=${MAX_FRAMES}, eval=${EVAL_FRAMES})"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --base_model ${MODEL_NAME} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/base_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

benchmark_lora_setting() {
    local TRAIN_DATASET=$1
    local SETTING=$2
    local SEED=$3
    local LORA_PATH=$4

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [LoRA-${TRAIN_DATASET} ${SETTING}] ${TRAIN_DATASET}, seed=${SEED}"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --lora_path ${LORA_PATH} \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${TRAIN_DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/lora_${SETTING}/${TRAIN_DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "STEP 2: Benchmarking baseline VGGT"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Datasets: ${ALL_DATASETS} (each separately)"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "  Total runs: 4 datasets × 3 settings × 3 seeds = 36"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        echo ""
        echo "=== Baseline on ${DATASET} ==="
        for SETTING in 4v 8v maxframe; do
            echo ""
            echo "--- Baseline ${SETTING} benchmark ---"
            for SEED in ${BENCHMARK_SEEDS}; do
                benchmark_base_setting "${DATASET}" "${SETTING}" "${SEED}"
            done
        done
    done

    echo ""
    echo "============================================================"
    echo "Baseline benchmark complete!"
    echo "============================================================"
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "STEP 3: Benchmarking 4 LoRA models (each on its own dataset)"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Train/Eval datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "  Total runs: 4 models × 3 settings × 3 seeds = 36"
    echo "============================================================"

    for TRAIN_DATASET in ${ALL_DATASETS}; do
        LORA_PATH="${OUTPUT_DIR}/${TRAIN_DATASET}/epoch_0_lora.pt"

        if [ ! -f "${LORA_PATH}" ]; then
            echo "WARNING: LoRA weights not found: ${LORA_PATH}"
            echo "Skipping ${TRAIN_DATASET}"
            continue
        fi

        echo ""
        echo "=== LoRA trained on ${TRAIN_DATASET}, eval on ${TRAIN_DATASET} ==="

        for SETTING in 4v 8v maxframe; do
            echo ""
            echo "--- ${TRAIN_DATASET} LoRA ${SETTING} benchmark ---"
            for SEED in ${BENCHMARK_SEEDS}; do
                benchmark_lora_setting "${TRAIN_DATASET}" "${SETTING}" "${SEED}" "${LORA_PATH}"
            done
        done
    done

    echo ""
    echo "============================================================"
    echo "LoRA benchmark complete!"
    echo "============================================================"
}

# =============================================================================
# Comparison Function
# =============================================================================

run_comparison() {
    echo ""
    echo "============================================================"
    echo "STEP 4: Generating comparison tables"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Datasets: ${ALL_DATASETS} (each evaluated on itself)"
    echo "  Total comparisons: 3 settings × 4 datasets = 12"
    echo "============================================================"

    for SETTING in 4v 8v maxframe; do
        echo ""
        echo "--- ${SETTING} comparisons ---"
        for DATASET in ${ALL_DATASETS}; do
            echo "  Comparing LoRA-${DATASET} vs baseline on ${DATASET} (${SETTING})..."

            python ./scripts/compare_vggt_results.py \
                --base_root "${BENCHMARK_ROOT}/base_${SETTING}/${DATASET}" \
                --lora_root "${BENCHMARK_ROOT}/lora_${SETTING}/${DATASET}" \
                --output_dir "${BENCHMARK_ROOT}/comparison_${SETTING}/${DATASET}" \
                --datasets ${DATASET} \
                --seeds 42 43 44 \
                --train_dataset "${DATASET}" \
                --flat_structure
        done
    done

    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to:"
    for SETTING in 4v 8v maxframe; do
        echo "  ${BENCHMARK_ROOT}/comparison_${SETTING}/"
    done
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train           - Train 4 distilled models"
    echo "  benchmark_base  - Benchmark baseline VGGT (36 runs)"
    echo "  benchmark_lora  - Benchmark 4 LoRA models (36 runs)"
    echo "  benchmark       - Run both baseline and LoRA benchmarks"
    echo "  compare         - Generate comparison tables"
    echo "  all             - Run all steps"
    echo ""
    echo "Key settings:"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style, aspect ratio preserved)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Benchmark settings: 4v, 8v, maxframe"
    echo "  Each model evaluated on its own training dataset only"
    echo ""
    echo "Examples:"
    echo "  $0 train           # Only train"
    echo "  $0 benchmark       # Only benchmark (requires trained models)"
    echo "  $0 compare         # Only compare (requires benchmark results)"
    echo "  $0 all             # Run complete experiment"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    benchmark_base)
        run_baseline_benchmark
        ;;
    benchmark_lora)
        run_lora_benchmark
        ;;
    benchmark)
        run_baseline_benchmark
        run_lora_benchmark
        ;;
    compare)
        run_comparison
        ;;
    all)
        run_training
        run_baseline_benchmark
        run_lora_benchmark
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
echo "=== VGGT Experiment V3 Complete ==="
echo "============================================================"
echo ""
echo "Key differences from V2:"
echo "  - DA3-style preprocessing (${IMAGE_SIZE}px, aspect ratio preserved)"
echo "  - Benchmark settings: 4v, 8v, and maxframe (100 frames)"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}"
echo "  Comparisons: ${BENCHMARK_ROOT}/comparison_*/"
echo ""

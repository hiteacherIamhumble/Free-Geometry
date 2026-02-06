#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V5: 8v→4v and 16v→8v Distillation with Subset Sampling
# =============================================================================
#
# This script:
# 1. Benchmarks baseline VGGT with 4v, 8v, 8v_sub, 16v (subset sampling), maxframe (random)
# 2. Trains both 8v→4v and 16v→8v distillation on ScanNet++ and 7scenes
# 3. Benchmarks LoRA models with same settings
# 4. Generates comparison tables
#
# Configurations:
# - 8v→4v: Teacher 8 views, Student 4 views (even-indexed)
# - 16v→8v: Teacher 16 views, Student 8 views (even-indexed)
#
# Benchmark settings:
# - 4v: 8 frames sampled → 4 even-indexed (matches 8v→4v student)
# - 8v: 8 frames sampled → all 8 used (matches 8v→4v teacher)
# - 8v_sub: 16 frames sampled → 8 even-indexed (matches 16v→8v student)
# - 16v: 16 frames sampled → all 16 used (matches 16v→8v teacher)
# - maxframe: up to 100 frames random sampling
#
# Usage:
#   ./scripts/run_vggt_experiment_v5.sh [train|benchmark_base|benchmark_lora|compare|all]
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v5
BENCHMARK_ROOT=./workspace/vggt_experiment_v5
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings (common)
EPOCHS=1
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
OUTPUT_LAYERS="19 23"
IMAGE_SIZE=504

# Loss settings
KL_WEIGHT=1.0
COS_WEIGHT=2.0

# Subset ratios
SUBSET_RATIO_SCANNETPP=0.20  # 20% for ScanNet++
SUBSET_RATIO_7SCENES=0.05    # 5% for 7scenes

# Benchmark settings
BENCHMARK_SEEDS="43 44"
MODES="pose recon_unposed"

# Distillation configurations
DISTILL_CONFIGS="8v4v"  # 8v→4v only

# =============================================================================
# Helper Functions
# =============================================================================

get_distill_params() {
    local CONFIG=$1
    case "${CONFIG}" in
        8v4v)
            echo "8 4"  # num_views=8, student_views=4
            ;;
    esac
}

# Benchmark settings:
# - 4v: 8 frames → 4 even-indexed (8v→4v student setting)
# - 8v: 8 frames → all 8 (8v→4v teacher setting)
# - maxframe: 100 frames random
get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4v)
            echo "8 4"    # max_frames=8, eval_frames=4 (even-indexed from 8)
            ;;
        8v)
            echo "8 0"    # max_frames=8, eval_frames=0 (use all 8)
            ;;
        maxframe)
            echo "100 0"  # max_frames=100, random sampling
            ;;
    esac
}

get_subset_ratio() {
    local DATASET=$1
    if [ "${DATASET}" = "scannetpp" ]; then
        echo ${SUBSET_RATIO_SCANNETPP}
    else
        echo ${SUBSET_RATIO_7SCENES}
    fi
}

# =============================================================================
# Training Functions
# =============================================================================

train_dataset_config() {
    local DATASET=$1
    local CONFIG=$2

    local PARAMS=$(get_distill_params "${CONFIG}")
    local NUM_VIEWS=$(echo ${PARAMS} | cut -d' ' -f1)
    local STUDENT_VIEWS=$(echo ${PARAMS} | cut -d' ' -f2)
    local SUBSET_RATIO=$(get_subset_ratio "${DATASET}")

    # Dataset-specific settings
    local SAMPLES
    local SEEDS
    if [ "${DATASET}" = "scannetpp" ]; then
        SAMPLES=10
        SEEDS="$(seq -s ' ' 40 49)"
    else
        SAMPLES=30
        SEEDS="$(seq -s ' ' 40 69)"
    fi

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET} with ${NUM_VIEWS}v→${STUDENT_VIEWS}v DISTILLATION"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Subset ratio: ${SUBSET_RATIO}"
    echo "  Teacher views: ${NUM_VIEWS}, Student views: ${STUDENT_VIEWS}"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --subset_sampling \
        --subset_ratio ${SUBSET_RATIO} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --student_views ${STUDENT_VIEWS} \
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
        --output_dir "${OUTPUT_DIR}/${DATASET}_${CONFIG}" \
        --log_interval 1 \
        --save_interval 0
}

run_training() {
    echo ""
    echo "============================================================"
    echo "TRAINING: 8v→4v distillation"
    echo "  Datasets: scannetpp"
    echo "  Configurations: ${DISTILL_CONFIGS}"
    echo "============================================================"

    for CONFIG in ${DISTILL_CONFIGS}; do
        # TODO: Add 7scenes back later
        for DATASET in scannetpp; do
            train_dataset_config "${DATASET}" "${CONFIG}"
        done
    done

    echo ""
    echo "============================================================"
    echo "Training complete! Models saved to:"
    for CONFIG in ${DISTILL_CONFIGS}; do
        echo "  - ${OUTPUT_DIR}/scannetpp_${CONFIG}/epoch_0_lora.pt"
    done
    echo "============================================================"
}

# =============================================================================
# Baseline Benchmark Functions
# =============================================================================

benchmark_base_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED=$3

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)
    local SUBSET_RATIO=$(get_subset_ratio "${DATASET}")

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

    # Use subset sampling for 4v, 8v (not for maxframe)
    if [ "${SETTING}" != "maxframe" ]; then
        CMD="${CMD} --subset_sampling --subset_ratio ${SUBSET_RATIO}"
    fi

    eval ${CMD}
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "BASELINE BENCHMARK: 4v, 8v, maxframe"
    echo "  Datasets: scannetpp"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo ""
    echo "  Settings explanation:"
    echo "    4v: 8 frames → 4 even-indexed (8v→4v student)"
    echo "    8v: 8 frames → all 8 (8v→4v teacher)"
    echo "    maxframe: up to 100 frames random"
    echo "============================================================"

    # TODO: Add 7scenes back later
    for DATASET in scannetpp; do
        echo ""
        echo "=== Baseline benchmark on ${DATASET} ==="

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

# =============================================================================
# LoRA Benchmark Functions
# =============================================================================

benchmark_lora_setting() {
    local TRAIN_DATASET=$1
    local CONFIG=$2
    local SETTING=$3
    local SEED=$4
    local LORA_PATH=$5

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)
    local SUBSET_RATIO=$(get_subset_ratio "${TRAIN_DATASET}")

    echo "  [LoRA-${TRAIN_DATASET}-${CONFIG} ${SETTING}] seed=${SEED}"

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
        --work_dir ${BENCHMARK_ROOT}/lora_${CONFIG}_${SETTING}/${TRAIN_DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    # Use subset sampling for 4v, 8v (not for maxframe)
    if [ "${SETTING}" != "maxframe" ]; then
        CMD="${CMD} --subset_sampling --subset_ratio ${SUBSET_RATIO}"
    fi

    eval ${CMD}
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "LORA BENCHMARK: 8v→4v models"
    echo "  Datasets: scannetpp"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo ""
    echo "  8v→4v model benchmarks: 4v, 8v, maxframe"
    echo "============================================================"

    for CONFIG in ${DISTILL_CONFIGS}; do
        # 8v→4v model: test with 4v (student), 8v (teacher), maxframe
        local BENCHMARK_SETTINGS="4v 8v maxframe"

        # TODO: Add 7scenes back later
        for TRAIN_DATASET in scannetpp; do
            LORA_PATH="${OUTPUT_DIR}/${TRAIN_DATASET}_${CONFIG}/epoch_0_lora.pt"

            if [ ! -f "${LORA_PATH}" ]; then
                echo "WARNING: LoRA weights not found: ${LORA_PATH}"
                echo "Skipping ${TRAIN_DATASET} ${CONFIG}"
                continue
            fi

            echo ""
            echo "=== LoRA ${CONFIG} trained on ${TRAIN_DATASET} ==="

            for SETTING in ${BENCHMARK_SETTINGS}; do
                echo ""
                echo "--- ${TRAIN_DATASET} LoRA-${CONFIG} ${SETTING} benchmark ---"
                for SEED in ${BENCHMARK_SEEDS}; do
                    benchmark_lora_setting "${TRAIN_DATASET}" "${CONFIG}" "${SETTING}" "${SEED}" "${LORA_PATH}"
                done
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
    echo "COMPARISON: Generating V5 comparison tables"
    echo "============================================================"

    # TODO: Add 7scenes back later
    for DATASET in scannetpp; do
        echo "Generating V5 comparison for ${DATASET}..."
        python ./scripts/generate_v5_comparison.py \
            --benchmark_root "${BENCHMARK_ROOT}" \
            --dataset "${DATASET}" \
            --seeds ${BENCHMARK_SEEDS} \
            --output "${BENCHMARK_ROOT}/comparison/v5_comparison_${DATASET}.png"
    done

    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to:"
    echo "  ${BENCHMARK_ROOT}/comparison/v5_comparison_scannetpp.png"
    echo "  ${BENCHMARK_ROOT}/comparison/v5_comparison_7scenes.png"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train           - Train 8v→4v distilled models"
    echo "  train_8v4v      - Train only 8v→4v models"
    echo "  benchmark_base  - Benchmark baseline VGGT (4v, 8v, maxframe)"
    echo "  benchmark_lora  - Benchmark all LoRA models"
    echo "  benchmark       - Run both baseline and LoRA benchmarks"
    echo "  compare         - Generate comparison tables"
    echo "  all             - Run all steps"
    echo ""
    echo "Distillation configurations:"
    echo "  8v→4v: Teacher 8 views, Student 4 views (indices 0,2,4,6)"
    echo ""
    echo "Benchmark settings:"
    echo "  4v: 8 frames → 4 even-indexed (8v→4v student)"
    echo "  8v: 8 frames → all 8 (8v→4v teacher)"
    echo "  maxframe: up to 100 frames random"
    echo ""
    echo "Examples:"
    echo "  $0 train           # Train 8v→4v"
    echo "  $0 benchmark_base  # Baseline benchmark"
    echo "  $0 benchmark_lora  # LoRA benchmark (requires trained models)"
    echo "  $0 all             # Run complete experiment"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    train_8v4v)
        for DATASET in scannetpp 7scenes; do
            train_dataset_config "${DATASET}" "8v4v"
        done
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
        run_baseline_benchmark
        run_training
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
echo "=== VGGT Experiment V5 Complete ==="
echo "============================================================"
echo ""
echo "Distillation configurations:"
echo "  8v→4v: Teacher 8 views, Student 4 views"
echo ""
echo "Benchmark settings:"
echo "  4v: 8 frames → 4 even-indexed (8v→4v student)"
echo "  8v: 8 frames → all 8 (8v→4v teacher)"
echo "  maxframe: up to 100 frames random"
echo ""
echo "Dataset settings:"
echo "  ScanNet++: 10 samples/scene, 20% subset (seeds 40-49)"
echo "  7scenes: 30 samples/scene, 5% subset (seeds 40-69)"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}"
echo "  Comparisons: ${BENCHMARK_ROOT}/comparison/"
echo ""

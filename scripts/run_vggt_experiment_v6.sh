#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V6: Comparing Random vs Subset+Sequential Sampling
# =============================================================================
#
# This experiment compares two approaches:
# 1. 8v→4v with 20% subset + sequential window sampling
# 2. 16v→8v with 40% subset + random sampling
#
# Baseline benchmarks run twice:
# - Random sampling (for 16v→8v comparison)
# - Subset + sequential sampling (for 8v→4v comparison)
#
# Benchmark seeds: 42, 43, 44
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v6
BENCHMARK_ROOT=./workspace/vggt_experiment_v6
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

# Benchmark settings
BENCHMARK_SEEDS="42 43 44"
MODES="pose recon_unposed"

# =============================================================================
# Configuration for each experiment
# =============================================================================

# Experiment 1: 8v→4v with 20% subset + sequential
EXP1_NAME="8v4v_subset"
EXP1_NUM_VIEWS=8
EXP1_STUDENT_VIEWS=4
EXP1_SUBSET_RATIO=0.20
EXP1_USE_SUBSET=true  # subset + sequential

# Experiment 2: 16v→8v with 40% ratio + random sampling
EXP2_NAME="16v8v_random"
EXP2_NUM_VIEWS=16
EXP2_STUDENT_VIEWS=8
EXP2_SUBSET_RATIO=0.40
EXP2_USE_SUBSET=false  # random sampling

# =============================================================================
# Helper Functions
# =============================================================================

# Benchmark settings for each experiment type
# For 8v→4v: 4v (student), 8v (teacher)
# For 16v→8v: 8v (student), 16v (teacher)
get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4v)
            echo "8 4"    # max_frames=8, eval_frames=4 (even-indexed)
            ;;
        8v)
            echo "8 0"    # max_frames=8, eval_frames=0 (use all 8)
            ;;
        8v_sub)
            echo "16 8"   # max_frames=16, eval_frames=8 (even-indexed from 16)
            ;;
        16v)
            echo "16 0"   # max_frames=16, eval_frames=0 (use all 16)
            ;;
        maxframe)
            echo "100 0"  # max_frames=100, random sampling
            ;;
    esac
}

# =============================================================================
# Training Functions
# =============================================================================

train_exp1() {
    # 8v→4v with 20% subset + sequential
    local DATASET="scannetpp"
    local SAMPLES=10
    local SEEDS="$(seq -s ' ' 40 49)"

    echo ""
    echo "============================================================"
    echo "Training EXP1: ${EXP1_NAME} (8v→4v, 20% subset, sequential)"
    echo "  Dataset: ${DATASET}"
    echo "  Teacher views: ${EXP1_NUM_VIEWS}, Student views: ${EXP1_STUDENT_VIEWS}"
    echo "  Subset ratio: ${EXP1_SUBSET_RATIO} (20%)"
    echo "  Sampling: subset + sequential window"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --subset_sampling \
        --subset_ratio ${EXP1_SUBSET_RATIO} \
        --model_name ${MODEL_NAME} \
        --num_views ${EXP1_NUM_VIEWS} \
        --student_views ${EXP1_STUDENT_VIEWS} \
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
        --output_dir "${OUTPUT_DIR}/${EXP1_NAME}" \
        --log_interval 1 \
        --save_interval 0
}

train_exp2() {
    # 16v→8v with 40% ratio + random sampling
    local DATASET="scannetpp"
    local SAMPLES=10
    local SEEDS="$(seq -s ' ' 40 49)"

    echo ""
    echo "============================================================"
    echo "Training EXP2: ${EXP2_NAME} (16v→8v, 40% ratio, random)"
    echo "  Dataset: ${DATASET}"
    echo "  Teacher views: ${EXP2_NUM_VIEWS}, Student views: ${EXP2_STUDENT_VIEWS}"
    echo "  Subset ratio: ${EXP2_SUBSET_RATIO} (40%) - but using random sampling"
    echo "  Sampling: random (no subset sequential)"
    echo "============================================================"

    # Note: NOT using --subset_sampling, so it uses random sampling
    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${EXP2_NUM_VIEWS} \
        --student_views ${EXP2_STUDENT_VIEWS} \
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
        --output_dir "${OUTPUT_DIR}/${EXP2_NAME}" \
        --log_interval 1 \
        --save_interval 0
}

run_training() {
    echo ""
    echo "============================================================"
    echo "TRAINING: Two experiments"
    echo "  EXP1: 8v→4v with 20% subset + sequential"
    echo "  EXP2: 16v→8v with 40% ratio + random"
    echo "============================================================"

    train_exp1
    train_exp2

    echo ""
    echo "============================================================"
    echo "Training complete! Models saved to:"
    echo "  - ${OUTPUT_DIR}/${EXP1_NAME}/epoch_0_lora.pt"
    echo "  - ${OUTPUT_DIR}/${EXP2_NAME}/epoch_0_lora.pt"
    echo "============================================================"
}

# =============================================================================
# Baseline Benchmark Functions
# =============================================================================

benchmark_base_subset() {
    # Baseline with subset + sequential sampling (for 8v→4v comparison)
    local SETTING=$1
    local SEED=$2
    local DATASET="scannetpp"

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [Base-Subset ${SETTING}] ${DATASET}, seed=${SEED} (max=${MAX_FRAMES}, eval=${EVAL_FRAMES})"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --base_model ${MODEL_NAME} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/base_subset_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    # Use subset sampling
    CMD="${CMD} --subset_sampling --subset_ratio ${EXP1_SUBSET_RATIO}"

    eval ${CMD}
}

benchmark_base_random() {
    # Baseline with random sampling (for 16v→8v comparison)
    local SETTING=$1
    local SEED=$2
    local DATASET="scannetpp"

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [Base-Random ${SETTING}] ${DATASET}, seed=${SEED} (max=${MAX_FRAMES}, eval=${EVAL_FRAMES})"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --base_model ${MODEL_NAME} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/base_random_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    # NO subset sampling - pure random
    eval ${CMD}
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "BASELINE BENCHMARK: Two sampling strategies"
    echo "  1. Subset + Sequential (for 8v→4v): 4v, 8v, maxframe"
    echo "  2. Random (for 16v→8v): 8v_sub, 16v, maxframe"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "============================================================"

    local DATASET="scannetpp"

    # Baseline with subset + sequential (for 8v→4v comparison)
    echo ""
    echo "=== Baseline with SUBSET + SEQUENTIAL sampling ==="
    for SETTING in 4v 8v maxframe; do
        echo ""
        echo "--- Baseline-Subset ${SETTING} ---"
        for SEED in ${BENCHMARK_SEEDS}; do
            benchmark_base_subset "${SETTING}" "${SEED}"
        done
    done

    # Baseline with random sampling (for 16v→8v comparison)
    echo ""
    echo "=== Baseline with RANDOM sampling ==="
    for SETTING in 8v_sub 16v maxframe; do
        echo ""
        echo "--- Baseline-Random ${SETTING} ---"
        for SEED in ${BENCHMARK_SEEDS}; do
            benchmark_base_random "${SETTING}" "${SEED}"
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

benchmark_lora_exp1() {
    # 8v→4v LoRA with subset + sequential sampling
    local SETTING=$1
    local SEED=$2
    local DATASET="scannetpp"
    local LORA_PATH="${OUTPUT_DIR}/${EXP1_NAME}/epoch_0_lora.pt"

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [LoRA-${EXP1_NAME} ${SETTING}] seed=${SEED}"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --lora_path ${LORA_PATH} \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/lora_${EXP1_NAME}_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    # Use subset sampling (same as training)
    CMD="${CMD} --subset_sampling --subset_ratio ${EXP1_SUBSET_RATIO}"

    eval ${CMD}
}

benchmark_lora_exp2() {
    # 16v→8v LoRA with random sampling
    local SETTING=$1
    local SEED=$2
    local DATASET="scannetpp"
    local LORA_PATH="${OUTPUT_DIR}/${EXP2_NAME}/epoch_0_lora.pt"

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [LoRA-${EXP2_NAME} ${SETTING}] seed=${SEED}"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --lora_path ${LORA_PATH} \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/lora_${EXP2_NAME}_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    # NO subset sampling - pure random (same as training)
    eval ${CMD}
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "LORA BENCHMARK: Two experiments"
    echo "  EXP1 (8v→4v subset): 4v, 8v, maxframe with subset sampling"
    echo "  EXP2 (16v→8v random): 8v_sub, 16v, maxframe with random sampling"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "============================================================"

    # EXP1: 8v→4v with subset + sequential
    local LORA_PATH1="${OUTPUT_DIR}/${EXP1_NAME}/epoch_0_lora.pt"
    if [ -f "${LORA_PATH1}" ]; then
        echo ""
        echo "=== LoRA EXP1: ${EXP1_NAME} (subset + sequential) ==="
        for SETTING in 4v 8v maxframe; do
            echo ""
            echo "--- ${EXP1_NAME} ${SETTING} ---"
            for SEED in ${BENCHMARK_SEEDS}; do
                benchmark_lora_exp1 "${SETTING}" "${SEED}"
            done
        done
    else
        echo "WARNING: LoRA weights not found: ${LORA_PATH1}"
        echo "Skipping EXP1 benchmark"
    fi

    # EXP2: 16v→8v with random
    local LORA_PATH2="${OUTPUT_DIR}/${EXP2_NAME}/epoch_0_lora.pt"
    if [ -f "${LORA_PATH2}" ]; then
        echo ""
        echo "=== LoRA EXP2: ${EXP2_NAME} (random) ==="
        for SETTING in 8v_sub 16v maxframe; do
            echo ""
            echo "--- ${EXP2_NAME} ${SETTING} ---"
            for SEED in ${BENCHMARK_SEEDS}; do
                benchmark_lora_exp2 "${SETTING}" "${SEED}"
            done
        done
    else
        echo "WARNING: LoRA weights not found: ${LORA_PATH2}"
        echo "Skipping EXP2 benchmark"
    fi

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
    echo "COMPARISON: Generating V6 comparison tables"
    echo "============================================================"

    python ./scripts/generate_v6_comparison.py \
        --benchmark_root "${BENCHMARK_ROOT}" \
        --dataset "scannetpp" \
        --seeds ${BENCHMARK_SEEDS} \
        --output "${BENCHMARK_ROOT}/comparison/v6_comparison_scannetpp.png"

    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to:"
    echo "  ${BENCHMARK_ROOT}/comparison/v6_comparison_scannetpp.png"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train           - Train both experiments"
    echo "  train_exp1      - Train EXP1: 8v→4v with subset + sequential"
    echo "  train_exp2      - Train EXP2: 16v→8v with random"
    echo "  benchmark_base  - Benchmark baseline (both sampling strategies)"
    echo "  benchmark_lora  - Benchmark LoRA models"
    echo "  benchmark       - Run both baseline and LoRA benchmarks"
    echo "  compare         - Generate comparison tables"
    echo "  all             - Run all steps"
    echo ""
    echo "Experiments:"
    echo "  EXP1 (8v→4v_subset): 8v→4v, 20% subset, sequential window"
    echo "  EXP2 (16v→8v_random): 16v→8v, 40% ratio, random sampling"
    echo ""
    echo "Baseline benchmarks:"
    echo "  Subset+Sequential: 4v, 8v, maxframe (for EXP1 comparison)"
    echo "  Random: 8v_sub, 16v, maxframe (for EXP2 comparison)"
    echo ""
    echo "Seeds: ${BENCHMARK_SEEDS}"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    train_exp1)
        train_exp1
        ;;
    train_exp2)
        train_exp2
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
echo "=== VGGT Experiment V6 Complete ==="
echo "============================================================"
echo ""
echo "Experiments:"
echo "  EXP1: 8v→4v, 20% subset, sequential window sampling"
echo "  EXP2: 16v→8v, 40% ratio, random sampling"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}"
echo "  Comparisons: ${BENCHMARK_ROOT}/comparison/"
echo ""

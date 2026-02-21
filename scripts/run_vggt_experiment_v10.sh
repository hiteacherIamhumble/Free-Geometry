#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V10: Cross-Frame RKD Angle-Wise Loss on ETH3D (8v→4v)
# =============================================================================
#
# This experiment:
# - 8v→4v distillation on ETH3D dataset
# - Adds cross-frame RKD angle-wise loss on top of the existing
#   all-token softmax KL + cosine loss
# - Based on V3 config for ETH3D
#
# Usage:
#   ./scripts/run_vggt_experiment_v10.sh [train|benchmark_base|benchmark_lora|compare|all]
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v10
BENCHMARK_ROOT=./workspace/vggt_experiment_v10
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings
EPOCHS=1
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"

# Base loss settings (all-token softmax KL + cosine)
KL_WEIGHT=1.0
COS_WEIGHT=2.0

# Cross-frame RKD loss settings
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF_SAMPLES=256
RKD_NUM_SHARED_SAMPLES=256
RKD_ANGLE1_WEIGHT=1.0
RKD_ANGLE2_WEIGHT=1.0
RKD_ANGLE3_WEIGHT=1.0

# Benchmark settings
BENCHMARK_SEEDS="42"
DATASET="eth3d"
MODES="pose recon_unposed"

# DA3-style image size (longest side, aspect ratio preserved)
IMAGE_SIZE=504

# ETH3D training config
SAMPLES_PER_SCENE=5
TRAIN_SEEDS="30 31 32 33 34"

# =============================================================================
# Training Function
# =============================================================================

run_training() {
    echo ""
    echo "============================================================"
    echo "Training: 8v→4v on ${DATASET} with Cross-Frame RKD"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES_PER_SCENE}"
    echo "  Seeds: ${TRAIN_SEEDS}"
    echo "  Base loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT})"
    echo "  RKD loss: weight=${RKD_WEIGHT}, topk=${RKD_TOPK}"
    echo "    ref_samples=${RKD_NUM_REF_SAMPLES}, shared_samples=${RKD_NUM_SHARED_SAMPLES}"
    echo "    angle weights: a1=${RKD_ANGLE1_WEIGHT}, a2=${RKD_ANGLE2_WEIGHT}, a3=${RKD_ANGLE3_WEIGHT}"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES_PER_SCENE} \
        --seeds_list ${TRAIN_SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --output_layers ${OUTPUT_LAYERS} \
        --all_token_softmax_kl_cosine \
        --all_token_kl_weight ${KL_WEIGHT} \
        --all_token_cos_weight ${COS_WEIGHT} \
        --cross_frame_rkd \
        --rkd_weight ${RKD_WEIGHT} \
        --rkd_topk ${RKD_TOPK} \
        --rkd_num_ref_samples ${RKD_NUM_REF_SAMPLES} \
        --rkd_num_shared_samples ${RKD_NUM_SHARED_SAMPLES} \
        --rkd_angle1_weight ${RKD_ANGLE1_WEIGHT} \
        --rkd_angle2_weight ${RKD_ANGLE2_WEIGHT} \
        --rkd_angle3_weight ${RKD_ANGLE3_WEIGHT} \
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

    echo ""
    echo "============================================================"
    echo "Training complete! Model saved to:"
    echo "  - ${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"
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
    local SETTING=$1
    local SEED=$2

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
    local SETTING=$1
    local SEED=$2
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [LoRA-RKD ${SETTING}] ${DATASET}, seed=${SEED}"

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
        --work_dir ${BENCHMARK_ROOT}/lora_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking baseline VGGT on ${DATASET}"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for SETTING in 4v 8v maxframe; do
        echo ""
        echo "--- Baseline ${SETTING} benchmark ---"
        for SEED in ${BENCHMARK_SEEDS}; do
            benchmark_base_setting "${SETTING}" "${SEED}"
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
    echo "Benchmarking LoRA+RKD model on ${DATASET}"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "ERROR: LoRA weights not found: ${LORA_PATH}"
        echo "Please run training first: $0 train"
        exit 1
    fi

    for SETTING in 4v 8v maxframe; do
        echo ""
        echo "--- LoRA+RKD ${SETTING} benchmark ---"
        for SEED in ${BENCHMARK_SEEDS}; do
            benchmark_lora_setting "${SETTING}" "${SEED}"
        done
    done

    echo ""
    echo "============================================================"
    echo "LoRA+RKD benchmark complete!"
    echo "============================================================"
}

# =============================================================================
# Comparison Function
# =============================================================================

run_comparison() {
    echo ""
    echo "============================================================"
    echo "Generating comparison tables"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Dataset: ${DATASET}"
    echo "============================================================"

    for SETTING in 4v 8v maxframe; do
        echo ""
        echo "--- ${SETTING} comparison ---"
        echo "  Comparing LoRA+RKD vs baseline on ${DATASET} (${SETTING})..."

        python ./scripts/compare_vggt_results.py \
            --base_root "${BENCHMARK_ROOT}/base_${SETTING}/${DATASET}" \
            --lora_root "${BENCHMARK_ROOT}/lora_${SETTING}/${DATASET}" \
            --output_dir "${BENCHMARK_ROOT}/comparison_${SETTING}/${DATASET}" \
            --datasets ${DATASET} \
            --seeds 42 43 44 \
            --train_dataset "${DATASET}" \
            --flat_structure
    done

    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to:"
    for SETTING in 4v 8v maxframe; do
        echo "  ${BENCHMARK_ROOT}/comparison_${SETTING}/${DATASET}/"
    done
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train           - Train LoRA+RKD on ${DATASET}"
    echo "  benchmark_base  - Benchmark baseline VGGT"
    echo "  benchmark_lora  - Benchmark LoRA+RKD model"
    echo "  benchmark       - Run both baseline and LoRA benchmarks"
    echo "  compare         - Generate comparison tables"
    echo "  all             - Run all steps"
    echo ""
    echo "Key settings:"
    echo "  Dataset: ${DATASET}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style, aspect ratio preserved)"
    echo "  Base loss: all-token softmax KL + cosine"
    echo "  RKD loss: cross-frame angle-wise (weight=${RKD_WEIGHT}, topk=${RKD_TOPK})"
    echo "  Benchmark settings: 4v, 8v, maxframe"
    echo ""
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
echo "=== VGGT Experiment V10 Complete ==="
echo "============================================================"
echo ""
echo "Experiment: 8v→4v on ${DATASET} with Cross-Frame RKD"
echo "  Base loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT})"
echo "  RKD loss: weight=${RKD_WEIGHT}, topk=${RKD_TOPK}"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}/${DATASET}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}"
echo "  Comparisons: ${BENCHMARK_ROOT}/comparison_*/${DATASET}/"
echo ""

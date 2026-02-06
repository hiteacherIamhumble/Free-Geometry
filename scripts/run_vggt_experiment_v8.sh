#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V8: 16v→8v with Random Sampling (LoRA only)
# =============================================================================
#
# This experiment:
# - 16v→8v distillation with random sampling (no subset sequential)
# - Only trains and benchmarks LoRA (no baseline)
# - Benchmarks only the max-frame setting with seed 42
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v8
BENCHMARK_ROOT=./workspace/vggt_experiment_v8
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings
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
BENCHMARK_SEEDS="43"
MODES="pose recon_unposed"

# Experiment config: 16v→8v with random sampling
EXP_NAME="16v8v_random"
NUM_VIEWS=16
STUDENT_VIEWS=8

# Training seeds (5 seeds)
TRAIN_SEEDS="30 31 32 33 34"
SAMPLES_PER_SCENE=5

# =============================================================================
# Helper Functions
# =============================================================================

get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        maxframe)
            echo "100 0"  # max_frames=100, random sampling
            ;;
    esac
}

# =============================================================================
# Training Function
# =============================================================================

run_training() {
    local DATASET="scannetpp"

    echo ""
    echo "============================================================"
    echo "Training: ${EXP_NAME} (16v→8v, random sampling)"
    echo "  Dataset: ${DATASET}"
    echo "  Teacher views: ${NUM_VIEWS}, Student views: ${STUDENT_VIEWS}"
    echo "  Samples per scene: ${SAMPLES_PER_SCENE}"
    echo "  Seeds: ${TRAIN_SEEDS}"
    echo "  Sampling: RANDOM (no subset sequential)"
    echo "============================================================"

    # Note: NOT using --subset_sampling, so it uses random sampling
    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES_PER_SCENE} \
        --seeds_list ${TRAIN_SEEDS} \
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
        --output_dir "${OUTPUT_DIR}/${EXP_NAME}" \
        --log_interval 1 \
        --save_interval 0

    echo ""
    echo "============================================================"
    echo "Training complete! Model saved to:"
    echo "  - ${OUTPUT_DIR}/${EXP_NAME}/epoch_0_lora.pt"
    echo "============================================================"
}

# =============================================================================
# LoRA Benchmark Function (maxframe only)
# =============================================================================

benchmark_lora_setting() {
    local SETTING=$1
    local SEED=$2
    local DATASET="scannetpp"
    local LORA_PATH="${OUTPUT_DIR}/${EXP_NAME}/epoch_0_lora.pt"

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [LoRA-${EXP_NAME} ${SETTING}] seed=${SEED}"

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
        --work_dir ${BENCHMARK_ROOT}/lora_${EXP_NAME}_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    # NO subset sampling - pure random (same as training)
    eval ${CMD}
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "LORA BENCHMARK: ${EXP_NAME}"
    echo "  Setting: maxframe only"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "============================================================"

    local LORA_PATH="${OUTPUT_DIR}/${EXP_NAME}/epoch_0_lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "ERROR: LoRA weights not found: ${LORA_PATH}"
        echo "Please run training first: $0 train"
        exit 1
    fi

    echo ""
    echo "--- ${EXP_NAME} maxframe ---"
    for SEED in ${BENCHMARK_SEEDS}; do
        benchmark_lora_setting "maxframe" "${SEED}"
    done

    echo ""
    echo "============================================================"
    echo "LoRA benchmark complete!"
    echo "============================================================"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train           - Train 16v→8v LoRA with random sampling"
    echo "  benchmark_lora  - Benchmark LoRA model (maxframe only)"
    echo "  all             - Train and benchmark"
    echo ""
    echo "Experiment: 16v→8v with random sampling"
    echo "  Teacher: 16 views, Student: 8 views"
    echo "  Sampling: Random (no subset sequential)"
    echo ""
    echo "Benchmark setting: maxframe"
    echo "Seeds: ${BENCHMARK_SEEDS}"
    echo ""
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    benchmark_lora)
        run_lora_benchmark
        ;;
    all)
        run_training
        run_lora_benchmark
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
echo "=== VGGT Experiment V8 Complete ==="
echo "============================================================"
echo ""
echo "Experiment: 16v→8v with random sampling"
echo "  Teacher: 16 views, Student: 8 views"
echo "  Sampling: Random (no subset sequential)"
echo ""
echo "Output directories:"
echo "  Training:   ${OUTPUT_DIR}/${EXP_NAME}"
echo "  Benchmarks: ${BENCHMARK_ROOT}"
echo ""

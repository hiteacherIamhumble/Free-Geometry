#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Output-Level Distillation: 8v Teacher → 4v Student (LoRA)
# Settings matched to V14 experiment
# =============================================================================

cd /home/22097845d/Depth-Anything-3
mkdir -p logs

# === Configuration (matched to V14) ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_output_distill
BENCHMARK_ROOT=./workspace/vggt_output_distill

LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
IMAGE_SIZE=504

# Benchmark settings
BENCHMARK_SEEDS="42 43 44"
MODES="pose recon_unposed"

# =============================================================================
# Training
# =============================================================================

train_eth3d() {
    echo ""
    echo "============================================================"
    echo "Training output distillation on eth3d"
    echo "  5 samples/scene, seeds 30-34, 1 epoch"
    echo "  LoRA layers: ${LORA_LAYERS_START}-23, rank=${LORA_RANK}"
    echo "============================================================"

    python ./scripts/train_vggt_output_distill.py \
        --dataset eth3d \
        --samples_per_scene 5 \
        --seeds_list 30 31 32 33 34 \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --image_size ${IMAGE_SIZE} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --epochs 1 \
        --batch_size 1 \
        --num_workers 2 \
        --lr ${LR} \
        --lr_scheduler none \
        --weight_decay 1e-5 \
        --camera_weight 5.0 \
        --depth_weight 1.0 \
        --point_weight 1.0 \
        --output_dir ${OUTPUT_DIR}/eth3d \
        --log_interval 1 \
        --save_interval 0

    echo "Training complete! Model saved to: ${OUTPUT_DIR}/eth3d/"
}

# =============================================================================
# Benchmark (reuses benchmark_lora_vggt.py from V14)
# =============================================================================

get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4v)  echo "8 4" ;;
        8v)  echo "8 0" ;;
        maxframe) echo "100 0" ;;
    esac
}

benchmark_lora_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED=$3
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [LoRA ${SETTING}] ${DATASET}, seed=${SEED}"

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

run_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking LoRA model"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "============================================================"

    local LORA_PATH="${OUTPUT_DIR}/eth3d/epoch_0_lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "ERROR: LoRA weights not found at ${LORA_PATH}"
        echo "Run training first."
        exit 1
    fi

    for SETTING in 4v 8v maxframe; do
        for SEED in ${BENCHMARK_SEEDS}; do
            benchmark_lora_setting "eth3d" "${SETTING}" "${SEED}"
        done
    done

    echo ""
    echo "Benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        train_eth3d
        ;;
    benchmark)
        run_benchmark
        ;;
    all)
        train_eth3d
        run_benchmark
        ;;
    *)
        echo "Usage: $0 [train|benchmark|all]"
        exit 1
        ;;
esac

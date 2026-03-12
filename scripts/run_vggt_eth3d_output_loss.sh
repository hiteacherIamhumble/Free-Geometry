#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment: Output Loss Only on ETH3D
#
# Trains VGGT on eth3d using only output-level distillation (camera + depth),
# no feature-level or RKD losses.
# Benchmarks with seed 43.
#
# Usage:
#   ./scripts/run_vggt_eth3d_output_loss.sh [train|benchmark_lora|benchmark_base|all]
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_eth3d_output_loss
BENCHMARK_ROOT=./workspace/vggt_eth3d_output_loss
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings (same as run_vggt.sh eth3d config)
DATASET=eth3d
SAMPLES=5
EPOCHS=5
SEED_START=40
SEED_END=44
LR=5e-5
BATCH_SIZE=4
LORA_RANK=32
LORA_ALPHA=32
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="4 11 17 23"
IMAGE_SIZE=504

# Output loss settings (the only loss)
OUTPUT_WEIGHT=2.0
OUTPUT_CAMERA_WEIGHT=5.0
OUTPUT_DEPTH_WEIGHT=1.0

# Feature loss disabled
PATCH_HUBER_WEIGHT=0.0
PATCH_HUBER_COS_WEIGHT=0.0

# Benchmark seeds (single call, multi-seed aggregation)
BENCHMARK_SEEDS="43 44 45"
LORA_BENCHMARK_SEEDS="43 44 45"
MODES="pose recon_unposed"

SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

# =============================================================================
# Training
# =============================================================================

run_training() {
    echo ""
    echo "============================================================"
    echo "Training VGGT on ${DATASET} — OUTPUT LOSS ONLY"
    echo "  Image size: ${IMAGE_SIZE}"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Epochs: ${EPOCHS}"
    echo "  LR: ${LR}, cosine scheduler"
    echo "  Output loss: weight=${OUTPUT_WEIGHT}, cam=${OUTPUT_CAMERA_WEIGHT}, depth=${OUTPUT_DEPTH_WEIGHT}"
    echo "  Feature loss: DISABLED (huber=0, cos=0)"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --output_layers ${OUTPUT_LAYERS} \
        --patch_huber_cosine \
        --patch_huber_weight ${PATCH_HUBER_WEIGHT} \
        --patch_huber_cos_weight ${PATCH_HUBER_COS_WEIGHT} \
        --output_weight ${OUTPUT_WEIGHT} \
        --output_camera_weight ${OUTPUT_CAMERA_WEIGHT} \
        --output_depth_weight ${OUTPUT_DEPTH_WEIGHT} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_workers 2 \
        --lr ${LR} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --lr_scheduler cosine \
        --warmup_ratio 0.15 \
        --eta_min 1e-7 \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}/${DATASET}" \
        --log_interval 1 \
        --save_interval 0

    echo ""
    echo "Training complete! Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

# =============================================================================
# Benchmark Functions (single call per setting, multi-seed)
# =============================================================================

get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4v)  echo "8 4" ;;
        8v)  echo "8 0" ;;
        16v) echo "16 0" ;;
        32v) echo "32 0" ;;
        64v) echo "64 0" ;;
    esac
}

benchmark_base_setting() {
    local SETTING=$1
    local SEED_LIST=$2  # space-separated seeds

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [Base ${SETTING}] ${DATASET}, seeds: ${SEED_LIST}"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --base_model ${MODEL_NAME} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seeds ${SEED_LIST} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/base_${SETTING}/${DATASET}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

benchmark_lora_setting() {
    local SETTING=$1
    local SEED_LIST=$2  # space-separated seeds
    local EPOCH=$3
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"

    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: LoRA weights not found at ${LORA_PATH}. Skipping."
        return 0
    fi

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [LoRA epoch=${EPOCH} ${SETTING}] ${DATASET}, seeds: ${SEED_LIST}"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --lora_path ${LORA_PATH} \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seeds ${SEED_LIST} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/lora_epoch${EPOCH}_${SETTING}/${DATASET}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking baseline VGGT on ${DATASET} (seeds: ${BENCHMARK_SEEDS})"
    echo "============================================================"

    for SETTING in 4v 8v 16v 32v 64v; do
        benchmark_base_setting "${SETTING}" "${BENCHMARK_SEEDS}"
    done

    echo "Baseline benchmark complete!"
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking LoRA models on ${DATASET} (seeds: ${LORA_BENCHMARK_SEEDS})"
    echo "============================================================"

    # Find all epoch files
    shopt -s nullglob
    local LORA_FILES=( "${OUTPUT_DIR}/${DATASET}/epoch_"*_lora.pt )
    shopt -u nullglob

    local EPOCHS=()
    for f in "${LORA_FILES[@]}"; do
        local bn
        bn="$(basename "${f}")"
        if [[ "${bn}" =~ ^epoch_([0-9]+)_lora\.pt$ ]]; then
            EPOCHS+=( "${BASH_REMATCH[1]}" )
        fi
    done

    if [ "${#EPOCHS[@]}" -eq 0 ]; then
        echo "WARNING: No LoRA epoch weights found under ${OUTPUT_DIR}/${DATASET}/. Skipping."
        return 0
    fi

    mapfile -t EPOCHS < <(printf '%s\n' "${EPOCHS[@]}" | sort -n)

    echo "  Epochs: ${EPOCHS[@]}"

    for EPOCH in "${EPOCHS[@]}"; do
        for SETTING in 4v 8v 16v 32v 64v; do
            benchmark_lora_setting "${SETTING}" "${LORA_BENCHMARK_SEEDS}" "${EPOCH}"
        done
    done

    echo "LoRA benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train            - Train on eth3d (output loss only)"
    echo "  benchmark_base   - Benchmark baseline VGGT"
    echo "  benchmark_lora   - Benchmark LoRA models"
    echo "  benchmark        - Both benchmarks"
    echo "  all              - Train + benchmark (default)"
    echo ""
    echo "Benchmark seeds: baseline=${BENCHMARK_SEEDS}, lora=${LORA_BENCHMARK_SEEDS}"
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
    all)
        run_baseline_benchmark
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
echo "=== VGGT ETH3D Output Loss Experiment Complete ==="
echo "============================================================"
echo "  Loss: output only (cam=${OUTPUT_CAMERA_WEIGHT}, depth=${OUTPUT_DEPTH_WEIGHT})"
echo "  Benchmark seeds: baseline=${BENCHMARK_SEEDS}, lora=${LORA_BENCHMARK_SEEDS}"
echo "  Training:   ${OUTPUT_DIR}/${DATASET}/"
echo "  Benchmarks: ${BENCHMARK_ROOT}/"
echo "============================================================"

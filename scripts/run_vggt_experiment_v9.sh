#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V9: Stride-2 Consecutive Window Sampling
# =============================================================================
#
# This experiment:
# - Trains on scannetpp only
# - Sampling: for each sample, randomly pick an anchor frame, then take
#   8 views at stride 2 from the sorted file list (anchor, +2, +4, ..., +14)
# - 10 samples per scene (10 random anchors per scene)
# - Loss: all-token softmax KL + cosine (same as V3/V6)
# - Student sees 4 views (indices [0,2,4,6] from the 8 teacher views)
#
# The stride-2 window ensures temporal consistency: each 8-view set covers
# a local temporal neighborhood (every other file in the sorted list).
#
# Training config:
#   scannetpp: 10 samples/scene, seeds 40-49, stride=2
#
# Usage:
#   ./scripts/run_vggt_experiment_v9.sh [train|benchmark_lora|all]
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v9
BENCHMARK_ROOT=./workspace/vggt_experiment_v9
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings
EPOCHS=1
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"

# Stride sampling settings
STRIDE=2
SAMPLES_PER_SCENE=10
SEEDS="$(seq -s ' ' 30 39)"

# Loss settings (all-token softmax KL + cosine)
KL_WEIGHT=1.0
COS_WEIGHT=2.0

# Benchmark settings (maxframe only)
BENCHMARK_SEEDS="42"
ALL_DATASETS="scannetpp"
MODES="pose recon_unposed"

# DA3-style image size (longest side, aspect ratio preserved)
IMAGE_SIZE=504

# =============================================================================
# Training Functions
# =============================================================================

train_scannetpp() {
    local DATASET="scannetpp"

    echo ""
    echo "============================================================"
    echo "Training: ${DATASET} with stride-2 consecutive window sampling"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES_PER_SCENE} (10 random anchors)"
    echo "  Seeds: ${SEEDS}"
    echo "  Stride: ${STRIDE} (every 2nd file in sorted list)"
    echo "  Teacher views: ${NUM_VIEWS}, Student views: 4"
    echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT})"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES_PER_SCENE} \
        --seeds_list ${SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --output_layers ${OUTPUT_LAYERS} \
        --stride_sampling \
        --stride ${STRIDE} \
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

    echo ""
    echo "Training on ${DATASET} complete!"
    echo "  Model saved to: ${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training: V9 stride-2 consecutive window sampling"
    echo "  Dataset: ${ALL_DATASETS}"
    echo "============================================================"

    train_scannetpp

    echo ""
    echo "============================================================"
    echo "Training complete! Model saved to:"
    echo "  - ${OUTPUT_DIR}/scannetpp/epoch_0_lora.pt"
    echo "============================================================"
}

# =============================================================================
# Benchmark Functions
# =============================================================================

get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        maxframe)
            echo "100 0"  # max_frames=100, eval_frames=0 (use all available up to 100)
            ;;
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

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking LoRA model"
    echo "  Setting: maxframe"
    echo "  Dataset: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

        if [ ! -f "${LORA_PATH}" ]; then
            echo "WARNING: LoRA weights not found: ${LORA_PATH}"
            echo "Skipping ${DATASET}"
            continue
        fi

        echo ""
        echo "--- ${DATASET} ---"

        for SEED in ${BENCHMARK_SEEDS}; do
            benchmark_lora_setting "${DATASET}" "maxframe" "${SEED}"
        done
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
    echo "  train           - Train LoRA model with stride-2 sampling"
    echo "  benchmark_lora  - Benchmark LoRA model"
    echo "  all             - Run all steps"
    echo ""
    echo "Key settings:"
    echo "  Dataset: scannetpp"
    echo "  Sampling: random anchor + 8 views at stride 2 from sorted file list"
    echo "  Samples per scene: ${SAMPLES_PER_SCENE} (10 random anchors)"
    echo "  Loss: all-token softmax KL + cosine"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style, aspect ratio preserved)"
    echo "  Benchmark setting: maxframe only"
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
echo "=== VGGT Experiment V9 Complete ==="
echo "============================================================"
echo ""
echo "Experiment: Stride-2 consecutive window sampling"
echo "  Sampling: random anchor + 8 views at stride ${STRIDE}"
echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT})"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}/{dataset}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}/lora_maxframe/{dataset}/"
echo ""

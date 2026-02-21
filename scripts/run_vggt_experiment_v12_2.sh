#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V12.2: Paired Anchor + Companion Sampling on ScanNet++
# =============================================================================
#
# This experiment:
# - Trains on scannetpp only
# - Loss: all-token softmax KL + cosine + cross-frame RKD (same as V12)
# - Paired sampling: 4 random anchors (student frames) + 4 companions at gap 3
#   in the sorted file list (teacher-only frames)
# - 15 samples/scene, seeds 30-44
#
# Training config:
#   scannetpp: 5 samples/scene, seeds 30 31 32 33 34
#
# Usage:
#   ./scripts/run_vggt_experiment_v12_2.sh [train|benchmark_lora|all]
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v12_2
BENCHMARK_ROOT=./workspace/vggt_experiment_v12_2
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings
EPOCHS=1
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"

# Paired sampling settings
PAIRED_GAP=2

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
BENCHMARK_SEEDS="79"
ALL_DATASETS="scannetpp"
MODES="pose recon_unposed"

# DA3-style image size (longest side, aspect ratio preserved)
IMAGE_SIZE=504

# =============================================================================
# Training Functions
# =============================================================================

train_scannetpp() {
    local DATASET="scannetpp"
    local SAMPLES=5
    local SEEDS="$(seq -s ' ' 79 83)"  # 79 80 81 82 83

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET}"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Paired sampling: gap=${PAIRED_GAP}"
    echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT}) + RKD (w=${RKD_WEIGHT})"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --paired_sampling \
        --paired_gap ${PAIRED_GAP} \
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
    echo "Training on ${DATASET} complete!"
    echo "  Model saved to: ${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training: scannetpp (paired sampling, gap=${PAIRED_GAP})"
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
    echo "  train           - Train LoRA model on scannetpp"
    echo "  benchmark_lora  - Benchmark LoRA model"
    echo "  all             - Run all steps"
    echo ""
    echo "Key settings:"
    echo "  Dataset: scannetpp"
    echo "  Sampling: paired anchor+companion (gap=${PAIRED_GAP})"
    echo "  Loss: all-token softmax KL + cosine + cross-frame RKD"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style, aspect ratio preserved)"
    echo "  Benchmark setting: maxframe only"
    echo ""
    echo "Training config:"
    echo "  scannetpp: 5 samples/scene, seeds 30-34"
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
echo "=== VGGT Experiment V12.2 Complete ==="
echo "============================================================"
echo ""
echo "Experiment: KL + cosine + RKD on scannetpp (paired sampling, gap=${PAIRED_GAP})"
echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT}) + RKD (w=${RKD_WEIGHT}, topk=${RKD_TOPK})"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}/scannetpp"
echo "  Benchmarks:  ${BENCHMARK_ROOT}/lora_maxframe/scannetpp/"
echo ""

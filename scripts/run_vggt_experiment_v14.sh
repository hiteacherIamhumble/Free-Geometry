#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V14: KL + Cosine + RKD on All 4 Datasets
# =============================================================================
#
# This experiment:
# - Trains on all 4 datasets with KL + cosine + cross-frame RKD loss
# - No baseline benchmark
# - LoRA benchmark only: 4v, 8v, maxframe with seeds 42, 43, 44
#
# Training configs (per-dataset):
#   scannetpp: 5 samples/scene, seeds 30-34, 2 epochs (from V12)
#   hiroom:    2 samples/scene, seeds 30 31, 1 epoch (from V11)
#   7scenes:   30 samples/scene, seeds 30-59, 1 epoch (from V11)
#   eth3d:     5 samples/scene, seeds 30-34, 1 epoch (from V10)
#
# Usage:
#   ./scripts/run_vggt_experiment_v14.sh [train|benchmark_lora|all]
#   ./scripts/run_vggt_experiment_v14.sh train_scannetpp
#   ./scripts/run_vggt_experiment_v14.sh train_hiroom
#   ./scripts/run_vggt_experiment_v14.sh train_7scenes
#   ./scripts/run_vggt_experiment_v14.sh train_eth3d
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v14
BENCHMARK_ROOT=./workspace/vggt_experiment_v14
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Common training settings
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
BENCHMARK_SEEDS="42 43 44"
ALL_DATASETS="scannetpp hiroom 7scenes eth3d"
MODES="pose recon_unposed"

# DA3-style image size (longest side, aspect ratio preserved)
IMAGE_SIZE=504

# =============================================================================
# Training Functions
# =============================================================================

# Per-dataset config: "samples epochs seed_start seed_end"
# (seeds are generated as seq seed_start seed_end)
get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        scannetpp)
            # V12: 5 samples/scene, seeds 30-34, 2 epochs
            echo "5 2 30 34"
            ;;
        hiroom)
            # V11: 2 samples/scene, seeds 30 31, 1 epoch
            echo "2 1 30 31"
            ;;
        7scenes)
            # V11: 30 samples/scene, seeds 30-59, 1 epoch
            echo "30 1 30 59"
            ;;
        eth3d)
            # V10: 5 samples/scene, seeds 30-34, 1 epoch
            echo "5 1 30 34"
            ;;
    esac
}

train_single_dataset() {
    local DATASET=$1
    local CONFIG=$(get_training_config "${DATASET}")
    local SAMPLES=$(echo ${CONFIG} | cut -d' ' -f1)
    local DS_EPOCHS=$(echo ${CONFIG} | cut -d' ' -f2)
    local SEED_START=$(echo ${CONFIG} | cut -d' ' -f3)
    local SEED_END=$(echo ${CONFIG} | cut -d' ' -f4)
    local SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET}"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Epochs: ${DS_EPOCHS}"
    echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT}) + RKD (w=${RKD_WEIGHT})"
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
        --cross_frame_rkd \
        --rkd_weight ${RKD_WEIGHT} \
        --rkd_topk ${RKD_TOPK} \
        --rkd_num_ref_samples ${RKD_NUM_REF_SAMPLES} \
        --rkd_num_shared_samples ${RKD_NUM_SHARED_SAMPLES} \
        --rkd_angle1_weight ${RKD_ANGLE1_WEIGHT} \
        --rkd_angle2_weight ${RKD_ANGLE2_WEIGHT} \
        --rkd_angle3_weight ${RKD_ANGLE3_WEIGHT} \
        --epochs ${DS_EPOCHS} \
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
    echo "  Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training: 4 datasets"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        train_single_dataset "${DATASET}"
    done

    echo ""
    echo "============================================================"
    echo "All training complete! Models saved to:"
    for DATASET in ${ALL_DATASETS}; do
        echo "  - ${OUTPUT_DIR}/${DATASET}/"
    done
    echo "============================================================"
}

# =============================================================================
# Benchmark Functions (LoRA only, no baseline)
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

benchmark_lora_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED=$3
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

    # For scannetpp with 2 epochs, use epoch_1
    if [ "${DATASET}" = "scannetpp" ] && [ -f "${OUTPUT_DIR}/${DATASET}/epoch_1_lora.pt" ]; then
        LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_1_lora.pt"
    fi

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
    echo "Benchmarking LoRA models (no baseline)"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

        # For scannetpp with 2 epochs, check epoch_1
        if [ "${DATASET}" = "scannetpp" ] && [ -f "${OUTPUT_DIR}/${DATASET}/epoch_1_lora.pt" ]; then
            LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_1_lora.pt"
        fi

        if [ ! -f "${LORA_PATH}" ]; then
            echo "WARNING: LoRA weights not found for ${DATASET}"
            echo "Skipping ${DATASET}"
            continue
        fi

        echo ""
        echo "--- ${DATASET} (using $(basename ${LORA_PATH})) ---"

        for SETTING in 4v 8v maxframe; do
            for SEED in ${BENCHMARK_SEEDS}; do
                benchmark_lora_setting "${DATASET}" "${SETTING}" "${SEED}"
            done
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
    echo "  train             - Train all 4 datasets"
    echo "  train_scannetpp   - Train scannetpp only"
    echo "  train_hiroom      - Train hiroom only"
    echo "  train_7scenes     - Train 7scenes only"
    echo "  train_eth3d       - Train eth3d only"
    echo "  benchmark_lora    - Benchmark all LoRA models (4v, 8v, maxframe)"
    echo "  all               - Train all + benchmark all"
    echo ""
    echo "Key settings:"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Loss: all-token softmax KL + cosine + cross-frame RKD"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style, aspect ratio preserved)"
    echo "  Benchmark: LoRA only, 4v/8v/maxframe, seeds ${BENCHMARK_SEEDS}"
    echo ""
    echo "Training configs:"
    echo "  scannetpp: 5 samples/scene, seeds 30-34, 2 epochs"
    echo "  hiroom:    2 samples/scene, seeds 30-31, 1 epoch"
    echo "  7scenes:   30 samples/scene, seeds 30-59, 1 epoch"
    echo "  eth3d:     5 samples/scene, seeds 30-34, 1 epoch"
    echo ""
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    train_scannetpp)
        train_single_dataset "scannetpp"
        ;;
    train_hiroom)
        train_single_dataset "hiroom"
        ;;
    train_7scenes)
        train_single_dataset "7scenes"
        ;;
    train_eth3d)
        train_single_dataset "eth3d"
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
echo "=== VGGT Experiment V14 Complete ==="
echo "============================================================"
echo ""
echo "Experiment: KL + cosine + RKD on all 4 datasets"
echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT}) + RKD (w=${RKD_WEIGHT}, topk=${RKD_TOPK})"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}/{dataset}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}/lora_{4v,8v,maxframe}/{dataset}/"
echo ""

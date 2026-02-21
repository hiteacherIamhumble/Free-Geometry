#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V15: Fixed-Subset Training on ScanNet++ and 7Scenes
# =============================================================================
#
# This experiment:
# - Restricts training to a fixed 100-frame subset per scene (seed=43)
# - Uses the same shuffle logic as the benchmark evaluator
# - Trains on ScanNet++ and 7Scenes only
# - Loss: KL + cosine + cross-frame RKD (same as V14)
# - LoRA benchmark only: 4v, 8v, maxframe with seeds 42, 43, 44
#
# Training configs (per-dataset):
#   scannetpp: 5 samples/scene, seeds 40-44, 1 epoch
#   7scenes:   10 samples/scene, seeds 40-49, 1 epoch
#
# Usage:
#   ./scripts/run_vggt_experiment_v15.sh [train|benchmark_lora|all]
#   ./scripts/run_vggt_experiment_v15.sh train_scannetpp
#   ./scripts/run_vggt_experiment_v15.sh train_7scenes
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v15
BENCHMARK_ROOT=./workspace/vggt_experiment_v15
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Common training settings
LR=1e-4
LR_SCHEDULER=cosine
WARMUP_RATIO=0.1
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"

# Fixed subset settings
FIXED_SUBSET_SEED=43
FIXED_SUBSET_MAX_FRAMES=100

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
BENCHMARK_SEEDS="43"
ALL_DATASETS="scannetpp 7scenes"
MODES="pose recon_unposed"

# DA3-style image size (longest side, aspect ratio preserved)
IMAGE_SIZE=504

# =============================================================================
# Training Functions
# =============================================================================

# Per-dataset config: "samples epochs seed_start seed_end num_scenes"
# (seeds are generated as seq seed_start seed_end)
get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        scannetpp)
            echo "5 2 30 39 20"
            ;;
        7scenes)
            echo "5 2 30 49 7"
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
    local NUM_SCENES=$(echo ${CONFIG} | cut -d' ' -f5)
    local SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    # Compute warmup steps: total_steps * warmup_ratio
    local TOTAL_STEPS=$(( SAMPLES * NUM_SCENES * DS_EPOCHS ))
    local WARMUP_STEPS=$(awk "BEGIN {printf \"%d\", ${TOTAL_STEPS} * ${WARMUP_RATIO}}")

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET}"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Epochs: ${DS_EPOCHS}"
    echo "  Fixed subset: seed=${FIXED_SUBSET_SEED}, max_frames=${FIXED_SUBSET_MAX_FRAMES}"
    echo "  LR: ${LR}, scheduler: ${LR_SCHEDULER}, warmup: ${WARMUP_STEPS}/${TOTAL_STEPS} steps"
    echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT}) + RKD (w=${RKD_WEIGHT})"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --fixed_subset_seed ${FIXED_SUBSET_SEED} \
        --fixed_subset_max_frames ${FIXED_SUBSET_MAX_FRAMES} \
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
        --lr_scheduler ${LR_SCHEDULER} \
        --warmup_steps ${WARMUP_STEPS} \
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
    echo "Training: ${ALL_DATASETS}"
    echo "  Fixed subset: seed=${FIXED_SUBSET_SEED}, max_frames=${FIXED_SUBSET_MAX_FRAMES}"
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
        maxframe)
            echo "100 0"  # max_frames=100, eval_frames=0 (use all available up to 100)
            ;;
    esac
}

benchmark_lora_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED=$3

    # Use the last epoch checkpoint
    local LORA_PATH=$(ls -1 "${OUTPUT_DIR}/${DATASET}"/epoch_*_lora.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)

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
    echo "  Settings: maxframe"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        local LORA_PATH=$(ls -1 "${OUTPUT_DIR}/${DATASET}"/epoch_*_lora.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)

        if [ -z "${LORA_PATH}" ]; then
            echo "WARNING: LoRA weights not found for ${DATASET}"
            echo "Skipping ${DATASET}"
            continue
        fi

        echo ""
        echo "--- ${DATASET} (using $(basename ${LORA_PATH})) ---"

        for SETTING in maxframe; do
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
    echo "  train             - Train scannetpp + 7scenes"
    echo "  train_scannetpp   - Train scannetpp only"
    echo "  train_7scenes     - Train 7scenes only"
    echo "  benchmark_lora    - Benchmark all LoRA models (maxframe)"
    echo "  all               - Train all + benchmark all"
    echo ""
    echo "Key settings:"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Fixed subset: seed=${FIXED_SUBSET_SEED}, max_frames=${FIXED_SUBSET_MAX_FRAMES}"
    echo "  Loss: all-token softmax KL + cosine + cross-frame RKD"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style, aspect ratio preserved)"
    echo "  Benchmark: LoRA only, maxframe, seeds ${BENCHMARK_SEEDS}"
    echo ""
    echo "Training configs:"
    echo "  scannetpp: 5 samples/scene, seeds 40-44, 1 epoch"
    echo "  7scenes:   10 samples/scene, seeds 40-49, 1 epoch"
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
    train_7scenes)
        train_single_dataset "7scenes"
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
echo "=== VGGT Experiment V15 Complete ==="
echo "============================================================"
echo ""
echo "Experiment: Fixed-subset training (seed=${FIXED_SUBSET_SEED}, max=${FIXED_SUBSET_MAX_FRAMES}) on ScanNet++ and 7Scenes"
echo "  Loss: all-token softmax KL (w=${KL_WEIGHT}) + cosine (w=${COS_WEIGHT}) + RKD (w=${RKD_WEIGHT}, topk=${RKD_TOPK})"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}/{dataset}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}/lora_maxframe/{dataset}/"
echo ""

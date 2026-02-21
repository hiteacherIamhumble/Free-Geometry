#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Combined Feature + Output Loss on All 4 Datasets
# =============================================================================
#
# Feature loss (V14): KL + cosine + cross-frame RKD
# Output loss (ablation_v2): camera + depth + point
# AMP enabled, maxframe=50
#
# Usage:
#   ./scripts/run_combined_all_datasets.sh [command]
#
# Commands:
#   train / train_{scannetpp,hiroom,7scenes,eth3d}
#   benchmark_lora / benchmark_baseline / benchmark_all
#   all  (train + benchmark_all)
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/combined_all_datasets
BENCHMARK_ROOT=./workspace/combined_all_datasets

# Common training settings
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"
IMAGE_SIZE=504

# Feature loss (V14 settings)
KL_WEIGHT=1.0
COS_WEIGHT=2.0
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF_SAMPLES=256
RKD_NUM_SHARED_SAMPLES=256
RKD_ANGLE1_WEIGHT=1.0
RKD_ANGLE2_WEIGHT=1.0
RKD_ANGLE3_WEIGHT=1.0

# Output loss (ablation_v2 settings)
OUTPUT_WEIGHT=1.0
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=1.0
POINT_WEIGHT=1.0

# Benchmark settings
BENCHMARK_SEEDS="42 43 44"
ALL_DATASETS="scannetpp hiroom 7scenes eth3d"
MODES="pose recon_unposed"

# =============================================================================
# Training
# =============================================================================

# Per-dataset config: "samples epochs seed_start seed_end"
get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        scannetpp) echo "10 1 30 39" ;;
        hiroom)    echo "2 1 30 31" ;;
        7scenes)   echo "30 1 30 59" ;;
        eth3d)     echo "5 1 30 34" ;;
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
    echo "Training on ${DATASET} (AMP)"
    echo "  Samples/scene: ${SAMPLES}, Seeds: ${SEEDS}, Epochs: ${DS_EPOCHS}"
    echo "  Loss: feature (KL+cos+RKD) + output (cam+depth+point)"
    echo "============================================================"

    python ./scripts/train_vggt_combined_distill.py \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --output_layers ${OUTPUT_LAYERS} \
        --image_size ${IMAGE_SIZE} \
        --use_amp \
        --all_token_kl_weight ${KL_WEIGHT} \
        --all_token_cos_weight ${COS_WEIGHT} \
        --rkd_weight ${RKD_WEIGHT} \
        --rkd_topk ${RKD_TOPK} \
        --rkd_num_ref_samples ${RKD_NUM_REF_SAMPLES} \
        --rkd_num_shared_samples ${RKD_NUM_SHARED_SAMPLES} \
        --rkd_angle1_weight ${RKD_ANGLE1_WEIGHT} \
        --rkd_angle2_weight ${RKD_ANGLE2_WEIGHT} \
        --rkd_angle3_weight ${RKD_ANGLE3_WEIGHT} \
        --output_weight ${OUTPUT_WEIGHT} \
        --camera_weight ${CAMERA_WEIGHT} \
        --depth_weight ${DEPTH_WEIGHT} \
        --point_weight ${POINT_WEIGHT} \
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

    echo "Training on ${DATASET} complete! Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training all 4 datasets: ${ALL_DATASETS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        train_single_dataset "${DATASET}"
    done

    echo ""
    echo "All training complete!"
}

# =============================================================================
# Benchmark
# =============================================================================

get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4v)       echo "8 4" ;;   # max_frames=8, eval_frames=4
        8v)       echo "8 0" ;;   # max_frames=8, eval_frames=0
        maxframe) echo "50 0" ;;  # max_frames=50, eval_frames=0
    esac
}

get_lora_path() {
    local DATASET=$1
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"
    # scannetpp trains 2 epochs, use epoch_1 if available
    if [ "${DATASET}" = "scannetpp" ] && [ -f "${OUTPUT_DIR}/${DATASET}/epoch_1_lora.pt" ]; then
        LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_1_lora.pt"
    fi
    echo "${LORA_PATH}"
}

benchmark_lora_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED=$3
    local LORA_PATH=$(get_lora_path "${DATASET}")

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

benchmark_baseline_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED=$3

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [Baseline ${SETTING}] ${DATASET}, seed=${SEED}"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/baseline_${SETTING}/${DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking LoRA models"
    echo "  Settings: 4v, 8v, maxframe (max_frames=50)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        local LORA_PATH=$(get_lora_path "${DATASET}")
        if [ ! -f "${LORA_PATH}" ]; then
            echo "WARNING: LoRA weights not found for ${DATASET} at ${LORA_PATH}. Skipping."
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
    echo "LoRA benchmark complete!"
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking baseline (original VGGT, no LoRA)"
    echo "  Settings: 4v, 8v, maxframe (max_frames=50)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        echo ""
        echo "--- ${DATASET} (baseline) ---"

        for SETTING in 4v 8v maxframe; do
            for SEED in ${BENCHMARK_SEEDS}; do
                benchmark_baseline_setting "${DATASET}" "${SETTING}" "${SEED}"
            done
        done
    done

    echo ""
    echo "Baseline benchmark complete!"
}

run_all_benchmarks() {
    run_lora_benchmark
    run_baseline_benchmark
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train               - Train all 4 datasets"
    echo "  train_scannetpp     - Train scannetpp only"
    echo "  train_hiroom        - Train hiroom only"
    echo "  train_7scenes       - Train 7scenes only"
    echo "  train_eth3d         - Train eth3d only"
    echo "  benchmark_lora      - Benchmark all LoRA models (4v, 8v, maxframe)"
    echo "  benchmark_baseline  - Benchmark original VGGT (4v, 8v, maxframe)"
    echo "  benchmark_all       - Benchmark LoRA + baseline"
    echo "  all                 - Train all + benchmark all"
    echo ""
    echo "Loss: feature (KL=${KL_WEIGHT}, cos=${COS_WEIGHT}, RKD=${RKD_WEIGHT})"
    echo "    + output  (cam=${CAMERA_WEIGHT}, depth=${DEPTH_WEIGHT}, point=${POINT_WEIGHT})"
    echo "AMP: enabled | maxframe: 50 | Seeds: ${BENCHMARK_SEEDS}"
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
    benchmark_baseline)
        run_baseline_benchmark
        ;;
    benchmark_all)
        run_all_benchmarks
        ;;
    all)
        run_training
        run_all_benchmarks
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
echo "=== Combined All Datasets Experiment Complete ==="
echo "============================================================"
echo ""
echo "  Training:   ${OUTPUT_DIR}/{dataset}"
echo "  LoRA bench: ${BENCHMARK_ROOT}/lora_{4v,8v,maxframe}/{dataset}/"
echo "  Baseline:   ${BENCHMARK_ROOT}/baseline_{4v,8v,maxframe}/{dataset}/"
echo ""

#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Experiment V4: ScanNet++ and 7scenes with Consecutive Window Sampling
# =============================================================================
#
# This script:
# 1. Trains on ScanNet++ with 10 samples per scene, 20% subset, consecutive windows
# 2. Trains on 7scenes with 30 samples per scene, 5% subset, consecutive windows
# 3. Benchmarks LoRA with 4v, 8v, maxframe settings (baseline from V3)
# 4. Generates comparison tables
#
# Key difference from V3:
# - Uses consecutive window sampling from subset (not random)
# - ScanNet++: 20% subset ratio (handles smaller scenes)
# - 7scenes: 5% subset ratio
#
# Usage:
#   ./scripts/run_vggt_experiment_v4.sh [train|benchmark|compare|all]
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_experiment_v4
BENCHMARK_ROOT=./workspace/vggt_experiment_v4
DATA_BASE=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset

# Training settings
EPOCHS=1
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"
IMAGE_SIZE=504

# Loss settings
KL_WEIGHT=1.0
COS_WEIGHT=2.0

# Benchmark settings
BENCHMARK_SEEDS="43 44"
MODES="pose recon_unposed"

# NEW: Subset sampling settings
SUBSET_RATIO_SCANNETPP=0.20  # 20% of frames for ScanNet++
SUBSET_RATIO_7SCENES=0.10    # 10% of frames for 7scenes

# =============================================================================
# Training Functions
# =============================================================================

train_scannetpp() {
    local DATASET="scannetpp"
    # 10 samples per scene, seeds 40-49
    local SAMPLES=10
    local SEEDS="$(seq -s ' ' 40 49)"

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET} with SUBSET SAMPLING"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Subset ratio: ${SUBSET_RATIO_SCANNETPP} (20% of frames)"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --subset_sampling \
        --subset_ratio ${SUBSET_RATIO_SCANNETPP} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
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
        --output_dir "${OUTPUT_DIR}/${DATASET}" \
        --log_interval 1 \
        --save_interval 0
}

train_7scenes() {
    local DATASET="7scenes"
    # 30 samples per scene, seeds 40-69
    local SAMPLES=30
    local SEEDS="$(seq -s ' ' 40 69)"

    echo ""
    echo "============================================================"
    echo "Training on ${DATASET} with SUBSET SAMPLING"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Seeds: ${SEEDS}"
    echo "  Subset ratio: ${SUBSET_RATIO_7SCENES} (10% of frames)"
    echo "============================================================"

    python ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --subset_sampling \
        --subset_ratio ${SUBSET_RATIO_7SCENES} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
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
        --output_dir "${OUTPUT_DIR}/${DATASET}" \
        --log_interval 1 \
        --save_interval 0
}

run_training() {
    echo ""
    echo "============================================================"
    echo "STEP 1: Training distilled models with CONSECUTIVE WINDOW SAMPLING"
    echo "  ScanNet++: 10 samples per scene, 20% subset (seeds 40-49)"
    echo "  7scenes: 30 samples per scene, 10% subset (seeds 40-69)"
    echo "  Image preprocessing: DA3-style (${IMAGE_SIZE}px)"
    echo "============================================================"

    train_scannetpp
    train_7scenes

    echo ""
    echo "============================================================"
    echo "Training complete! Models saved to:"
    echo "  - ${OUTPUT_DIR}/scannetpp/epoch_0_lora.pt"
    echo "  - ${OUTPUT_DIR}/7scenes/epoch_0_lora.pt"
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
    local DATASET=$1
    local SETTING=$2
    local SEED=$3

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
    local TRAIN_DATASET=$1
    local SETTING=$2
    local SEED=$3
    local LORA_PATH=$4

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    # Get subset ratio based on dataset
    local SUBSET_RATIO
    if [ "${TRAIN_DATASET}" = "scannetpp" ]; then
        SUBSET_RATIO=${SUBSET_RATIO_SCANNETPP}
    else
        SUBSET_RATIO=${SUBSET_RATIO_7SCENES}
    fi

    echo "  [LoRA-${TRAIN_DATASET} ${SETTING}] ${TRAIN_DATASET}, seed=${SEED}"

    local CMD="python ./scripts/benchmark_lora_vggt.py \
        --lora_path ${LORA_PATH} \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${TRAIN_DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/lora_${SETTING}/${TRAIN_DATASET}/seed${SEED}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    # Use subset sampling for 4v and 8v settings (not maxframe)
    if [ "${SETTING}" = "4v" ] || [ "${SETTING}" = "8v" ]; then
        CMD="${CMD} --subset_sampling --subset_ratio ${SUBSET_RATIO}"
    fi

    eval ${CMD}
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "STEP 2: Baseline benchmark SKIPPED"
    echo "  (Using existing baseline results from V3)"
    echo "============================================================"
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "STEP 3: Benchmarking LoRA models (each on its own dataset)"
    echo "  Settings: 4v, 8v, maxframe"
    echo "  Datasets: scannetpp, 7scenes"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for TRAIN_DATASET in scannetpp 7scenes; do
        LORA_PATH="${OUTPUT_DIR}/${TRAIN_DATASET}/epoch_0_lora.pt"

        if [ ! -f "${LORA_PATH}" ]; then
            echo "WARNING: LoRA weights not found: ${LORA_PATH}"
            echo "Skipping ${TRAIN_DATASET}"
            continue
        fi

        echo ""
        echo "=== LoRA trained on ${TRAIN_DATASET}, eval on ${TRAIN_DATASET} ==="

        for SETTING in 4v 8v maxframe; do
            echo ""
            echo "--- ${TRAIN_DATASET} LoRA ${SETTING} benchmark ---"
            for SEED in ${BENCHMARK_SEEDS}; do
                benchmark_lora_setting "${TRAIN_DATASET}" "${SETTING}" "${SEED}" "${LORA_PATH}"
            done
        done
    done

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
    echo "STEP 3: Generating comparison tables"
    echo "  (Using baseline results from V3)"
    echo "============================================================"

    for DATASET in scannetpp 7scenes; do
        echo "Generating comparison for ${DATASET}..."
        python ./scripts/generate_dataset_comparison.py \
            --benchmark_root "${BENCHMARK_ROOT}" \
            --baseline_root "./workspace/vggt_experiment_v3" \
            --dataset "${DATASET}" \
            --settings 4v 8v maxframe \
            --seeds ${BENCHMARK_SEEDS} \
            --output "${BENCHMARK_ROOT}/comparison/comparison_${DATASET}.png"
    done

    echo ""
    echo "============================================================"
    echo "Comparison complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to:"
    echo "  ${BENCHMARK_ROOT}/comparison/comparison_scannetpp.png"
    echo "  ${BENCHMARK_ROOT}/comparison/comparison_7scenes.png"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train           - Train 2 distilled models (scannetpp, 7scenes)"
    echo "  train_scannetpp - Train only scannetpp"
    echo "  train_7scenes   - Train only 7scenes"
    echo "  benchmark_lora  - Benchmark LoRA models"
    echo "  benchmark       - Run LoRA benchmarks (baseline skipped)"
    echo "  compare         - Generate comparison tables"
    echo "  all             - Run all steps"
    echo ""
    echo "Key settings:"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style, aspect ratio preserved)"
    echo "  ScanNet++: 10 samples/scene, 20% subset (seeds 40-49)"
    echo "  7scenes: 30 samples/scene, 10% subset (seeds 40-69)"
    echo "  Consecutive window sampling from subset"
    echo "  Benchmark settings: 4v, 8v, maxframe"
    echo ""
    echo "Examples:"
    echo "  $0 train           # Only train"
    echo "  $0 benchmark       # Only benchmark (requires trained models)"
    echo "  $0 compare         # Only compare (requires benchmark results)"
    echo "  $0 all             # Run complete experiment"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    train_scannetpp)
        train_scannetpp
        ;;
    train_7scenes)
        train_7scenes
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
echo "=== VGGT Experiment V4 Complete ==="
echo "============================================================"
echo ""
echo "Configuration:"
echo "  ScanNet++: 10 samples/scene, 20% subset (seeds 40-49)"
echo "  7scenes: 30 samples/scene, 10% subset (seeds 40-69)"
echo "  Consecutive window sampling from subset"
echo "  DA3-style preprocessing (${IMAGE_SIZE}px)"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}"
echo "  Comparisons: ${BENCHMARK_ROOT}/comparison/"
echo ""

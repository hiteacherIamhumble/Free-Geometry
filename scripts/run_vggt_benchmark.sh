#!/usr/bin/env bash
set -e

# VGGT Benchmark Evaluation Script
# This script benchmarks LoRA-finetuned VGGT models on various datasets

TRAIN_ROOT=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset
BASE_MODEL=facebook/vggt-1b
LORA_RANK=16
LORA_ALPHA=16
IMAGE_SIZE=518

# Default settings
MAX_FRAMES=8
MODES="pose"

run_benchmark() {
    local LORA_PATH=$1
    local DATASETS=$2
    local WORK_DIR=$3
    local EXTRA_ARGS=${4:-""}

    echo "=== Running VGGT Benchmark ==="
    echo "  LoRA: ${LORA_PATH}"
    echo "  Datasets: ${DATASETS}"
    echo "  Work dir: ${WORK_DIR}"
    echo "  Max frames: ${MAX_FRAMES}"
    echo "  Modes: ${MODES}"

    python ./scripts/benchmark_lora_vggt.py \
        --lora_path ${LORA_PATH} \
        --base_model ${BASE_MODEL} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --datasets ${DATASETS} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${WORK_DIR} \
        ${EXTRA_ARGS}
}

run_base_model() {
    local DATASETS=$1
    local WORK_DIR=$2

    echo "=== Running Base VGGT Benchmark (no LoRA) ==="
    echo "  Model: ${BASE_MODEL}"
    echo "  Datasets: ${DATASETS}"
    echo "  Work dir: ${WORK_DIR}"

    python ./scripts/benchmark_lora_vggt.py \
        --base_model ${BASE_MODEL} \
        --datasets ${DATASETS} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${WORK_DIR}
}

# Benchmark on ScanNet++
benchmark_scannetpp() {
    local LORA_PATH=$1
    local EPOCHS=${2:-2}
    local LR=${3:-1e-4}

    WORK_DIR=./workspace/vggt_eval_scannetpp_${EPOCHS}epoch_${LR}

    run_benchmark \
        "${LORA_PATH}" \
        "scannetpp" \
        "${WORK_DIR}"
}

# Benchmark on ETH3D
benchmark_eth3d() {
    local LORA_PATH=$1
    local EPOCHS=${2:-2}
    local LR=${3:-1e-4}

    WORK_DIR=./workspace/vggt_eval_eth3d_${EPOCHS}epoch_${LR}

    run_benchmark \
        "${LORA_PATH}" \
        "eth3d" \
        "${WORK_DIR}"
}

# Benchmark on 7Scenes
benchmark_7scenes() {
    local LORA_PATH=$1
    local EPOCHS=${2:-2}
    local LR=${3:-1e-4}

    WORK_DIR=./workspace/vggt_eval_7scenes_${EPOCHS}epoch_${LR}

    run_benchmark \
        "${LORA_PATH}" \
        "7scenes" \
        "${WORK_DIR}"
}

# Benchmark on all datasets
benchmark_all() {
    local LORA_PATH=$1
    local EPOCHS=${2:-2}
    local LR=${3:-1e-4}

    WORK_DIR=./workspace/vggt_eval_all_${EPOCHS}epoch_${LR}

    run_benchmark \
        "${LORA_PATH}" \
        "scannetpp eth3d 7scenes" \
        "${WORK_DIR}"
}

# Benchmark with reconstruction
benchmark_with_recon() {
    local LORA_PATH=$1
    local DATASET=$2
    local EPOCHS=${3:-2}
    local LR=${4:-1e-4}

    MODES="pose recon_unposed"
    WORK_DIR=./workspace/vggt_eval_${DATASET}_recon_${EPOCHS}epoch_${LR}

    run_benchmark \
        "${LORA_PATH}" \
        "${DATASET}" \
        "${WORK_DIR}"
}

# Print usage
usage() {
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  scannetpp <lora_path> [epochs] [lr]  - Benchmark on ScanNet++"
    echo "  eth3d <lora_path> [epochs] [lr]      - Benchmark on ETH3D"
    echo "  7scenes <lora_path> [epochs] [lr]    - Benchmark on 7Scenes"
    echo "  all <lora_path> [epochs] [lr]        - Benchmark on all datasets"
    echo "  recon <lora_path> <dataset> [epochs] [lr] - Benchmark with reconstruction"
    echo "  base <datasets>                      - Benchmark base model (no LoRA)"
    echo ""
    echo "Examples:"
    echo "  $0 scannetpp ./checkpoints/vggt_distill/epoch_1_lora.pt 2 1e-4"
    echo "  $0 all ./checkpoints/vggt_distill/epoch_1_lora.pt"
    echo "  $0 base scannetpp"
    echo "  $0 recon ./checkpoints/vggt_distill/epoch_1_lora.pt eth3d"
}

# Main entry point
case "${1:-}" in
    scannetpp)
        benchmark_scannetpp "${2}" "${3:-2}" "${4:-1e-4}"
        ;;
    eth3d)
        benchmark_eth3d "${2}" "${3:-2}" "${4:-1e-4}"
        ;;
    7scenes)
        benchmark_7scenes "${2}" "${3:-2}" "${4:-1e-4}"
        ;;
    all)
        benchmark_all "${2}" "${3:-2}" "${4:-1e-4}"
        ;;
    recon)
        benchmark_with_recon "${2}" "${3}" "${4:-2}" "${5:-1e-4}"
        ;;
    base)
        run_base_model "${2:-scannetpp}" "./workspace/vggt_eval_base"
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo "Benchmark complete."

#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Benchmark Final LoRA Models (DA3 + VGGT)
#
# - 4v/8v/16v: all seeds in one call (faster, model loaded once)
# - 32v: each seed in separate subprocess (prevents OOM)
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
DA3_MODEL="depth-anything/DA3-GIANT-1.1"
VGGT_MODEL="facebook/vggt-1b"

DA3_CHECKPOINT_DIR="./checkpoints/da3_lora_final"
VGGT_CHECKPOINT_DIR="./checkpoints/vggt_lora_final"

DA3_WORK_DIR="./workspace/da3_final_benchmark"
VGGT_WORK_DIR="./workspace/vggt_final_benchmark"

# LoRA settings
DA3_LORA_RANK=32
DA3_LORA_ALPHA=32
VGGT_LORA_RANK=32
VGGT_LORA_ALPHA=32
VGGT_LORA_LAYERS_START=0

# Benchmark settings
SEEDS="43 44 45"
VIEW_SETTINGS="${VIEW_SETTINGS:-4 8 16 32}"
MODES="pose recon_unposed"

DA3_DATASETS="${DA3_DATASETS:-eth3d scannetpp 7scenes hiroom}"
VGGT_DATASETS="eth3d scannetpp 7scenes hiroom"
VGGT_IMAGE_SIZE=504

# =============================================================================
# DA3 Benchmark
# =============================================================================

benchmark_da3_single() {
    local DATASET=$1
    local MAX_FRAMES=$2

    local LORA_PATH="${DA3_CHECKPOINT_DIR}/${DATASET}/lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: DA3 LoRA not found at ${LORA_PATH}. Skipping."
        return 0
    fi

    if [ "${MAX_FRAMES}" -ge 32 ]; then
        # 32v: run each seed separately to prevent OOM
        echo "  [DA3] ${DATASET}, ${MAX_FRAMES}v — per-seed mode"
        for SEED in ${SEEDS}; do
            local SEED_WORK_DIR="${DA3_WORK_DIR}/${DATASET}/frames_${MAX_FRAMES}/seed${SEED}"
            if [ -f "${SEED_WORK_DIR}/metrics.json" ]; then
                echo "    [Seed ${SEED}] Already completed. Skipping."
                continue
            fi
            echo "    [Seed ${SEED}]"
            python -u ./scripts/benchmark_lora.py \
                --lora_path "${LORA_PATH}" \
                --base_model "${DA3_MODEL}" \
                --lora_rank ${DA3_LORA_RANK} \
                --lora_alpha ${DA3_LORA_ALPHA} \
                --datasets "${DATASET}" \
                --modes ${MODES} \
                --max_frames ${MAX_FRAMES} \
                --seeds ${SEED} \
                --work_dir "${SEED_WORK_DIR}"
            echo "    [Seed ${SEED}] Done."
        done
    else
        # 4v/8v/16v: all seeds in one call
        echo "  [DA3] ${DATASET}, ${MAX_FRAMES}v, seeds: ${SEEDS}"
        python -u ./scripts/benchmark_lora.py \
            --lora_path "${LORA_PATH}" \
            --base_model "${DA3_MODEL}" \
            --lora_rank ${DA3_LORA_RANK} \
            --lora_alpha ${DA3_LORA_ALPHA} \
            --datasets "${DATASET}" \
            --modes ${MODES} \
            --max_frames ${MAX_FRAMES} \
            --seeds ${SEEDS} \
            --work_dir "${DA3_WORK_DIR}/${DATASET}/frames_${MAX_FRAMES}"
    fi
}

run_da3_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 Final LoRA Models"
    echo "  Datasets: ${DA3_DATASETS}"
    echo "  View settings: ${VIEW_SETTINGS}"
    echo "  Seeds: ${SEEDS}"
    echo "============================================================"

    for DATASET in ${DA3_DATASETS}; do
        echo ""
        echo "=== DA3: ${DATASET} ==="
        for MAX_FRAMES in ${VIEW_SETTINGS}; do
            benchmark_da3_single "${DATASET}" "${MAX_FRAMES}"
        done
    done

    echo ""
    echo "DA3 benchmark complete!"
}

# =============================================================================
# VGGT Benchmark
# =============================================================================

get_vggt_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4)  echo "8 4" ;;
        8)  echo "8 0" ;;
        16) echo "16 0" ;;
        32) echo "32 0" ;;
    esac
}

benchmark_vggt_single() {
    local DATASET=$1
    local VIEW_SETTING=$2

    local LORA_PATH="${VGGT_CHECKPOINT_DIR}/${DATASET}/lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: VGGT LoRA not found at ${LORA_PATH}. Skipping."
        return 0
    fi

    local PARAMS=$(get_vggt_benchmark_params "${VIEW_SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    local EVAL_FRAMES_ARG=""
    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        EVAL_FRAMES_ARG="--eval_frames ${EVAL_FRAMES}"
    fi

    if [ "${VIEW_SETTING}" -ge 32 ]; then
        # 32v: run each seed separately to prevent OOM
        echo "  [VGGT] ${DATASET}, ${VIEW_SETTING}v — per-seed mode"
        for SEED in ${SEEDS}; do
            local SEED_WORK_DIR="${VGGT_WORK_DIR}/${DATASET}/${VIEW_SETTING}v/seed${SEED}"
            if [ -f "${SEED_WORK_DIR}/metrics.json" ]; then
                echo "    [Seed ${SEED}] Already completed. Skipping."
                continue
            fi
            echo "    [Seed ${SEED}]"
            python -u ./scripts/benchmark_lora_vggt.py \
                --lora_path "${LORA_PATH}" \
                --base_model ${VGGT_MODEL} \
                --lora_rank ${VGGT_LORA_RANK} \
                --lora_alpha ${VGGT_LORA_ALPHA} \
                --lora_layers_start ${VGGT_LORA_LAYERS_START} \
                --datasets ${DATASET} \
                --modes ${MODES} \
                --max_frames ${MAX_FRAMES} \
                --seeds ${SEED} \
                --image_size ${VGGT_IMAGE_SIZE} \
                ${EVAL_FRAMES_ARG} \
                --work_dir "${SEED_WORK_DIR}"
            echo "    [Seed ${SEED}] Done."
        done
    else
        # 4v/8v/16v: all seeds in one call
        echo "  [VGGT] ${DATASET}, ${VIEW_SETTING}v, seeds: ${SEEDS}"
        python -u ./scripts/benchmark_lora_vggt.py \
            --lora_path "${LORA_PATH}" \
            --base_model ${VGGT_MODEL} \
            --lora_rank ${VGGT_LORA_RANK} \
            --lora_alpha ${VGGT_LORA_ALPHA} \
            --lora_layers_start ${VGGT_LORA_LAYERS_START} \
            --datasets ${DATASET} \
            --modes ${MODES} \
            --max_frames ${MAX_FRAMES} \
            --seeds ${SEEDS} \
            --image_size ${VGGT_IMAGE_SIZE} \
            ${EVAL_FRAMES_ARG} \
            --work_dir "${VGGT_WORK_DIR}/${DATASET}/${VIEW_SETTING}v"
    fi
}

run_vggt_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking VGGT Final LoRA Models"
    echo "  Datasets: ${VGGT_DATASETS}"
    echo "  View settings: ${VIEW_SETTINGS}"
    echo "  Seeds: ${SEEDS}"
    echo "  Image size: ${VGGT_IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for DATASET in ${VGGT_DATASETS}; do
        echo ""
        echo "=== VGGT: ${DATASET} ==="
        for VIEW_SETTING in ${VIEW_SETTINGS}; do
            benchmark_vggt_single "${DATASET}" "${VIEW_SETTING}"
        done
    done

    echo ""
    echo "VGGT benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  da3              - Benchmark DA3 final models only"
    echo "  vggt             - Benchmark VGGT final models only"
    echo "  all              - Benchmark both DA3 and VGGT (default)"
    echo ""
    echo "  4v/8v/16v: all seeds in one call"
    echo "  32v: each seed in separate subprocess (OOM prevention)"
    echo ""
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    da3)
        run_da3_benchmark
        ;;
    vggt)
        run_vggt_benchmark
        ;;
    all)
        run_da3_benchmark
        run_vggt_benchmark
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
echo "=== Final Model Benchmark Complete ==="
echo "============================================================"
echo ""
echo "Output directories:"
echo "  DA3:  ${DA3_WORK_DIR}/{dataset}/frames_{4,8,16,32}/"
echo "  VGGT: ${VGGT_WORK_DIR}/{dataset}/{4v,8v,16v,32v}/"
echo ""

#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS="${DATASETS:-hiroom scannetpp}"
VIEW_COUNTS="${VIEW_COUNTS:-8 16 32}"
SEED="${SEED:-43}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

DA3_MODEL_NAME="${DA3_MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
VGGT_MODEL_NAME="${VGGT_MODEL_NAME:-facebook/vggt-1b}"
VGGT_IMAGE_SIZE="${VGGT_IMAGE_SIZE:-504}"

DA3_WORK_DIR="${DA3_WORK_DIR:-./results/multiview_da3_hiroom_scannetpp}"
VGGT_WORK_DIR="${VGGT_WORK_DIR:-./results/multiview_vggt_hiroom_scannetpp}"

# Default LoRA roots. The benchmark script resolves `<root>/<dataset>/lora.pt`.
USE_LORA="${USE_LORA:-1}"
DA3_LORA_ROOT="${DA3_LORA_ROOT:-./checkpoints/da3_lora_final}"
VGGT_LORA_ROOT="${VGGT_LORA_ROOT:-./checkpoints/vggt_lora_final}"
DA3_LORA_PATH="${DA3_LORA_PATH:-}"
VGGT_LORA_PATH="${VGGT_LORA_PATH:-}"
VGGT_LORA_RANK="${VGGT_LORA_RANK:-32}"
VGGT_LORA_ALPHA="${VGGT_LORA_ALPHA:-32.0}"
VGGT_LORA_LAYERS_START="${VGGT_LORA_LAYERS_START:-0}"

mkdir -p logs

build_optional_args() {
    local -n out_ref=$1
    local lora_path=$2
    local lora_root=$3

    out_ref=()
    if [[ "${USE_LORA}" == "1" ]]; then
        if [[ -n "${lora_path}" ]]; then
            out_ref+=(--lora_path "${lora_path}")
        else
            out_ref+=(--lora_root "${lora_root}")
        fi
    else
        out_ref+=(--no_lora)
    fi
    if [[ "${SKIP_EXISTING}" == "1" ]]; then
        out_ref+=(--skip_existing)
    fi
}

run_da3() {
    local da3_args=()
    build_optional_args da3_args "${DA3_LORA_PATH}" "${DA3_LORA_ROOT}"

    echo ""
    echo "============================================================"
    echo "DA3 multi-view benchmark"
    echo "  Datasets: ${DATASETS}"
    echo "  View counts: ${VIEW_COUNTS}"
    echo "  Seed: ${SEED}"
    echo "  Work dir: ${DA3_WORK_DIR}"
    if [[ "${USE_LORA}" == "1" ]]; then
        if [[ -n "${DA3_LORA_PATH}" ]]; then
            echo "  Mode: LoRA (${DA3_LORA_PATH})"
        else
            echo "  Mode: LoRA root (${DA3_LORA_ROOT}/{dataset}/lora.pt)"
        fi
    else
        echo "  Mode: base model"
    fi
    echo "============================================================"

    python -u ./scripts/benchmark_multiview_all_datasets.py \
        --model_family da3 \
        --datasets ${DATASETS} \
        --view_counts ${VIEW_COUNTS} \
        --seed "${SEED}" \
        --model_name "${DA3_MODEL_NAME}" \
        --work_dir "${DA3_WORK_DIR}" \
        "${da3_args[@]}"
}

run_vggt() {
    local vggt_args=()
    build_optional_args vggt_args "${VGGT_LORA_PATH}" "${VGGT_LORA_ROOT}"

    if [[ "${USE_LORA}" == "1" ]]; then
        vggt_args+=(
            --lora_rank "${VGGT_LORA_RANK}"
            --lora_alpha "${VGGT_LORA_ALPHA}"
            --lora_layers_start "${VGGT_LORA_LAYERS_START}"
        )
    fi

    echo ""
    echo "============================================================"
    echo "VGGT multi-view benchmark"
    echo "  Datasets: ${DATASETS}"
    echo "  View counts: ${VIEW_COUNTS}"
    echo "  Seed: ${SEED}"
    echo "  Work dir: ${VGGT_WORK_DIR}"
    echo "  Image size: ${VGGT_IMAGE_SIZE}"
    if [[ "${USE_LORA}" == "1" ]]; then
        if [[ -n "${VGGT_LORA_PATH}" ]]; then
            echo "  Mode: LoRA (${VGGT_LORA_PATH})"
        else
            echo "  Mode: LoRA root (${VGGT_LORA_ROOT}/{dataset}/lora.pt)"
        fi
    else
        echo "  Mode: base model"
    fi
    echo "============================================================"

    python -u ./scripts/benchmark_multiview_all_datasets.py \
        --model_family vggt \
        --datasets ${DATASETS} \
        --view_counts ${VIEW_COUNTS} \
        --seed "${SEED}" \
        --model_name "${VGGT_MODEL_NAME}" \
        --image_size "${VGGT_IMAGE_SIZE}" \
        --work_dir "${VGGT_WORK_DIR}" \
        "${vggt_args[@]}"
}

usage() {
    echo "Usage: $0 [da3|vggt|all]"
    echo ""
    echo "Defaults:"
    echo "  datasets: hiroom scannetpp"
    echo "  view counts: 8 16 32"
    echo ""
    echo "Optional env vars:"
    echo "  USE_LORA=0|1"
    echo "  DA3_LORA_ROOT, VGGT_LORA_ROOT"
    echo "  DA3_LORA_PATH, VGGT_LORA_PATH"
    echo "  DA3_WORK_DIR, VGGT_WORK_DIR"
    echo "  SKIP_EXISTING=0|1"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    da3)
        run_da3
        ;;
    vggt)
        run_vggt
        ;;
    all)
        run_da3
        run_vggt
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
echo "Completed multi-view benchmark run."
echo "DA3 results:  ${DA3_WORK_DIR}"
echo "VGGT results: ${VGGT_WORK_DIR}"

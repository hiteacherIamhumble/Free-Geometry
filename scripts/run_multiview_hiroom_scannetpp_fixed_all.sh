#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate da3
PYTHON_BIN="${CONDA_PREFIX:-/root/miniconda3/envs/da3}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: da3 env python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

DATASETS=(hiroom scannetpp)
VIEW_COUNTS=(8 16 32)
SEED="${SEED:-43}"

DA3_BASE_WORK_DIR="${DA3_BASE_WORK_DIR:-/root/autodl-tmp/da3/results/multiview_da3_hiroom_scannetpp_base_fixed}"
DA3_LORA_WORK_DIR="${DA3_LORA_WORK_DIR:-/root/autodl-tmp/da3/results/multiview_da3_hiroom_scannetpp_lora_fixed}"
VGGT_BASE_WORK_DIR="${VGGT_BASE_WORK_DIR:-/root/autodl-tmp/da3/results/multiview_vggt_hiroom_scannetpp_base_fixed}"
VGGT_LORA_WORK_DIR="${VGGT_LORA_WORK_DIR:-/root/autodl-tmp/da3/results/multiview_vggt_hiroom_scannetpp_lora_fixed}"

run_job() {
    local title="$1"
    shift
    echo
    echo "============================================================"
    echo "${title}"
    echo "============================================================"
    "${PYTHON_BIN}" scripts/benchmark_multiview_all_datasets.py "$@"
}

run_job \
    "DA3 base: hiroom + scannetpp @ 8v/16v/32v" \
    --model_family da3 \
    --no_lora \
    --datasets "${DATASETS[@]}" \
    --all_scenes \
    --view_counts "${VIEW_COUNTS[@]}" \
    --seed "${SEED}" \
    --work_dir "${DA3_BASE_WORK_DIR}"

run_job \
    "DA3 LoRA: hiroom + scannetpp @ 8v/16v/32v" \
    --model_family da3 \
    --datasets "${DATASETS[@]}" \
    --all_scenes \
    --view_counts "${VIEW_COUNTS[@]}" \
    --seed "${SEED}" \
    --work_dir "${DA3_LORA_WORK_DIR}"

run_job \
    "VGGT base: hiroom + scannetpp @ 8v/16v/32v" \
    --model_family vggt \
    --no_lora \
    --datasets "${DATASETS[@]}" \
    --all_scenes \
    --view_counts "${VIEW_COUNTS[@]}" \
    --seed "${SEED}" \
    --work_dir "${VGGT_BASE_WORK_DIR}"

run_job \
    "VGGT LoRA: hiroom + scannetpp @ 8v/16v/32v" \
    --model_family vggt \
    --datasets "${DATASETS[@]}" \
    --all_scenes \
    --view_counts "${VIEW_COUNTS[@]}" \
    --seed "${SEED}" \
    --work_dir "${VGGT_LORA_WORK_DIR}"

echo
echo "Finished all four fixed multiview benchmark runs."
echo "DA3 base:  ${DA3_BASE_WORK_DIR}"
echo "DA3 LoRA:  ${DA3_LORA_WORK_DIR}"
echo "VGGT base: ${VGGT_BASE_WORK_DIR}"
echo "VGGT LoRA: ${VGGT_LORA_WORK_DIR}"

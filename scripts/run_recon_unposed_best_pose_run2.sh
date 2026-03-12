#!/usr/bin/env bash
set -euo pipefail

# Re-run recon_unposed evaluation (eval-only) at the best pose epochs from run2.
# Best epochs were selected by max AUC@3 on 4-view pose:
# - eth3d: 1
# - scannetpp: 1
# - hiroom: 2
# - 7scenes: 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/root/miniconda3/envs/da3/bin/python"
BASE_MODEL="depth-anything/DA3-GIANT-1.1"
MAX_FRAMES=4
SEED=43

run_one() {
    local DATASET="$1"
    local EPOCH="$2"
    local WEIGHTS="./checkpoints/da3_finetune_run2/${DATASET}/epoch_${EPOCH}_lora.pt"
    local WORK_DIR="./workspace/da3_finetune_run2/epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"

    echo ""
    echo "============================================================"
    echo "Recon_unposed eval-only: ${DATASET}, epoch=${EPOCH}, max_frames=${MAX_FRAMES}"
    echo "  work_dir=${WORK_DIR}"
    echo "============================================================"

    "${PYTHON_BIN}" -u ./scripts/benchmark_lora.py \
        --lora_path "${WEIGHTS}" \
        --base_model "${BASE_MODEL}" \
        --finetune \
        --datasets "${DATASET}" \
        --modes recon_unposed \
        --max_frames "${MAX_FRAMES}" \
        --seed "${SEED}" \
        --work_dir "${WORK_DIR}" \
        --eval_only
}

run_one "eth3d" "1"
run_one "scannetpp" "1"
run_one "hiroom" "2"
run_one "7scenes" "2"

echo ""
echo "ALL_DONE recon_unposed best-pose epochs"

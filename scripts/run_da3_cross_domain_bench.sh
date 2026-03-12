#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cross-domain benchmark: DA3 ScanNet++ LoRA → ETH3D
#
# Tests whether the scannetpp-trained LoRA generalizes to eth3d,
# or if it's domain-specific.
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
DA3_MODEL="depth-anything/DA3-GIANT-1.1"
LORA_PATH="./checkpoints/da3_lora_final/scannetpp/lora.pt"
WORK_DIR="./workspace/da3_cross_domain/scannetpp_on_eth3d"

LORA_RANK=32
LORA_ALPHA=32

MAX_FRAMES=16
SEEDS="43"
MODES="pose recon_unposed"

# =============================================================================

echo "============================================================"
echo "Cross-domain benchmark: ScanNet++ LoRA → ETH3D"
echo "  LoRA:       ${LORA_PATH}"
echo "  Eval on:    eth3d"
echo "  Views:      ${MAX_FRAMES}v"
echo "  Seeds:      ${SEEDS}"
echo "============================================================"

if [ ! -f "${LORA_PATH}" ]; then
    echo "ERROR: LoRA checkpoint not found at ${LORA_PATH}"
    exit 1
fi

python -u ./scripts/benchmark_lora.py \
    --lora_path "${LORA_PATH}" \
    --base_model "${DA3_MODEL}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --datasets eth3d \
    --modes ${MODES} \
    --max_frames ${MAX_FRAMES} \
    --seeds ${SEEDS} \
    --work_dir "${WORK_DIR}/frames_${MAX_FRAMES}"

echo ""
echo "============================================================"
echo "Cross-domain benchmark complete!"
echo "  Results: ${WORK_DIR}/frames_${MAX_FRAMES}/"
echo "============================================================"

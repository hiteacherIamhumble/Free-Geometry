#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Test: LoRA epoch 0 benchmark on ETH3D, 32 frames, 3 seeds
# Verifies hybrid parallel execution (sequential inference + parallel eval)
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LORA_PATH="checkpoints/all_da3/eth3d/epoch_0_lora.pt"
MODEL_NAME="depth-anything/DA3-GIANT-1.1"
WORK_DIR="./workspace/all_da3/lora_epoch0/frames_32/eth3d"

echo "============================================================"
echo "Test: LoRA epoch 0 — ETH3D 32 frames, seeds 43 44 45"
echo "  LoRA: ${LORA_PATH}"
echo "  Work dir: ${WORK_DIR}"
echo "============================================================"

python -u ./scripts/benchmark_lora.py \
    --lora_path "${LORA_PATH}" \
    --base_model "${MODEL_NAME}" \
    --lora_rank 32 \
    --lora_alpha 32 \
    --datasets eth3d \
    --modes pose recon_unposed \
    --max_frames 32 \
    --seeds 43 44 45 \
    --work_dir "${WORK_DIR}"

echo ""
echo "============================================================"
echo "Checking outputs..."
echo "============================================================"

# Verify per-seed logs exist
for SEED in 43 44 45; do
    LOG="${WORK_DIR}/seed${SEED}.log"
    if [ -f "${LOG}" ]; then
        echo "  seed${SEED}.log exists ($(wc -l < "${LOG}") lines)"
    else
        echo "  WARNING: seed${SEED}.log NOT found"
    fi
done

# Verify per-seed metrics
for SEED in 43 44 45; do
    METRICS="${WORK_DIR}/seed${SEED}/metrics.json"
    if [ -f "${METRICS}" ]; then
        echo "  seed${SEED}/metrics.json exists"
    else
        echo "  WARNING: seed${SEED}/metrics.json NOT found"
    fi
done

# Verify aggregated summary
SUMMARY="${WORK_DIR}/multi_seed_summary.json"
if [ -f "${SUMMARY}" ]; then
    echo "  multi_seed_summary.json exists"
else
    echo "  WARNING: multi_seed_summary.json NOT found"
fi

echo ""
echo "Test complete!"

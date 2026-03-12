#!/usr/bin/env bash
set -euo pipefail

# Benchmark scannetpp lora_5s at 4v, 8v, 50v
# (100v already done in workspace/da3_combined_all/lora_5s/scannetpp/)

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
LORA_PATH="./checkpoints/da3_combined_all/scannetpp_5s/epoch_0_lora.pt"
LORA_RANK=16
LORA_ALPHA=16
MODES="pose recon_unposed"

for VIEWS in 4 8 50; do
    echo ""
    echo "============================================================"
    echo "Scannetpp lora_5s benchmark: max_frames=${VIEWS}"
    echo "============================================================"

    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --datasets scannetpp \
        --modes ${MODES} \
        --max_frames ${VIEWS} \
        --work_dir "./workspace/da3_${VIEWS}v/lora_5s/scannetpp"
done

echo ""
echo "Done! Results at:"
echo "  ./workspace/da3_4v/lora_5s/scannetpp/"
echo "  ./workspace/da3_8v/lora_5s/scannetpp/"
echo "  ./workspace/da3_50v/lora_5s/scannetpp/"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME=facebook/vggt-1b
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
IMAGE_SIZE=504
OUTPUT_DIR=./checkpoints/feature_only_all_datasets
BENCHMARK_ROOT=./workspace/feature_only_all_datasets
MODES="pose recon_unposed"
SEED=43

for DATASET in scannetpp hiroom 7scenes eth3d; do
    LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"
    echo ""
    echo "=== [LoRA maxframe] ${DATASET}, seed=${SEED} ==="
    python ./scripts/benchmark_lora_vggt.py \
        --lora_path "${LORA_PATH}" \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames 50 \
        --seed ${SEED} \
        --image_size ${IMAGE_SIZE} \
        --work_dir "${BENCHMARK_ROOT}/lora_maxframe/${DATASET}/seed${SEED}"
done

echo ""
echo "All benchmarks done!"

#!/usr/bin/env bash
set -euo pipefail

# Run seeds 43 and 44 benchmarks (baseline + best lora) for all datasets
# at 4v/8v/50v/100v
#
# Best configs:
#   scannetpp: lora_5s (5 samples/scene)
#   hiroom:    lora    (2 samples/scene)
#   7scenes:   lora    (30 samples/scene)
#   eth3d:     lora    (5 samples/scene)

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
LORA_RANK=16
LORA_ALPHA=16
MODES="pose recon_unposed"

# Best lora checkpoint per dataset
declare -A LORA_PATHS
LORA_PATHS[scannetpp]="./checkpoints/da3_combined_all/scannetpp_5s/epoch_0_lora.pt"
LORA_PATHS[hiroom]="./checkpoints/da3_combined_all/hiroom/epoch_0_lora.pt"
LORA_PATHS[7scenes]="./checkpoints/da3_combined_all/7scenes/epoch_0_lora.pt"
LORA_PATHS[eth3d]="./checkpoints/da3_combined_all/eth3d/epoch_0_lora.pt"

ALL_DATASETS="scannetpp hiroom 7scenes eth3d"
SEEDS="43 44"
VIEWS_LIST="4 8 50 100"

benchmark_lora() {
    local DATASET=$1
    local MAX_FRAMES=$2
    local SEED=$3
    local WORK_DIR=$4
    local LORA_PATH=${LORA_PATHS[$DATASET]}

    echo "  [LoRA] ${DATASET}, max_frames=${MAX_FRAMES}, seed=${SEED}"
    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --datasets "${DATASET}" \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed ${SEED} \
        --work_dir "${WORK_DIR}"
}

benchmark_baseline() {
    local DATASET=$1
    local MAX_FRAMES=$2
    local SEED=$3
    local WORK_DIR=$4

    echo "  [Baseline] ${DATASET}, max_frames=${MAX_FRAMES}, seed=${SEED}"
    python -u -c "
import sys, os, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)
evaluator = Evaluator(
    work_dir='${WORK_DIR}',
    datas=['${DATASET}'],
    modes=['pose', 'recon_unposed'],
    max_frames=${MAX_FRAMES},
    seed=${SEED},
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
"
}

for SEED in ${SEEDS}; do
    echo ""
    echo "============================================================"
    echo "Seed ${SEED}"
    echo "============================================================"

    for VIEWS in ${VIEWS_LIST}; do
        echo ""
        echo "------------------------------------------------------------"
        echo "max_frames=${VIEWS}, seed=${SEED}"
        echo "------------------------------------------------------------"

        for DATASET in ${ALL_DATASETS}; do
            WORK_BASE="./workspace/seed_runs/seed${SEED}/${VIEWS}v"

            benchmark_baseline "${DATASET}" ${VIEWS} ${SEED} "${WORK_BASE}/baseline/${DATASET}"
            benchmark_lora "${DATASET}" ${VIEWS} ${SEED} "${WORK_BASE}/lora/${DATASET}"
        done
    done
done

echo ""
echo "============================================================"
echo "All done! Results at: ./workspace/seed_runs/seed{43,44}/{4,8,50,100}v/{baseline,lora}/{dataset}/"
echo "============================================================"

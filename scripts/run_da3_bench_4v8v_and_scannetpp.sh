#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 1) Benchmark all 4 datasets at 4v and 8v (LoRA + baseline)
# 2) Retrain scannetpp (5 samples/scene) then re-benchmark at max_frames=100
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
CHECKPOINT_DIR=./checkpoints/da3_combined_all
LORA_RANK=16
LORA_ALPHA=16

ALL_DATASETS="scannetpp hiroom 7scenes eth3d"
MODES="pose recon_unposed"

# =============================================================================
# Part 1: 4v and 8v benchmarks for all datasets
# =============================================================================

benchmark_lora() {
    local DATASET=$1
    local MAX_FRAMES=$2
    local WORK_DIR=$3
    local LORA_PATH="${CHECKPOINT_DIR}/${DATASET}/epoch_0_lora.pt"

    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: LoRA not found at ${LORA_PATH}, skipping."
        return 0
    fi

    echo "  [LoRA] ${DATASET}, max_frames=${MAX_FRAMES}"
    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --datasets "${DATASET}" \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --work_dir "${WORK_DIR}"
}

benchmark_baseline() {
    local DATASET=$1
    local MAX_FRAMES=$2
    local WORK_DIR=$3

    echo "  [Baseline] ${DATASET}, max_frames=${MAX_FRAMES}"
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
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
"
}

echo ""
echo "============================================================"
echo "Part 1: 4v, 8v, 50v benchmarks (LoRA + baseline) x 4 datasets"
echo "============================================================"

for VIEWS in 4 8 50; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Benchmarking with max_frames=${VIEWS}"
    echo "------------------------------------------------------------"

    for DATASET in ${ALL_DATASETS}; do
        benchmark_lora "${DATASET}" ${VIEWS} "./workspace/da3_${VIEWS}v/lora/${DATASET}"
        benchmark_baseline "${DATASET}" ${VIEWS} "./workspace/da3_${VIEWS}v/baseline/${DATASET}"
    done
done

echo ""
echo "Part 1 complete! 4v, 8v, and 50v benchmarks done."

# =============================================================================
# Part 2: Retrain scannetpp (5 samples/scene) + re-benchmark max_frames=100
# =============================================================================

echo ""
echo "============================================================"
echo "Part 2: Retrain scannetpp (5 samples/scene) + benchmark 100v"
echo "============================================================"

SCANNETPP_OUTPUT="${CHECKPOINT_DIR}/scannetpp_5s"

python -u ./scripts/train_distill.py \
    --dataset scannetpp \
    --samples_per_scene 5 \
    --seeds_list 30 31 32 33 34 \
    --use_scannetpp_test_split \
    --model_name ${MODEL_NAME} \
    --num_views 8 \
    --combined_loss \
    --all_token_softmax_kl_cosine \
    --all_token_softmax_kl_weight 1.0 \
    --all_token_softmax_cos_weight 2.0 \
    --rkd_weight 2.0 \
    --rkd_topk 4 \
    --rkd_num_ref_samples 256 \
    --rkd_num_shared_samples 256 \
    --rkd_angle1_weight 1.0 \
    --rkd_angle2_weight 1.0 \
    --rkd_angle3_weight 1.0 \
    --output_weight 2.0 \
    --camera_weight 5.0 \
    --depth_weight 1.0 \
    --depth_gradient_loss grad \
    --depth_valid_range 0.98 \
    --epochs 1 \
    --batch_size 1 \
    --num_workers 2 \
    --lr 1e-4 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir "${SCANNETPP_OUTPUT}"

echo "Scannetpp retrain done!"

# Benchmark the new scannetpp LoRA at max_frames=100
benchmark_lora_path="${SCANNETPP_OUTPUT}/epoch_0_lora.pt"
echo ""
echo "Benchmarking new scannetpp LoRA at max_frames=100..."
python -u ./scripts/benchmark_lora.py \
    --lora_path "${benchmark_lora_path}" \
    --base_model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --datasets scannetpp \
    --modes ${MODES} \
    --max_frames 100 \
    --work_dir "./workspace/da3_combined_all/lora_5s/scannetpp"

echo ""
echo "============================================================"
echo "All done!"
echo "  4v results:  ./workspace/da3_4v/{lora,baseline}/{dataset}/"
echo "  8v results:  ./workspace/da3_8v/{lora,baseline}/{dataset}/"
echo "  50v results: ./workspace/da3_50v/{lora,baseline}/{dataset}/"
echo "  Scannetpp 5s checkpoint: ${SCANNETPP_OUTPUT}/"
echo "  Scannetpp 5s bench:     ./workspace/da3_combined_all/lora_5s/scannetpp/"
echo "============================================================"

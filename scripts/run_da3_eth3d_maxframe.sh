#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DA3 Giant: Train on ETH3D + Benchmark (max-frame setting)
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/da3_eth3d_maxframe
BENCHMARK_ROOT=./workspace/da3_eth3d_maxframe

LORA_RANK=16
LORA_ALPHA=16
LR=1e-4
NUM_VIEWS=8

# Feature loss
KL_WEIGHT=1.0
COS_WEIGHT=2.0
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256

# Output loss
OUTPUT_WEIGHT=2.0
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=1.0
DEPTH_GRAD=grad
DEPTH_VALID_RANGE=0.98

# ETH3D training
ETH3D_SAMPLES=5
ETH3D_SEEDS="30 31 32 33 34"
EPOCHS=1

# Benchmark
MAX_FRAMES=100

# =============================================================================
# Train
# =============================================================================

train() {
    echo ""
    echo "============================================================"
    echo "Training DA3 on ETH3D (combined loss, max-frame)"
    echo "  Samples/scene: ${ETH3D_SAMPLES}, Seeds: ${ETH3D_SEEDS}"
    echo "  Loss: feature (KL+cos+RKD) + output (cam+depth)"
    echo "============================================================"

    python -u ./scripts/train_distill.py \
        --dataset eth3d \
        --samples_per_scene ${ETH3D_SAMPLES} \
        --seeds_list ${ETH3D_SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --combined_loss \
        --all_token_softmax_kl_cosine \
        --all_token_softmax_kl_weight ${KL_WEIGHT} \
        --all_token_softmax_cos_weight ${COS_WEIGHT} \
        --rkd_weight ${RKD_WEIGHT} \
        --rkd_topk ${RKD_TOPK} \
        --rkd_num_ref_samples ${RKD_NUM_REF} \
        --rkd_num_shared_samples ${RKD_NUM_SHARED} \
        --rkd_angle1_weight 1.0 \
        --rkd_angle2_weight 1.0 \
        --rkd_angle3_weight 1.0 \
        --output_weight ${OUTPUT_WEIGHT} \
        --camera_weight ${CAMERA_WEIGHT} \
        --depth_weight ${DEPTH_WEIGHT} \
        --depth_gradient_loss ${DEPTH_GRAD} \
        --depth_valid_range ${DEPTH_VALID_RANGE} \
        --epochs ${EPOCHS} \
        --batch_size 1 \
        --num_workers 2 \
        --lr ${LR} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lr_scheduler none \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}"

    echo "Training complete! Model saved to: ${OUTPUT_DIR}/"
}

# =============================================================================
# Benchmark
# =============================================================================

benchmark_lora() {
    local LORA_PATH="${OUTPUT_DIR}/epoch_0_lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "ERROR: LoRA weights not found at ${LORA_PATH}"
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 LoRA (max_frames=${MAX_FRAMES})"
    echo "============================================================"

    python ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --datasets eth3d \
        --modes pose recon_unposed \
        --max_frames ${MAX_FRAMES} \
        --work_dir "${BENCHMARK_ROOT}/lora/eth3d"

    echo "LoRA benchmark complete!"
}

benchmark_baseline() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 baseline (no LoRA, max_frames=${MAX_FRAMES})"
    echo "============================================================"

    python -c "
import sys, os, torch
sys.path.insert(0, os.path.join('src'))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)
evaluator = Evaluator(
    work_dir='${BENCHMARK_ROOT}/baseline/eth3d',
    datas=['eth3d'],
    modes=['pose', 'recon_unposed'],
    max_frames=${MAX_FRAMES},
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
"

    echo "Baseline benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        train
        ;;
    benchmark_lora)
        benchmark_lora
        ;;
    benchmark_baseline)
        benchmark_baseline
        ;;
    benchmark_all)
        benchmark_lora
        benchmark_baseline
        ;;
    all)
        train
        benchmark_lora
        benchmark_baseline
        ;;
    -h|--help|help)
        echo "Usage: $0 [train|benchmark_lora|benchmark_baseline|benchmark_all|all]"
        echo ""
        echo "  train              - Train DA3 on ETH3D with combined loss"
        echo "  benchmark_lora     - Benchmark LoRA model (max_frames=${MAX_FRAMES})"
        echo "  benchmark_baseline - Benchmark original DA3 (max_frames=${MAX_FRAMES})"
        echo "  benchmark_all      - Both benchmarks"
        echo "  all                - Train + both benchmarks (default)"
        exit 0
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        echo "Usage: $0 [train|benchmark_lora|benchmark_baseline|benchmark_all|all]"
        exit 1
        ;;
esac

echo ""
echo "Done! Results in:"
echo "  Checkpoints: ${OUTPUT_DIR}/"
echo "  Benchmarks:  ${BENCHMARK_ROOT}/"

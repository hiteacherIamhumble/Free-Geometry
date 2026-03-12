#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DA3 Giant: Feature-Only (no output loss) — Train + Benchmark all datasets
#
# Datasets: scannetpp, eth3d, 7scenes, hiroom
# Loss: feature (KL+cos+RKD), output_weight=0
#
# Usage:
#   ./scripts/run_da3_feature_only_all_datasets.sh [command]
#
# Commands:
#   train / train_{scannetpp,eth3d,7scenes,hiroom}
#   benchmark_lora / benchmark_baseline / benchmark_all
#   all  (train all + benchmark all)
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/da3_kl_cos_topk_rkd_l1
BENCHMARK_ROOT=./workspace/da3_kl_cos_topk_rkd_l1

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

# Output loss DISABLED
OUTPUT_WEIGHT=0.0
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=1.0
DEPTH_GRAD=grad
DEPTH_VALID_RANGE=0.98

# Benchmark
MAX_FRAMES=100
ALL_DATASETS="scannetpp hiroom 7scenes eth3d"

# =============================================================================
# Per-dataset config: "samples epochs seed_start seed_end"
# =============================================================================
get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        scannetpp) echo "5 1 30 34" ;;
        hiroom)    echo "2 1 30 31" ;;
        7scenes)   echo "20 1 30 49" ;;
        eth3d)     echo "5 1 30 34" ;;
    esac
}

# =============================================================================
# Train
# =============================================================================

train_single_dataset() {
    local DATASET=$1
    local CONFIG=$(get_training_config "${DATASET}")
    local SAMPLES=$(echo ${CONFIG} | cut -d' ' -f1)
    local DS_EPOCHS=$(echo ${CONFIG} | cut -d' ' -f2)
    local SEED_START=$(echo ${CONFIG} | cut -d' ' -f3)
    local SEED_END=$(echo ${CONFIG} | cut -d' ' -f4)
    local SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    echo ""
    echo "============================================================"
    echo "Training DA3 on ${DATASET} — FEATURE ONLY (no output loss)"
    echo "  Samples/scene: ${SAMPLES}, Seeds: ${SEEDS}, Epochs: ${DS_EPOCHS}"
    echo "  Loss: feature (KL+cos+RKD), output_weight=0"
    echo "============================================================"

    local EXTRA_ARGS=""
    if [ "${DATASET}" = "scannetpp" ]; then
        EXTRA_ARGS="--use_scannetpp_test_split"
    fi

    python -u ./scripts/train_distill.py \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        ${EXTRA_ARGS} \
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
        --epochs ${DS_EPOCHS} \
        --batch_size 1 \
        --num_workers 2 \
        --lr ${LR} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lr_scheduler none \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}/${DATASET}"

    echo "Training on ${DATASET} complete! Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training all datasets (FEATURE ONLY): ${ALL_DATASETS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        train_single_dataset "${DATASET}"
    done

    echo ""
    echo "All training complete!"
}

# =============================================================================
# Benchmark
# =============================================================================

benchmark_lora_single() {
    local DATASET=$1
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: LoRA weights not found for ${DATASET} at ${LORA_PATH}. Skipping."
        return 0
    fi

    echo "  [LoRA] ${DATASET}, max_frames=${MAX_FRAMES}"
    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --datasets "${DATASET}" \
        --modes pose recon_unposed \
        --max_frames ${MAX_FRAMES} \
        --seed 43 \
        --work_dir "${BENCHMARK_ROOT}/lora/${DATASET}"
}

benchmark_baseline_single() {
    local DATASET=$1

    echo "  [Baseline] ${DATASET}, max_frames=${MAX_FRAMES}"
    python -u -c "
import sys, os, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)
evaluator = Evaluator(
    work_dir='${BENCHMARK_ROOT}/baseline/${DATASET}',
    datas=['${DATASET}'],
    modes=['pose', 'recon_unposed'],
    max_frames=${MAX_FRAMES},
    seed=43,
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
"
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 LoRA feature-only (max_frames=${MAX_FRAMES})"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        benchmark_lora_single "${DATASET}"
    done

    echo "LoRA benchmark complete!"
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 baseline (max_frames=${MAX_FRAMES})"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        benchmark_baseline_single "${DATASET}"
    done

    echo "Baseline benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train               - Train all 4 datasets"
    echo "  train_scannetpp     - Train scannetpp only"
    echo "  train_hiroom        - Train hiroom only"
    echo "  train_7scenes       - Train 7scenes only"
    echo "  train_eth3d         - Train eth3d only"
    echo "  benchmark_lora      - Benchmark all LoRA models"
    echo "  benchmark_baseline  - Benchmark original DA3"
    echo "  benchmark_all       - Both benchmarks"
    echo "  all                 - Train all + benchmark all (default)"
    echo ""
    echo "Loss: feature ONLY (KL=${KL_WEIGHT}, cos=${COS_WEIGHT}, RKD=${RKD_WEIGHT})"
    echo "      output_weight=${OUTPUT_WEIGHT} (disabled)"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    train_scannetpp)
        train_single_dataset "scannetpp"
        ;;
    train_hiroom)
        train_single_dataset "hiroom"
        ;;
    train_7scenes)
        train_single_dataset "7scenes"
        ;;
    train_eth3d)
        train_single_dataset "eth3d"
        ;;
    benchmark_lora)
        run_lora_benchmark
        ;;
    benchmark_baseline)
        run_baseline_benchmark
        ;;
    benchmark_all)
        run_lora_benchmark
        run_baseline_benchmark
        ;;
    all)
        run_training
        run_lora_benchmark
        run_baseline_benchmark
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
echo "============================================================"
echo "Done!"
echo "  Checkpoints: ${OUTPUT_DIR}/{dataset}/"
echo "  LoRA bench:  ${BENCHMARK_ROOT}/lora/{dataset}/"
echo "  Baseline:    ${BENCHMARK_ROOT}/baseline/{dataset}/"
echo "============================================================"

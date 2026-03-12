#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DA3 Giant: Tuned LoRA Distillation — per-dataset hyperparameter tuning
#
# Based on analysis of run_da3_strong_lora.sh results:
#   - scannetpp: lower LR (3e-5), fewer samples (3), 1 epoch (was overfitting)
#   - hiroom:    more samples (5), more epochs (5) (was underfitting at E0)
#   - 7scenes:   higher LR (3e-4), rank 64 (small gains suggest more capacity)
#   - eth3d:     lower LR (5e-5), keep rest (was peaking at E0-E1)
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/da3_tuned_lora
BENCHMARK_ROOT=./workspace/da3_tuned_lora

NUM_VIEWS=8

# Feature loss (shared across all datasets)
KL_WEIGHT=1.0
COS_WEIGHT=2.0
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256

# Output loss DISABLED
OUTPUT_WEIGHT=0.0
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=3.0
DEPTH_GRAD=grad
DEPTH_VALID_RANGE=0.98

# Benchmark
MAX_FRAMES_LIST="4 8 16 32 64"

ALL_DATASETS="eth3d scannetpp hiroom 7scenes"

# =============================================================================
# Per-dataset config: "samples epochs seed_start seed_end lr lora_rank lora_alpha"
# =============================================================================
get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        scannetpp) echo "3 1 40 44 3e-5 32 32" ;;
        hiroom)    echo "5 5 42 44 1e-4 32 32" ;;
        7scenes)   echo "10 3 40 49 3e-4 64 64" ;;
        eth3d)     echo "5 3 40 44 5e-5 32 32" ;;
        *)
            echo "ERROR: Unknown dataset '${DATASET}'" >&2
            return 1
            ;;
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
    local DS_LR=$(echo ${CONFIG} | cut -d' ' -f5)
    local DS_LORA_RANK=$(echo ${CONFIG} | cut -d' ' -f6)
    local DS_LORA_ALPHA=$(echo ${CONFIG} | cut -d' ' -f7)
    local SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    echo ""
    echo "============================================================"
    echo "Training DA3 on ${DATASET} — TUNED LoRA DISTILL"
    echo "  Samples/scene: ${SAMPLES}, Seeds: ${SEEDS}, Epochs: ${DS_EPOCHS}"
    echo "  LoRA rank=${DS_LORA_RANK}, alpha=${DS_LORA_ALPHA}, LR=${DS_LR}, cosine scheduler"
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
        --distill_all_layers \
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
        --lr ${DS_LR} \
        --lora_rank ${DS_LORA_RANK} \
        --lora_alpha ${DS_LORA_ALPHA} \
        --lr_scheduler cosine \
        --warmup_steps 50 \
        --weight_decay 1e-5 \
        --resample_per_epoch \
        --output_dir "${OUTPUT_DIR}/${DATASET}"

    echo "Training on ${DATASET} complete! Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training all datasets (TUNED): ${ALL_DATASETS}"
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
    local EPOCH=$2
    local MAX_FRAMES=$3
    local CONFIG=$(get_training_config "${DATASET}")
    local DS_LORA_RANK=$(echo ${CONFIG} | cut -d' ' -f6)
    local DS_LORA_ALPHA=$(echo ${CONFIG} | cut -d' ' -f7)
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"

    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: LoRA weights not found at ${LORA_PATH}. Skipping."
        return 0
    fi

    echo "  [LoRA epoch=${EPOCH}] ${DATASET}, max_frames=${MAX_FRAMES}"
    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${DS_LORA_RANK} \
        --lora_alpha ${DS_LORA_ALPHA} \
        --datasets "${DATASET}" \
        --modes pose recon_unposed \
        --max_frames ${MAX_FRAMES} \
        --seed 43 \
        --work_dir "${BENCHMARK_ROOT}/lora_epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"
}

benchmark_baseline_single() {
    local DATASET=$1
    local MAX_FRAMES=$2

    echo "  [Baseline] ${DATASET}, max_frames=${MAX_FRAMES}"
    python -u -c "
import sys, os, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)
evaluator = Evaluator(
    work_dir='${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}',
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
    echo "Benchmarking DA3 Tuned LoRA — all epochs"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        shopt -s nullglob
        local LORA_FILES=( "${OUTPUT_DIR}/${DATASET}/epoch_"*_lora.pt )
        shopt -u nullglob

        local EPOCHS=()
        for f in "${LORA_FILES[@]}"; do
            local bn
            bn="$(basename "${f}")"
            if [[ "${bn}" =~ ^epoch_([0-9]+)_lora\.pt$ ]]; then
                EPOCHS+=( "${BASH_REMATCH[1]}" )
            fi
        done

        if [ "${#EPOCHS[@]}" -eq 0 ]; then
            echo "WARNING: No LoRA epoch weights found under ${OUTPUT_DIR}/${DATASET}/. Skipping ${DATASET}."
            continue
        fi

        mapfile -t EPOCHS < <(printf '%s\n' "${EPOCHS[@]}" | sort -n)

        for EPOCH in "${EPOCHS[@]}"; do
            for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
                benchmark_lora_single "${DATASET}" "${EPOCH}" "${MAX_FRAMES}"
            done
        done
    done

    echo "LoRA benchmark complete!"
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 baseline"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
            benchmark_baseline_single "${DATASET}" "${MAX_FRAMES}"
        done
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
    echo "  benchmark_all       - Both benchmarks (baseline first)"
    echo "  all                 - Train all + benchmark all (default)"
    echo ""
    echo "Per-dataset tuned configs:"
    echo "  scannetpp:  3 samp, 1 ep, LR=3e-5, rank=32"
    echo "  hiroom:     5 samp, 5 ep, LR=1e-4, rank=32"
    echo "  7scenes:   10 samp, 3 ep, LR=3e-4, rank=64"
    echo "  eth3d:      5 samp, 3 ep, LR=5e-5, rank=32"
    echo ""
    echo "Benchmark frames: ${MAX_FRAMES_LIST}"
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
        run_baseline_benchmark
        run_lora_benchmark
        ;;
    all)
        run_training
        run_baseline_benchmark
        run_lora_benchmark
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
echo "  LoRA bench:  ${BENCHMARK_ROOT}/lora_epoch{epoch}/frames_{max_frames}/{dataset}/"
echo "  Baseline:    ${BENCHMARK_ROOT}/baseline/frames_{max_frames}/{dataset}/"
echo "============================================================"

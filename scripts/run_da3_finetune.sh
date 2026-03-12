#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DA3 Giant: Full Fine-tuning (layers 13-39) + RKD Distance Loss
#
# Same loss setup as run_da3_tuned_lora_with_dist.sh but uses full
# fine-tuning instead of LoRA. Lower LR since ~765M params are unfrozen.
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR="${DA3_OUTPUT_DIR:-./checkpoints/da3_finetune}"
BENCHMARK_ROOT="${DA3_BENCHMARK_ROOT:-./workspace/da3_finetune}"

NUM_VIEWS=8
BATCH_SIZE=4

# Feature loss
KL_WEIGHT=1.0
GAUSSIAN_KL_WEIGHT=1.0
COS_WEIGHT=2.0
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256

# Use Huber+cosine feature loss (best from ablation)
USE_PATCH_HUBER=1
PATCH_HUBER_WEIGHT=1.0
PATCH_HUBER_COS_WEIGHT=2.0
PATCH_HUBER_DELTA=1.0

# RKD Distance loss (KL mode)
RKD_DIST_WEIGHT=1.0
RKD_DIST_TEMP=10.0
RKD_D1_WEIGHT=1.0
RKD_D2_WEIGHT=1.0
RKD_D3_WEIGHT=0.0
RKD_DIST_MODE=kl

# Output loss DISABLED
OUTPUT_WEIGHT=0.0
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=3.0
DEPTH_GRAD=grad
DEPTH_VALID_RANGE=0.98

# Benchmark (4-view only)
MAX_FRAMES_LIST="4"
BENCHMARK_MODES="pose"

ALL_DATASETS="eth3d scannetpp hiroom 7scenes"
ALL_DATASETS_5="eth3d scannetpp hiroom 7scenes dtu"

# =============================================================================
# Per-dataset config: "samples epochs seed_start seed_end lr eta_min"
# LRs are ~10-20x lower than LoRA since we're updating 765M params directly
# =============================================================================
get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        # Use 10 samples/scene (10 fixed seeds).
        scannetpp) echo "10 5 40 49 2e-6 1e-8" ;;
        hiroom)    echo "5 5 42 46 3e-6 1e-8" ;;
        7scenes)   echo "10 5 40 49 5e-6 1e-8" ;;
        # ETH3D override: higher start LR with eta_min=1e-6.
        eth3d)     echo "10 5 40 49 5e-6 1e-6" ;;
        # DTU: 5 samples/scene (5 fixed seeds).
        dtu)       echo "5 5 40 44 2e-6 1e-8" ;;
        *)
            echo "ERROR: Unknown dataset '${DATASET}'" >&2
            return 1
            ;;
    esac
}

get_scene_count() {
    local DATASET=$1
    case "${DATASET}" in
        eth3d) echo "11" ;;
        scannetpp) echo "20" ;;
        hiroom) echo "30" ;;
        7scenes) echo "7" ;;
        dtu) echo "22" ;;
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
    local SAMPLES DS_EPOCHS SEED_START SEED_END DS_LR DS_ETA_MIN
    read -r SAMPLES DS_EPOCHS SEED_START SEED_END DS_LR DS_ETA_MIN <<< "${CONFIG}"
    local SCENE_COUNT NUM_SEEDS STEPS_PER_EPOCH TOTAL_STEPS DS_WARMUP
    SCENE_COUNT=$(get_scene_count "${DATASET}")
    NUM_SEEDS=$((SEED_END - SEED_START + 1))
    STEPS_PER_EPOCH=$(( (SCENE_COUNT * NUM_SEEDS) / BATCH_SIZE ))
    if [ "${STEPS_PER_EPOCH}" -lt 1 ]; then
        STEPS_PER_EPOCH=1
    fi
    TOTAL_STEPS=$((STEPS_PER_EPOCH * DS_EPOCHS))
    # Warmup is 20% of total steps.
    DS_WARMUP=$(( (TOTAL_STEPS + 4) / 5 ))
    local SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    echo ""
    echo "============================================================"
    echo "Training DA3 on ${DATASET} — FULL FINETUNE (layers 13-39)"
    echo "  Samples/scene: ${SAMPLES}, Seeds: ${SEEDS}, Epochs: ${DS_EPOCHS}"
    echo "  LR=${DS_LR}, warmup=${DS_WARMUP}/${TOTAL_STEPS} (20%), eta_min=${DS_ETA_MIN}, batch=${BATCH_SIZE}, cosine scheduler"
    echo "  Feature: Huber+cosine, RKD angle+distance (KL)"
    echo "  Resampling: disabled (fixed seed set per run)"
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
        --finetune \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --combined_loss \
        --patch_huber_cosine \
        --patch_huber_weight ${PATCH_HUBER_WEIGHT} \
        --patch_huber_cos_weight ${PATCH_HUBER_COS_WEIGHT} \
        --patch_huber_delta ${PATCH_HUBER_DELTA} \
        --distill_all_layers \
        --rkd_weight ${RKD_WEIGHT} \
        --rkd_topk ${RKD_TOPK} \
        --rkd_num_ref_samples ${RKD_NUM_REF} \
        --rkd_num_shared_samples ${RKD_NUM_SHARED} \
        --rkd_angle1_weight 1.0 \
        --rkd_angle2_weight 1.0 \
        --rkd_angle3_weight 1.0 \
        --use_rkd_distance \
        --rkd_distance_weight ${RKD_DIST_WEIGHT} \
        --rkd_distance_temperature ${RKD_DIST_TEMP} \
        --rkd_distance_mode ${RKD_DIST_MODE} \
        --rkd_d1_weight ${RKD_D1_WEIGHT} \
        --rkd_d2_weight ${RKD_D2_WEIGHT} \
        --rkd_d3_weight ${RKD_D3_WEIGHT} \
        --output_weight ${OUTPUT_WEIGHT} \
        --camera_weight ${CAMERA_WEIGHT} \
        --depth_weight ${DEPTH_WEIGHT} \
        --depth_gradient_loss ${DEPTH_GRAD} \
        --depth_valid_range ${DEPTH_VALID_RANGE} \
        --epochs ${DS_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_workers 2 \
        --lr ${DS_LR} \
        --lr_scheduler cosine \
        --eta_min ${DS_ETA_MIN} \
        --warmup_steps ${DS_WARMUP} \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}/${DATASET}"

    echo "Training on ${DATASET} complete! Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training all datasets (FULL FINETUNE): ${ALL_DATASETS}"
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

benchmark_finetune_single() {
    local DATASET=$1
    local EPOCH=$2
    local MAX_FRAMES=$3
    local EVAL_DATASETS="${4:-${DATASET}}"
    local WEIGHTS_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"

    if [ ! -f "${WEIGHTS_PATH}" ]; then
        echo "WARNING: Finetune weights not found at ${WEIGHTS_PATH}. Skipping."
        return 0
    fi

    echo "  [Finetune epoch=${EPOCH}] weights=${DATASET}, eval=${EVAL_DATASETS}, max_frames=${MAX_FRAMES}"
    python -u ./scripts/benchmark_lora.py \
        --lora_path "${WEIGHTS_PATH}" \
        --base_model "${MODEL_NAME}" \
        --finetune \
        --datasets ${EVAL_DATASETS} \
        --modes ${BENCHMARK_MODES} \
        --max_frames ${MAX_FRAMES} \
        --seed 43 \
        --work_dir "${BENCHMARK_ROOT}/epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"
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
    datasets=['${DATASET}'],
    modes=['${BENCHMARK_MODES}'],
    max_frames=${MAX_FRAMES},
    seed=43,
    work_dir='${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}',
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
"
}

run_finetune_benchmark_dataset() {
    local DATASET=$1
    shift || true
    local EVAL_DATASETS="${*:-${DATASET}}"

    if ! echo " ${ALL_DATASETS} " | grep -q " ${DATASET} "; then
        echo "ERROR: Unknown dataset '${DATASET}'"
        echo "  Valid datasets: ${ALL_DATASETS}"
        return 1
    fi

    echo "Benchmarking weights from ${DATASET} on: ${EVAL_DATASETS}"

    shopt -s nullglob
    local WEIGHT_FILES=( "${OUTPUT_DIR}/${DATASET}/epoch_"*_lora.pt )
    shopt -u nullglob

    local EPOCHS=()
    for f in "${WEIGHT_FILES[@]}"; do
        local bn
        bn="$(basename "${f}")"
        if [[ "${bn}" =~ ^epoch_([0-9]+)_lora\.pt$ ]]; then
            EPOCHS+=( "${BASH_REMATCH[1]}" )
        fi
    done

    if [ "${#EPOCHS[@]}" -eq 0 ]; then
        echo "WARNING: No finetune weights found under ${OUTPUT_DIR}/${DATASET}/. Skipping."
        return 0
    fi

    mapfile -t EPOCHS < <(printf '%s\n' "${EPOCHS[@]}" | sort -n)

    for EPOCH in "${EPOCHS[@]}"; do
        for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
            benchmark_finetune_single "${DATASET}" "${EPOCH}" "${MAX_FRAMES}" "${EVAL_DATASETS}"
        done
    done
}

run_train_then_benchmark_single_dataset() {
    local DATASET=$1
    train_single_dataset "${DATASET}"
    run_finetune_benchmark_dataset "${DATASET}"
}

run_train_then_benchmark_scannetpp_dtu64() {
    local DATASET="scannetpp"
    train_single_dataset "${DATASET}"
    run_finetune_benchmark_dataset "${DATASET}" "scannetpp dtu64"
}

run_train_then_benchmark_all_datasets() {
    echo ""
    echo "============================================================"
    echo "Train + benchmark (interleaved per dataset)"
    echo "  Order: ${ALL_DATASETS}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "  Modes: ${BENCHMARK_MODES}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        run_train_then_benchmark_single_dataset "${DATASET}"
    done

    echo "Interleaved train+benchmark complete!"
}

run_train_then_benchmark_all_5datasets() {
    echo ""
    echo "============================================================"
    echo "Train + benchmark (interleaved, 5 datasets)"
    echo "  Order: ${ALL_DATASETS_5}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "  Modes: ${BENCHMARK_MODES}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS_5}; do
        train_single_dataset "${DATASET}"
        if [ "${DATASET}" = "dtu" ]; then
            # DTU-64 is pose-only; evaluate dtu-trained weights there.
            run_finetune_benchmark_dataset "${DATASET}" "dtu64"
        else
            run_finetune_benchmark_dataset "${DATASET}"
        fi
    done

    echo "Interleaved 5-dataset train+benchmark complete!"
}

run_finetune_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 Full Finetune — all epochs"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "  Modes: ${BENCHMARK_MODES}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        run_finetune_benchmark_dataset "${DATASET}"
    done

    echo "Finetune benchmark complete!"
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

CMD="${1:-all}"

case "${CMD}" in
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
    train_benchmark)
        DATASET="${2:-}"
        if [ -z "${DATASET}" ]; then
            echo "ERROR: Missing dataset name."
            echo "Usage: $0 train_benchmark [eth3d|scannetpp|hiroom|7scenes]"
            exit 1
        fi
        run_train_then_benchmark_single_dataset "${DATASET}"
        ;;
    train_benchmark_all)
        run_train_then_benchmark_all_datasets
        ;;
    train_benchmark_all_5datasets)
        run_train_then_benchmark_all_5datasets
        ;;
    train_benchmark_scannetpp_dtu64)
        run_train_then_benchmark_scannetpp_dtu64
        ;;
    benchmark)
        run_finetune_benchmark
        ;;
    benchmark_baseline)
        run_baseline_benchmark
        ;;
    benchmark_all)
        run_baseline_benchmark
        run_finetune_benchmark
        ;;
    all)
        run_training
        run_finetune_benchmark
        ;;
    *)
        echo "Unknown command: ${CMD}"
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  train               - Train all 4 datasets"
        echo "  train_scannetpp     - Train scannetpp only"
        echo "  train_hiroom        - Train hiroom only"
        echo "  train_7scenes       - Train 7scenes only"
        echo "  train_eth3d         - Train eth3d only"
        echo "  train_benchmark X   - Train one dataset then benchmark it (X in: eth3d/scannetpp/hiroom/7scenes)"
        echo "  train_benchmark_all - Train each dataset then benchmark it before moving on"
        echo "  train_benchmark_all_5datasets - Interleaved train+benchmark for eth3d/scannetpp/hiroom/7scenes/dtu"
        echo "  train_benchmark_scannetpp_dtu64 - Train ScanNet++ then benchmark pose on ScanNet++ + DTU-64"
        echo "  benchmark           - Benchmark all finetune models"
        echo "  benchmark_baseline  - Benchmark original DA3"
        echo "  benchmark_all       - Both benchmarks"
        echo "  all                 - Train all + benchmark (default)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Done!"
echo "  Checkpoints: ${OUTPUT_DIR}/{dataset}/"
echo "  Finetune:    ${BENCHMARK_ROOT}/epoch{N}/frames_{max_frames}/{dataset}/"
echo "  Baseline:    ${BENCHMARK_ROOT}/baseline/frames_{max_frames}/{dataset}/"
echo "============================================================"

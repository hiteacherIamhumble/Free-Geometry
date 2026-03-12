#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DA3 Giant: Tuned LoRA Distillation + RKD Distance Loss
#
# Same as run_da3_tuned_lora.sh but adds KL-based RKD distance loss
# with T=10, Huber(beta=0.5), d3_weight=0 (d1+d2 only).
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Configuration ===
MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/all_da3_v2
BENCHMARK_ROOT=./workspace/all_da3_v2

NUM_VIEWS=8

# Feature + RKD settings (shared across all datasets)
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256

# Patch Huber + cosine (always enabled)
PATCH_HUBER_WEIGHT=${PATCH_HUBER_WEIGHT:-1.0}
PATCH_HUBER_COS_WEIGHT=${PATCH_HUBER_COS_WEIGHT:-2.0}
PATCH_HUBER_DELTA=${PATCH_HUBER_DELTA:-1.0}

# Full finetune mode (set USE_FINETUNE=1 to unfreeze layers 13-39 instead of LoRA)
USE_FINETUNE=${USE_FINETUNE:-0}
FINETUNE_LR=${FINETUNE_LR:-5e-6}

# RKD Distance loss
RKD_DIST_WEIGHT=1.0
RKD_DIST_TEMP=10.0
RKD_D1_WEIGHT=1.0
RKD_D2_WEIGHT=1.0
RKD_D3_WEIGHT=0.0
RKD_DIST_MODE=kl

# Output loss (disabled for feature-only distillation)
OUTPUT_WEIGHT=0.0
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=3.0
DEPTH_GRAD=grad
DEPTH_VALID_RANGE=0.98


# Benchmark
MAX_FRAMES_LIST="16"
BASELINE_SEEDS="43 44 45"
LORA_SEEDS="${LORA_SEEDS:-43}"

ALL_DATASETS="eth3d scannetpp 7scenes"

# =============================================================================
# Per-dataset config: "samples epochs seed_start seed_end lr lora_rank lora_alpha"
# =============================================================================
get_training_config() {
    local DATASET=$1
    # "samples epochs seed_start seed_end lr lora_rank lora_alpha eta_min"
    case "${DATASET}" in
        scannetpp) echo "10 3 40 49 1e-5 32 32 1e-8" ;;
        7scenes)   echo "20 3 40 49 5e-5 32 32 1e-8" ;;
        eth3d)     echo "10 3 40 44 1e-4 32 32 1e-7" ;;
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
    local DS_ETA_MIN=$(echo ${CONFIG} | cut -d' ' -f8)
    local SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    echo ""
    echo "============================================================"
    echo "Training DA3 on ${DATASET} — TUNED LoRA DISTILL + RKD DIST"
    echo "  Samples/scene: ${SAMPLES}, Seeds: ${SEEDS}, Epochs: ${DS_EPOCHS}"
    echo "  LoRA rank=${DS_LORA_RANK}, alpha=${DS_LORA_ALPHA}, LR=${DS_LR}, cosine scheduler"
    echo "  RKD dist: weight=${RKD_DIST_WEIGHT}, T=${RKD_DIST_TEMP}, d1=${RKD_D1_WEIGHT}, d2=${RKD_D2_WEIGHT}, d3=${RKD_D3_WEIGHT}"
    echo "============================================================"

    local EXTRA_ARGS=""
    if [ "${DATASET}" = "scannetpp" ]; then
        EXTRA_ARGS="--use_scannetpp_test_split"
    fi

    # Finetune mode: override LR and add --finetune flag
    local FINETUNE_ARGS=""
    if [ "${USE_FINETUNE}" = "1" ]; then
        DS_LR=${FINETUNE_LR}
        FINETUNE_ARGS="--finetune"
        echo "  MODE: Full finetune (layers 13-39), LR=${DS_LR}"
    fi

    python -u ./scripts/train_distill.py \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        ${EXTRA_ARGS} \
        ${FINETUNE_ARGS} \
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
        --rkd_selection_mode mixed \
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
        --batch_size 4 \
        --num_workers 2 \
        --lr ${DS_LR} \
        --lora_rank ${DS_LORA_RANK} \
        --lora_alpha ${DS_LORA_ALPHA} \
        --lr_scheduler cosine \
        --warmup_ratio 0.15 \
        --eta_min ${DS_ETA_MIN} \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}/${DATASET}"

    echo "Training on ${DATASET} complete! Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Training all datasets (TUNED + RKD DIST): ${ALL_DATASETS}"
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
    local SEED_LIST=$4  # space-separated seeds
    local CONFIG=$(get_training_config "${DATASET}")
    local DS_LORA_RANK=$(echo ${CONFIG} | cut -d' ' -f6)
    local DS_LORA_ALPHA=$(echo ${CONFIG} | cut -d' ' -f7)
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"

    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: LoRA weights not found at ${LORA_PATH}. Skipping."
        return 0
    fi

    local BENCH_FINETUNE_ARGS=""
    if [ "${USE_FINETUNE}" = "1" ]; then
        BENCH_FINETUNE_ARGS="--finetune"
    fi

    echo "  [LoRA epoch=${EPOCH}] ${DATASET}, max_frames=${MAX_FRAMES}, seeds: ${SEED_LIST}"
    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${DS_LORA_RANK} \
        --lora_alpha ${DS_LORA_ALPHA} \
        ${BENCH_FINETUNE_ARGS} \
        --datasets "${DATASET}" \
        --modes pose recon_unposed \
        --max_frames ${MAX_FRAMES} \
        --seeds ${SEED_LIST} \
        --work_dir "${BENCHMARK_ROOT}/lora_epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"
}

benchmark_baseline_single() {
    local DATASET=$1
    local MAX_FRAMES=$2
    local SEED_LIST=$3  # space-separated seeds

    echo "  [Baseline] ${DATASET}, max_frames=${MAX_FRAMES}, seeds: ${SEED_LIST}"

    # Run each seed in a separate subprocess to prevent memory accumulation
    for SEED in ${SEED_LIST}; do
        echo "    [Baseline] seed=${SEED}"
        python -u -c "
import sys, os, json, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)

seed = ${SEED}
seed_work_dir = '${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed' + str(seed)
evaluator = Evaluator(
    work_dir=seed_work_dir,
    datas=['${DATASET}'],
    modes=['pose', 'recon_unposed'],
    max_frames=${MAX_FRAMES},
    seed=seed,
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)

# Save metrics for aggregation
os.makedirs(seed_work_dir, exist_ok=True)
with open(os.path.join(seed_work_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
"
    done

    # Aggregate results across seeds
    python -u -c "
import sys, os, json
import numpy as np
sys.path.insert(0, 'src')

seeds = [${SEED_LIST// /, }]
all_seed_metrics = {}
for seed in seeds:
    mf = '${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed' + str(seed) + '/metrics.json'
    if os.path.exists(mf):
        with open(mf) as f:
            all_seed_metrics[seed] = json.load(f)
    else:
        print(f'WARNING: no metrics for seed {seed}')

if len(all_seed_metrics) <= 1:
    sys.exit(0)

all_keys = set()
for sm in all_seed_metrics.values():
    all_keys.update(sm.keys())
for mk in sorted(all_keys):
    print(f'\n  {mk} — Multi-seed summary')
    all_scenes = set()
    for s in all_seed_metrics:
        if mk in all_seed_metrics[s]:
            for sc in all_seed_metrics[s][mk]:
                if sc != 'mean':
                    all_scenes.add(sc)
    all_scenes = sorted(all_scenes)
    if not all_scenes:
        continue
    mnames = None
    for s in all_seed_metrics:
        if mk in all_seed_metrics[s] and all_scenes[0] in all_seed_metrics[s][mk]:
            mnames = list(all_seed_metrics[s][mk][all_scenes[0]].keys())
            break
    if not mnames:
        continue
    seed_means = {s: {n: [] for n in mnames} for s in all_seed_metrics}
    for sc in all_scenes:
        for s in all_seed_metrics:
            sd = all_seed_metrics[s].get(mk, {}).get(sc)
            if sd is None:
                continue
            for n in mnames:
                seed_means[s][n].append(sd.get(n, float('nan')))
    for s in all_seed_metrics:
        vals_str = ' | '.join(f'{n}: {np.mean(seed_means[s][n]):.4f}' for n in mnames if seed_means[s][n])
        print(f'  mean/seed{s}: {vals_str}')
    all_vals_mean = ' | '.join(f'{n}: {np.mean([v for s in all_seed_metrics for v in seed_means[s][n]]):.4f}' for n in mnames)
    all_vals_std = ' | '.join(f'{n}: {np.std([v for s in all_seed_metrics for v in seed_means[s][n]]):.4f}' for n in mnames)
    print(f'  mean/overall: {all_vals_mean}')
    print(f'  std/overall: {all_vals_std}')
"
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 Tuned LoRA + RKD Dist — all epochs"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "  Seeds: ${LORA_SEEDS}"
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

        echo ""
        echo "--- ${DATASET} (epochs: ${EPOCHS[@]}) ---"

        for EPOCH in "${EPOCHS[@]}"; do
            for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
                benchmark_lora_single "${DATASET}" "${EPOCH}" "${MAX_FRAMES}" "${LORA_SEEDS}"
            done
        done
    done

    echo "LoRA benchmark complete!"
}

run_baseline_benchmark_seed() {
    local SEED_LIST=$1
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 baseline (seeds: ${SEED_LIST})"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
            benchmark_baseline_single "${DATASET}" "${MAX_FRAMES}" "${SEED_LIST}"
        done
    done
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking DA3 baseline"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Max frames: ${MAX_FRAMES_LIST}"
    echo "  Seeds: ${BASELINE_SEEDS}"
    echo "============================================================"

    run_baseline_benchmark_seed "${BASELINE_SEEDS}"

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
    echo "Same as run_da3_tuned_lora.sh + RKD distance loss (T=${RKD_DIST_TEMP}, d1+d2 only)"
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
        # Train + LoRA benchmark (seed 43, 16v only)
        run_training
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

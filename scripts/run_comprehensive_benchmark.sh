#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Comprehensive Benchmark Script
# 1. VGGT output loss benchmark on eth3d (3 seeds)
# 2. DA3 baseline on hiroom, 7scenes, scannetpp (3 seeds each)
# 3. DA3 LoRA benchmark for all epochs on all 4 datasets (seed 43 only)
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# Configuration
DA3_MODEL="depth-anything/DA3-GIANT-1.1"
VGGT_MODEL="facebook/vggt-1b"
DA3_CHECKPOINT_ROOT="./checkpoints/all_da3"
VGGT_CHECKPOINT_ROOT="./checkpoints/all_vggt"
DA3_BENCHMARK_ROOT="./workspace/all_da3"
VGGT_BENCHMARK_ROOT="./workspace/all_vggt"

MAX_FRAMES=32
SEEDS="43 44 45"
LORA_SEED=43

# =============================================================================
# Part 1: VGGT Output Loss Training + Benchmark on ETH3D (3 seeds)
# =============================================================================

run_vggt_eth3d_training() {
    echo ""
    echo "============================================================"
    echo "Part 1a: Training VGGT on ETH3D — OUTPUT LOSS ONLY"
    echo "  Samples per scene: 5"
    echo "  Seeds: 40 41 42 43 44"
    echo "  Epochs: 5"
    echo "  Output loss: weight=2.0, cam=5.0, depth=1.0"
    echo "  Feature loss: DISABLED (huber=0, cos=0)"
    echo "============================================================"

    local DATA_BASE=./workspace/benchmark_dataset
    local OUTPUT_DIR=./checkpoints/vggt_eth3d_output_loss
    local DATASET=eth3d
    local SAMPLES=5
    local EPOCHS=5
    local SEED_START=40
    local SEED_END=44
    local LR=5e-5
    local BATCH_SIZE=4
    local LORA_RANK=32
    local LORA_ALPHA=32
    local LORA_LAYERS_START=0
    local NUM_VIEWS=8
    local OUTPUT_LAYERS="4 11 17 23"
    local IMAGE_SIZE=504
    local OUTPUT_WEIGHT=2.0
    local OUTPUT_CAMERA_WEIGHT=5.0
    local OUTPUT_DEPTH_WEIGHT=1.0
    local PATCH_HUBER_WEIGHT=0.0
    local PATCH_HUBER_COS_WEIGHT=0.0
    local TRAIN_SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    python -u ./scripts/train_distill_vggt.py \
        --data_root "${DATA_BASE}/${DATASET}" \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${TRAIN_SEEDS} \
        --model_name ${VGGT_MODEL} \
        --num_views ${NUM_VIEWS} \
        --output_layers ${OUTPUT_LAYERS} \
        --patch_huber_cosine \
        --patch_huber_weight ${PATCH_HUBER_WEIGHT} \
        --patch_huber_cos_weight ${PATCH_HUBER_COS_WEIGHT} \
        --output_weight ${OUTPUT_WEIGHT} \
        --output_camera_weight ${OUTPUT_CAMERA_WEIGHT} \
        --output_depth_weight ${OUTPUT_DEPTH_WEIGHT} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_workers 2 \
        --lr ${LR} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --lr_scheduler cosine \
        --warmup_ratio 0.15 \
        --eta_min 1e-7 \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}/${DATASET}" \
        --log_interval 1 \
        --save_interval 0

    echo "VGGT training complete! Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_vggt_eth3d_benchmark() {
    echo ""
    echo "============================================================"
    echo "Part 1b: VGGT Output Loss Benchmark on ETH3D"
    echo "  Seeds: ${SEEDS}"
    echo "  Max frames: ${MAX_FRAMES}"
    echo "============================================================"

    local OUTPUT_DIR=./checkpoints/vggt_eth3d_output_loss
    local BENCHMARK_ROOT=./workspace/vggt_eth3d_output_loss
    local DATASET=eth3d
    local LORA_RANK=32
    local LORA_ALPHA=32
    local LORA_LAYERS_START=0
    local IMAGE_SIZE=504

    # Find all epoch files
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
        echo "WARNING: No LoRA epoch weights found under ${OUTPUT_DIR}/${DATASET}/. Skipping VGGT benchmark."
        return 0
    fi

    mapfile -t EPOCHS < <(printf '%s\n' "${EPOCHS[@]}" | sort -n)
    echo "  Epochs: ${EPOCHS[@]}"

    for EPOCH in "${EPOCHS[@]}"; do
        local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"
        echo "  [LoRA epoch=${EPOCH}] ${DATASET}, max_frames=${MAX_FRAMES}, seeds: ${SEEDS}"

        python -u ./scripts/benchmark_lora_vggt.py \
            --lora_path "${LORA_PATH}" \
            --base_model "${VGGT_MODEL}" \
            --lora_rank ${LORA_RANK} \
            --lora_alpha ${LORA_ALPHA} \
            --lora_layers_start ${LORA_LAYERS_START} \
            --datasets ${DATASET} \
            --modes pose recon_unposed \
            --max_frames ${MAX_FRAMES} \
            --seeds ${SEEDS} \
            --image_size ${IMAGE_SIZE} \
            --work_dir "${BENCHMARK_ROOT}/lora_epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"
    done

    echo "VGGT ETH3D benchmark complete!"
}

# =============================================================================
# Part 2: DA3 Baseline on hiroom, 7scenes, scannetpp (3 seeds each)
# =============================================================================

run_da3_baseline_single() {
    local DATASET=$1
    local SEED_LIST=$2

    echo ""
    echo "------------------------------------------------------------"
    echo "DA3 Baseline: ${DATASET}, seeds: ${SEED_LIST}"
    echo "------------------------------------------------------------"

    for SEED in ${SEED_LIST}; do
        echo "  [Baseline] ${DATASET} seed ${SEED}"
        python -u -c "
import sys, os, json, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${DA3_MODEL}').to(device)

seed = ${SEED}
seed_work_dir = '${DA3_BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed' + str(seed)
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
    echo "  Aggregating ${DATASET} results..."
    python -u -c "
import sys, os, json
import numpy as np
sys.path.insert(0, 'src')

seeds = [${SEED_LIST// /, }]
all_seed_metrics = {}
for seed in seeds:
    mf = '${DA3_BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed' + str(seed) + '/metrics.json'
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

run_da3_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Part 2: DA3 Baseline on hiroom, 7scenes, scannetpp"
    echo "  Seeds: ${SEEDS}"
    echo "  Max frames: ${MAX_FRAMES}"
    echo "============================================================"

    for DATASET in hiroom 7scenes scannetpp; do
        run_da3_baseline_single "${DATASET}" "${SEEDS}"
    done

    echo "DA3 baseline benchmark complete!"
}

# =============================================================================
# Part 3: DA3 LoRA Benchmark for all epochs on all datasets (seed 43 only)
# =============================================================================

get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        scannetpp) echo "32 32" ;;
        hiroom)    echo "32 32" ;;
        7scenes)   echo "32 32" ;;
        eth3d)     echo "32 32" ;;
        *)
            echo "ERROR: Unknown dataset '${DATASET}'" >&2
            return 1
            ;;
    esac
}

run_da3_lora_benchmark_single() {
    local DATASET=$1
    local EPOCH=$2
    local CONFIG=$(get_training_config "${DATASET}")
    local DS_LORA_RANK=$(echo ${CONFIG} | cut -d' ' -f1)
    local DS_LORA_ALPHA=$(echo ${CONFIG} | cut -d' ' -f2)
    local LORA_PATH="${DA3_CHECKPOINT_ROOT}/${DATASET}/epoch_${EPOCH}_lora.pt"

    if [ ! -f "${LORA_PATH}" ]; then
        echo "  WARNING: LoRA weights not found at ${LORA_PATH}. Skipping."
        return 0
    fi

    echo "  [LoRA epoch=${EPOCH}] ${DATASET}, seed=${LORA_SEED}"
    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${DA3_MODEL}" \
        --lora_rank ${DS_LORA_RANK} \
        --lora_alpha ${DS_LORA_ALPHA} \
        --datasets "${DATASET}" \
        --modes pose recon_unposed \
        --max_frames ${MAX_FRAMES} \
        --seed ${LORA_SEED} \
        --work_dir "${DA3_BENCHMARK_ROOT}/lora_epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"
}

run_da3_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "Part 3: DA3 LoRA Benchmark (all epochs, seed ${LORA_SEED})"
    echo "  Datasets: eth3d, hiroom, 7scenes, scannetpp"
    echo "  Max frames: ${MAX_FRAMES}"
    echo "============================================================"

    for DATASET in eth3d hiroom 7scenes scannetpp; do
        shopt -s nullglob
        local LORA_FILES=( "${DA3_CHECKPOINT_ROOT}/${DATASET}/epoch_"*_lora.pt )
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
            echo "WARNING: No LoRA epoch weights found under ${DA3_CHECKPOINT_ROOT}/${DATASET}/. Skipping ${DATASET}."
            continue
        fi

        mapfile -t EPOCHS < <(printf '%s\n' "${EPOCHS[@]}" | sort -n)

        echo ""
        echo "--- ${DATASET} (epochs: ${EPOCHS[@]}) ---"

        for EPOCH in "${EPOCHS[@]}"; do
            run_da3_lora_benchmark_single "${DATASET}" "${EPOCH}"
        done
    done

    echo "DA3 LoRA benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

main() {
    local START_TIME=$(date +%s)

    echo "============================================================"
    echo "Comprehensive Benchmark Pipeline"
    echo "  Started at: $(date)"
    echo "============================================================"

    # Part 1a: VGGT training with output loss only on eth3d
    run_vggt_eth3d_training

    # Part 1b: VGGT output loss benchmark on eth3d (3 seeds)
    run_vggt_eth3d_benchmark

    # Part 2: DA3 baseline on hiroom, 7scenes, scannetpp
    run_da3_baseline_benchmark

    # Part 3: DA3 LoRA all epochs on all datasets (seed 43)
    run_da3_lora_benchmark

    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    local HOURS=$((DURATION / 3600))
    local MINUTES=$(((DURATION % 3600) / 60))
    local SECONDS=$((DURATION % 60))

    echo ""
    echo "============================================================"
    echo "✅ All benchmarks complete!"
    echo "  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "  Finished at: $(date)"
    echo "============================================================"
    echo ""
    echo "Results:"
    echo "  VGGT:         ./workspace/vggt_eth3d_output_loss/lora_epoch{epoch}/frames_${MAX_FRAMES}/eth3d/"
    echo "  DA3 baseline: ${DA3_BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/{dataset}/"
    echo "  DA3 LoRA:     ${DA3_BENCHMARK_ROOT}/lora_epoch{epoch}/frames_${MAX_FRAMES}/{dataset}/"
    echo "============================================================"
}

main "$@"

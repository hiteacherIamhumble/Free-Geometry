#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 64v & 100v Benchmark: DA3 + VGGT, Baseline + LoRA
#
# Runs all 4 datasets (eth3d, scannetpp, hiroom, 7scenes) with 3 seeds.
# Each benchmark runs one-by-one (sequential) to avoid CPU memory OOM.
#
# Pipeline:
#   Part 1: DA3 Baseline  (64v, 100v) x 4 datasets x 3 seeds
#   Part 2: DA3 LoRA      (64v, 100v) x 4 datasets x 3 seeds
#   Part 3: VGGT Baseline (64v, 100v) x 4 datasets x 3 seeds
#   Part 4: VGGT LoRA     (64v, 100v) x 4 datasets x 3 seeds
#
# Usage:
#   nohup bash scripts/run_64v_100v_benchmark.sh > logs/run_64v_100v.log 2>&1 &
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Models ===
DA3_MODEL="depth-anything/DA3-GIANT-1.1"
VGGT_MODEL="facebook/vggt-1b"

# === Checkpoints ===
DA3_LORA_ROOT="./checkpoints/da3_lora_final"
VGGT_LORA_ROOT="./checkpoints/vggt_lora_final"

# === Output dirs ===
DA3_BENCHMARK_ROOT="./workspace/da3_lora_final"
VGGT_BENCHMARK_ROOT="./workspace/vggt_lora_final"

# === Benchmark settings ===
SEEDS="43 44 45"
ALL_DATASETS="eth3d scannetpp hiroom 7scenes"
MAX_FRAMES_LIST="64 100"

# VGGT settings
VGGT_LORA_RANK=32
VGGT_LORA_ALPHA=32
VGGT_LORA_LAYERS_START=0
VGGT_IMAGE_SIZE=504

# =============================================================================
# Part 1: DA3 Baseline
# =============================================================================

run_da3_baseline() {
    echo ""
    echo "============================================================"
    echo "Part 1: DA3 Baseline (64v, 100v)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${SEEDS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
            for SEED in ${SEEDS}; do
                local WORK_DIR="${DA3_BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed${SEED}"

                # Skip if already done
                if [ -f "${WORK_DIR}/metrics.json" ]; then
                    echo "  [SKIP] DA3 baseline ${DATASET} ${MAX_FRAMES}v seed${SEED} — already done"
                    continue
                fi

                echo "  [DA3 Baseline] ${DATASET}, ${MAX_FRAMES}v, seed=${SEED}"
                python -u -c "
import sys, os, json, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${DA3_MODEL}').to(device)

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

os.makedirs('${WORK_DIR}', exist_ok=True)
with open(os.path.join('${WORK_DIR}', 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
"
            done

            # Aggregate seeds
            _aggregate_seeds "${DA3_BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}" "${SEEDS}"
        done
    done

    echo "DA3 baseline benchmark complete!"
}

# =============================================================================
# Part 2: DA3 LoRA
# =============================================================================

run_da3_lora() {
    echo ""
    echo "============================================================"
    echo "Part 2: DA3 LoRA (64v, 100v)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${SEEDS}"
    echo "  Checkpoint root: ${DA3_LORA_ROOT}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        local LORA_PATH="${DA3_LORA_ROOT}/${DATASET}/lora.pt"

        if [ ! -f "${LORA_PATH}" ]; then
            echo "  WARNING: DA3 LoRA not found at ${LORA_PATH}. Skipping ${DATASET}."
            continue
        fi

        for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
            echo "  [DA3 LoRA] ${DATASET}, ${MAX_FRAMES}v, seeds: ${SEEDS}"
            python -u ./scripts/benchmark_lora.py \
                --lora_path "${LORA_PATH}" \
                --base_model "${DA3_MODEL}" \
                --datasets "${DATASET}" \
                --modes pose recon_unposed \
                --max_frames ${MAX_FRAMES} \
                --seeds ${SEEDS} \
                --work_dir "${DA3_BENCHMARK_ROOT}/lora/frames_${MAX_FRAMES}/${DATASET}"
        done
    done

    echo "DA3 LoRA benchmark complete!"
}

# =============================================================================
# Part 3: VGGT Baseline
# =============================================================================

run_vggt_baseline() {
    echo ""
    echo "============================================================"
    echo "Part 3: VGGT Baseline (64v, 100v)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${SEEDS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
            echo "  [VGGT Baseline] ${DATASET}, ${MAX_FRAMES}v, seeds: ${SEEDS}"
            python -u ./scripts/benchmark_lora_vggt.py \
                --base_model "${VGGT_MODEL}" \
                --datasets "${DATASET}" \
                --modes pose recon_unposed \
                --max_frames ${MAX_FRAMES} \
                --seeds ${SEEDS} \
                --image_size ${VGGT_IMAGE_SIZE} \
                --work_dir "${VGGT_BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}"
        done
    done

    echo "VGGT baseline benchmark complete!"
}

# =============================================================================
# Part 4: VGGT LoRA
# =============================================================================

run_vggt_lora() {
    echo ""
    echo "============================================================"
    echo "Part 4: VGGT LoRA (64v, 100v)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${SEEDS}"
    echo "  Checkpoint root: ${VGGT_LORA_ROOT}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        local LORA_PATH="${VGGT_LORA_ROOT}/${DATASET}/lora.pt"

        if [ ! -f "${LORA_PATH}" ]; then
            echo "  WARNING: VGGT LoRA not found at ${LORA_PATH}. Skipping ${DATASET}."
            continue
        fi

        for MAX_FRAMES in ${MAX_FRAMES_LIST}; do
            echo "  [VGGT LoRA] ${DATASET}, ${MAX_FRAMES}v, seeds: ${SEEDS}"
            python -u ./scripts/benchmark_lora_vggt.py \
                --lora_path "${LORA_PATH}" \
                --base_model "${VGGT_MODEL}" \
                --lora_rank ${VGGT_LORA_RANK} \
                --lora_alpha ${VGGT_LORA_ALPHA} \
                --lora_layers_start ${VGGT_LORA_LAYERS_START} \
                --datasets "${DATASET}" \
                --modes pose recon_unposed \
                --max_frames ${MAX_FRAMES} \
                --seeds ${SEEDS} \
                --image_size ${VGGT_IMAGE_SIZE} \
                --work_dir "${VGGT_BENCHMARK_ROOT}/lora/frames_${MAX_FRAMES}/${DATASET}"
        done
    done

    echo "VGGT LoRA benchmark complete!"
}

# =============================================================================
# Seed aggregation helper
# =============================================================================

_aggregate_seeds() {
    local BASE_DIR=$1
    local SEED_LIST=$2

    python -u -c "
import sys, os, json
import numpy as np
sys.path.insert(0, 'src')

seeds = [${SEED_LIST// /, }]
all_seed_metrics = {}
for seed in seeds:
    mf = '${BASE_DIR}/seed' + str(seed) + '/metrics.json'
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

# =============================================================================
# Main
# =============================================================================

main() {
    local START_TIME=$(date +%s)

    echo "============================================================"
    echo "64v & 100v Benchmark Pipeline"
    echo "  Started at: $(date)"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Frame configs: ${MAX_FRAMES_LIST}"
    echo "  Seeds: ${SEEDS}"
    echo "============================================================"

    # Part 1: DA3 Baseline
    run_da3_baseline

    # Part 2: DA3 LoRA
    run_da3_lora

    # Part 3: VGGT Baseline
    run_vggt_baseline

    # Part 4: VGGT LoRA
    run_vggt_lora

    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    local HOURS=$((DURATION / 3600))
    local MINUTES=$(((DURATION % 3600) / 60))
    local SECONDS=$((DURATION % 60))

    echo ""
    echo "============================================================"
    echo "All benchmarks complete!"
    echo "  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "  Finished at: $(date)"
    echo "============================================================"
    echo ""
    echo "Results:"
    echo "  DA3 baseline:  ${DA3_BENCHMARK_ROOT}/baseline/frames_{64,100}/{dataset}/"
    echo "  DA3 LoRA:      ${DA3_BENCHMARK_ROOT}/lora/frames_{64,100}/{dataset}/"
    echo "  VGGT baseline: ${VGGT_BENCHMARK_ROOT}/baseline/frames_{64,100}/{dataset}/"
    echo "  VGGT LoRA:     ${VGGT_BENCHMARK_ROOT}/lora/frames_{64,100}/{dataset}/"
    echo "============================================================"
}

main "$@"

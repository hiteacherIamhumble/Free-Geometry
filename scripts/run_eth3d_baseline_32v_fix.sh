#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Run ETH3D baseline benchmark for 32 frames with 3 seeds
# Each seed runs in a separate process to prevent memory accumulation
# =============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
BENCHMARK_ROOT=./workspace/all_da3
DATASET="eth3d"
MAX_FRAMES=32
SEEDS="43 44 45"

echo "============================================================"
echo "ETH3D Baseline Benchmark - 32 frames"
echo "  Seeds: ${SEEDS}"
echo "  Each seed runs in separate subprocess"
echo "============================================================"

# Run each seed in a separate subprocess
for SEED in ${SEEDS}; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Running seed ${SEED}"
    echo "------------------------------------------------------------"

    python -u -c "
import sys, os, json, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading model on {device}...')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)

seed = ${SEED}
seed_work_dir = '${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed' + str(seed)
print(f'Work dir: {seed_work_dir}')

evaluator = Evaluator(
    work_dir=seed_work_dir,
    datas=['${DATASET}'],
    modes=['pose', 'recon_unposed'],
    max_frames=${MAX_FRAMES},
    seed=seed,
)

print('Running inference...')
evaluator.infer(api)

print('Running evaluation...')
metrics = evaluator.eval()
evaluator.print_metrics(metrics)

# Save metrics for aggregation
os.makedirs(seed_work_dir, exist_ok=True)
metrics_file = os.path.join(seed_work_dir, 'metrics.json')
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'Saved metrics to: {metrics_file}')
"

    echo "Seed ${SEED} complete!"
done

# Aggregate results across seeds
echo ""
echo "============================================================"
echo "Aggregating results across seeds"
echo "============================================================"

python -u -c "
import sys, os, json
import numpy as np
sys.path.insert(0, 'src')

seeds = [${SEEDS// /, }]
all_seed_metrics = {}

for seed in seeds:
    mf = '${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed' + str(seed) + '/metrics.json'
    if os.path.exists(mf):
        with open(mf) as f:
            all_seed_metrics[seed] = json.load(f)
        print(f'Loaded metrics for seed {seed}')
    else:
        print(f'WARNING: no metrics for seed {seed}')

if len(all_seed_metrics) == 0:
    print('ERROR: No metrics found!')
    sys.exit(1)

print(f'\nAggregating {len(all_seed_metrics)} seeds...\n')

all_keys = set()
for sm in all_seed_metrics.values():
    all_keys.update(sm.keys())

for mk in sorted(all_keys):
    print(f'\n{'='*60}')
    print(f'{mk} — Multi-seed summary')
    print(f'{'='*60}')

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

    # Print header
    header = f'{'Scene':<30s}'
    for n in mnames:
        header += f'  {n:>10s}'
    print(header)
    print('-' * len(header))

    seed_means = {s: {n: [] for n in mnames} for s in all_seed_metrics}

    # Print per-scene per-seed results
    for sc in all_scenes:
        for s in all_seed_metrics:
            sd = all_seed_metrics[s].get(mk, {}).get(sc)
            if sd is None:
                continue

            row = f'{sc}/seed{s}'
            row = f'{row:<30s}'
            for n in mnames:
                val = sd.get(n, float('nan'))
                row += f'  {val:>10.4f}'
                seed_means[s][n].append(val)
            print(row)

    print('-' * len(header))

    # Print per-seed means
    for s in all_seed_metrics:
        row = f'mean/seed{s}'
        row = f'{row:<30s}'
        for n in mnames:
            vals = seed_means[s][n]
            mean_val = np.mean(vals) if vals else float('nan')
            row += f'  {mean_val:>10.4f}'
        print(row)

    print('-' * len(header))

    # Print overall mean and std
    row_mean = f'mean/overall'
    row_mean = f'{row_mean:<30s}'
    row_std = f'std/overall'
    row_std = f'{row_std:<30s}'

    for n in mnames:
        all_vals = []
        for s in all_seed_metrics:
            all_vals.extend(seed_means[s][n])
        mean_val = np.mean(all_vals) if all_vals else float('nan')
        std_val = np.std(all_vals) if all_vals else float('nan')
        row_mean += f'  {mean_val:>10.4f}'
        row_std += f'  {std_val:>10.4f}'

    print(row_mean)
    print(row_std)

print(f'\n{'='*60}')
print('✅ All done!')
print(f'{'='*60}')
"

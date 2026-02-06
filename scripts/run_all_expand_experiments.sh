#!/usr/bin/env bash
set -e

# =============================================================================
# Run ALL Expand Experiments: Train on each dataset, benchmark on all
# =============================================================================
#
# This script runs the expand experiment for all 5 training configurations:
# 1. eth3d (3 samples)
# 2. scannetpp (3 samples)
# 3. hiroom (2 samples)
# 4. 7scenes (12 samples)
# 5. 7scenes_3samples (3 samples)
#
# Usage:
#   ./scripts/run_all_expand_experiments.sh
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"

mkdir -p "${LOG_DIR}"

DATASETS="eth3d scannetpp hiroom 7scenes 7scenes_3samples"

echo ""
echo "============================================================"
echo "Running ALL Expand Experiments"
echo "============================================================"
echo "Datasets to train on: ${DATASETS}"
echo "Each will be benchmarked on all 4 datasets"
echo "Logs will be saved to: ${LOG_DIR}"
echo "============================================================"
echo ""

for DATASET in ${DATASETS}; do
    echo ""
    echo "============================================================"
    echo "Starting experiment: Train on ${DATASET}, benchmark on all"
    echo "============================================================"

    "${SCRIPT_DIR}/run_vggt_full_experiment_expand.sh" "${DATASET}" all 2>&1 | tee "${LOG_DIR}/vggt_expand_${DATASET}.log"

    echo ""
    echo "============================================================"
    echo "Completed: ${DATASET}"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "ALL EXPAND EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
for DATASET in ${DATASETS}; do
    echo "  - workspace/vggt_expand_${DATASET}/comparison/"
done
echo ""

#!/usr/bin/env bash
set -euo pipefail

# 5-dataset interleaved training+benchmark script.
# Order: eth3d -> scannetpp -> hiroom -> 7scenes -> dtu
# Benchmark mode: pose only, 4 views.
# For dtu weights, benchmark runs on dtu64 (pose-only).
# Paths/log are hard-coded for this run variant.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DA3_OUTPUT_DIR="./checkpoints/da3_finetune_run2"
export DA3_BENCHMARK_ROOT="./workspace/da3_finetune_run2"
LOG_FILE="./logs/da3_finetune_5datasets_pose4v_run2.log"

mkdir -p logs

bash ./scripts/run_da3_finetune.sh train_benchmark_all_5datasets 2>&1 | tee -a "${LOG_FILE}"

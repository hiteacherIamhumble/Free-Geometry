#!/usr/bin/env bash
set -euo pipefail

# Train ScanNet++ with ETH3D-like settings, then benchmark pose-only on
# ScanNet++ + DTU-64 at 4 views. This uses eta_min=1e-8 from run_da3_finetune.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

bash ./scripts/run_da3_finetune.sh train_benchmark_scannetpp_dtu64

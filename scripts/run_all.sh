#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Combined runner: DA3 + VGGT full pipeline
#
# Pipeline:
#   1. DA3 ETH3D baseline (3 seeds) - DONE
#   2. DA3 all LoRA training - DONE
#   3. DA3 baseline (3 seeds, all 4 datasets, all frame configs)
#   4. DA3 LoRA (every epoch, seed 43 only, all frame configs)
#   5. VGGT baseline (3 seeds, all 4 datasets, all frame configs)
#   6. VGGT LoRA training (all 4 datasets)
#   7. VGGT LoRA (every epoch, seed 43 only, all frame configs)
#
# Usage:
#   nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# Export missing DA3 variables (from run_da3_tuned_lora.sh defaults)
export OUTPUT_WEIGHT=0.0
export CAMERA_WEIGHT=5.0
export DEPTH_WEIGHT=3.0
export DEPTH_GRAD=grad
export DEPTH_VALID_RANGE=0.98

echo "============================================================"
echo "=== Phase 1: DA3 ETH3D Baseline (3 seeds) ==="
echo "=== STATUS: DONE ✓ ==="
echo "============================================================"

echo ""
echo "============================================================"
echo "=== Phase 2: DA3 All LoRA Training ==="
echo "=== STATUS: DONE ✓ ==="
echo "============================================================"

echo ""
echo "============================================================"
echo "=== Phase 3: DA3 Baseline (3 seeds, all datasets) ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_da3.sh" benchmark_baseline 2>&1 | tee logs/run_da3_baseline_bench.log

echo ""
echo "============================================================"
echo "=== Phase 4: DA3 LoRA (every epoch, seed 43 only) ==="
echo "============================================================"
# Override LORA_SEEDS to only use seed 43
export LORA_SEEDS="43"
bash "${SCRIPT_DIR}/run_da3.sh" benchmark_lora 2>&1 | tee logs/run_da3_lora_bench_seed43.log

echo ""
echo "============================================================"
echo "=== Phase 5: VGGT Baseline (3 seeds, all datasets) ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_vggt.sh" benchmark_base 2>&1 | tee logs/run_vggt_baseline_bench.log

echo ""
echo "============================================================"
echo "=== Phase 6: VGGT LoRA Training (all 4 datasets) ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_vggt.sh" train 2>&1 | tee logs/run_vggt_train.log

echo ""
echo "============================================================"
echo "=== Phase 7: VGGT LoRA (every epoch, seed 43 only) ==="
echo "============================================================"
# Override LORA_BENCHMARK_SEEDS to only use seed 43
export LORA_BENCHMARK_SEEDS="43"
bash "${SCRIPT_DIR}/run_vggt.sh" benchmark_lora 2>&1 | tee logs/run_vggt_lora_bench_seed43.log

echo ""
echo "============================================================"
echo "=== All phases complete ==="
echo "============================================================"

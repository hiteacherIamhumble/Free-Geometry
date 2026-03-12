#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Continue from where run_all.sh stopped
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# Export missing DA3 variables
export OUTPUT_WEIGHT=0.0
export CAMERA_WEIGHT=5.0
export DEPTH_WEIGHT=3.0
export DEPTH_GRAD=grad
export DEPTH_VALID_RANGE=0.98

echo "============================================================"
echo "=== Continuing DA3: baseline benchmark (scannetpp, hiroom, 7scenes) ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_da3.sh" benchmark_baseline 2>&1 | tee -a logs/run_da3_all.log

echo ""
echo "============================================================"
echo "=== DA3: Training all datasets ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_da3.sh" train 2>&1 | tee -a logs/run_da3_all.log

echo ""
echo "============================================================"
echo "=== DA3: LoRA benchmark ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_da3.sh" benchmark_lora 2>&1 | tee -a logs/run_da3_all.log

echo ""
echo "============================================================"
echo "=== Phase 2: VGGT (all) ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_vggt.sh" all 2>&1 | tee logs/run_vggt_all.log

echo ""
echo "============================================================"
echo "=== Phase 3: VGGT ETH3D Output Loss Only ==="
echo "============================================================"
bash "${SCRIPT_DIR}/run_vggt_eth3d_output_loss.sh" all 2>&1 | tee logs/run_vggt_eth3d_output_loss.log

echo ""
echo "============================================================"
echo "=== All done ==="
echo "============================================================"

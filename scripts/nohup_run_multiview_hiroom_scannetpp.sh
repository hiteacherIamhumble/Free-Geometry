#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

COMMAND="${1:-all}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-./logs/nohup_multiview_hiroom_scannetpp_${COMMAND}_${TIMESTAMP}.log}"

nohup bash ./scripts/run_multiview_hiroom_scannetpp.sh "${COMMAND}" > "${LOG_PATH}" 2>&1 &
PID=$!

echo "Started background job."
echo "  PID: ${PID}"
echo "  Log: ${LOG_PATH}"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_PATH}"

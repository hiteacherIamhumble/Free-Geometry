#!/usr/bin/env bash
# =============================================================================
# Watchdog: monitors run_all.sh for OOM or crashes every 10 minutes.
#
# Usage:
#   nohup bash scripts/watchdog.sh > logs/watchdog.log 2>&1 &
#
# What it does:
#   - Every 10 min, checks if run_all.sh is still alive
#   - Scans the latest log for OOM / CUDA out of memory / crash signals
#   - If OOM detected: patches batch_size to 2 in the shell scripts,
#     kills the current run, and restarts run_all.sh
#   - If process died without OOM: logs the error and exits
# =============================================================================

set -u

INTERVAL=600  # 10 minutes
LOG_FILE="logs/run_all.log"
SCRIPTS=("scripts/run_da3.sh" "scripts/run_vggt.sh" "scripts/run_vggt_eth3d_output_loss.sh")
PATCHED_BS=0  # track if we already patched

get_main_pid() {
    pgrep -f "bash scripts/run_all.sh" | head -1
}

check_oom() {
    # Check last 200 lines of log for OOM signals
    if [ -f "${LOG_FILE}" ]; then
        tail -200 "${LOG_FILE}" | grep -qi \
            -e "CUDA out of memory" \
            -e "OutOfMemoryError" \
            -e "RuntimeError.*out of memory" \
            -e "torch.cuda.OutOfMemoryError"
        return $?
    fi
    return 1
}

check_crash() {
    # Check last 50 lines for common crash patterns (not OOM)
    if [ -f "${LOG_FILE}" ]; then
        tail -50 "${LOG_FILE}" | grep -qi \
            -e "Traceback (most recent call last)" \
            -e "Error.*Killed" \
            -e "Segmentation fault"
        return $?
    fi
    return 1
}

patch_batch_size() {
    local NEW_BS=$1
    for script in "${SCRIPTS[@]}"; do
        if [ -f "${script}" ]; then
            # Patch --batch_size N or BATCH_SIZE=N
            sed -i "s/--batch_size [0-9]\+/--batch_size ${NEW_BS}/g" "${script}"
            sed -i "s/^BATCH_SIZE=[0-9]\+/BATCH_SIZE=${NEW_BS}/g" "${script}"
            echo "[$(date)] Patched ${script}: batch_size -> ${NEW_BS}"
        fi
    done
}

restart_run_all() {
    echo "[$(date)] Restarting run_all.sh ..."
    nohup bash scripts/run_all.sh >> "${LOG_FILE}" 2>&1 &
    echo "[$(date)] Restarted with PID $!"
}

kill_tree() {
    local PID=$1
    # Kill the whole process group
    pkill -P "${PID}" 2>/dev/null
    kill "${PID}" 2>/dev/null
    # Also kill any lingering python training processes
    pkill -f "train_distill" 2>/dev/null
    pkill -f "benchmark_lora" 2>/dev/null
    sleep 3
}

echo "[$(date)] Watchdog started. Checking every ${INTERVAL}s."
echo "[$(date)] Monitoring log: ${LOG_FILE}"

while true; do
    sleep "${INTERVAL}"

    MAIN_PID=$(get_main_pid)

    if [ -z "${MAIN_PID}" ]; then
        echo "[$(date)] run_all.sh is NOT running."

        if check_oom; then
            echo "[$(date)] OOM detected in log!"
            if [ "${PATCHED_BS}" -eq 0 ]; then
                echo "[$(date)] Patching batch_size to 2 and restarting..."
                patch_batch_size 2
                PATCHED_BS=1
                restart_run_all
            else
                echo "[$(date)] Already patched batch_size. OOM persists. Giving up."
                exit 1
            fi
        elif check_crash; then
            echo "[$(date)] Crash detected (not OOM). Check logs. Exiting watchdog."
            tail -30 "${LOG_FILE}"
            exit 1
        else
            echo "[$(date)] Process ended (possibly completed). Exiting watchdog."
            exit 0
        fi
    else
        # Process is running — still check for OOM in case subprocess died
        # but parent shell hasn't noticed yet
        if check_oom; then
            echo "[$(date)] OOM detected while run_all.sh (PID ${MAIN_PID}) still running!"
            if [ "${PATCHED_BS}" -eq 0 ]; then
                echo "[$(date)] Killing current run, patching batch_size to 2, restarting..."
                kill_tree "${MAIN_PID}"
                patch_batch_size 2
                PATCHED_BS=1
                restart_run_all
            else
                echo "[$(date)] Already patched. Waiting for current run to proceed..."
            fi
        else
            echo "[$(date)] OK — run_all.sh running (PID ${MAIN_PID}), no OOM."
        fi
    fi
done

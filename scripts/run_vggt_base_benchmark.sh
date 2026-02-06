#!/usr/bin/env bash
set -e

# Benchmark script for base VGGT model (without LoRA)
# This establishes baseline performance on benchmark datasets

MODEL_NAME=facebook/vggt-1b
IMAGE_SIZE=518

# Default settings
MAX_FRAMES=8
MODES="pose"

usage() {
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  eth3d [max_frames]        - Benchmark on ETH3D"
    echo "  7scenes [max_frames]      - Benchmark on 7Scenes"
    echo "  scannetpp [max_frames]    - Benchmark on ScanNet++"
    echo "  hiroom [max_frames]       - Benchmark on HiRoom"
    echo "  dtu [max_frames]          - Benchmark on DTU"
    echo "  all [max_frames]          - Benchmark on all datasets"
    echo "  recon <dataset>           - Benchmark with reconstruction"
    echo ""
    echo "Examples:"
    echo "  $0 eth3d 8               # ETH3D with 8 frames per scene"
    echo "  $0 all 16                # All datasets with 16 frames"
    echo "  $0 recon eth3d           # ETH3D with reconstruction"
}

run_benchmark() {
    local DATASETS=$1
    local WORK_DIR=$2
    local MAX_FRAMES_ARG=${3:-$MAX_FRAMES}
    local MODES_ARG=${4:-$MODES}

    echo "=== Benchmarking Base VGGT Model ==="
    echo "  Model: ${MODEL_NAME}"
    echo "  Datasets: ${DATASETS}"
    echo "  Max frames: ${MAX_FRAMES_ARG}"
    echo "  Modes: ${MODES_ARG}"
    echo "  Work dir: ${WORK_DIR}"
    echo ""

    python ./scripts/benchmark_vggt_base.py \
        --model_name ${MODEL_NAME} \
        --datasets ${DATASETS} \
        --modes ${MODES_ARG} \
        --max_frames ${MAX_FRAMES_ARG} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${WORK_DIR}
}

case "${1:-}" in
    eth3d)
        MAX_FRAMES=${2:-8}
        run_benchmark "eth3d" "./workspace/vggt_base_eval_eth3d" ${MAX_FRAMES}
        ;;
    7scenes)
        MAX_FRAMES=${2:-8}
        run_benchmark "7scenes" "./workspace/vggt_base_eval_7scenes" ${MAX_FRAMES}
        ;;
    scannetpp)
        MAX_FRAMES=${2:-8}
        run_benchmark "scannetpp" "./workspace/vggt_base_eval_scannetpp" ${MAX_FRAMES}
        ;;
    hiroom)
        MAX_FRAMES=${2:-8}
        run_benchmark "hiroom" "./workspace/vggt_base_eval_hiroom" ${MAX_FRAMES}
        ;;
    dtu)
        MAX_FRAMES=${2:-8}
        run_benchmark "dtu" "./workspace/vggt_base_eval_dtu" ${MAX_FRAMES}
        ;;
    all)
        MAX_FRAMES=${2:-8}
        run_benchmark "eth3d 7scenes scannetpp" "./workspace/vggt_base_eval_all" ${MAX_FRAMES}
        ;;
    recon)
        DATASET=${2:-eth3d}
        MAX_FRAMES=${3:-8}
        run_benchmark "${DATASET}" "./workspace/vggt_base_eval_${DATASET}_recon" ${MAX_FRAMES} "pose recon_unposed"
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo ""
echo "=== Benchmark Complete ==="

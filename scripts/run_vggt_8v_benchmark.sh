#!/usr/bin/env bash
set -e

# =============================================================================
# Run 8-view benchmark for vggt_full_experiment
# =============================================================================
#
# This script runs the 8-view benchmark (no even-indexing) for comparison
# with the 4-view results.
#
# Usage:
#   ./scripts/run_vggt_8v_benchmark.sh
#
# =============================================================================

MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/vggt_full_experiment
BENCHMARK_ROOT=./workspace/vggt_full_experiment

# Benchmark settings
BENCHMARK_SEEDS="42 43 44"
MAX_FRAMES=8
EVAL_FRAMES=0  # 0 means use all 8 frames (no even-indexing)
MODES="pose recon_unposed"

LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0

echo ""
echo "============================================================"
echo "Running 8-view benchmark for vggt_full_experiment"
echo "============================================================"
echo "Max frames: ${MAX_FRAMES}"
echo "Eval frames: ${EVAL_FRAMES} (0 = use all ${MAX_FRAMES})"
echo "Seeds: ${BENCHMARK_SEEDS}"
echo "============================================================"

for DATASET in eth3d scannetpp hiroom 7scenes; do
    echo ""
    echo "--- Dataset: ${DATASET} ---"

    LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_0_lora.pt"

    for SEED in ${BENCHMARK_SEEDS}; do
        # Base model 8v
        echo "  Benchmarking base model 8v on ${DATASET} (seed=${SEED})..."
        python ./scripts/benchmark_lora_vggt.py \
            --base_model ${MODEL_NAME} \
            --datasets ${DATASET} \
            --modes ${MODES} \
            --max_frames ${MAX_FRAMES} \
            --seed ${SEED} \
            --image_size 518 \
            --work_dir "${BENCHMARK_ROOT}/base_8v/${DATASET}/seed${SEED}"

        # LoRA model 8v
        if [ -f "${LORA_PATH}" ]; then
            echo "  Benchmarking LoRA model 8v on ${DATASET} (seed=${SEED})..."
            python ./scripts/benchmark_lora_vggt.py \
                --lora_path "${LORA_PATH}" \
                --base_model ${MODEL_NAME} \
                --lora_rank ${LORA_RANK} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_layers_start ${LORA_LAYERS_START} \
                --datasets ${DATASET} \
                --modes ${MODES} \
                --max_frames ${MAX_FRAMES} \
                --seed ${SEED} \
                --image_size 518 \
                --work_dir "${BENCHMARK_ROOT}/lora_8v/${DATASET}/seed${SEED}"
        fi
    done
done

echo ""
echo "============================================================"
echo "8-view benchmark complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  Base 8v: ${BENCHMARK_ROOT}/base_8v/"
echo "  LoRA 8v: ${BENCHMARK_ROOT}/lora_8v/"
echo ""
echo "Run comparison with:"
echo "  python ./scripts/compare_vggt_results.py \\"
echo "      --base_root ${BENCHMARK_ROOT}/base_8v \\"
echo "      --lora_root ${BENCHMARK_ROOT}/lora_8v \\"
echo "      --output_dir ${BENCHMARK_ROOT}/comparison_8v \\"
echo "      --datasets eth3d scannetpp hiroom 7scenes \\"
echo "      --seeds 42 43 44"
echo ""

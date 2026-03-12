#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 1. Run all ablation_v2 experiments
./scripts/run_ablation_v2.sh all

# 2. Run baseline benchmark on all 4 datasets
./scripts/run_combined_all_datasets.sh benchmark_baseline

# 3. Run LoRA benchmark on eth3d and hiroom only
MODEL=facebook/vggt-1b
SEEDS="42 43 44"
MODES="pose recon_unposed"

for DS in eth3d hiroom; do
    CKPT=./checkpoints/combined_all_datasets/${DS}/epoch_0_lora.pt
    if [ ! -f "${CKPT}" ]; then
        echo "WARNING: LoRA weights not found for ${DS} at ${CKPT}. Skipping."
        continue
    fi
    echo ""
    echo "--- LoRA benchmark: ${DS} ---"

    for SETTING in 4v 8v maxframe; do
        case ${SETTING} in
            4v)       MF=8;  EF=4 ;;
            8v)       MF=8;  EF=0 ;;
            maxframe) MF=50; EF=0 ;;
        esac

        for SEED in ${SEEDS}; do
            echo "  [LoRA ${SETTING}] ${DS}, seed=${SEED}"
            CMD="python ./scripts/benchmark_lora_vggt.py \
                --lora_path ${CKPT} \
                --base_model ${MODEL} \
                --lora_rank 16 \
                --lora_alpha 16 \
                --lora_layers_start 0 \
                --datasets ${DS} \
                --modes ${MODES} \
                --max_frames ${MF} \
                --seed ${SEED} \
                --image_size 504 \
                --work_dir ./workspace/combined_all_datasets/lora_${SETTING}/${DS}/seed${SEED}"

            if [ "${EF}" -gt 0 ]; then
                CMD="${CMD} --eval_frames ${EF}"
            fi

            eval ${CMD}
        done
    done
done

echo ""
echo "All done!"

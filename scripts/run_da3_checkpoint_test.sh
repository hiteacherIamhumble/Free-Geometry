#!/usr/bin/env bash
set -euo pipefail

# Quick test: train eth3d with checkpointed RKD angle loss, then benchmark 4v/8v
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/da3_checkpoint_test
BENCHMARK_ROOT=./workspace/da3_checkpoint_test

DATASET=eth3d
SAMPLES=5
EPOCHS=3
SEEDS="40 41 42 43 44"

echo "=== Training eth3d with checkpointed RKD angle loss ==="
python -u ./scripts/train_distill.py \
    --dataset ${DATASET} \
    --samples_per_scene ${SAMPLES} \
    --seeds_list ${SEEDS} \
    --model_name ${MODEL_NAME} \
    --num_views 8 \
    --combined_loss \
    --all_token_softmax_kl_cosine \
    --all_token_softmax_kl_weight 1.0 \
    --all_token_softmax_cos_weight 2.0 \
    --distill_all_layers \
    --rkd_weight 2.0 \
    --rkd_topk 4 \
    --rkd_num_ref_samples 256 \
    --rkd_num_shared_samples 256 \
    --rkd_angle1_weight 1.0 \
    --rkd_angle2_weight 1.0 \
    --rkd_angle3_weight 1.0 \
    --output_weight 0.0 \
    --camera_weight 5.0 \
    --depth_weight 3.0 \
    --depth_gradient_loss grad \
    --depth_valid_range 0.98 \
    --epochs ${EPOCHS} \
    --batch_size 1 \
    --num_workers 2 \
    --lr 1e-4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lr_scheduler cosine \
    --warmup_steps 50 \
    --weight_decay 1e-5 \
    --resample_per_epoch \
    --output_dir "${OUTPUT_DIR}/${DATASET}"

echo ""
echo "=== Benchmarking ==="
for EPOCH in 0 1 2; do
    LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "Skipping epoch ${EPOCH} (no checkpoint)"
        continue
    fi
    for MF in 4 8; do
        echo "  [LoRA epoch=${EPOCH}] ${DATASET}, max_frames=${MF}"
        python -u ./scripts/benchmark_lora.py \
            --lora_path "${LORA_PATH}" \
            --base_model "${MODEL_NAME}" \
            --lora_rank 32 \
            --lora_alpha 32 \
            --datasets "${DATASET}" \
            --modes pose recon_unposed \
            --max_frames ${MF} \
            --seed 43 \
            --work_dir "${BENCHMARK_ROOT}/lora_epoch${EPOCH}/frames_${MF}/${DATASET}"
    done
done

echo ""
echo "=== Done ==="

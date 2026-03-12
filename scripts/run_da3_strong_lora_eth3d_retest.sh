#!/usr/bin/env bash
set -euo pipefail

# Retest strong LoRA (KL+Cos+RKD angle, NO distance) on eth3d
# to verify the forward_features_only code path doesn't regress.

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/da3_strong_lora_retest
BENCHMARK_ROOT=./workspace/da3_strong_lora_retest

DATASET=eth3d
SAMPLES=5
EPOCHS=3
SEEDS="40 41 42 43 44"

LORA_RANK=32
LORA_ALPHA=32
LR=1e-4

echo ""
echo "============================================================"
echo "Training ${DATASET} — Strong LoRA (no distance) RETEST"
echo "============================================================"

python -u ./scripts/train_distill.py \
    --dataset "${DATASET}" \
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
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lr_scheduler cosine \
    --warmup_steps 50 \
    --weight_decay 1e-5 \
    --resample_per_epoch \
    --output_dir "${OUTPUT_DIR}/${DATASET}"

echo "Training complete!"

echo ""
echo "============================================================"
echo "Benchmarking LoRA epochs on ${DATASET}"
echo "============================================================"

for EPOCH in 0 1 2; do
    LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "Skipping epoch ${EPOCH} (not found)"
        continue
    fi
    for MAX_FRAMES in 4 8; do
        echo "  [LoRA epoch=${EPOCH}] ${DATASET}, max_frames=${MAX_FRAMES}"
        python -u ./scripts/benchmark_lora.py \
            --lora_path "${LORA_PATH}" \
            --base_model "${MODEL_NAME}" \
            --lora_rank ${LORA_RANK} \
            --lora_alpha ${LORA_ALPHA} \
            --datasets "${DATASET}" \
            --modes pose recon_unposed \
            --max_frames ${MAX_FRAMES} \
            --seed 43 \
            --work_dir "${BENCHMARK_ROOT}/lora_epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"
    done
done

echo ""
echo "Done! Results in: ${BENCHMARK_ROOT}/"

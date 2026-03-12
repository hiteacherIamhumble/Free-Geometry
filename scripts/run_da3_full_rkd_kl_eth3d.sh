#!/usr/bin/env bash
set -euo pipefail

# KL+Cos + RKD Angle + RKD Distance KL on eth3d
# Full combo: feature KL(T=1) + cosine + angle-wise RKD + distance-wise RKD (KL variant)

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/da3_full_rkd_kl_eth3d/eth3d
BENCHMARK_ROOT=./workspace/da3_full_rkd_kl_eth3d

echo "============================================================"
echo "Training eth3d — KL+Cos + RKD Angle + RKD Distance KL"
echo "============================================================"

python -u ./scripts/train_distill.py \
    --dataset eth3d \
    --samples_per_scene 5 \
    --seeds_list 40 41 42 43 44 \
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
    --use_rkd_distance \
    --rkd_distance_weight 1.0 \
    --output_weight 0.0 \
    --camera_weight 5.0 \
    --depth_weight 3.0 \
    --depth_gradient_loss grad \
    --depth_valid_range 0.98 \
    --epochs 3 \
    --batch_size 1 \
    --num_workers 2 \
    --lr 1e-4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lr_scheduler cosine \
    --warmup_steps 50 \
    --weight_decay 1e-5 \
    --resample_per_epoch \
    --output_dir ${OUTPUT_DIR}

echo "Training complete!"

echo ""
echo "============================================================"
echo "Benchmarking LoRA epochs — eth3d, 4v and 8v"
echo "============================================================"

for EPOCH in 0 1 2; do
    LORA_PATH="${OUTPUT_DIR}/epoch_${EPOCH}_lora.pt"
    [ ! -f "${LORA_PATH}" ] && continue
    for MF in 4 8; do
        echo "  [LoRA epoch=${EPOCH}] eth3d, max_frames=${MF}"
        python -u ./scripts/benchmark_lora.py \
            --lora_path "${LORA_PATH}" \
            --base_model ${MODEL_NAME} \
            --lora_rank 32 \
            --lora_alpha 32 \
            --datasets eth3d \
            --modes pose recon_unposed \
            --max_frames ${MF} \
            --seed 43 \
            --work_dir "${BENCHMARK_ROOT}/lora_epoch${EPOCH}/frames_${MF}/eth3d"
    done
done

echo ""
echo "============================================================"
echo "Done! Results in: ${BENCHMARK_ROOT}/"
echo "============================================================"

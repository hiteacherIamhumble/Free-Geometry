#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
OUTPUT_DIR=./checkpoints/da3_kl_cos_cam_rkd/scannetpp
BENCHMARK_ROOT=./workspace/da3_kl_cos_cam_rkd

# Train: KL + cosine + camera token RKD
python -u ./scripts/train_distill.py \
    --dataset scannetpp \
    --samples_per_scene 5 \
    --seeds_list 30 31 32 33 34 \
    --use_scannetpp_test_split \
    --model_name ${MODEL_NAME} \
    --num_views 8 \
    --combined_loss \
    --all_token_softmax_kl_cosine \
    --all_token_softmax_kl_weight 1.0 \
    --all_token_softmax_cos_weight 2.0 \
    --use_cam_rkd \
    --rkd_weight 2.0 \
    --rkd_angle1_weight 1.0 \
    --rkd_angle2_weight 1.0 \
    --rkd_angle3_weight 1.0 \
    --output_weight 0.0 \
    --camera_weight 5.0 \
    --depth_weight 1.0 \
    --depth_gradient_loss grad \
    --depth_valid_range 0.98 \
    --epochs 1 \
    --batch_size 1 \
    --num_workers 2 \
    --lr 1e-4 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --log_interval 1 \
    --save_interval 0 \
    --output_dir ${OUTPUT_DIR}

# Benchmark
python -u ./scripts/benchmark_lora.py \
    --lora_path ${OUTPUT_DIR}/epoch_0_lora.pt \
    --base_model ${MODEL_NAME} \
    --lora_rank 16 \
    --lora_alpha 16 \
    --datasets scannetpp \
    --modes pose recon_unposed \
    --max_frames 100 \
    --seed 43 \
    --work_dir ${BENCHMARK_ROOT}/lora/scannetpp

echo "Done!"

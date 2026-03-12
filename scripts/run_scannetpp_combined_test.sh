#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Train scannetpp with output loss only (feat=0, output=1.0)
python ./scripts/train_vggt_combined_distill.py \
    --dataset scannetpp \
    --samples_per_scene 10 \
    --seeds_list $(seq -s ' ' 30 39) \
    --model_name facebook/vggt-1b \
    --num_views 8 \
    --output_layers 23 \
    --image_size 504 \
    --use_amp \
    --all_token_kl_weight 0.0 \
    --all_token_cos_weight 0.0 \
    --rkd_weight 0.0 \
    --rkd_topk 4 \
    --rkd_num_ref_samples 256 \
    --rkd_num_shared_samples 256 \
    --rkd_angle1_weight 1.0 \
    --rkd_angle2_weight 1.0 \
    --rkd_angle3_weight 1.0 \
    --output_weight 1.0 \
    --camera_weight 5.0 \
    --depth_weight 1.0 \
    --point_weight 1.0 \
    --epochs 1 \
    --batch_size 1 \
    --num_workers 2 \
    --lr 1e-4 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_layers_start 0 \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir ./checkpoints/combined_all_datasets/scannetpp \
    --log_interval 1 \
    --save_interval 0

# Benchmark maxframe=50, seed 42
python ./scripts/benchmark_lora_vggt.py \
    --lora_path ./checkpoints/combined_all_datasets/scannetpp/epoch_0_lora.pt \
    --base_model facebook/vggt-1b \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_layers_start 0 \
    --datasets scannetpp \
    --modes pose recon_unposed \
    --max_frames 50 \
    --seed 42 \
    --image_size 504 \
    --work_dir ./workspace/combined_all_datasets/lora_maxframe/scannetpp/seed42

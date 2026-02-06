#!/usr/bin/env bash
set -e

# VGGT Knowledge Distillation Training Script
# This script trains a student VGGT model (4 views + LoRA) to match a teacher (8 views, frozen)

TRAIN_ROOT=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset
MODEL_NAME=facebook/vggt-1b
EPOCHS=2
LR=1e-4
BS=1
LORA_RANK=16
LORA_ALPHA=16
SEEDS_LIST="39 40 41"
SAMPLES_PER_SCENE=3

# Loss weights
FRAME_KL_WEIGHT=1.0
FRAME_COS_WEIGHT=2.0
GLOBAL_KL_WEIGHT=1.0
GLOBAL_COS_WEIGHT=2.0

run_scannetpp() {
  DATASET=scannetpp
  OUTPUT_DIR=./checkpoints/vggt_distill_${DATASET}_${EPOCHS}epoch_${LR}

  echo "=== Training VGGT ${DATASET} LoRA (combined frame+global softmax KL+cos) ==="
  python ./scripts/train_distill_vggt.py \
    --data_root ${TRAIN_ROOT}/${DATASET} \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS_LIST} \
    --first_frame_ref \
    --num_views 8 \
    --output_layers 19 23 \
    --combined_token_softmax_kl_cosine \
    --frame_kl_weight ${FRAME_KL_WEIGHT} \
    --frame_cos_weight ${FRAME_COS_WEIGHT} \
    --global_kl_weight ${GLOBAL_KL_WEIGHT} \
    --global_cos_weight ${GLOBAL_COS_WEIGHT} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --num_workers 4 \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start 12 \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir ${OUTPUT_DIR}

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  echo "Training complete. LoRA weights saved to: ${LORA_PATH}"
}

run_eth3d() {
  DATASET=eth3d
  OUTPUT_DIR=./checkpoints/vggt_distill_${DATASET}_${EPOCHS}epoch_${LR}

  echo "=== Training VGGT ${DATASET} LoRA (combined frame+global softmax KL+cos) ==="
  python ./scripts/train_distill_vggt.py \
    --data_root ${TRAIN_ROOT}/${DATASET} \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS_LIST} \
    --first_frame_ref \
    --num_views 8 \
    --output_layers 19 23 \
    --combined_token_softmax_kl_cosine \
    --frame_kl_weight ${FRAME_KL_WEIGHT} \
    --frame_cos_weight ${FRAME_COS_WEIGHT} \
    --global_kl_weight ${GLOBAL_KL_WEIGHT} \
    --global_cos_weight ${GLOBAL_COS_WEIGHT} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --num_workers 4 \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start 12 \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir ${OUTPUT_DIR}

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  echo "Training complete. LoRA weights saved to: ${LORA_PATH}"
}

run_7scenes() {
  DATASET=7scenes
  OUTPUT_DIR=./checkpoints/vggt_distill_${DATASET}_${EPOCHS}epoch_${LR}

  echo "=== Training VGGT ${DATASET} LoRA (combined frame+global softmax KL+cos) ==="
  python ./scripts/train_distill_vggt.py \
    --data_root ${TRAIN_ROOT}/${DATASET} \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS_LIST} \
    --first_frame_ref \
    --num_views 8 \
    --output_layers 19 23 \
    --combined_token_softmax_kl_cosine \
    --frame_kl_weight ${FRAME_KL_WEIGHT} \
    --frame_cos_weight ${FRAME_COS_WEIGHT} \
    --global_kl_weight ${GLOBAL_KL_WEIGHT} \
    --global_cos_weight ${GLOBAL_COS_WEIGHT} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --num_workers 4 \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start 12 \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir ${OUTPUT_DIR}

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  echo "Training complete. LoRA weights saved to: ${LORA_PATH}"
}

run_hiroom() {
  DATASET=hiroom
  OUTPUT_DIR=./checkpoints/vggt_distill_${DATASET}_${EPOCHS}epoch_${LR}

  echo "=== Training VGGT ${DATASET} LoRA (combined frame+global softmax KL+cos) ==="
  python ./scripts/train_distill_vggt.py \
    --data_root ${TRAIN_ROOT}/${DATASET} \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS_LIST} \
    --first_frame_ref \
    --num_views 8 \
    --output_layers 19 23 \
    --combined_token_softmax_kl_cosine \
    --frame_kl_weight ${FRAME_KL_WEIGHT} \
    --frame_cos_weight ${FRAME_COS_WEIGHT} \
    --global_kl_weight ${GLOBAL_KL_WEIGHT} \
    --global_cos_weight ${GLOBAL_COS_WEIGHT} \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --num_workers 4 \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start 12 \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir ${OUTPUT_DIR}

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  echo "Training complete. LoRA weights saved to: ${LORA_PATH}"
}

# Run with all-token loss (alternative)
run_scannetpp_all_token() {
  DATASET=scannetpp
  OUTPUT_DIR=./checkpoints/vggt_distill_${DATASET}_all_token_${EPOCHS}epoch_${LR}

  echo "=== Training VGGT ${DATASET} LoRA (all-token softmax KL+cos) ==="
  python ./scripts/train_distill_vggt.py \
    --data_root ${TRAIN_ROOT}/${DATASET} \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS_LIST} \
    --first_frame_ref \
    --num_views 8 \
    --output_layers 19 23 \
    --all_token_softmax_kl_cosine \
    --all_token_kl_weight 1.0 \
    --all_token_cos_weight 2.0 \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --num_workers 4 \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start 12 \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir ${OUTPUT_DIR}

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  echo "Training complete. LoRA weights saved to: ${LORA_PATH}"
}

# Default: run scannetpp
run_scannetpp

echo "All done."

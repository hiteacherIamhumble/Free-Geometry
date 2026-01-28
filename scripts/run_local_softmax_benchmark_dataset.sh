#!/usr/bin/env bash
set -e

# Train one LoRA per benchmark dataset (eth3d, scannetpp, 7scenes, hiroom) using the benchmark_dataset folder.
# For scannetpp, use the 'test' split as requested. Seeds 37-41 => 5 samples/scene.

TRAIN_ROOT=/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset
BASE_MODEL=depth-anything/DA3-GIANT-1.1
EPOCHS=2
LR=1e-4
BS=1
# SEEDS_LIST="34 35 36 37 38 39 40 41"
# SEEDS_LIST="40 41"
SEEDS_LIST="39 40 41"
SAMPLES_PER_SCENE=3
WARMUP=0
LORA_RANK=16
LORA_ALPHA=16
L_KL_WEIGHT=2.0
L_COS_WEIGHT=4.0
G_KL_WEIGHT=0.0
G_COS_WEIGHT=0.0

run_eth3d() {
  DATASET=eth3d
  OUTPUT_DIR=./checkpoints/distill_${DATASET}_local_softmax_${EPOCHS}epoch_${LR}
  WORK_DIR=./workspace/eval_${DATASET}_local_softmax_seed42_${EPOCHS}epoch_${LR}

  # echo "=== Training ${DATASET} LoRA (local softmax KL+cos) ==="
  # python ./scripts/train_distill.py \
  #   --data_root ${TRAIN_ROOT}/${DATASET} \
  #   --dataset ${DATASET} \
  #   --model_name ${BASE_MODEL} \
  #   --samples_per_scene ${SAMPLES_PER_SCENE} \
  #   --seeds_list ${SEEDS_LIST} \
  #   --first_frame_ref \
  #   --num_views 8 \
  #   --local_token_softmax_kl_cosine \
  #   --local_token_softmax_kl_weight ${L_KL_WEIGHT} \
  #   --local_token_softmax_cos_weight ${L_COS_WEIGHT} \
  #   --epochs ${EPOCHS} \
  #   --batch_size ${BS} \
  #   --num_workers 4 \
  #   --lr ${LR} \
  #   --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
  #   --warmup_steps ${WARMUP} \
  #   --output_dir ${OUTPUT_DIR}
  # echo "=== Training ${DATASET} LoRA (local softmax KL+cos) ==="
  # python ./scripts/train_distill.py \
  #   --data_root ${TRAIN_ROOT}/${DATASET} \
  #   --dataset ${DATASET} \
  #   --model_name ${BASE_MODEL} \
  #   --samples_per_scene ${SAMPLES_PER_SCENE} \
  #   --seeds_list ${SEEDS_LIST} \
  #   --first_frame_ref \
  #   --num_views 8 \
  #   --combined_token_softmax_kl_cosine \
  #   --local_token_softmax_kl_weight ${L_KL_WEIGHT} \
  #   --local_token_softmax_cos_weight ${L_COS_WEIGHT} \
  #   --global_token_softmax_kl_weight ${G_KL_WEIGHT} \
  #   --global_token_softmax_cos_weight ${G_COS_WEIGHT} \
  #   --epochs ${EPOCHS} \
  #   --batch_size ${BS} \
  #   --num_workers 4 \
  #   --lr ${LR} \
  #   --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
  #   --lr_scheduler none \
  #   --weight_decay 1e-5 \
  #   --output_dir ${OUTPUT_DIR}

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  if [ ! -f "${LORA_PATH}" ]; then
    echo "LoRA not found at ${LORA_PATH}, skipping ${DATASET} benchmark."
    return
  fi
  python ./scripts/benchmark_lora.py \
    --lora_path ${LORA_PATH} \
    --base_model ${BASE_MODEL} \
    --datasets ${DATASET} \
    --modes pose recon_unposed \
    --max_frames 8 \
    --seeds 42 \
    --work_dir ${WORK_DIR}
  # echo "=== Benchmarking ${DATASET} (seed 42) ==="
  # python ./scripts/benchmark_lora.py \
  #   --lora_path ${LORA_PATH} \
  #   --base_model ${BASE_MODEL} \
  #   --datasets ${DATASET} \
  #   --modes pose recon_unposed \
  #   --max_frames 4 \
  #   --seeds 42 \
  #   --work_dir ${WORK_DIR}
}

run_scannetpp() {
  DATASET=scannetpp
  OUTPUT_DIR=./checkpoints/distill_${DATASET}_local_softmax_${EPOCHS}epoch_1e-3_v1
  WORK_DIR=./workspace/eval_${DATASET}_local_softmax_seed42_${EPOCHS}epoch_1e-3_v1

  # echo "=== Training ${DATASET} LoRA (local softmax KL+cos) on TEST split ==="
  # python ./scripts/train_distill.py \
  #   --data_root ${TRAIN_ROOT}/${DATASET} \
  #   --dataset ${DATASET} \
  #   --model_name ${BASE_MODEL} \
  #   --samples_per_scene ${SAMPLES_PER_SCENE} \
  #   --seeds_list ${SEEDS_LIST} \
  #   --first_frame_ref \
  #   --num_views 8 \
  #   --combined_token_softmax_kl_cosine \
  #   --local_token_softmax_kl_weight 1 \
  #   --local_token_softmax_cos_weight 2 \
  #   --global_token_softmax_kl_weight 2 \
  #   --global_token_softmax_cos_weight 4 \
  #   --epochs ${EPOCHS} \
  #   --batch_size ${BS} \
  #   --num_workers 4 \
  #   --lr ${LR} \
  #   --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
  #   --warmup_steps ${WARMUP} \
  #   --output_dir ${OUTPUT_DIR} \
  #   --use_scannetpp_test_split

  # LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt

  # echo "=== Benchmarking ${DATASET} (seed 42) ==="
  # python ./scripts/benchmark_lora.py \
  #   --lora_path ${LORA_PATH} \
  #   --base_model ${BASE_MODEL} \
  #   --datasets ${DATASET} \
  #   --modes pose recon_unposed \
  #   --max_frames 4 \
  #   --seeds 42 \
  #   --work_dir ${WORK_DIR}

  # echo "=== Training ${DATASET} LoRA (local softmax KL+cos) ==="
  python ./scripts/train_distill.py \
    --data_root ${TRAIN_ROOT}/${DATASET} \
    --dataset ${DATASET} \
    --model_name ${BASE_MODEL} \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS_LIST} \
    --first_frame_ref \
    --num_views 8 \
    --all_token_softmax_kl_cosine \
    --all_token_softmax_kl_weight 1 \
    --all_token_softmax_cos_weight 2 \
    --epochs ${EPOCHS} \
    --batch_size ${BS} \
    --num_workers 4 \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir ${OUTPUT_DIR} \
    --use_scannetpp_test_split
    # --combined_token_softmax_kl_cosine \
    # --local_token_softmax_kl_weight ${L_KL_WEIGHT} \
    # --local_token_softmax_cos_weight ${L_COS_WEIGHT} \
    # --global_token_softmax_kl_weight ${G_KL_WEIGHT} \
    # --global_token_softmax_cos_weight ${G_COS_WEIGHT} \

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  if [ ! -f "${LORA_PATH}" ]; then
    echo "LoRA not found at ${LORA_PATH}, skipping ${DATASET} benchmark."
    return
  fi

  python ./scripts/benchmark_lora.py \
    --lora_path ${LORA_PATH} \
    --base_model ${BASE_MODEL} \
    --datasets ${DATASET} \
    --modes pose \
    --max_frames 4 \
    --seeds 42 \
    --work_dir ${WORK_DIR}

  python ./scripts/benchmark_lora.py \
    --lora_path ${LORA_PATH} \
    --base_model ${BASE_MODEL} \
    --datasets ${DATASET} \
    --modes pose \
    --max_frames 8 \
    --seeds 42 \
    --work_dir ${WORK_DIR}_8v
}

run_7scenes() {
  DATASET=7scenes
  OUTPUT_DIR=./checkpoints/distill_${DATASET}_local_softmax_${EPOCHS}epoch_${LR}
  WORK_DIR=./workspace/eval_${DATASET}_local_softmax_seed42_${EPOCHS}epoch_${LR}

  # echo "=== Training ${DATASET} LoRA (local softmax KL+cos) ==="
  # python ./scripts/train_distill.py \
  #   --data_root ${TRAIN_ROOT}/${DATASET} \
  #   --dataset ${DATASET} \
  #   --model_name ${BASE_MODEL} \
  #   --samples_per_scene ${SAMPLES_PER_SCENE} \
  #   --seeds_list ${SEEDS_LIST} \
  #   --first_frame_ref \
  #   --num_views 8 \
  #   --epochs ${EPOCHS} \
  #   --batch_size ${BS} \
  #   --num_workers 4 \
  #   --lr ${LR} \
  #   --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
  #   --lr_scheduler none \
  #   --weight_decay 1e-5 \
  #   --output_dir ${OUTPUT_DIR} \
  #   --all_token_softmax_kl_cosine \
  #   --all_token_softmax_kl_weight 2 \
  #   --all_token_softmax_cos_weight 2 
    # --combined_token_softmax_kl_cosine \
    # --local_token_softmax_kl_weight ${L_KL_WEIGHT} \
    # --local_token_softmax_cos_weight ${L_COS_WEIGHT} \
    # --global_token_softmax_kl_weight ${G_KL_WEIGHT} \
    # --global_token_softmax_cos_weight ${G_COS_WEIGHT} \

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt

  # echo "=== Benchmarking ${DATASET} (seed 42) ==="
  # python ./scripts/benchmark_lora.py \
  #   --lora_path ${LORA_PATH} \
  #   --base_model ${BASE_MODEL} \
  #   --datasets ${DATASET} \
  #   --modes pose recon_unposed \
  #   --max_frames 4 \
  #   --seeds 42 \
  #   --work_dir ${WORK_DIR}
  # python ./scripts/benchmark_lora.py \
  #   --lora_path ${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt \
  #   --base_model ${BASE_MODEL} \
  #   --datasets ${DATASET} \
  #   --modes pose recon_unposed\
  #   --max_frames 4 \
  #   --seeds 42 \
  #   --work_dir ${WORK_DIR}
  if [ ! -f "${LORA_PATH}" ]; then
    echo "LoRA not found at ${LORA_PATH}, skipping ${DATASET} benchmark."
    return
  fi
  python ./scripts/benchmark_lora.py \
    --lora_path ${LORA_PATH} \
    --base_model ${BASE_MODEL} \
    --datasets ${DATASET} \
    --modes pose recon_unposed\
    --max_frames 8 \
    --seeds 42 \
    --work_dir ${WORK_DIR}_8v
  # python ./scripts/benchmark_lora.py \
  #   --lora_path ${OUTPUT_DIR}/epoch_2_lora.pt \
  #   --base_model ${BASE_MODEL} \
  #   --datasets ${DATASET} \
  #   --modes pose \
  #   --max_frames 4 \
  #   --seeds 42 \
  #   --work_dir ${WORK_DIR}
}

run_hiroom() {
  DATASET=hiroom
  OUTPUT_DIR=./checkpoints/distill_${DATASET}_local_softmax_${EPOCHS}epoch_${LR}
  WORK_DIR=./workspace/eval_${DATASET}_local_softmax_seed42_${EPOCHS}epoch_${LR}

  # echo "=== Training ${DATASET} LoRA (local softmax KL+cos) ==="
  # python ./scripts/train_distill.py \
  #   --data_root ${TRAIN_ROOT}/${DATASET} \
  #   --dataset ${DATASET} \
  #   --model_name ${BASE_MODEL} \
  #   --samples_per_scene ${SAMPLES_PER_SCENE} \
  #   --seeds_list ${SEEDS_LIST} \
  #   --first_frame_ref \
  #   --num_views 8 \
  #   --combined_token_softmax_kl_cosine \
  #   --local_token_softmax_kl_weight 1 \
  #   --local_token_softmax_cos_weight 2 \
  #   --global_token_softmax_kl_weight 2 \
  #   --global_token_softmax_cos_weight 4 \
  #   --epochs ${EPOCHS} \
  #   --batch_size ${BS} \
  #   --num_workers 4 \
  #   --lr ${LR} \
  #   --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
  #   --warmup_steps ${WARMUP} \
  #   --output_dir ${OUTPUT_DIR}

  # LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt

  # echo "=== Benchmarking ${DATASET} (seed 42) ==="
  # python ./scripts/benchmark_lora.py \
  #   --lora_path ${LORA_PATH} \
  #   --base_model ${BASE_MODEL} \
  #   --datasets ${DATASET} \
  #   --modes pose recon_unposed \
  #   --max_frames 4 \
  #   --seeds 42 \
  #   --work_dir ${WORK_DIR}
  # echo "=== Training ${DATASET} LoRA (local softmax KL+cos) ==="
  # python ./scripts/train_distill.py \
  #   --data_root ${TRAIN_ROOT}/${DATASET} \
  #   --dataset ${DATASET} \
  #   --model_name ${BASE_MODEL} \
  #   --samples_per_scene ${SAMPLES_PER_SCENE} \
  #   --seeds_list ${SEEDS_LIST} \
  #   --first_frame_ref \
  #   --num_views 8 \
  #   --combined_token_softmax_kl_cosine \
  #   --local_token_softmax_kl_weight ${L_KL_WEIGHT} \
  #   --local_token_softmax_cos_weight ${L_COS_WEIGHT} \
  #   --global_token_softmax_kl_weight ${G_KL_WEIGHT} \
  #   --global_token_softmax_cos_weight ${G_COS_WEIGHT} \
  #   --epochs ${EPOCHS} \
  #   --batch_size ${BS} \
  #   --num_workers 4 \
  #   --lr ${LR} \
  #   --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} \
  #   --lr_scheduler none \
  #   --weight_decay 1e-5 \
  #   --output_dir ${OUTPUT_DIR} \
  #   # --use_scannetpp_test_split

  LORA_PATH=${OUTPUT_DIR}/epoch_$((EPOCHS-1))_lora.pt
  if [ ! -f "${LORA_PATH}" ]; then
    echo "LoRA not found at ${LORA_PATH}, skipping ${DATASET} benchmark."
    return
  fi
  python ./scripts/benchmark_lora.py \
    --lora_path ${LORA_PATH} \
    --base_model ${BASE_MODEL} \
    --datasets ${DATASET} \
    --modes pose recon_unposed \
    --max_frames 8 \
    --seeds 42 \
    --work_dir ${WORK_DIR}
}

# run_eth3d
run_scannetpp
# run_7scenes
# run_hiroom

echo "All done."

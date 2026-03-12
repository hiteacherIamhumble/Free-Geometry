#!/usr/bin/env bash
set -euo pipefail

# Retrain hiroom with 5 samples/scene, then benchmark at 50v (seed 42)

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
LORA_RANK=16
LORA_ALPHA=16
MODES="pose recon_unposed"
OUTPUT_DIR=./checkpoints/da3_combined_all/hiroom_5s

# =============================================================================
# Train
# =============================================================================

echo ""
echo "============================================================"
echo "Training hiroom (5 samples/scene, seeds 30-34)"
echo "============================================================"

python -u ./scripts/train_distill.py \
    --dataset hiroom \
    --samples_per_scene 5 \
    --seeds_list 30 31 32 33 34 \
    --model_name ${MODEL_NAME} \
    --num_views 8 \
    --combined_loss \
    --all_token_softmax_kl_cosine \
    --all_token_softmax_kl_weight 1.0 \
    --all_token_softmax_cos_weight 2.0 \
    --rkd_weight 2.0 \
    --rkd_topk 4 \
    --rkd_num_ref_samples 256 \
    --rkd_num_shared_samples 256 \
    --rkd_angle1_weight 1.0 \
    --rkd_angle2_weight 1.0 \
    --rkd_angle3_weight 1.0 \
    --output_weight 2.0 \
    --camera_weight 5.0 \
    --depth_weight 1.0 \
    --depth_gradient_loss grad \
    --depth_valid_range 0.98 \
    --epochs 1 \
    --batch_size 1 \
    --num_workers 2 \
    --lr 1e-4 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --output_dir "${OUTPUT_DIR}"

echo "Training done!"

# =============================================================================
# Benchmark at 50v (baseline + lora_5s)
# =============================================================================

LORA_PATH="${OUTPUT_DIR}/epoch_0_lora.pt"

echo ""
echo "============================================================"
echo "Benchmark hiroom lora_5s at 50v (seed 42)"
echo "============================================================"

python -u ./scripts/benchmark_lora.py \
    --lora_path "${LORA_PATH}" \
    --base_model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --datasets hiroom \
    --modes ${MODES} \
    --max_frames 50 \
    --seed 42 \
    --work_dir "./workspace/da3_50v/lora_5s/hiroom"

echo ""
echo "============================================================"
echo "Done!"
echo "  Checkpoint: ${OUTPUT_DIR}/"
echo "  Results:    ./workspace/da3_50v/lora_5s/hiroom/"
echo "============================================================"

#!/usr/bin/env bash
set -euo pipefail

# Retrain 7scenes with 20 samples/scene, then benchmark at 4v/8v/50v/100v

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
LORA_RANK=16
LORA_ALPHA=16
MODES="pose recon_unposed"
OUTPUT_DIR=./checkpoints/da3_combined_all/7scenes_20s

# =============================================================================
# Train
# =============================================================================

echo ""
echo "============================================================"
echo "Training 7scenes (20 samples/scene, seeds 30-49)"
echo "============================================================"

python -u ./scripts/train_distill.py \
    --dataset 7scenes \
    --samples_per_scene 20 \
    --seeds_list $(seq -s ' ' 30 49) \
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

echo "7scenes retrain done!"

# =============================================================================
# Benchmark at 4v/8v/50v/100v
# =============================================================================

LORA_PATH="${OUTPUT_DIR}/epoch_0_lora.pt"

for VIEWS in 4 8 50 100; do
    echo ""
    echo "============================================================"
    echo "7scenes lora_20s benchmark: max_frames=${VIEWS}"
    echo "============================================================"

    if [ "${VIEWS}" -eq 100 ]; then
        WORK_DIR="./workspace/da3_combined_all/lora_20s/7scenes"
    else
        WORK_DIR="./workspace/da3_${VIEWS}v/lora_20s/7scenes"
    fi

    python -u ./scripts/benchmark_lora.py \
        --lora_path "${LORA_PATH}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --datasets 7scenes \
        --modes ${MODES} \
        --max_frames ${VIEWS} \
        --work_dir "${WORK_DIR}"
done

echo ""
echo "Done!"
echo "  Checkpoint: ${OUTPUT_DIR}/"
echo "  Results:"
echo "    ./workspace/da3_4v/lora_20s/7scenes/"
echo "    ./workspace/da3_8v/lora_20s/7scenes/"
echo "    ./workspace/da3_50v/lora_20s/7scenes/"
echo "    ./workspace/da3_combined_all/lora_20s/7scenes/"

#!/usr/bin/env bash
set -euo pipefail

# ETH3D VGGT LoRA ablation (1 epoch, constant LR, 5 samples/scene)
#
# Experiments:
# 1) all_losses: all-token KL+cos + RKD angle + RKD distance
# 2) no_feat_kl_cos: remove KL+cos (set both weights to 0), keep RKD losses
# 3) no_rkd: remove RKD angle + RKD distance, keep KL+cos
#
# Benchmark:
# - dataset: eth3d
# - modes: pose + recon_unposed
# - max_frames: 4
# - seed: 42

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME="facebook/vggt-1b"
DATASET="eth3d"
SEEDS="40 41 42 43 44"
SAMPLES_PER_SCENE=5
EPOCHS=1
LR=1e-4
LORA_RANK=32
LORA_ALPHA=32
LORA_LAYERS_START=0
IMAGE_SIZE=504

NUM_VIEWS=8
BATCH_SIZE=4
NUM_WORKERS=2
WEIGHT_DECAY=1e-5

OUTPUT_ROOT="./checkpoints/eth3d_vggt_ablation_lora_1epoch"
WORK_ROOT="./workspace/eth3d_vggt_ablation_lora_1epoch"

# Base feature loss settings (VGGT)
ALL_TOKEN_KL_WEIGHT=1.0
ALL_TOKEN_COS_WEIGHT=2.0

# RKD angle settings
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256
RKD_ANGLE1_WEIGHT=1.0
RKD_ANGLE2_WEIGHT=1.0
RKD_ANGLE3_WEIGHT=0.0
RKD_SHARED_CHUNK=64

# RKD distance settings (DA3-style)
RKD_DIST_WEIGHT=1.0
RKD_DIST_CHUNK=16
RKD_DIST_TYPE=l2
RKD_DIST_TEMP=2.0
RKD_DIST_MODE=kl
RKD_DIST_HUBER_BETA=0.5
RKD_D1_WEIGHT=1.0
RKD_D2_WEIGHT=1.0
RKD_D3_WEIGHT=0.0

BENCH_MAX_FRAMES=4
BENCH_SEED=42
BENCH_MODES="pose recon_unposed"

train_and_bench() {
  local EXP_NAME="$1"
  local EXP_KL_W="$2"
  local EXP_COS_W="$3"
  local USE_RKD_ANGLE="$4"  # 1 or 0
  local USE_RKD_DIST="$5"   # 1 or 0

  local OUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
  local WORK_DIR="${WORK_ROOT}/${EXP_NAME}/frames_${BENCH_MAX_FRAMES}/${DATASET}"

  echo ""
  echo "============================================================"
  echo "Experiment: ${EXP_NAME}"
  echo "  all_token_kl_weight=${EXP_KL_W}, all_token_cos_weight=${EXP_COS_W}"
  echo "  use_rkd_angle=${USE_RKD_ANGLE}, use_rkd_distance=${USE_RKD_DIST}"
  echo "============================================================"

  local RKD_ANGLE_ARGS=""
  if [ "${USE_RKD_ANGLE}" = "1" ]; then
    RKD_ANGLE_ARGS="--cross_frame_rkd \
      --rkd_weight ${RKD_WEIGHT} \
      --rkd_topk ${RKD_TOPK} \
      --rkd_num_ref_samples ${RKD_NUM_REF} \
      --rkd_num_shared_samples ${RKD_NUM_SHARED} \
      --rkd_angle1_weight ${RKD_ANGLE1_WEIGHT} \
      --rkd_angle2_weight ${RKD_ANGLE2_WEIGHT} \
      --rkd_angle3_weight ${RKD_ANGLE3_WEIGHT} \
      --rkd_shared_chunk_size ${RKD_SHARED_CHUNK}"
  fi

  local RKD_DIST_ARGS=""
  if [ "${USE_RKD_DIST}" = "1" ]; then
    RKD_DIST_ARGS="--use_rkd_distance \
      --rkd_distance_weight ${RKD_DIST_WEIGHT} \
      --rkd_distance_chunk_size ${RKD_DIST_CHUNK} \
      --rkd_distance_type ${RKD_DIST_TYPE} \
      --rkd_distance_temperature ${RKD_DIST_TEMP} \
      --rkd_distance_mode ${RKD_DIST_MODE} \
      --rkd_distance_huber_beta ${RKD_DIST_HUBER_BETA} \
      --rkd_d1_weight ${RKD_D1_WEIGHT} \
      --rkd_d2_weight ${RKD_D2_WEIGHT} \
      --rkd_d3_weight ${RKD_D3_WEIGHT}"
  fi

  python -u ./scripts/train_distill_vggt.py \
    --dataset "${DATASET}" \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS} \
    --model_name "${MODEL_NAME}" \
    --num_views ${NUM_VIEWS} \
    --image_size ${IMAGE_SIZE} \
    --output_layers 19 23 \
    --all_token_softmax_kl_cosine \
    --all_token_kl_weight ${EXP_KL_W} \
    --all_token_cos_weight ${EXP_COS_W} \
    ${RKD_ANGLE_ARGS} \
    ${RKD_DIST_ARGS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start ${LORA_LAYERS_START} \
    --lr_scheduler constant \
    --warmup_steps 0 \
    --weight_decay ${WEIGHT_DECAY} \
    --output_dir "${OUT_DIR}" \
    --log_interval 1 \
    --save_interval 0

  local LORA_PATH="${OUT_DIR}/epoch_0_lora.pt"
  if [ ! -f "${LORA_PATH}" ]; then
    echo "ERROR: Missing ${LORA_PATH}"
    return 1
  fi

  python -u ./scripts/benchmark_lora_vggt.py \
    --lora_path "${LORA_PATH}" \
    --base_model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start ${LORA_LAYERS_START} \
    --datasets "${DATASET}" \
    --modes ${BENCH_MODES} \
    --max_frames ${BENCH_MAX_FRAMES} \
    --seed ${BENCH_SEED} \
    --image_size ${IMAGE_SIZE} \
    --work_dir "${WORK_DIR}"
}

# 1) All losses
train_and_bench "all_losses" "1.0" "2.0" "1" "1"

# 2) Remove KL + cosine feature loss
train_and_bench "no_feat_kl_cos" "0.0" "0.0" "1" "1"

# 3) Remove RKD angle + RKD distance
train_and_bench "no_rkd" "1.0" "2.0" "0" "0"

echo ""
echo "============================================================"
echo "Ablation summary (ETH3D pose/recon_unposed mean)"
echo "============================================================"
python - <<PY
import json
from pathlib import Path

base = Path("./workspace/eth3d_vggt_ablation_lora_1epoch")
exps = ["all_losses", "no_feat_kl_cos", "no_rkd"]
for exp in exps:
    pose_p = base / exp / "frames_4" / "eth3d" / "metric_results" / "eth3d_pose.json"
    recon_p = base / exp / "frames_4" / "eth3d" / "metric_results" / "eth3d_recon_unposed.json"

    pose_str = "pose: missing"
    recon_str = "recon_unposed: missing"

    if pose_p.exists():
        pose_m = json.loads(pose_p.read_text()).get("mean", {})
        pose_str = f"pose AUC@3={pose_m.get(auc03, float(nan)):.4f}, AUC@30={pose_m.get(auc30, float(nan)):.4f}"
    if recon_p.exists():
        recon_m = json.loads(recon_p.read_text()).get("mean", {})
        recon_str = f"recon_unposed d1={recon_m.get(d1, float(nan)):.4f}, d2={recon_m.get(d2, float(nan)):.4f}"

    print(f"{exp}: {pose_str} | {recon_str}")
PY

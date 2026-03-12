#!/usr/bin/env bash
set -euo pipefail

# ETH3D VGGT layer comparison:
#  - all4_layers: patch huber+cos over [4,11,17,23]
#  - last_layer_only: patch huber+cos over [23]
# RKD angle + RKD distance are always computed on the last output layer.

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME="facebook/vggt-1b"
DATASET="eth3d"
SEEDS="30 31 32 33 34"
SAMPLES_PER_SCENE=5
EPOCHS=1
LR=1e-4
BATCH_SIZE=2
NUM_WORKERS=2
WEIGHT_DECAY=1e-5

LORA_RANK=32
LORA_ALPHA=32
LORA_LAYERS_START=0
IMAGE_SIZE=504
NUM_VIEWS=8

PATCH_HUBER_WEIGHT=1.0
PATCH_HUBER_COS_WEIGHT=2.0
PATCH_HUBER_DELTA=1.0

RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256
RKD_ANGLE1_WEIGHT=1.0
RKD_ANGLE2_WEIGHT=1.0
RKD_ANGLE3_WEIGHT=0.0
RKD_SHARED_CHUNK=64

RKD_DIST_WEIGHT=1.0
RKD_DIST_CHUNK=16
RKD_DIST_TYPE=l2
RKD_DIST_TEMP=2.0
RKD_DIST_MODE=kl
RKD_DIST_HUBER_BETA=0.5
RKD_D1_WEIGHT=1.0
RKD_D2_WEIGHT=1.0
RKD_D3_WEIGHT=0.0

BENCH_MAX_FRAMES=8
BENCH_EVAL_FRAMES=4
BENCH_SEED=42
BENCH_MODES="pose recon_unposed"

OUTPUT_ROOT="./checkpoints/eth3d_vggt_patchhuber_layers_compare"
WORK_ROOT="./workspace/eth3d_vggt_patchhuber_layers_compare"

benchmark_baseline() {
  local WORK_DIR="${WORK_ROOT}/baseline/frames_${BENCH_MAX_FRAMES}/${DATASET}"

  echo ""
  echo "============================================================"
  echo "Benchmark: baseline (original VGGT)"
  echo "============================================================"

  python -u ./scripts/benchmark_lora_vggt.py \
    --base_model "${MODEL_NAME}" \
    --datasets "${DATASET}" \
    --modes ${BENCH_MODES} \
    --max_frames ${BENCH_MAX_FRAMES} \
    --eval_frames ${BENCH_EVAL_FRAMES} \
    --seed ${BENCH_SEED} \
    --image_size ${IMAGE_SIZE} \
    --work_dir "${WORK_DIR}"
}

train_and_bench() {
  local EXP_NAME="$1"
  local OUT_LAYERS="$2"

  local OUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
  local WORK_DIR="${WORK_ROOT}/${EXP_NAME}/frames_${BENCH_MAX_FRAMES}/${DATASET}"

  echo ""
  echo "============================================================"
  echo "Experiment: ${EXP_NAME}"
  echo "  output_layers: ${OUT_LAYERS}"
  echo "  patch_huber over all listed layers (averaged)"
  echo "  rkd layer: last output layer only"
  echo "============================================================"

  python -u ./scripts/train_distill_vggt.py \
    --dataset "${DATASET}" \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS} \
    --model_name "${MODEL_NAME}" \
    --num_views ${NUM_VIEWS} \
    --image_size ${IMAGE_SIZE} \
    --output_layers ${OUT_LAYERS} \
    --patch_huber_cosine \
    --patch_huber_weight ${PATCH_HUBER_WEIGHT} \
    --patch_huber_cos_weight ${PATCH_HUBER_COS_WEIGHT} \
    --patch_huber_delta ${PATCH_HUBER_DELTA} \
    --cross_frame_rkd \
    --rkd_weight ${RKD_WEIGHT} \
    --rkd_topk ${RKD_TOPK} \
    --rkd_num_ref_samples ${RKD_NUM_REF} \
    --rkd_num_shared_samples ${RKD_NUM_SHARED} \
    --rkd_angle1_weight ${RKD_ANGLE1_WEIGHT} \
    --rkd_angle2_weight ${RKD_ANGLE2_WEIGHT} \
    --rkd_angle3_weight ${RKD_ANGLE3_WEIGHT} \
    --rkd_shared_chunk_size ${RKD_SHARED_CHUNK} \
    --use_rkd_distance \
    --rkd_distance_weight ${RKD_DIST_WEIGHT} \
    --rkd_distance_chunk_size ${RKD_DIST_CHUNK} \
    --rkd_distance_type ${RKD_DIST_TYPE} \
    --rkd_distance_temperature ${RKD_DIST_TEMP} \
    --rkd_distance_mode ${RKD_DIST_MODE} \
    --rkd_distance_huber_beta ${RKD_DIST_HUBER_BETA} \
    --rkd_d1_weight ${RKD_D1_WEIGHT} \
    --rkd_d2_weight ${RKD_D2_WEIGHT} \
    --rkd_d3_weight ${RKD_D3_WEIGHT} \
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
    --eval_frames ${BENCH_EVAL_FRAMES} \
    --seed ${BENCH_SEED} \
    --image_size ${IMAGE_SIZE} \
    --work_dir "${WORK_DIR}"
}

benchmark_baseline
train_and_bench "all4_layers" "4 11 17 23"
train_and_bench "last_layer_only" "23"

echo ""
echo "============================================================"
echo "Comparison summary (ETH3D, seed42, max_frames=8, eval_frames=4)"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

base = Path("./workspace/eth3d_vggt_patchhuber_layers_compare")
for exp in ["baseline", "all4_layers", "last_layer_only"]:
    pose_p = base / exp / "frames_8" / "eth3d" / "metric_results" / "eth3d_pose.json"
    recon_p = base / exp / "frames_8" / "eth3d" / "metric_results" / "eth3d_recon_unposed.json"

    if pose_p.exists():
        pose = json.loads(pose_p.read_text()).get("mean", {})
        pose_msg = f"pose AUC@3={pose.get('auc03', float('nan')):.4f}, AUC@30={pose.get('auc30', float('nan')):.4f}"
    else:
        pose_msg = "pose missing"

    if recon_p.exists():
        recon = json.loads(recon_p.read_text()).get("mean", {})
        recon_msg = f"recon_unposed d1={recon.get('d1', float('nan')):.4f}, d2={recon.get('d2', float('nan')):.4f}"
    else:
        recon_msg = "recon_unposed missing"

    print(f"{exp}: {pose_msg} | {recon_msg}")
PY

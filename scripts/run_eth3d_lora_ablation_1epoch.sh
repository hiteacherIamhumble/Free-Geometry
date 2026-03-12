#!/usr/bin/env bash
set -euo pipefail

# ETH3D LoRA ablation (3 epochs, cosine LR with warmup, 5 samples/scene)
#
# Experiments:
# 1) all_losses: patch_huber+cos + RKD angle + RKD distance
# 2) no_feat_patchhuber_cos: remove patch_huber+cos (set both weights to 0)
# 3) no_rkd: remove RKD angle + RKD distance (rkd_weight=0, no distance loss)
#
# Benchmark:
# - dataset: eth3d
# - mode: pose recon_unposed
# - max_frames: 4
# - seed: 42

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
DATASET="eth3d"
SEEDS="40 41 42 43 44"
SAMPLES_PER_SCENE=5
EPOCHS=3
LR=5e-5
LORA_RANK=32
LORA_ALPHA=32

NUM_VIEWS=8
BATCH_SIZE=4
NUM_WORKERS=2
WEIGHT_DECAY=1e-5

OUTPUT_ROOT="./checkpoints/eth3d_ablation_lora_3epoch"
WORK_ROOT="./workspace/eth3d_ablation_lora_3epoch"

RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256
RKD_DIST_WEIGHT=1.0
RKD_DIST_TEMP=10.0
RKD_DIST_MODE=kl
RKD_D1_WEIGHT=1.0
RKD_D2_WEIGHT=1.0
RKD_D3_WEIGHT=0.0

PATCH_HUBER_WEIGHT=1.0
PATCH_HUBER_COS_WEIGHT=2.0
PATCH_HUBER_DELTA=1.0

OUTPUT_WEIGHT=0.0
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=3.0
DEPTH_GRAD=grad
DEPTH_VALID_RANGE=0.98

BENCH_MAX_FRAMES=4
BENCH_SEED=42
BENCH_MODES="pose recon_unposed"

train_and_bench() {
  local EXP_NAME="$1"
  local FEAT_HUBER_W="$2"
  local FEAT_COS_W="$3"
  local EXP_RKD_WEIGHT="$4"
  local USE_RKD_DIST="$5"      # 1 or 0
  local SELECTION_MODE="${6:-topk}"  # topk, random, mixed

  local OUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
  local WORK_DIR="${WORK_ROOT}/${EXP_NAME}/frames_${BENCH_MAX_FRAMES}/${DATASET}"

  echo ""
  echo "============================================================"
  echo "Experiment: ${EXP_NAME}"
  echo "  feature: patch_huber_w=${FEAT_HUBER_W}, patch_huber_cos_w=${FEAT_COS_W}"
  echo "  rkd_weight=${EXP_RKD_WEIGHT}, use_rkd_distance=${USE_RKD_DIST}, selection=${SELECTION_MODE}"
  echo "  epochs=${EPOCHS}, lr=${LR}, scheduler=cosine, warmup=50"
  echo "============================================================"

  local RKD_DIST_ARGS=""
  if [ "${USE_RKD_DIST}" = "1" ]; then
    RKD_DIST_ARGS="--use_rkd_distance \
      --rkd_distance_weight ${RKD_DIST_WEIGHT} \
      --rkd_distance_temperature ${RKD_DIST_TEMP} \
      --rkd_distance_mode ${RKD_DIST_MODE} \
      --rkd_d1_weight ${RKD_D1_WEIGHT} \
      --rkd_d2_weight ${RKD_D2_WEIGHT} \
      --rkd_d3_weight ${RKD_D3_WEIGHT}"
  fi

  python -u ./scripts/train_distill.py \
    --dataset "${DATASET}" \
    --samples_per_scene ${SAMPLES_PER_SCENE} \
    --seeds_list ${SEEDS} \
    --model_name "${MODEL_NAME}" \
    --num_views ${NUM_VIEWS} \
    --combined_loss \
    --patch_huber_cosine \
    --patch_huber_weight ${FEAT_HUBER_W} \
    --patch_huber_cos_weight ${FEAT_COS_W} \
    --patch_huber_delta ${PATCH_HUBER_DELTA} \
    --distill_all_layers \
    --rkd_weight ${EXP_RKD_WEIGHT} \
    --rkd_topk ${RKD_TOPK} \
    --rkd_num_ref_samples ${RKD_NUM_REF} \
    --rkd_num_shared_samples ${RKD_NUM_SHARED} \
    --rkd_angle1_weight 1.0 \
    --rkd_angle2_weight 1.0 \
    --rkd_angle3_weight 1.0 \
    --rkd_selection_mode ${SELECTION_MODE} \
    ${RKD_DIST_ARGS} \
    --output_weight ${OUTPUT_WEIGHT} \
    --camera_weight ${CAMERA_WEIGHT} \
    --depth_weight ${DEPTH_WEIGHT} \
    --depth_gradient_loss ${DEPTH_GRAD} \
    --depth_valid_range ${DEPTH_VALID_RANGE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LR} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lr_scheduler cosine \
    --warmup_steps 50 \
    --weight_decay ${WEIGHT_DECAY} \
    --output_dir "${OUT_DIR}"

  # Benchmark the last epoch
  local LAST_EPOCH=$((EPOCHS - 1))
  local LORA_PATH="${OUT_DIR}/epoch_${LAST_EPOCH}_lora.pt"
  if [ ! -f "${LORA_PATH}" ]; then
    echo "ERROR: Missing ${LORA_PATH}"
    return 1
  fi

  python -u ./scripts/benchmark_lora.py \
    --lora_path "${LORA_PATH}" \
    --base_model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --datasets "${DATASET}" \
    --modes ${BENCH_MODES} \
    --max_frames ${BENCH_MAX_FRAMES} \
    --seed ${BENCH_SEED} \
    --work_dir "${WORK_DIR}"
}

# # 1) All losses (topk selection - default) — already have results
# train_and_bench "all_losses" "1.0" "2.0" "2.0" "1" "topk"

# # 2) Remove patch huber + cosine loss — skip for now
# train_and_bench "no_feat_patchhuber_cos" "0.0" "0.0" "2.0" "1" "topk"

# # 3) Remove RKD angle + RKD distance losses — skip for now
# train_and_bench "no_rkd" "1.0" "2.0" "0.0" "0" "topk"

# 4) All losses with random patch selection
train_and_bench "all_losses_random" "1.0" "2.0" "2.0" "1" "random"

# 5) All losses with mixed (top2 + bottom2) patch selection
train_and_bench "all_losses_mixed" "1.0" "2.0" "2.0" "1" "mixed"

echo ""
echo "============================================================"
echo "Ablation summary (ETH3D, 3 epochs, cosine LR)"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

base = Path("./workspace/eth3d_ablation_lora_3epoch")
exps = ["all_losses", "no_feat_patchhuber_cos", "no_rkd", "all_losses_random", "all_losses_mixed"]
header = f"{'experiment':<30s} {'AUC@3':>8s} {'AUC@30':>8s} {'F-score':>8s} {'Overall':>8s}"
print(header)
print("-" * len(header))
for exp in exps:
    pose_p = base / exp / "frames_4" / "eth3d" / "metric_results" / "eth3d_pose.json"
    recon_p = base / exp / "frames_4" / "eth3d" / "metric_results" / "eth3d_recon_unposed.json"

    auc03 = auc30 = fscore = overall = float('nan')
    if pose_p.exists():
        m = json.loads(pose_p.read_text()).get("mean", {})
        auc03 = m.get("auc03", float('nan'))
        auc30 = m.get("auc30", float('nan'))
    if recon_p.exists():
        m = json.loads(recon_p.read_text()).get("mean", {})
        fscore = m.get("fscore", float('nan'))
        overall = m.get("overall", float('nan'))

    print(f"{exp:<30s} {auc03:>8.4f} {auc30:>8.4f} {fscore:>8.4f} {overall:>8.4f}")
PY

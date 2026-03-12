#!/usr/bin/env bash
set -euo pipefail

# Compare RKD distance modes on 7Scenes (same settings, only mode differs):
# - mode=kl
# - mode=huber
#
# Training/benchmark settings are aligned with current DA3 settings for 7scenes:
# - samples_per_scene=10, seeds=40..49, epochs=5
# - batch_size=4, lr=5e-6, warmup=20% total steps, eta_min=1e-8
# - LoRA mode + patch_huber + RKD angle + RKD distance, output_weight=0
# - benchmark: pose only, 4 views

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

MODEL_NAME="depth-anything/DA3-GIANT-1.1"
DATASET="7scenes"

NUM_VIEWS=8
BATCH_SIZE=4
SAMPLES=10
SEED_START=40
SEED_END=49
SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"
EPOCHS=5
LR=5e-6
ETA_MIN=1e-8

# 7Scenes has 7 scenes; warmup = 20% of total steps.
SCENE_COUNT=7
NUM_SEEDS=$((SEED_END - SEED_START + 1))
STEPS_PER_EPOCH=$(( (SCENE_COUNT * NUM_SEEDS) / BATCH_SIZE ))
if [ "${STEPS_PER_EPOCH}" -lt 1 ]; then
  STEPS_PER_EPOCH=1
fi
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
WARMUP_STEPS=$(( (TOTAL_STEPS + 4) / 5 ))

RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF=256
RKD_NUM_SHARED=256
RKD_DIST_WEIGHT=1.0
RKD_DIST_TEMP=10.0
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

MAX_FRAMES=4
BENCH_MODES="pose"
BENCH_SEED=43

run_mode() {
  local MODE="$1"   # kl | huber
  local OUTPUT_DIR="./checkpoints/da3_7scenes_lora_rkd_${MODE}"
  local BENCH_ROOT="./workspace/da3_7scenes_lora_rkd_${MODE}"

  echo ""
  echo "============================================================"
  echo "Training 7Scenes with RKD distance mode=${MODE}"
  echo "  Seeds: ${SEEDS}"
  echo "  Steps/epoch=${STEPS_PER_EPOCH}, total=${TOTAL_STEPS}, warmup=${WARMUP_STEPS} (20%)"
  echo "  Output: ${OUTPUT_DIR}"
  echo "============================================================"

  python -u ./scripts/train_distill.py \
    --dataset "${DATASET}" \
    --samples_per_scene ${SAMPLES} \
    --seeds_list ${SEEDS} \
    --model_name "${MODEL_NAME}" \
    --num_views ${NUM_VIEWS} \
    --combined_loss \
    --patch_huber_cosine \
    --patch_huber_weight ${PATCH_HUBER_WEIGHT} \
    --patch_huber_cos_weight ${PATCH_HUBER_COS_WEIGHT} \
    --patch_huber_delta ${PATCH_HUBER_DELTA} \
    --distill_all_layers \
    --rkd_weight ${RKD_WEIGHT} \
    --rkd_topk ${RKD_TOPK} \
    --rkd_num_ref_samples ${RKD_NUM_REF} \
    --rkd_num_shared_samples ${RKD_NUM_SHARED} \
    --rkd_angle1_weight 1.0 \
    --rkd_angle2_weight 1.0 \
    --rkd_angle3_weight 1.0 \
    --use_rkd_distance \
    --rkd_distance_weight ${RKD_DIST_WEIGHT} \
    --rkd_distance_temperature ${RKD_DIST_TEMP} \
    --rkd_distance_mode ${MODE} \
    --rkd_d1_weight ${RKD_D1_WEIGHT} \
    --rkd_d2_weight ${RKD_D2_WEIGHT} \
    --rkd_d3_weight ${RKD_D3_WEIGHT} \
    --output_weight ${OUTPUT_WEIGHT} \
    --camera_weight ${CAMERA_WEIGHT} \
    --depth_weight ${DEPTH_WEIGHT} \
    --depth_gradient_loss ${DEPTH_GRAD} \
    --depth_valid_range ${DEPTH_VALID_RANGE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers 2 \
    --lr ${LR} \
    --lr_scheduler cosine \
    --eta_min ${ETA_MIN} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay 1e-5 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --output_dir "${OUTPUT_DIR}"

  echo ""
  echo "Benchmarking 7Scenes pose for mode=${MODE}"
  for EPOCH in 0 1 2 3 4; do
    local WEIGHTS_PATH="${OUTPUT_DIR}/epoch_${EPOCH}_lora.pt"
    if [ ! -f "${WEIGHTS_PATH}" ]; then
      echo "  WARNING: missing ${WEIGHTS_PATH}, skip."
      continue
    fi
    python -u ./scripts/benchmark_lora.py \
      --lora_path "${WEIGHTS_PATH}" \
      --base_model "${MODEL_NAME}" \
      --datasets "${DATASET}" \
      --modes ${BENCH_MODES} \
      --max_frames ${MAX_FRAMES} \
      --seed ${BENCH_SEED} \
      --work_dir "${BENCH_ROOT}/epoch${EPOCH}/frames_${MAX_FRAMES}/${DATASET}"
  done
}

run_mode "kl"
run_mode "huber"

echo ""
echo "============================================================"
echo "Compare summary (AUC@3/AUC@30 by epoch)"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

def load_rows(mode):
    rows = []
    for ep in range(5):
        p = Path(f"./workspace/da3_7scenes_lora_rkd_{mode}/epoch{ep}/frames_4/7scenes/metric_results/7scenes_pose.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        m = d.get("mean", {})
        rows.append((ep, float(m.get("auc03", float("nan"))), float(m.get("auc30", float("nan")))))
    return rows

rows_kl = load_rows("kl")
rows_hb = load_rows("huber")
all_eps = sorted(set([r[0] for r in rows_kl] + [r[0] for r in rows_hb]))

def to_map(rows):
    return {ep: (a3, a30) for ep, a3, a30 in rows}

mk = to_map(rows_kl)
mh = to_map(rows_hb)

print("epoch |   kl_auc3  kl_auc30 | huber_auc3 huber_auc30")
for ep in all_eps:
    ka3, ka30 = mk.get(ep, (float("nan"), float("nan")))
    ha3, ha30 = mh.get(ep, (float("nan"), float("nan")))
    print(f"{ep:>5} | {ka3:>8.4f} {ka30:>9.4f} | {ha3:>10.4f} {ha30:>11.4f}")

if rows_kl:
    b = max(rows_kl, key=lambda x: (x[1], x[2], -x[0]))
    print(f"best_kl: epoch={b[0]}, auc3={b[1]:.4f}, auc30={b[2]:.4f}")
if rows_hb:
    b = max(rows_hb, key=lambda x: (x[1], x[2], -x[0]))
    print(f"best_huber: epoch={b[0]}, auc3={b[1]:.4f}, auc30={b[2]:.4f}")
PY

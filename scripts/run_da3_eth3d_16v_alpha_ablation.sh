#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DA3 ETH3D 16-view LoRA alpha ablation
#
# Reference training recipe:
#   scripts/run_all.sh -> scripts/run_da3.sh
#
# This script keeps the current DA3 loss stack from run_da3.sh:
#   - combined_loss
#   - patch_huber_cosine
#   - distill_all_layers
#   - RKD + RKD distance
#   - output/camera/depth weights from the current DA3 recipe
#
# Requested changes for this ablation:
#   - dataset: ETH3D
#   - teacher input views: 16
#   - batch_size: 1
#   - LoRA (rank, alpha) sweep: (8,8), (16,16), (32,32), (64,64)
#   - LoRA stays on the default DA3 layers 13-39 in StudentModel
#   - one additional full-finetune experiment with layers 13-39 unfrozen
#   - benchmark each LoRA model at 16v
#   - benchmark frozen baseline at 16v
# =============================================================================

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate da3
PYTHON_BIN="${CONDA_PREFIX:-/root/miniconda3/envs/da3}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: da3 env python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p logs

MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
DATASET="eth3d"
NUM_VIEWS=16
MAX_FRAMES=16
BATCH_SIZE=1
NUM_WORKERS="${NUM_WORKERS:-2}"

# ETH3D settings from the current DA3 training recipe in scripts/run_da3.sh
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-5}"
TRAIN_SEEDS="${TRAIN_SEEDS:-40 41 42 43 44}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-5}"
LORA_VALUES=(${LORA_VALUES:-8 16 32 64})
FINETUNE_LR="${FINETUNE_LR:-5e-6}"

# Bench seed kept explicit and simple for this ablation.
BENCH_SEED="${BENCH_SEED:-43}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints/da3_eth3d_16v_alpha_ablation}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-./workspace/da3_eth3d_16v_alpha_ablation}"

# Current DA3 loss recipe from scripts/run_da3.sh
RKD_WEIGHT="${RKD_WEIGHT:-2.0}"
RKD_TOPK="${RKD_TOPK:-4}"
RKD_NUM_REF="${RKD_NUM_REF:-256}"
RKD_NUM_SHARED="${RKD_NUM_SHARED:-256}"

PATCH_HUBER_WEIGHT="${PATCH_HUBER_WEIGHT:-1.0}"
PATCH_HUBER_COS_WEIGHT="${PATCH_HUBER_COS_WEIGHT:-2.0}"
PATCH_HUBER_DELTA="${PATCH_HUBER_DELTA:-1.0}"

RKD_DIST_WEIGHT="${RKD_DIST_WEIGHT:-1.0}"
RKD_DIST_TEMP="${RKD_DIST_TEMP:-10.0}"
RKD_D1_WEIGHT="${RKD_D1_WEIGHT:-1.0}"
RKD_D2_WEIGHT="${RKD_D2_WEIGHT:-1.0}"
RKD_D3_WEIGHT="${RKD_D3_WEIGHT:-0.0}"
RKD_DIST_MODE="${RKD_DIST_MODE:-kl}"

OUTPUT_WEIGHT="${OUTPUT_WEIGHT:-0.0}"
CAMERA_WEIGHT="${CAMERA_WEIGHT:-5.0}"
DEPTH_WEIGHT="${DEPTH_WEIGHT:-3.0}"
DEPTH_GRAD="${DEPTH_GRAD:-grad}"
DEPTH_VALID_RANGE="${DEPTH_VALID_RANGE:-0.98}"

train_alpha() {
    local lora_value="$1"
    local out_dir="${OUTPUT_ROOT}/rank_${lora_value}_alpha_${lora_value}"

    echo
    echo "============================================================"
    echo "Training DA3 ETH3D 16v LoRA"
    echo "  rank=${lora_value}, alpha=${lora_value}"
    echo "  dataset=${DATASET}, num_views=${NUM_VIEWS}, batch_size=${BATCH_SIZE}"
    echo "  output_dir=${out_dir}"
    echo "  note: DA3 StudentModel applies LoRA to layers 13-39 by default"
    echo "============================================================"

    "${PYTHON_BIN}" -u ./scripts/train_distill.py \
        --dataset "${DATASET}" \
        --samples_per_scene "${SAMPLES_PER_SCENE}" \
        --seeds_list ${TRAIN_SEEDS} \
        --model_name "${MODEL_NAME}" \
        --num_views "${NUM_VIEWS}" \
        --combined_loss \
        --patch_huber_cosine \
        --patch_huber_weight "${PATCH_HUBER_WEIGHT}" \
        --patch_huber_cos_weight "${PATCH_HUBER_COS_WEIGHT}" \
        --patch_huber_delta "${PATCH_HUBER_DELTA}" \
        --distill_all_layers \
        --rkd_weight "${RKD_WEIGHT}" \
        --rkd_topk "${RKD_TOPK}" \
        --rkd_num_ref_samples "${RKD_NUM_REF}" \
        --rkd_num_shared_samples "${RKD_NUM_SHARED}" \
        --rkd_angle1_weight 1.0 \
        --rkd_angle2_weight 1.0 \
        --rkd_angle3_weight 1.0 \
        --rkd_selection_mode mixed \
        --use_rkd_distance \
        --rkd_distance_weight "${RKD_DIST_WEIGHT}" \
        --rkd_distance_temperature "${RKD_DIST_TEMP}" \
        --rkd_distance_mode "${RKD_DIST_MODE}" \
        --rkd_d1_weight "${RKD_D1_WEIGHT}" \
        --rkd_d2_weight "${RKD_D2_WEIGHT}" \
        --rkd_d3_weight "${RKD_D3_WEIGHT}" \
        --output_weight "${OUTPUT_WEIGHT}" \
        --camera_weight "${CAMERA_WEIGHT}" \
        --depth_weight "${DEPTH_WEIGHT}" \
        --depth_gradient_loss "${DEPTH_GRAD}" \
        --depth_valid_range "${DEPTH_VALID_RANGE}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --lr "${LR}" \
        --lora_rank "${lora_value}" \
        --lora_alpha "${lora_value}" \
        --lr_scheduler cosine \
        --warmup_ratio 0.15 \
        --eta_min 1e-7 \
        --weight_decay 1e-5 \
        --output_dir "${out_dir}"
}

benchmark_lora_alpha() {
    local lora_value="$1"
    local epoch="${2:-$((EPOCHS - 1))}"
    local lora_path="${OUTPUT_ROOT}/rank_${lora_value}_alpha_${lora_value}/epoch_${epoch}_lora.pt"
    local work_dir="${BENCHMARK_ROOT}/lora_rank_${lora_value}_alpha_${lora_value}/frames_${MAX_FRAMES}/${DATASET}"

    if [[ ! -f "${lora_path}" ]]; then
        echo "ERROR: Missing LoRA weights: ${lora_path}"
        return 1
    fi

    echo
    echo "============================================================"
    echo "Benchmarking DA3 ETH3D 16v LoRA"
    echo "  rank=${lora_value}, alpha=${lora_value}, epoch=${epoch}, seed=${BENCH_SEED}"
    echo "  work_dir=${work_dir}"
    echo "============================================================"

    "${PYTHON_BIN}" -u ./scripts/benchmark_lora.py \
        --lora_path "${lora_path}" \
        --base_model "${MODEL_NAME}" \
        --lora_rank "${lora_value}" \
        --lora_alpha "${lora_value}" \
        --datasets "${DATASET}" \
        --modes pose recon_unposed \
        --max_frames "${MAX_FRAMES}" \
        --seed "${BENCH_SEED}" \
        --work_dir "${work_dir}"
}

benchmark_baseline() {
    local work_dir="${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed${BENCH_SEED}"

    echo
    echo "============================================================"
    echo "Benchmarking frozen DA3 baseline"
    echo "  dataset=${DATASET}, max_frames=${MAX_FRAMES}, seed=${BENCH_SEED}"
    echo "  work_dir=${work_dir}"
    echo "============================================================"

    "${PYTHON_BIN}" -u -c "
import json, os, sys, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
api = DepthAnything3.from_pretrained('${MODEL_NAME}').to(device)
evaluator = Evaluator(
    work_dir='${work_dir}',
    datas=['${DATASET}'],
    modes=['pose', 'recon_unposed'],
    max_frames=${MAX_FRAMES},
    seed=${BENCH_SEED},
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
os.makedirs('${work_dir}', exist_ok=True)
with open(os.path.join('${work_dir}', 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
"
}

train_finetune() {
    local out_dir="${OUTPUT_ROOT}/finetune_13_39"

    echo
    echo "============================================================"
    echo "Training DA3 ETH3D 16v full finetune"
    echo "  unfrozen layers: 13-39"
    echo "  dataset=${DATASET}, num_views=${NUM_VIEWS}, batch_size=${BATCH_SIZE}"
    echo "  lr=${FINETUNE_LR}"
    echo "  output_dir=${out_dir}"
    echo "============================================================"

    "${PYTHON_BIN}" -u ./scripts/train_distill.py \
        --dataset "${DATASET}" \
        --samples_per_scene "${SAMPLES_PER_SCENE}" \
        --seeds_list ${TRAIN_SEEDS} \
        --model_name "${MODEL_NAME}" \
        --num_views "${NUM_VIEWS}" \
        --finetune \
        --combined_loss \
        --patch_huber_cosine \
        --patch_huber_weight "${PATCH_HUBER_WEIGHT}" \
        --patch_huber_cos_weight "${PATCH_HUBER_COS_WEIGHT}" \
        --patch_huber_delta "${PATCH_HUBER_DELTA}" \
        --distill_all_layers \
        --rkd_weight "${RKD_WEIGHT}" \
        --rkd_topk "${RKD_TOPK}" \
        --rkd_num_ref_samples "${RKD_NUM_REF}" \
        --rkd_num_shared_samples "${RKD_NUM_SHARED}" \
        --rkd_angle1_weight 1.0 \
        --rkd_angle2_weight 1.0 \
        --rkd_angle3_weight 1.0 \
        --rkd_selection_mode mixed \
        --use_rkd_distance \
        --rkd_distance_weight "${RKD_DIST_WEIGHT}" \
        --rkd_distance_temperature "${RKD_DIST_TEMP}" \
        --rkd_distance_mode "${RKD_DIST_MODE}" \
        --rkd_d1_weight "${RKD_D1_WEIGHT}" \
        --rkd_d2_weight "${RKD_D2_WEIGHT}" \
        --rkd_d3_weight "${RKD_D3_WEIGHT}" \
        --output_weight "${OUTPUT_WEIGHT}" \
        --camera_weight "${CAMERA_WEIGHT}" \
        --depth_weight "${DEPTH_WEIGHT}" \
        --depth_gradient_loss "${DEPTH_GRAD}" \
        --depth_valid_range "${DEPTH_VALID_RANGE}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --lr "${FINETUNE_LR}" \
        --lr_scheduler cosine \
        --warmup_ratio 0.15 \
        --eta_min 1e-7 \
        --weight_decay 1e-5 \
        --output_dir "${out_dir}"
}

benchmark_finetune() {
    local epoch="${1:-$((EPOCHS - 1))}"
    local weights_path="${OUTPUT_ROOT}/finetune_13_39/epoch_${epoch}.pt"
    local work_dir="${BENCHMARK_ROOT}/finetune_13_39/frames_${MAX_FRAMES}/${DATASET}"

    if [[ ! -f "${weights_path}" ]]; then
        echo "ERROR: Missing finetune weights: ${weights_path}"
        return 1
    fi

    echo
    echo "============================================================"
    echo "Benchmarking DA3 ETH3D 16v full finetune"
    echo "  unfrozen layers: 13-39, epoch=${epoch}, seed=${BENCH_SEED}"
    echo "  work_dir=${work_dir}"
    echo "============================================================"

    "${PYTHON_BIN}" -u ./scripts/benchmark_lora.py \
        --finetune \
        --lora_path "${weights_path}" \
        --base_model "${MODEL_NAME}" \
        --datasets "${DATASET}" \
        --modes pose recon_unposed \
        --max_frames "${MAX_FRAMES}" \
        --seed "${BENCH_SEED}" \
        --work_dir "${work_dir}"
}

train_all() {
    for lora_value in "${LORA_VALUES[@]}"; do
        train_alpha "${lora_value}"
    done
    train_finetune
}

benchmark_all_lora() {
    for lora_value in "${LORA_VALUES[@]}"; do
        benchmark_lora_alpha "${lora_value}"
    done
    benchmark_finetune
}

usage() {
    echo "Usage: $0 [train|benchmark_lora|benchmark_baseline|benchmark_all|all]"
    echo
    echo "Default ablation:"
    echo "  dataset: eth3d"
    echo "  teacher views: 16"
    echo "  benchmark max_frames: 16"
    echo "  batch size: 1"
    echo "  LoRA rank/alpha sweep: ${LORA_VALUES[*]} (matched pairs)"
    echo "  Extra experiment: full finetune with layers 13-39 unfrozen"
    echo "  benchmark seed: ${BENCH_SEED}"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        train_all
        ;;
    benchmark_lora)
        benchmark_all_lora
        ;;
    benchmark_baseline)
        benchmark_baseline
        ;;
    benchmark_all)
        benchmark_baseline
        benchmark_all_lora
        ;;
    all)
        benchmark_baseline
        train_all
        benchmark_all_lora
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        usage
        exit 1
        ;;
esac

echo
echo "============================================================"
echo "Done!"
echo "  Checkpoints: ${OUTPUT_ROOT}/rank_{8,16,32,64}_alpha_{8,16,32,64}/"
echo "  Finetune:    ${OUTPUT_ROOT}/finetune_13_39/"
echo "  Baseline:    ${BENCHMARK_ROOT}/baseline/frames_${MAX_FRAMES}/${DATASET}/seed${BENCH_SEED}/"
echo "  LoRA bench:  ${BENCHMARK_ROOT}/lora_rank_{8,16,32,64}_alpha_{8,16,32,64}/frames_${MAX_FRAMES}/${DATASET}/"
echo "  FT bench:    ${BENCHMARK_ROOT}/finetune_13_39/frames_${MAX_FRAMES}/${DATASET}/"
echo "============================================================"

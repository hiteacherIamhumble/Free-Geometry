#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# VGGT Distillation Ablation: Feature-only, Output-only, Combined
# All fp32 (no --use_amp), same hyperparams, ETH3D maxframe benchmark seed 42
# =============================================================================
#
# Commands:
#   feature    - train feature-only + benchmark
#   output     - train output-only + benchmark
#   combined   - train combined + benchmark
#   all        - run all three
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# === Shared configuration ===
MODEL_NAME=facebook/vggt-1b
LR=1e-4
LORA_RANK=16
LORA_ALPHA=16
LORA_LAYERS_START=0
NUM_VIEWS=8
OUTPUT_LAYERS="19 23"
IMAGE_SIZE=504

# Feature loss settings (V14)
KL_WEIGHT=1.0
COS_WEIGHT=2.0
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF_SAMPLES=256
RKD_NUM_SHARED_SAMPLES=256

# Output loss settings
CAMERA_WEIGHT=5.0
DEPTH_WEIGHT=1.0
POINT_WEIGHT=1.0

# Benchmark
BENCHMARK_SEEDS="42"
MODES="pose recon_unposed"

# === Common training args ===
COMMON_TRAIN_ARGS="--dataset eth3d \
    --samples_per_scene 5 \
    --seeds_list 30 31 32 33 34 \
    --model_name ${MODEL_NAME} \
    --num_views ${NUM_VIEWS} \
    --output_layers ${OUTPUT_LAYERS} \
    --image_size ${IMAGE_SIZE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_layers_start ${LORA_LAYERS_START} \
    --epochs 1 \
    --batch_size 1 \
    --num_workers 2 \
    --lr ${LR} \
    --lr_scheduler none \
    --weight_decay 1e-5 \
    --log_interval 1 \
    --save_interval 0"

# =============================================================================
# Training functions
# =============================================================================

train_feature() {
    local out_dir="./checkpoints/vggt_distill_feature/eth3d"
    echo ""
    echo "============================================================"
    echo "[Feature-only] Training on eth3d (fp32)"
    echo "  output_weight=0 (no output loss)"
    echo "============================================================"

    eval python ./scripts/train_vggt_combined_distill.py \
        ${COMMON_TRAIN_ARGS} \
        --all_token_kl_weight "${KL_WEIGHT}" \
        --all_token_cos_weight "${COS_WEIGHT}" \
        --rkd_weight "${RKD_WEIGHT}" \
        --rkd_topk "${RKD_TOPK}" \
        --rkd_num_ref_samples "${RKD_NUM_REF_SAMPLES}" \
        --rkd_num_shared_samples "${RKD_NUM_SHARED_SAMPLES}" \
        --output_weight 0.0 \
        --camera_weight "${CAMERA_WEIGHT}" \
        --depth_weight "${DEPTH_WEIGHT}" \
        --point_weight "${POINT_WEIGHT}" \
        --output_dir "${out_dir}"

    echo "Feature-only training complete! Model saved to: ${out_dir}/"
}

train_output() {
    local out_dir="./checkpoints/vggt_distill_output/eth3d"
    echo ""
    echo "============================================================"
    echo "[Output-only] Training on eth3d (fp32)"
    echo "  kl=0, cos=0, rkd=0 (no feature loss)"
    echo "============================================================"

    eval python ./scripts/train_vggt_combined_distill.py \
        ${COMMON_TRAIN_ARGS} \
        --all_token_kl_weight 0.0 \
        --all_token_cos_weight 0.0 \
        --rkd_weight 0.0 \
        --rkd_topk "${RKD_TOPK}" \
        --rkd_num_ref_samples "${RKD_NUM_REF_SAMPLES}" \
        --rkd_num_shared_samples "${RKD_NUM_SHARED_SAMPLES}" \
        --output_weight 1.0 \
        --camera_weight "${CAMERA_WEIGHT}" \
        --depth_weight "${DEPTH_WEIGHT}" \
        --point_weight "${POINT_WEIGHT}" \
        --output_dir "${out_dir}"

    echo "Output-only training complete! Model saved to: ${out_dir}/"
}

train_combined() {
    local out_dir="./checkpoints/vggt_distill_combined/eth3d"
    echo ""
    echo "============================================================"
    echo "[Combined] Training on eth3d (fp32)"
    echo "  feature + output loss"
    echo "============================================================"

    eval python ./scripts/train_vggt_combined_distill.py \
        ${COMMON_TRAIN_ARGS} \
        --all_token_kl_weight "${KL_WEIGHT}" \
        --all_token_cos_weight "${COS_WEIGHT}" \
        --rkd_weight "${RKD_WEIGHT}" \
        --rkd_topk "${RKD_TOPK}" \
        --rkd_num_ref_samples "${RKD_NUM_REF_SAMPLES}" \
        --rkd_num_shared_samples "${RKD_NUM_SHARED_SAMPLES}" \
        --output_weight 1.0 \
        --camera_weight "${CAMERA_WEIGHT}" \
        --depth_weight "${DEPTH_WEIGHT}" \
        --point_weight "${POINT_WEIGHT}" \
        --output_dir "${out_dir}"

    echo "Combined training complete! Model saved to: ${out_dir}/"
}

# =============================================================================
# Benchmark
# =============================================================================

get_benchmark_params() {
    local setting="$1"
    case "${setting}" in
        4v) echo "8 4" ;;
        8v) echo "8 0" ;;
        maxframe) echo "100 0" ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            return 1
            ;;
    esac
}

benchmark_lora() {
    local label="$1"
    local ckpt_dir="$2"
    local bench_dir="$3"
    local lora_path="${ckpt_dir}/epoch_0_lora.pt"

    if [ ! -f "${lora_path}" ]; then
        echo "ERROR: LoRA weights not found at ${lora_path}. Skipping ${label}."
        return 1
    fi

    echo ""
    echo "============================================================"
    echo "[${label}] Benchmarking maxframe, seed ${BENCHMARK_SEEDS}"
    echo "============================================================"

    for seed in ${BENCHMARK_SEEDS}; do
        local params
        params="$(get_benchmark_params maxframe)"
        local max_frames
        max_frames="$(echo "${params}" | cut -d' ' -f1)"
        local eval_frames
        eval_frames="$(echo "${params}" | cut -d' ' -f2)"

        echo "  [LoRA maxframe] eth3d, seed=${seed}"

        local cmd
        cmd="python ./scripts/benchmark_lora_vggt.py \
            --lora_path ${lora_path} \
            --base_model ${MODEL_NAME} \
            --lora_rank ${LORA_RANK} \
            --lora_alpha ${LORA_ALPHA} \
            --lora_layers_start ${LORA_LAYERS_START} \
            --datasets eth3d \
            --modes ${MODES} \
            --max_frames ${max_frames} \
            --seed ${seed} \
            --image_size ${IMAGE_SIZE} \
            --work_dir ${bench_dir}/lora_maxframe/eth3d/seed${seed}"

        if [ "${eval_frames}" -gt 0 ]; then
            cmd="${cmd} --eval_frames ${eval_frames}"
        fi

        eval "${cmd}"
    done

    echo "[${label}] Benchmark complete!"
}

# =============================================================================
# Main
# =============================================================================

run_feature() {
    train_feature
    benchmark_lora "Feature-only" \
        "./checkpoints/vggt_distill_feature/eth3d" \
        "./workspace/vggt_distill_feature"
}

run_output() {
    train_output
    benchmark_lora "Output-only" \
        "./checkpoints/vggt_distill_output/eth3d" \
        "./workspace/vggt_distill_output"
}

run_combined() {
    train_combined
    benchmark_lora "Combined" \
        "./checkpoints/vggt_distill_combined/eth3d" \
        "./workspace/vggt_distill_combined"
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    feature)
        run_feature
        ;;
    output)
        run_output
        ;;
    combined)
        run_combined
        ;;
    all)
        run_feature
        run_output
        run_combined
        ;;
    *)
        echo "Usage: $0 [feature|output|combined|all]"
        exit 1
        ;;
esac

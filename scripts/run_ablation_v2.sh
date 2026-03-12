#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Ablation V2: Remove one loss at a time from the full combined loss
#
# Full = KL + Cosine + RKD + Output
#   no_kl     = remove KL       → Cosine + RKD + Output
#   no_cos    = remove Cosine   → KL + RKD + Output
#   no_rkd    = remove RKD      → KL + Cosine + Output
#   no_output = remove Output   → KL + Cosine + RKD
#   baseline  = original VGGT (no training, benchmark only)
#
# All trained with AMP, lr=1e-4, ETH3D, 1 epoch, benchmark seed 42
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

# Feature loss (V14)
KL_WEIGHT=1.0
COS_WEIGHT=2.0
RKD_WEIGHT=2.0
RKD_TOPK=4
RKD_NUM_REF_SAMPLES=256
RKD_NUM_SHARED_SAMPLES=256

# Output loss
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
    --save_interval 0 \
    --use_amp"

# =============================================================================
# Training functions
# =============================================================================

# Full: KL + Cosine + RKD + Output
train_full() {
    local out_dir="./checkpoints/ablation_v2_full/eth3d"
    echo ""
    echo "============================================================"
    echo "[Full] KL + Cosine + RKD + Output (AMP, lr=${LR})"
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
}

# No-KL: Cosine + RKD + Output (remove KL only)
train_no_kl() {
    local out_dir="./checkpoints/ablation_v2_no_kl/eth3d"
    echo ""
    echo "============================================================"
    echo "[No-KL] Cosine + RKD + Output (AMP, lr=${LR})"
    echo "============================================================"

    eval python ./scripts/train_vggt_combined_distill.py \
        ${COMMON_TRAIN_ARGS} \
        --all_token_kl_weight 0.0 \
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
}

# No-Cosine: KL + RKD + Output (remove Cosine only)
train_no_cos() {
    local out_dir="./checkpoints/ablation_v2_no_cos/eth3d"
    echo ""
    echo "============================================================"
    echo "[No-Cos] KL + RKD + Output (AMP, lr=${LR})"
    echo "============================================================"

    eval python ./scripts/train_vggt_combined_distill.py \
        ${COMMON_TRAIN_ARGS} \
        --all_token_kl_weight "${KL_WEIGHT}" \
        --all_token_cos_weight 0.0 \
        --rkd_weight "${RKD_WEIGHT}" \
        --rkd_topk "${RKD_TOPK}" \
        --rkd_num_ref_samples "${RKD_NUM_REF_SAMPLES}" \
        --rkd_num_shared_samples "${RKD_NUM_SHARED_SAMPLES}" \
        --output_weight 1.0 \
        --camera_weight "${CAMERA_WEIGHT}" \
        --depth_weight "${DEPTH_WEIGHT}" \
        --point_weight "${POINT_WEIGHT}" \
        --output_dir "${out_dir}"
}

# No-RKD: KL + Cosine + Output (remove RKD only)
train_no_rkd() {
    local out_dir="./checkpoints/ablation_v2_no_rkd/eth3d"
    echo ""
    echo "============================================================"
    echo "[No-RKD] KL + Cosine + Output (AMP, lr=${LR})"
    echo "============================================================"

    eval python ./scripts/train_vggt_combined_distill.py \
        ${COMMON_TRAIN_ARGS} \
        --all_token_kl_weight "${KL_WEIGHT}" \
        --all_token_cos_weight "${COS_WEIGHT}" \
        --rkd_weight 0.0 \
        --rkd_topk "${RKD_TOPK}" \
        --rkd_num_ref_samples "${RKD_NUM_REF_SAMPLES}" \
        --rkd_num_shared_samples "${RKD_NUM_SHARED_SAMPLES}" \
        --output_weight 1.0 \
        --camera_weight "${CAMERA_WEIGHT}" \
        --depth_weight "${DEPTH_WEIGHT}" \
        --point_weight "${POINT_WEIGHT}" \
        --output_dir "${out_dir}"
}

# No-Output: KL + Cosine + RKD (remove Output only)
train_no_output() {
    local out_dir="./checkpoints/ablation_v2_no_output/eth3d"
    echo ""
    echo "============================================================"
    echo "[No-Output] KL + Cosine + RKD (AMP, lr=${LR})"
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
}

# =============================================================================
# Benchmark
# =============================================================================

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
        echo "  [LoRA maxframe] eth3d, seed=${seed}"

        python ./scripts/benchmark_lora_vggt.py \
            --lora_path "${lora_path}" \
            --base_model "${MODEL_NAME}" \
            --lora_rank "${LORA_RANK}" \
            --lora_alpha "${LORA_ALPHA}" \
            --lora_layers_start "${LORA_LAYERS_START}" \
            --datasets eth3d \
            --modes ${MODES} \
            --max_frames 100 \
            --seed "${seed}" \
            --image_size "${IMAGE_SIZE}" \
            --work_dir "${bench_dir}/lora_maxframe/eth3d/seed${seed}"
    done

    echo "[${label}] Benchmark complete!"
}

benchmark_baseline() {
    local bench_dir="./workspace/ablation_v2_baseline"

    echo ""
    echo "============================================================"
    echo "[Baseline] Benchmarking original VGGT (no LoRA), seed ${BENCHMARK_SEEDS}"
    echo "============================================================"

    for seed in ${BENCHMARK_SEEDS}; do
        echo "  [Baseline maxframe] eth3d, seed=${seed}"

        python ./scripts/benchmark_lora_vggt.py \
            --base_model "${MODEL_NAME}" \
            --lora_rank "${LORA_RANK}" \
            --lora_alpha "${LORA_ALPHA}" \
            --lora_layers_start "${LORA_LAYERS_START}" \
            --datasets eth3d \
            --modes ${MODES} \
            --max_frames 100 \
            --seed "${seed}" \
            --image_size "${IMAGE_SIZE}" \
            --work_dir "${bench_dir}/lora_maxframe/eth3d/seed${seed}"
    done

    echo "[Baseline] Benchmark complete!"
}

# =============================================================================
# Run functions (train + benchmark)
# =============================================================================

run_full() {
    train_full
    benchmark_lora "Full" \
        "./checkpoints/ablation_v2_full/eth3d" \
        "./workspace/ablation_v2_full"
}

run_no_kl() {
    train_no_kl
    benchmark_lora "No-KL" \
        "./checkpoints/ablation_v2_no_kl/eth3d" \
        "./workspace/ablation_v2_no_kl"
}

run_no_cos() {
    train_no_cos
    benchmark_lora "No-Cos" \
        "./checkpoints/ablation_v2_no_cos/eth3d" \
        "./workspace/ablation_v2_no_cos"
}

run_no_rkd() {
    train_no_rkd
    benchmark_lora "No-RKD" \
        "./checkpoints/ablation_v2_no_rkd/eth3d" \
        "./workspace/ablation_v2_no_rkd"
}

run_no_output() {
    train_no_output
    benchmark_lora "No-Output" \
        "./checkpoints/ablation_v2_no_output/eth3d" \
        "./workspace/ablation_v2_no_output"
}

# =============================================================================
# Main
# =============================================================================

COMMAND="${1:-all}"

case "${COMMAND}" in
    full)
        run_full
        ;;
    no_kl)
        run_no_kl
        ;;
    no_cos)
        run_no_cos
        ;;
    no_rkd)
        run_no_rkd
        ;;
    no_output)
        run_no_output
        ;;
    baseline)
        benchmark_baseline
        ;;
    all)
        run_full
        run_no_kl
        run_no_cos
        run_no_rkd
        run_no_output
        benchmark_baseline
        ;;
    *)
        echo "Usage: $0 [full|no_kl|no_cos|no_rkd|no_output|baseline|all]"
        exit 1
        ;;
esac

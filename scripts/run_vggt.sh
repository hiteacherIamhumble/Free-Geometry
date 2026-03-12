#!/usr/bin/env bash
set -e

# =============================================================================
# VGGT Free-Geometry training and benchmarking.
# =============================================================================
#
# Active VGGT workflow:
# - Trains on all 4 datasets with patch huber + cosine + cross-frame CF angle + CF distance
# - Free-Geometry adaptation checkpoints are saved as *_lora.pt
# - baseline and adapted benchmarking are both supported
#
# Training configs (per-dataset):
#   scannetpp: 5 samples/scene, seeds 30-34, 2 epochs (from V12)
#   hiroom:    2 samples/scene, seeds 30 31, 1 epoch (from V11)
#   7scenes:   30 samples/scene, seeds 30-59, 1 epoch (from V11)
#   eth3d:     5 samples/scene, seeds 30-34, 1 epoch (from V10)
#
# Usage:
#   ./scripts/run_vggt.sh [train|benchmark_lora|all]
#   ./scripts/run_vggt.sh train_scannetpp
#   ./scripts/run_vggt.sh train_hiroom
#   ./scripts/run_vggt.sh train_7scenes
#   ./scripts/run_vggt.sh train_eth3d
#
# =============================================================================

# === Configuration ===
MODEL_NAME=facebook/vggt-1b
OUTPUT_DIR=./checkpoints/all_vggt_v3
BENCHMARK_ROOT=./workspace/all_vggt_v3
# Common training settings
LR=1e-4
BATCH_SIZE=2
LORA_RANK=32
LORA_ALPHA=32
LORA_LAYERS_START=0
NUM_VIEWS=8
# DPT decoder feature layers: [4, 11, 17, 23]
OUTPUT_LAYERS="4 11 17 23"

# Free-Geometry base loss settings (patch huber + cosine)
PATCH_HUBER_WEIGHT=1.0
PATCH_HUBER_COS_WEIGHT=2.0
PATCH_HUBER_DELTA=1.0

# Cross-frame CF angle loss settings
CF_WEIGHT=2.0
CF_TOPK=4
CF_NUM_REF_SAMPLES=256
CF_NUM_SHARED_SAMPLES=256
CF_ANGLE1_WEIGHT=1.0
CF_ANGLE2_WEIGHT=1.0
CF_ANGLE3_WEIGHT=1.0

# Free-Geometry cross-frame CF distance loss settings
CF_DIST_WEIGHT=1.0
CF_DIST_CHUNK_SIZE=16
CF_DIST_TYPE=l2
CF_DIST_TEMP=1.0
CF_DIST_MODE=kl
CF_DIST_HUBER_BETA=0.5
CF_D1_WEIGHT=1.0
CF_D2_WEIGHT=1.0
CF_D3_WEIGHT=0.0

# Benchmark settings
BENCHMARK_SEEDS="43 44 45"
LORA_BENCHMARK_SEEDS="${LORA_BENCHMARK_SEEDS:-43}"
ALL_DATASETS="scannetpp 7scenes eth3d"
MODES="pose recon_unposed"

# Free-Geometry image size (longest side, aspect ratio preserved)
IMAGE_SIZE=504

# =============================================================================
# Training Functions
# =============================================================================

# Per-dataset config: "samples epochs seed_start seed_end lr"
get_training_config() {
    local DATASET=$1
    case "${DATASET}" in
        scannetpp) echo "10 3 40 49 3e-5" ;;
        hiroom)    echo "5 3 40 44 1e-4" ;;
        7scenes)   echo "10 3 40 49 1e-5" ;;
        eth3d)     echo "10 3 40 44 1e-5" ;;
    esac
}

train_single_dataset() {
    local DATASET=$1
    local CONFIG=$(get_training_config "${DATASET}")
    local SAMPLES=$(echo ${CONFIG} | cut -d' ' -f1)
    local DS_EPOCHS=$(echo ${CONFIG} | cut -d' ' -f2)
    local SEED_START=$(echo ${CONFIG} | cut -d' ' -f3)
    local SEED_END=$(echo ${CONFIG} | cut -d' ' -f4)
    local DS_LR=$(echo ${CONFIG} | cut -d' ' -f5)
    local SEEDS="$(seq -s ' ' ${SEED_START} ${SEED_END})"

    echo ""
    echo "============================================================"
    echo "Training Free-Geometry on ${DATASET}"
    echo "  Image size: ${IMAGE_SIZE} (longest side, aspect ratio preserved)"
    echo "  Samples per scene: ${SAMPLES}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Seeds: ${SEEDS}"
    echo "  Epochs: ${DS_EPOCHS}"
    echo "  LR: ${DS_LR}, cosine scheduler with warmup"
    echo "  Loss: patch huber (w=${PATCH_HUBER_WEIGHT}, delta=${PATCH_HUBER_DELTA}) + cosine (w=${PATCH_HUBER_COS_WEIGHT}) + CF-angle (w=${CF_WEIGHT}) + CF-dist (w=${CF_DIST_WEIGHT}, mode=${CF_DIST_MODE})"
    echo "============================================================"

    python ./scripts/train_vggt.py \
        --dataset "${DATASET}" \
        --samples_per_scene ${SAMPLES} \
        --seeds_list ${SEEDS} \
        --model_name ${MODEL_NAME} \
        --num_views ${NUM_VIEWS} \
        --output_layers ${OUTPUT_LAYERS} \
        --patch_huber_weight ${PATCH_HUBER_WEIGHT} \
        --patch_huber_cos_weight ${PATCH_HUBER_COS_WEIGHT} \
        --patch_huber_delta ${PATCH_HUBER_DELTA} \
        --cf_weight ${CF_WEIGHT} \
        --cf_topk ${CF_TOPK} \
        --cf_num_ref_samples ${CF_NUM_REF_SAMPLES} \
        --cf_num_shared_samples ${CF_NUM_SHARED_SAMPLES} \
        --cf_angle1_weight ${CF_ANGLE1_WEIGHT} \
        --cf_angle2_weight ${CF_ANGLE2_WEIGHT} \
        --cf_angle3_weight ${CF_ANGLE3_WEIGHT} \
        --cf_selection_mode mixed \
        --use_cf_distance \
        --cf_distance_weight ${CF_DIST_WEIGHT} \
        --cf_distance_chunk_size ${CF_DIST_CHUNK_SIZE} \
        --cf_distance_type ${CF_DIST_TYPE} \
        --cf_distance_temperature ${CF_DIST_TEMP} \
        --cf_distance_mode ${CF_DIST_MODE} \
        --cf_distance_huber_beta ${CF_DIST_HUBER_BETA} \
        --cf_d1_weight ${CF_D1_WEIGHT} \
        --cf_d2_weight ${CF_D2_WEIGHT} \
        --cf_d3_weight ${CF_D3_WEIGHT} \
        --epochs ${DS_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_workers 2 \
        --lr ${DS_LR} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --lr_scheduler cosine \
        --warmup_ratio 0.15 \
        --eta_min 1e-8 \
        --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}/${DATASET}" \
        --log_interval 1 \
        --save_interval 0

    echo ""
    echo "Free-Geometry training on ${DATASET} complete!"
    echo "  Model saved to: ${OUTPUT_DIR}/${DATASET}/"
}

run_training() {
    echo ""
    echo "============================================================"
    echo "Free-Geometry training: 4 datasets"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        train_single_dataset "${DATASET}"
    done

    echo ""
    echo "============================================================"
    echo "All Free-Geometry training complete! Models saved to:"
    for DATASET in ${ALL_DATASETS}; do
        echo "  - ${OUTPUT_DIR}/${DATASET}/"
    done
    echo "============================================================"
}

# =============================================================================
# Benchmark Functions
# =============================================================================

get_benchmark_params() {
    local SETTING=$1
    case "${SETTING}" in
        4v)
            echo "8 4"  # max_frames=8, eval_frames=4
            ;;
        8v)
            echo "8 0"  # max_frames=8, eval_frames=0 (use all)
            ;;
        16v)
            echo "16 0"  # max_frames=16
            ;;
        32v)
            echo "32 0"  # max_frames=32
            ;;
        64v)
            echo "64 0"  # max_frames=64
            ;;
        maxframe)
            echo "100 0"  # max_frames=100, eval_frames=0 (use all available up to 100)
            ;;
    esac
}

benchmark_base_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED_LIST=$3  # space-separated seeds

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [Base ${SETTING}] ${DATASET}, seeds: ${SEED_LIST}"

    local CMD="python ./scripts/benchmark_vggt.py \
        --base_model ${MODEL_NAME} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seeds ${SEED_LIST} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/base_${SETTING}/${DATASET}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

benchmark_lora_setting() {
    local DATASET=$1
    local SETTING=$2
    local SEED_LIST=$3  # space-separated seeds
    local EPOCH=$4
    local LORA_PATH="${OUTPUT_DIR}/${DATASET}/epoch_${EPOCH}_lora.pt"

    if [ ! -f "${LORA_PATH}" ]; then
        echo "WARNING: LoRA weights not found at ${LORA_PATH}. Skipping."
        return 0
    fi

    local PARAMS=$(get_benchmark_params "${SETTING}")
    local MAX_FRAMES=$(echo ${PARAMS} | cut -d' ' -f1)
    local EVAL_FRAMES=$(echo ${PARAMS} | cut -d' ' -f2)

    echo "  [Free-Geometry epoch=${EPOCH} ${SETTING}] ${DATASET}, seeds: ${SEED_LIST}"

    local CMD="python ./scripts/benchmark_vggt.py \
        --lora_path ${LORA_PATH} \
        --base_model ${MODEL_NAME} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_layers_start ${LORA_LAYERS_START} \
        --datasets ${DATASET} \
        --modes ${MODES} \
        --max_frames ${MAX_FRAMES} \
        --seeds ${SEED_LIST} \
        --image_size ${IMAGE_SIZE} \
        --work_dir ${BENCHMARK_ROOT}/lora_epoch${EPOCH}_${SETTING}/${DATASET}"

    if [ "${EVAL_FRAMES}" -gt 0 ]; then
        CMD="${CMD} --eval_frames ${EVAL_FRAMES}"
    fi

    eval ${CMD}
}

run_baseline_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking baseline VGGT"
    echo "  Settings: 4v, 8v, 16v, 32v, 64v"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        echo ""
        echo "--- baseline ${DATASET} ---"
        for SETTING in 4v 8v 16v 32v; do
            benchmark_base_setting "${DATASET}" "${SETTING}" "${BENCHMARK_SEEDS}"
        done
    done

    echo ""
    echo "============================================================"
    echo "Baseline benchmark complete!"
    echo "============================================================"
}

run_lora_benchmark() {
    echo ""
    echo "============================================================"
    echo "Benchmarking Free-Geometry checkpoints (all epochs, seed 43 only, 16v)"
    echo "  Settings: 16v"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Seeds: ${LORA_BENCHMARK_SEEDS}"
    echo "  Image size: ${IMAGE_SIZE} (DA3-style)"
    echo "============================================================"

    for DATASET in ${ALL_DATASETS}; do
        # Find all epoch files for this dataset
        shopt -s nullglob
        local LORA_FILES=( "${OUTPUT_DIR}/${DATASET}/epoch_"*_lora.pt )
        shopt -u nullglob

        local EPOCHS=()
        for f in "${LORA_FILES[@]}"; do
            local bn
            bn="$(basename "${f}")"
            if [[ "${bn}" =~ ^epoch_([0-9]+)_lora\.pt$ ]]; then
                EPOCHS+=( "${BASH_REMATCH[1]}" )
            fi
        done

        if [ "${#EPOCHS[@]}" -eq 0 ]; then
            echo "WARNING: No LoRA epoch weights found under ${OUTPUT_DIR}/${DATASET}/. Skipping ${DATASET}."
            continue
        fi

        # Sort epochs numerically
        mapfile -t EPOCHS < <(printf '%s\n' "${EPOCHS[@]}" | sort -n)

        echo ""
        echo "--- ${DATASET} (epochs: ${EPOCHS[@]}) ---"

        for EPOCH in "${EPOCHS[@]}"; do
            benchmark_lora_setting "${DATASET}" "16v" "${LORA_BENCHMARK_SEEDS}" "${EPOCH}"
        done
    done

    echo ""
    echo "============================================================"
    echo "Free-Geometry benchmark complete!"
    echo "============================================================"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  train             - Train all 4 datasets"
    echo "  train_scannetpp   - Train scannetpp only"
    echo "  train_hiroom      - Train hiroom only"
    echo "  train_7scenes     - Train 7scenes only"
    echo "  train_eth3d       - Train eth3d only"
    echo "  benchmark_base    - Benchmark baseline VGGT (4v, 8v, maxframe)"
    echo "  benchmark_lora    - Benchmark all Free-Geometry checkpoints"
    echo "  benchmark         - Run both baseline and Free-Geometry benchmarks"
    echo "  all               - Train all + benchmark all"
    echo ""
    echo "Key settings:"
    echo "  Datasets: ${ALL_DATASETS}"
    echo "  Loss: patch huber + cosine + cross-frame CF angle + CF distance"
    echo "  Image size: ${IMAGE_SIZE} (Free-Geometry preprocessing, aspect ratio preserved)"
    echo "  Benchmark: baseline + Free-Geometry, 4v/8v/16v/32v, seeds ${BENCHMARK_SEEDS}"
    echo ""
    echo "Training configs:"
    echo "  scannetpp: 5 samples/scene, seeds 30-34, 2 epochs"
    echo "  hiroom:    2 samples/scene, seeds 30-31, 1 epoch"
    echo "  7scenes:   30 samples/scene, seeds 30-59, 1 epoch"
    echo "  eth3d:     5 samples/scene, seeds 30-34, 1 epoch"
    echo ""
}

COMMAND="${1:-all}"

case "${COMMAND}" in
    train)
        run_training
        ;;
    train_scannetpp)
        train_single_dataset "scannetpp"
        ;;
    train_hiroom)
        train_single_dataset "hiroom"
        ;;
    train_7scenes)
        train_single_dataset "7scenes"
        ;;
    train_eth3d)
        train_single_dataset "eth3d"
        ;;
    benchmark_base)
        run_baseline_benchmark
        ;;
    benchmark_lora)
        run_lora_benchmark
        ;;
    benchmark)
        run_baseline_benchmark
        run_lora_benchmark
        ;;
    all)
        # Train all datasets, then benchmark all Free-Geometry epochs on 16v seed 43
        run_training
        run_lora_benchmark
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

echo ""
echo "============================================================"
echo "=== VGGT Experiment Complete ==="
echo "============================================================"
echo ""
echo "Experiment: Free-Geometry patch huber + cosine + CF angle + CF distance on all 4 datasets"
echo "  Loss: patch huber (w=${PATCH_HUBER_WEIGHT}, delta=${PATCH_HUBER_DELTA}) + cosine (w=${PATCH_HUBER_COS_WEIGHT}) + CF-angle (w=${CF_WEIGHT}, topk=${CF_TOPK}) + CF-dist (w=${CF_DIST_WEIGHT}, mode=${CF_DIST_MODE})"
echo ""
echo "Output directories:"
echo "  Training:    ${OUTPUT_DIR}/{dataset}"
echo "  Benchmarks:  ${BENCHMARK_ROOT}/lora_epoch{epoch}_16v/{dataset}/ and ${BENCHMARK_ROOT}/base_{setting}/{dataset}/"
echo ""

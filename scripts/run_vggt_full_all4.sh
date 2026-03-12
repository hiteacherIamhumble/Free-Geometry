#!/bin/bash
# Run VGGT teacher vs student benchmark on all 4 datasets with per-dataset LoRA checkpoints
set -e

mkdir -p logs

for ds in eth3d hiroom 7scenes scannetpp; do
    echo ""
    echo "########################################################################"
    echo "  VGGT Benchmark: $ds (all scenes)"
    echo "  LoRA: checkpoints/vggt_lora_final/$ds/lora.pt"
    echo "########################################################################"
    echo ""

    python scripts/benchmark_teacher_student_all_datasets_vggt.py \
        --datasets $ds \
        --all_scenes \
        --lora_path checkpoints/vggt_lora_final/$ds/lora.pt \
        --work_dir ./results/vggt_full/$ds \
        --seed 43 \
        --lora_rank 32 \
        --lora_alpha 32.0 \
        --lora_layers_start 0
done

echo ""
echo "========================================"
echo "VGGT full benchmark complete!"
echo "========================================"

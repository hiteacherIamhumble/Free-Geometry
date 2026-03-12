#!/bin/bash
# Run DA3 teacher vs student benchmark on all 4 datasets with per-dataset LoRA checkpoints
set -e

mkdir -p logs

for ds in eth3d hiroom 7scenes scannetpp; do
    echo ""
    echo "########################################################################"
    echo "  DA3 Benchmark: $ds (all scenes)"
    echo "  LoRA: checkpoints/da3_lora_final/$ds/lora.pt"
    echo "########################################################################"
    echo ""

    python scripts/benchmark_teacher_student_all_datasets.py \
        --datasets $ds \
        --all_scenes \
        --lora_path checkpoints/da3_lora_final/$ds/lora.pt \
        --work_dir ./results/da3_full/$ds \
        --seed 43
done

echo ""
echo "========================================"
echo "DA3 full benchmark complete!"
echo "========================================"

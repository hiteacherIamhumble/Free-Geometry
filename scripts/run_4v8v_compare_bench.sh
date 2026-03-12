#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

cd /root/autodl-tmp/da3

MODEL="depth-anything/DA3-GIANT-1.1"
LORA_PATH="checkpoints/da3_all_layers_distill/scannetpp/epoch_0_lora.pt"

echo "=== 1. 4v baseline (max_frames=4, seed=43) ==="
python -u -c "
import sys, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator
api = DepthAnything3.from_pretrained('${MODEL}').to('cuda')
evaluator = Evaluator(work_dir='./workspace/da3_4v8v_compare/4v_baseline', datas=['scannetpp'], modes=['pose','recon_unposed'], max_frames=4, seed=43)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
"

echo ""
echo "=== 2. 8v baseline (max_frames=8, seed=43) ==="
python -u -c "
import sys, torch
sys.path.insert(0, 'src')
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator
api = DepthAnything3.from_pretrained('${MODEL}').to('cuda')
evaluator = Evaluator(work_dir='./workspace/da3_4v8v_compare/8v_baseline', datas=['scannetpp'], modes=['pose','recon_unposed'], max_frames=8, seed=43)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
"

echo ""
echo "=== 3. 4v LoRA (max_frames=4, seed=43) ==="
python -u scripts/benchmark_lora.py \
    --lora_path "${LORA_PATH}" \
    --base_model "${MODEL}" \
    --lora_rank 16 --lora_alpha 16 \
    --datasets scannetpp \
    --modes pose recon_unposed \
    --max_frames 4 --seed 43 \
    --work_dir ./workspace/da3_4v8v_compare/4v_lora

echo ""
echo "=== DONE ==="

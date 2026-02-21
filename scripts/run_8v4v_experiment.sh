#!/usr/bin/env bash

# 8v vs 4v Frame Extraction Experiment on ALL benchmark datasets
# Uses DA3-GIANT-1.1 (no LoRA), seed 43
# DTU is first so its slow recon eval runs first

cd /home/22097845d/Depth-Anything-3
mkdir -p logs

nohup python scripts/run_8v4v_extraction_experiment.py \
  --seed 43 \
  --work_dir ./workspace/extraction_exp_all \
  --model_name depth-anything/DA3-GIANT-1.1 \
  --datasets dtu eth3d 7scenes scannetpp hiroom dtu64 \
  --experiments 8v_all 8v_extract_result 8v_extract_feat 4v_all \
  --eval_only \
  > logs/extraction_exp_all.log 2>&1 &

echo "Started with PID: $!"
echo "Monitor: tail -f logs/extraction_exp_all.log"

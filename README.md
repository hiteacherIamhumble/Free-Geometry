# Free-Geometry

This repository is the active Free-Geometry workflow for DA3 and VGGT:

- training: `scripts/train_da3.py`, `scripts/train_vggt.py`
- single-model benchmark: `scripts/benchmark_da3.py`, `scripts/benchmark_vggt.py`
- all-in-one runners: `scripts/run_da3.sh`, `scripts/run_vggt.sh`
- shared multiview benchmark + visualization:
  `scripts/benchmark_multiview_all_datasets.py` and `scripts/visualize_free_geometry.py`

The multiview benchmark and the visualizer must use the same `results/` root.

## Table of Contents

- [1. Installation](#1-installation)
- [2. Free-Geometry Training](#2-free-geometry-training)
- [3. Free-Geometry Benchmark](#3-free-geometry-benchmark)
- [4. All-In-One Runners](#4-all-in-one-runners)
- [5. Shared Results Benchmark + Visualization](#5-shared-results-benchmark--visualization)
- [Directory Summary](#directory-summary)

## 1. Installation

```bash
conda create -n Free-Geo python=3.10 -y
conda activate Free-Geo

pip install xformers torch\>=2 torchvision
pip install -e .
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install -e ".[app]"
pip install -e ".[all]"
```

## 2. Free-Geometry Training

### DA3

Canonical DA3 training entry point:

```bash
python scripts/train_da3.py \
  --dataset hiroom \
  --samples_per_scene 5 \
  --seeds_list 40 41 42 43 44 \
  --model_name depth-anything/DA3-GIANT-1.1 \
  --num_views 8 \
  --patch_huber_weight 1.0 \
  --patch_huber_cos_weight 2.0 \
  --patch_huber_delta 1.0 \
  --cf_weight 2.0 \
  --cf_topk 4 \
  --cf_num_ref_samples 256 \
  --cf_num_shared_samples 256 \
  --cf_angle1_weight 1.0 \
  --cf_angle2_weight 1.0 \
  --cf_angle3_weight 1.0 \
  --cf_selection_mode mixed \
  --use_cf_distance \
  --cf_distance_weight 1.0 \
  --cf_distance_temperature 10.0 \
  --cf_distance_mode kl \
  --cf_d1_weight 1.0 \
  --cf_d2_weight 1.0 \
  --cf_d3_weight 0.0 \
  --epochs 5 \
  --batch_size 4 \
  --num_workers 2 \
  --lr 1e-4 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lr_scheduler cosine \
  --warmup_ratio 0.15 \
  --eta_min 1e-7 \
  --weight_decay 1e-5 \
  --output_dir checkpoints/da3_hiroom
```

### VGGT

Canonical VGGT training entry point:

```bash
python scripts/train_vggt.py \
  --dataset hiroom \
  --samples_per_scene 5 \
  --seeds_list 40 41 42 43 44 \
  --model_name facebook/vggt-1b \
  --num_views 8 \
  --image_size 504 \
  --output_layers 4 11 17 23 \
  --patch_huber_weight 1.0 \
  --patch_huber_cos_weight 2.0 \
  --patch_huber_delta 1.0 \
  --cf_weight 2.0 \
  --cf_topk 4 \
  --cf_num_ref_samples 256 \
  --cf_num_shared_samples 256 \
  --cf_angle1_weight 1.0 \
  --cf_angle2_weight 1.0 \
  --cf_angle3_weight 1.0 \
  --cf_selection_mode mixed \
  --use_cf_distance \
  --cf_distance_weight 1.0 \
  --cf_distance_chunk_size 16 \
  --cf_distance_type l2 \
  --cf_distance_temperature 1.0 \
  --cf_distance_mode kl \
  --cf_distance_huber_beta 0.5 \
  --cf_d1_weight 1.0 \
  --cf_d2_weight 1.0 \
  --cf_d3_weight 0.0 \
  --epochs 3 \
  --batch_size 2 \
  --num_workers 2 \
  --lr 1e-4 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_layers_start 0 \
  --lr_scheduler cosine \
  --warmup_ratio 0.15 \
  --eta_min 1e-8 \
  --weight_decay 1e-5 \
  --output_dir checkpoints/vggt_hiroom
```

Notes:

- `scripts/run_da3.sh` and `scripts/run_vggt.sh` are the easiest way to reproduce the repo’s tuned dataset settings.
- Adapter checkpoints are saved as `epoch_*_lora.pt` and `latest_lora.pt`.

## 3. Free-Geometry Benchmark

### DA3 benchmark

```bash
python scripts/benchmark_da3.py \
  --lora_path checkpoints/da3_hiroom/epoch_5_lora.pt \
  --base_model depth-anything/DA3-GIANT-1.1 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --datasets hiroom \
  --modes pose recon_unposed \
  --max_frames 16 \
  --seeds 43 \
  --work_dir workspace/benchmark_da3_hiroom
```

### VGGT benchmark

```bash
python scripts/benchmark_vggt.py \
  --lora_path checkpoints/vggt_hiroom/epoch_3_lora.pt \
  --base_model facebook/vggt-1b \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_layers_start 0 \
  --datasets hiroom \
  --modes pose recon_unposed \
  --max_frames 16 \
  --seeds 43 \
  --image_size 504 \
  --work_dir workspace/benchmark_vggt_hiroom
```

Use baseline evaluation by omitting `--lora_path` for VGGT, or by using the baseline path in the all-in-one DA3 runner.

## 4. All-In-One Runners

### DA3

```bash
bash scripts/run_da3.sh train
bash scripts/run_da3.sh train_hiroom
bash scripts/run_da3.sh benchmark_baseline
bash scripts/run_da3.sh benchmark_lora
bash scripts/run_da3.sh benchmark_all
bash scripts/run_da3.sh all
```

Default DA3 outputs:

- checkpoints: `checkpoints/all_da3_v2/{dataset}/`
- benchmark outputs: `workspace/all_da3_v2/`

### VGGT

```bash
bash scripts/run_vggt.sh train
bash scripts/run_vggt.sh train_hiroom
bash scripts/run_vggt.sh benchmark_base
bash scripts/run_vggt.sh benchmark_lora
bash scripts/run_vggt.sh benchmark
bash scripts/run_vggt.sh all
```

Default VGGT outputs:

- checkpoints: `checkpoints/all_vggt_v3/{dataset}/`
- benchmark outputs: `workspace/all_vggt_v3/`

## 5. Shared Results Benchmark + Visualization

Use these two scripts together:

- `scripts/benchmark_multiview_all_datasets.py`
- `scripts/visualize_free_geometry.py`

They must point to the same `results/` root.

### Example: DA3 baseline and Free-Geometry results

Run baseline:

```bash
python scripts/benchmark_multiview_all_datasets.py \
  --model_family da3 \
  --no_lora \
  --datasets hiroom scannetpp \
  --view_counts 8 16 32 \
  --seed 43 \
  --results_root results \
  --work_dir results/multiview_da3_hiroom_scannetpp_base_fixed
```

Run Free-Geometry:

```bash
python scripts/benchmark_multiview_all_datasets.py \
  --model_family da3 \
  --datasets hiroom scannetpp \
  --view_counts 8 16 32 \
  --seed 43 \
  --lora_path checkpoints/da3_hiroom/epoch_5_lora.pt \
  --results_root results \
  --work_dir results/multiview_da3_hiroom_scannetpp_lora_fixed
```

Visualize from the same `results/` root:

```bash
python scripts/visualize_free_geometry.py \
  --model_family da3 \
  --dataset hiroom \
  --frames 16 \
  --seed 43 \
  --results_root results \
  --host 127.0.0.1 \
  --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

### Example: VGGT baseline and Free-Geometry results

Run baseline:

```bash
python scripts/benchmark_multiview_all_datasets.py \
  --model_family vggt \
  --no_lora \
  --datasets hiroom scannetpp \
  --view_counts 8 16 32 \
  --seed 43 \
  --results_root results \
  --work_dir results/multiview_vggt_hiroom_scannetpp_base_fixed
```

Run Free-Geometry:

```bash
python scripts/benchmark_multiview_all_datasets.py \
  --model_family vggt \
  --datasets hiroom scannetpp \
  --view_counts 8 16 32 \
  --seed 43 \
  --lora_path checkpoints/vggt_hiroom/epoch_3_lora.pt \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_layers_start 0 \
  --results_root results \
  --work_dir results/multiview_vggt_hiroom_scannetpp_lora_fixed
```

Visualize from the same `results/` root:

```bash
python scripts/visualize_free_geometry.py \
  --model_family vggt \
  --dataset hiroom \
  --frames 16 \
  --seed 43 \
  --results_root results \
  --host 127.0.0.1 \
  --port 7860
```

## Directory Summary

```text
results/
  multiview_da3_hiroom_scannetpp_base_fixed/
    8v/
    16v/
    32v/
  multiview_da3_hiroom_scannetpp_lora_fixed/
    8v/
    16v/
    32v/
  multiview_vggt_hiroom_scannetpp_base_fixed/
    8v/
    16v/
    32v/
  multiview_vggt_hiroom_scannetpp_lora_fixed/
    8v/
    16v/
    32v/
```

This is the expected layout for `scripts/visualize_free_geometry.py`.

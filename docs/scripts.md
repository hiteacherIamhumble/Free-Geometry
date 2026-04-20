# Scripts

This document contains the environment setup, training, and benchmarking commands for Free Geometry. The README stays focused on the paper, qualitative results, and demos.

## Environment Setup

```bash
conda create -n Free-Geo python=3.10 -y
conda activate Free-Geo

pip install xformers "torch>=2" torchvision
pip install -e .
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install -e ".[app]"
pip install -e ".[all]"
```

## Training And Benchmarking

Only the maintained bash runners are documented here. The long direct `python scripts/train_*.py` and `python scripts/benchmark_*.py` commands are intentionally omitted.

### DA3

```bash
bash scripts/run_da3.sh train
bash scripts/run_da3.sh train_scannetpp
bash scripts/run_da3.sh train_hiroom
bash scripts/run_da3.sh train_7scenes
bash scripts/run_da3.sh train_eth3d

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
bash scripts/run_vggt.sh train_scannetpp
bash scripts/run_vggt.sh train_hiroom
bash scripts/run_vggt.sh train_7scenes
bash scripts/run_vggt.sh train_eth3d

bash scripts/run_vggt.sh benchmark_base
bash scripts/run_vggt.sh benchmark_lora
bash scripts/run_vggt.sh benchmark
bash scripts/run_vggt.sh all
```

Default VGGT outputs:

- checkpoints: `checkpoints/all_vggt_v3/{dataset}/`
- benchmark outputs: `workspace/all_vggt_v3/`

# ETH3D LoRA Experiment: One-Epoch Local-Token Distill + Per-Scene Metrics

This notebook outlines a reproducible flow to:
- Sample **5× 8-view** clips per ETH3D scene using seeds **37–41**, first frame as ref.
- Compute per-scene token metrics **before training** (epoch -1).
- Train **1 epoch** with **local per-token loss (cosine + KL)** on the 8→4 teacher vs 4-view student.
- Re-compute per-scene metrics **after epoch 0**.
- Summarize how many scenes improved (KL ↓, cosine ↑) and emit a per-scene table (JSON + display).
- Benchmark the resulting LoRA on ETH3D with seeds **42, 43, 44** (pose + recon_nopose), averaging 4-view inputs.

> Note: This is a driver notebook; heavy training/benchmarking is not executed here. Run the code cells in your environment with GPUs and ETH3D data present at `ETH3D_EVAL_DATA_ROOT`.

---

## Cell 1: Imports and Paths

```python
import os, json, random, copy
import numpy as np
import torch
import pandas as pd
from PIL import Image

from depth_anything_3.bench.datasets.eth3d import ETH3D
from depth_anything_3.api import DepthAnything3
from depth_anything_3.distillation.models import StudentModel
from depth_anything_3.utils.constants import ETH3D_SCENES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

OUTPUT_DIR = "explore-attn/outputs/eth3d_lora_exp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL = "depth-anything/DA3-GIANT-1.1"
```

---

## Cell 2: Sampling 8-View Clips per Scene (Seeds 37–41)

```python
dataset = ETH3D()
target_size = (518, 518)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_tensor(img_np):
    f = img_np.astype(np.float32) / 255.0
    img_norm = (f - mean) / std
    return torch.from_numpy(img_norm).permute(2, 0, 1)

def sample_8_views(scene_data, seed):
    rng = random.Random(seed)
    idxs = list(range(len(scene_data.image_files)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:8])  # keep ascending order; first is ref
    image_paths = [scene_data.image_files[i] for i in idxs]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    images_resized = [img.resize(target_size, Image.BILINEAR) for img in images]
    images_np = [np.array(img) for img in images_resized]
    images_tensor = torch.stack([to_tensor(img) for img in images_np], dim=0).to(device)
    images_batch = images_tensor.unsqueeze(0).float()
    return images_batch, images_resized

def prepare_batches_per_scene():
    batches = {}
    for scene in ETH3D_SCENES:
        sd = dataset.get_data(scene)
        scene_batches = []
        for seed in [37, 38, 39, 40, 41]:
            images_batch, images_resized = sample_8_views(sd, seed)
            # 4-view = even indices [0,2,4,6]
            subset = [images_resized[i] for i in [0,2,4,6]]
            subset_tensor = torch.stack([to_tensor(np.array(img)) for img in subset], dim=0).to(device)
            subset_batch = subset_tensor.unsqueeze(0).float()
            scene_batches.append({"seed": seed, "images_batch": images_batch, "subset_batch": subset_batch})
        batches[scene] = scene_batches
    return batches

scene_batches = prepare_batches_per_scene()
print("Prepared batches for scenes:", list(scene_batches.keys()))
```

---

## Cell 3: Token Extraction and Metrics (Local per-token KL/Cos)

```python
tokens_per_view = 37 * 37  # 1369 patch tokens

def get_tokens(model_obj, images_batch, subset_batch):
    base = model_obj.da3 if hasattr(model_obj, "da3") else model_obj
    with torch.no_grad():
        tokens_8 = base.model.backbone.forward_with_attention(images_batch, attn_layers=[], ref_view_strategy="first")[0][-1][0]
        tokens_8_even = tokens_8[:, [0,2,4,6]]
        tokens_4 = base.model.backbone.forward_with_attention(subset_batch, attn_layers=[], ref_view_strategy="first")[0][-1][0]
    return tokens_8, tokens_8_even, tokens_4

def split_tokens(tokens):
    C_half = tokens.shape[-1] // 2
    return tokens[..., :C_half], tokens[..., C_half:]

def per_token_softmax(arr):
    x = arr - arr.max(axis=-1, keepdims=True)
    expx = np.exp(x)
    return expx / (expx.sum(axis=-1, keepdims=True) + 1e-8)

def mean_token_metrics(a_raw, b_raw):
    a_sm = per_token_softmax(a_raw)
    b_sm = per_token_softmax(b_raw)
    kl_map = a_sm * (np.log(a_sm + 1e-8) - np.log(b_sm + 1e-8))
    kl_mean = float(kl_map.sum(axis=-1).mean())
    l2_mean = float(np.sqrt(((a_sm - b_sm) ** 2).sum(axis=-1)).mean())
    cos_map = (a_sm * b_sm).sum(axis=-1) / (np.linalg.norm(a_sm, axis=-1) * np.linalg.norm(b_sm, axis=-1) + 1e-8)
    cos_mean = float(cos_map.mean())
    return {"KL_token": kl_mean, "L2_token": l2_mean, "cos_token": cos_mean}

def compute_pair_metrics(tokens_dict):
    # tokens_dict: {"all_8":..., "loc_8":..., ...} from run_tokens-like logic
    base_8even_loc = tokens_dict["base_loc_8even"]
    base_4_loc = tokens_dict["base_loc_4"]
    lora_8even_loc = tokens_dict["lora_loc_8even"]
    lora_4_loc = tokens_dict["lora_loc_4"]
    rows = [
        ("baseline_8even vs baseline_4v", base_8even_loc, base_4_loc),
        ("baseline_8even vs lora_4v", base_8even_loc, lora_4_loc),
        ("baseline_4v vs lora_4v", base_4_loc, lora_4_loc),
        ("lora_8even vs baseline_8even", lora_8even_loc, base_8even_loc),
    ]
    return [{ "pair": name, **mean_token_metrics(a, b) } for name, a, b in rows]
```

---

## Cell 4: Models (Baseline, Fresh Student LoRA Init) and Epoch -1 Metrics

```python
baseline_model = DepthAnything3.from_pretrained(BASE_MODEL).to(device).eval()
student_init = StudentModel(model_name=BASE_MODEL, lora_rank=16, lora_alpha=16.0, train_camera_token=True).to(device)
student_init.eval()

def run_scene_metrics(model_base, model_lora, batches):
    results = {}
    for scene, clips in batches.items():
        metrics_scene = []
        for clip in clips:
            tokens_base = get_tokens(model_base, clip["images_batch"], clip["subset_batch"])
            tokens_lora = get_tokens(model_lora, clip["images_batch"], clip["subset_batch"])

            # split locals
            base_loc_8, _ = split_tokens(tokens_base[0]); base_loc_8even, _ = split_tokens(tokens_base[1]); base_loc_4, _ = split_tokens(tokens_base[2])
            lora_loc_8, _ = split_tokens(tokens_lora[0]); lora_loc_8even, _ = split_tokens(tokens_lora[1]); lora_loc_4, _ = split_tokens(tokens_lora[2])

            tokens_dict = {
                "base_loc_8even": base_loc_8even.detach().cpu().numpy(),
                "base_loc_4": base_loc_4.detach().cpu().numpy(),
                "lora_loc_8even": lora_loc_8even.detach().cpu().numpy(),
                "lora_loc_4": lora_loc_4.detach().cpu().numpy(),
            }
            metrics_scene.append(compute_pair_metrics(tokens_dict))
        results[scene] = metrics_scene
    return results

metrics_epoch_neg1 = run_scene_metrics(baseline_model, student_init, scene_batches)
with open(os.path.join(OUTPUT_DIR, "metrics_epoch-1.json"), "w") as f:
    json.dump(metrics_epoch_neg1, f, indent=2)
print("Saved epoch -1 metrics")
```

---

## Cell 5: Train 1 Epoch (Local Per-Token Loss) — Command to Run

Train a fresh LoRA from DA3-GIANT-1.1 with:
- Seeds 37–41, 5 samples/scene (one 8-view clip per seed), first frame as ref.
- Teacher 8 views, student 4 views (even indices).
- Loss: local per-token cosine + KL (channel softmax).
- LR 1e-4, epochs 1, batch size 2 (adjust to your GPU).

Run this shell cell on your machine (not executed here):

```bash
OUTPUT_DIR=./checkpoints/distill_eth3d_seeds37_41_local_token

python scripts/train_distill.py \
  --data_root /home/22097845d/Depth-Anything-3/workspace/benchmark_dataset/eth3d \
  --dataset eth3d \
  --model_name depth-anything/DA3-GIANT-1.1 \
  --samples_per_scene 5 \
  --seeds 37 38 39 40 41 \
  --use_first_frame_ref \
  --num_views 8 \
  --student_views 4 \
  --loss_local_token_cos_kl 1.0 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 4 \
  --lr 1e-4 \
  --lora_rank 16 --lora_alpha 16 \
  --output_dir ${OUTPUT_DIR}
```

After it finishes, set `LORA_PATH_EPOCH0=${OUTPUT_DIR}/epoch_0_lora.pt` (adjust name if different) and rerun metrics.

---

## Cell 6: Load Epoch-0 LoRA and Recompute Metrics

```python
LORA_PATH_EPOCH0 = os.path.join(OUTPUT_DIR, "epoch_0_lora.pt")  # update if your filename differs
lora_model_epoch0 = StudentModel(model_name=BASE_MODEL, lora_rank=16, lora_alpha=16.0, train_camera_token=True).to(device)
lora_model_epoch0.load_lora_weights(LORA_PATH_EPOCH0)
lora_model_epoch0.eval()

metrics_epoch0 = run_scene_metrics(baseline_model, lora_model_epoch0, scene_batches)
with open(os.path.join(OUTPUT_DIR, "metrics_epoch0.json"), "w") as f:
    json.dump(metrics_epoch0, f, indent=2)
print("Saved epoch 0 metrics")
```

---

## Cell 7: Summarize Improvements (KL/Cos)

```python
def summarize_improvements(metrics_neg1, metrics_0):
    rows = []
    kl_improved = 0
    cos_improved = 0
    total = 0
    for scene in metrics_neg1:
        m_neg1 = metrics_neg1[scene]
        m_0 = metrics_0.get(scene, [])
        if not m_0 or not m_neg1:
            continue
        # take seed-averaged KL/Cos for baseline_8even vs lora_4v (row index 1)
        kl_neg1 = np.mean([entry[1]["KL_token"] for entry in m_neg1])
        cos_neg1 = np.mean([entry[1]["cos_token"] for entry in m_neg1])
        kl_0 = np.mean([entry[1]["KL_token"] for entry in m_0])
        cos_0 = np.mean([entry[1]["cos_token"] for entry in m_0])
        kl_improved += int(kl_0 < kl_neg1)
        cos_improved += int(cos_0 > cos_neg1)
        total += 1
        rows.append({
            "scene": scene,
            "KL_epoch-1": kl_neg1,
            "KL_epoch0": kl_0,
            "KL_delta": kl_0 - kl_neg1,
            "Cos_epoch-1": cos_neg1,
            "Cos_epoch0": cos_0,
            "Cos_delta": cos_0 - cos_neg1,
        })
    summary = {
        "kl_improved_scenes": kl_improved,
        "cos_improved_scenes": cos_improved,
        "total_scenes": total,
    }
    return rows, summary

rows_table, summary = summarize_improvements(metrics_epoch_neg1, metrics_epoch0)
display(pd.DataFrame(rows_table))
print("Summary:", summary)
```

---

## Cell 8: Benchmark LoRA on ETH3D (Seeds 42, 43, 44)

Use `scripts/benchmark_lora.py` (pose + recon_unposed) for 4-view inputs; run three seeds and average:

```bash
for seed in 42 43 44; do
  python scripts/benchmark_lora.py \
    --lora_path ${LORA_PATH_EPOCH0} \
    --base_model ${BASE_MODEL} \
    --datasets eth3d \
    --modes pose recon_unposed \
    --max_frames 4 \
    --work_dir ${OUTPUT_DIR}/eval_seed${seed} \
    --scenes "" \
    --eval_only
done
# After runs, aggregate results from the work_dir outputs (pose/recon metrics) and average.
```

---

## Notes
- Sampling: seeds 37–41 per scene, first frame as ref, 8→4 teacher vs 4-view student.
- Loss: focus on local per-token channel-softmax KL/Cos. Globals logged too for visibility.
- Metrics: per-scene JSON saved for epoch -1 (init) and epoch 0 (post-1-epoch). Summary counts scenes where KL decreased and cosine increased.
- Benchmark: run 4-view evaluation seeds 42/43/44 via `benchmark_lora.py`, average metrics externally.

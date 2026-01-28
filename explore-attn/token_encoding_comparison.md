# Token Encoding Comparison: 8-View vs 8→4 Slice vs 4-View (Layer-39 Tokens Before Decoder)

This notebook compares encoder tokens (before the depth decoder) across:
- **8-view** run (views 0–7).
- **8→4 slice**: take only even views (0,2,4,6) from the 8-view run and renormalize.
- **4-view** run (views 0,2,4,6 as input).

For each set, it computes cosine, L1, L2, and KL between flattened token vectors:
- **Global tokens** (second half of the cat_token output).
- **Local tokens** (first half).
- **All tokens** (full concatenation).

Tables:
- Each table has 4 columns (cos, L1, L2, KL) and 2 rows:
  - Row 1: 8v vs 8→4 slice
  - Row 2: 8→4 slice vs 4v

---

## Cell 1: Imports and Setup

```python
import os
import sys
import numpy as np
import torch
import pandas as pd
from PIL import Image

sys.path.insert(0, '/home/22097845d/Depth-Anything-3/src')
from depth_anything_3.bench.datasets.eth3d import ETH3D
from depth_anything_3.api import DepthAnything3
from depth_anything_3.distillation.models import StudentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("explore-attn/outputs", exist_ok=True)
print(f"Using device: {device}")
```

---

## Cell 3: Load ETH3D Playground Scene (8 Frames)

```python
dataset = ETH3D()
scene = dataset.get_data("playground")

num_frames = 8
idxs = list(range(num_frames))
image_paths = [scene.image_files[i] for i in idxs]

images = [Image.open(path).convert("RGB") for path in image_paths]

print(f"Loaded {len(images)} frames, size={images[0].size}")
```

---

## Cell 4: Preprocess Images (Image-Only Mode)

```python
target_size = (518, 518)
images_resized = [img.resize(target_size, Image.BILINEAR) for img in images]

images_np = [np.array(img) for img in images_resized]
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_tensor(img_np):
    f = img_np.astype(np.float32) / 255.0
    img_norm = (f - mean) / std
    return torch.from_numpy(img_norm).permute(2, 0, 1)

images_tensor = torch.stack([to_tensor(img) for img in images_np], dim=0).to(device)
images_batch = images_tensor.unsqueeze(0).float()

print(f"Batch: {images_batch.shape}")
```

---

## Cell 5: Helpers for Token Extraction and Metrics

```python
tokens_per_view = 37 * 37  # 1369 patch tokens (no CLS)

def get_backbone_tokens(model_obj, x_batch):
    """
    Run backbone with attention collection disabled, return last out_layer tokens (B,S,N,Ccat).
    Supports DepthAnything3 and StudentModel (via model_obj.da3).
    """
    base = model_obj.da3 if hasattr(model_obj, "da3") else model_obj
    with torch.no_grad():
        outputs, _ = base.model.backbone.forward_with_attention(x_batch, attn_layers=[], ref_view_strategy="first")
    tokens = outputs[-1][0]  # shape: (B, S, N, Ccat)
    return tokens

def split_tokens(tokens):
    """Split concatenated local/global tokens assuming cat_token=True."""
    C = tokens.shape[-1]
    C_half = C // 2
    local = tokens[..., :C_half]
    global_ = tokens[..., C_half:]
    return local, global_

def flatten_tokens(tokens):
    """Flatten tokens over batch, views, patches, channels -> 1D numpy."""
    return tokens.reshape(-1).detach().cpu().numpy()

def normalize_dist(vec):
    """Softmax to a distribution for KL."""
    v = vec - vec.max()
    expv = np.exp(v)
    return expv / (expv.sum() + 1e-8)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def l1(a, b):
    return float(np.abs(a - b).sum())

def l2(a, b):
    return float(np.sqrt(((a - b) ** 2).sum()))

def kl(p, q):
    return float((p * (np.log(p + 1e-8) - np.log(q + 1e-8))).sum())

def metric_row(vec_a, vec_b):
    return {
        "cos": cosine(vec_a, vec_b),
        "L1": l1(vec_a, vec_b),
        "L2": l2(vec_a, vec_b),
        "KL": kl(normalize_dist(vec_a), normalize_dist(vec_b)),
    }
```

---

## Cell 6: Run 8-View, Slice Even, and Run 4-View

```python
# Utility: load baseline or LoRA model
def load_baseline(model_path):
    m = DepthAnything3.from_pretrained(model_path).to(device)
    m.eval()
    return m

def load_lora(model_path, lora_path):
    student = StudentModel(model_name=model_path, lora_rank=16, lora_alpha=16.0, train_camera_token=True).to(device)
    student.load_lora_weights(lora_path)
    student.eval()
    return student

# Prepare shared tensors
even_idx = [0, 2, 4, 6]
images_subset = [images_resized[i] for i in even_idx]
images_subset_tensor = torch.stack([to_tensor(np.array(img)) for img in images_subset], dim=0).to(device)
images_subset_batch = images_subset_tensor.unsqueeze(0).float()

def run_tokens(model_obj, images_batch_in, images_subset_batch_in):
    # 8-view
    tokens_8 = get_backbone_tokens(model_obj, images_batch_in)
    local_8, global_8 = split_tokens(tokens_8)
    # 8→4 slice
    tokens_8_even = tokens_8[:, even_idx]
    local_8_even, global_8_even = split_tokens(tokens_8_even)
    # 4-view
    tokens_4 = get_backbone_tokens(model_obj, images_subset_batch_in)
    local_4, global_4 = split_tokens(tokens_4)
    # Flatten to CPU
    def to_vecs(all_tokens, loc_tokens, glob_tokens):
        return (
            flatten_tokens(all_tokens),
            flatten_tokens(loc_tokens),
            flatten_tokens(glob_tokens),
        )
    all_8, loc_8, glob_8 = to_vecs(tokens_8, local_8, global_8)
    all_8_even, loc_8_even, glob_8_even = to_vecs(tokens_8_even, local_8_even, global_8_even)
    all_4, loc_4, glob_4 = to_vecs(tokens_4, local_4, global_4)
    # Keep raw local tensors for per-token channel softmax KL
    loc_raw = {
        "loc_8": local_8.detach().cpu().numpy(),
        "loc_8_even": local_8_even.detach().cpu().numpy(),
        "loc_4": local_4.detach().cpu().numpy(),
    }
    glob_raw = {
        "glob_8": global_8.detach().cpu().numpy(),
        "glob_8_even": global_8_even.detach().cpu().numpy(),
        "glob_4": global_4.detach().cpu().numpy(),
    }
    del tokens_8, local_8, global_8, tokens_8_even, local_8_even, global_8_even
    del tokens_4, local_4, global_4
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {
        "all_8": all_8, "loc_8": loc_8, "glob_8": glob_8,
        "all_8_even": all_8_even, "loc_8_even": loc_8_even, "glob_8_even": glob_8_even,
        "all_4": all_4, "loc_4": loc_4, "glob_4": glob_4,
        "loc_raw": loc_raw,
        "glob_raw": glob_raw,
    }

# Baseline (DA3-GIANT-1.1)
baseline_model = load_baseline("depth-anything/DA3-GIANT-1.1")
baseline_tokens = run_tokens(baseline_model, images_batch, images_subset_batch)
del baseline_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# LoRA
lora_model = load_lora("depth-anything/DA3-GIANT-1.1", "/home/22097845d/Depth-Anything-3/checkpoints/distill_experiments/08_local_robust/best_lora.pt")
lora_tokens = run_tokens(lora_model, images_batch, images_subset_batch)
del lora_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Cleanup shared batches
del images_subset_tensor, images_subset_batch, images_batch, images_tensor
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

---

## Cell 7: Compute Metrics and Show Tables

```python
def make_row(a, b):
    return {
        "pair": f"{a['name']} vs {b['name']}",
        "all": metric_row(a["all"], b["all"]),
        "loc": metric_row(a["loc"], b["loc"]),
        "glob": metric_row(a["glob"], b["glob"]),
    }

baseline_8even = {"name": "baseline_8even", "all": baseline_tokens["all_8_even"], "loc": baseline_tokens["loc_8_even"], "glob": baseline_tokens["glob_8_even"]}
baseline_4v    = {"name": "baseline_4v",    "all": baseline_tokens["all_4"],      "loc": baseline_tokens["loc_4"],      "glob": baseline_tokens["glob_4"]}
lora_8even     = {"name": "lora_8even",     "all": lora_tokens["all_8_even"],     "loc": lora_tokens["loc_8_even"],     "glob": lora_tokens["glob_8_even"]}
lora_4v        = {"name": "lora_4v",        "all": lora_tokens["all_4"],          "loc": lora_tokens["loc_4"],          "glob": lora_tokens["glob_4"]}

rows = [
    make_row(baseline_8even, baseline_4v),  # start
    make_row(baseline_8even, lora_4v),      # current
    make_row(baseline_4v,    lora_4v),      # changes
    make_row(lora_8even,     baseline_8even),  # last
]

table_all = [{"pair": r["pair"], **r["all"]} for r in rows]
table_loc = [{"pair": r["pair"], **r["loc"]} for r in rows]
table_glob = [{"pair": r["pair"], **r["glob"]} for r in rows]

print("All tokens:")
display(pd.DataFrame(table_all))

print("\nLocal tokens:")
display(pd.DataFrame(table_loc))

print("\nGlobal tokens:")
display(pd.DataFrame(table_glob))
```

---

## Cell 8: Local Tokens KL with Per-Token Channel Softmax (Training-Style Normalization)

```python
def per_token_softmax(arr):
    # arr: (B, S, P, C)
    x = arr - arr.max(axis=-1, keepdims=True)
    expx = np.exp(x)
    return expx / (expx.sum(axis=-1, keepdims=True) + 1e-8)

def mean_token_kl(a, b):
    # a, b: (B, S, P, C) distributions
    kl_map = a * (np.log(a + 1e-8) - np.log(b + 1e-8))
    return float(kl_map.sum(axis=-1).mean())  # mean over tokens

def tokenwise_metrics(a_raw, b_raw):
    a_sm = per_token_softmax(a_raw)
    b_sm = per_token_softmax(b_raw)
    cos_map = (a_sm * b_sm).sum(axis=-1) / (np.linalg.norm(a_sm, axis=-1) * np.linalg.norm(b_sm, axis=-1) + 1e-8)
    return {
        "KL_token": mean_token_kl(a_sm, b_sm),
        "L2_token": float(np.sqrt(((a_sm - b_sm) ** 2).sum(axis=-1)).mean()),
        "cos_token": float(cos_map.mean()),
    }

def make_token_row(a, b):
    return {
        "pair": f"{a['name']} vs {b['name']}",
        **tokenwise_metrics(a["loc"], b["loc"]),
    }

baseline_8even_loc = {"name": "baseline_8even", "loc": baseline_tokens["loc_raw"]["loc_8_even"]}
baseline_4v_loc    = {"name": "baseline_4v",    "loc": baseline_tokens["loc_raw"]["loc_4"]}
lora_8even_loc     = {"name": "lora_8even",     "loc": lora_tokens["loc_raw"]["loc_8_even"]}
lora_4v_loc        = {"name": "lora_4v",        "loc": lora_tokens["loc_raw"]["loc_4"]}

token_rows = [
    make_token_row(baseline_8even_loc, baseline_4v_loc),
    make_token_row(baseline_8even_loc, lora_4v_loc),
    make_token_row(baseline_4v_loc,    lora_4v_loc),
    make_token_row(lora_8even_loc,     baseline_8even_loc),
]

print("Local tokens (per-token channel softmax):")
display(pd.DataFrame(token_rows))

# Global tokens (per-token channel softmax)
global_rows = [
    make_token_row({"name": "baseline_8even", "loc": baseline_tokens["glob_raw"]["glob_8_even"]},
                   {"name": "baseline_4v",    "loc": baseline_tokens["glob_raw"]["glob_4"]}),
    make_token_row({"name": "baseline_8even", "loc": baseline_tokens["glob_raw"]["glob_8_even"]},
                   {"name": "lora_4v",        "loc": lora_tokens["glob_raw"]["glob_4"]}),
    make_token_row({"name": "baseline_4v",    "loc": baseline_tokens["glob_raw"]["glob_4"]},
                   {"name": "lora_4v",        "loc": lora_tokens["glob_raw"]["glob_4"]}),
    make_token_row({"name": "lora_8even",     "loc": lora_tokens["glob_raw"]["glob_8_even"]},
                   {"name": "baseline_8even", "loc": baseline_tokens["glob_raw"]["glob_8_even"]}),
]

print("\nGlobal tokens (per-token channel softmax):")
display(pd.DataFrame(global_rows))
```

---

## Cell 9: Multi-Scene Loop, JSON Export, and Mean Metrics

```python
import json
from depth_anything_3.utils.constants import ETH3D_SCENES

def prepare_scene_batches(scene_name):
    data = dataset.get_data(scene_name)
    image_paths_scene = data.image_files
    images_scene = [Image.open(p).convert("RGB") for p in image_paths_scene]
    images_resized_scene = [img.resize(target_size, Image.BILINEAR) for img in images_scene]
    images_np_scene = [np.array(img) for img in images_resized_scene]
    images_tensor_scene = torch.stack([to_tensor(img) for img in images_np_scene], dim=0).to(device)
    images_batch_scene = images_tensor_scene.unsqueeze(0).float()
    images_subset_scene = [images_resized_scene[i] for i in even_idx]
    images_subset_tensor_scene = torch.stack([to_tensor(np.array(img)) for img in images_subset_scene], dim=0).to(device)
    images_subset_batch_scene = images_subset_tensor_scene.unsqueeze(0).float()
    return images_batch_scene, images_subset_batch_scene

def compute_scene_metrics(images_batch_scene, images_subset_batch_scene):
    base_tokens = run_tokens(baseline_model_ref, images_batch_scene, images_subset_batch_scene)
    lora_tokens = run_tokens(lora_model_ref, images_batch_scene, images_subset_batch_scene)

    base_8even = {"name": "baseline_8even", "all": base_tokens["all_8_even"], "loc": base_tokens["loc_8_even"], "glob": base_tokens["glob_8_even"]}
    base_4v    = {"name": "baseline_4v",    "all": base_tokens["all_4"],      "loc": base_tokens["loc_4"],      "glob": base_tokens["glob_4"]}
    lora_8even = {"name": "lora_8even",     "all": lora_tokens["all_8_even"], "loc": lora_tokens["loc_8_even"], "glob": lora_tokens["glob_8_even"]}
    lora_4v    = {"name": "lora_4v",        "all": lora_tokens["all_4"],      "loc": lora_tokens["loc_4"],      "glob": lora_tokens["glob_4"]}

    rows_scene = [
        make_row(base_8even, base_4v),
        make_row(base_8even, lora_4v),
        make_row(base_4v,    lora_4v),
        make_row(lora_8even, base_8even),
    ]
    table_all_scene = [{"pair": r["pair"], **r["all"]} for r in rows_scene]
    table_loc_scene = [{"pair": r["pair"], **r["loc"]} for r in rows_scene]
    table_glob_scene = [{"pair": r["pair"], **r["glob"]} for r in rows_scene]

    # Per-token local/global metrics
    b_loc = {"name": "baseline_8even", "loc": base_tokens["loc_raw"]["loc_8_even"]}
    b_loc4 = {"name": "baseline_4v", "loc": base_tokens["loc_raw"]["loc_4"]}
    l_loc = {"name": "lora_8even", "loc": lora_tokens["loc_raw"]["loc_8_even"]}
    l_loc4 = {"name": "lora_4v", "loc": lora_tokens["loc_raw"]["loc_4"]}

    token_rows_scene = [
        make_token_row(b_loc, b_loc4),
        make_token_row(b_loc, l_loc4),
        make_token_row(b_loc4, l_loc4),
        make_token_row(l_loc, b_loc),
    ]

    b_glob = {"name": "baseline_8even", "loc": base_tokens["glob_raw"]["glob_8_even"]}
    b_glob4 = {"name": "baseline_4v", "loc": base_tokens["glob_raw"]["glob_4"]}
    l_glob = {"name": "lora_8even", "loc": lora_tokens["glob_raw"]["glob_8_even"]}
    l_glob4 = {"name": "lora_4v", "loc": lora_tokens["glob_raw"]["glob_4"]}

    global_rows_scene = [
        make_token_row(b_glob, b_glob4),
        make_token_row(b_glob, l_glob4),
        make_token_row(b_glob4, l_glob4),
        make_token_row(l_glob, b_glob),
    ]

    # Improvement flag: local_token KL reduced for 8even vs 4v when using LoRA
    kl_baseline = token_rows_scene[0]["KL_token"]
    kl_lora = token_rows_scene[1]["KL_token"]
    improved = kl_lora < kl_baseline

    return {
        "all": table_all_scene,
        "local": table_loc_scene,
        "global": table_glob_scene,
        "local_token": token_rows_scene,
        "global_token": global_rows_scene,
        "improved_local": improved,
    }

# Keep models on device for loop
baseline_model_ref = load_baseline("depth-anything/DA3-GIANT-1.1")
lora_model_ref = load_lora("depth-anything/DA3-GIANT-1.1", "/home/22097845d/Depth-Anything-3/checkpoints/distill_experiments/08_local_robust/best_lora.pt")

scene_results = {}
for scene_name in ETH3D_SCENES:
    try:
        ib, isub = prepare_scene_batches(scene_name)
        scene_results[scene_name] = compute_scene_metrics(ib, isub)
        del ib, isub
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Scene {scene_name} failed: {e}")

# Save JSON
with open("explore-attn/outputs/token_metrics.json", "w") as f:
    json.dump(scene_results, f, indent=2)

# Mean across scenes for key metrics (local_token KL/cos)
kl_list = []
cos_list = []
for v in scene_results.values():
    if "local_token" in v and v["local_token"]:
        kl_list.append(v["local_token"][1]["KL_token"])  # baseline_8even vs lora_4v
        cos_list.append(v["local_token"][1]["cos_token"])

print("\nMean local-token metrics (baseline_8even vs lora_4v) across scenes:")
print(f"  KL_token mean: {np.mean(kl_list):.6f} over {len(kl_list)} scenes")
print(f"  cos_token mean: {np.mean(cos_list):.6f} over {len(cos_list)} scenes")
```

---

## Notes
- Tokens are taken from the last backbone out_layer (cat_token output). Local/global split assumes first/second half of the channel dimension.
- Metrics (Cells 7) use flattened vectors: Cos/L1/L2 on raw values; KL on softmax-normalized flattened vectors.
- Cell 8 uses per-token channel softmax (training-style normalization) on local tokens and reports mean KL/L2 across tokens.
- Cell 9 loops over ETH3D scenes, saves per-scene metrics to `explore-attn/outputs/token_metrics.json`, marks whether LoRA reduced the local-token KL (8even vs 4v), and prints mean local-token KL/cos across scenes.
- Both 8-view and 4-view runs are **image-only** (no extrinsics/intrinsics).

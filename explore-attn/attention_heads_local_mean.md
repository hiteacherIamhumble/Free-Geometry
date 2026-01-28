# Local Attention Heads (Layer 38) — Per-Head, CLS or Patch Query (View 0)

This notebook visualizes **per-head attention for layer 38** using one query per head:
- Choose **CLS query** (DINO-style) or a **specific patch query**.
- Optional **top-mass mask** (keep strongest x% attention) for the classic sparse DINO look.
- Per-head heatmaps and overlays on view 0.

Outputs go to `explore-attn/outputs/`.

---

## Cell 1: Imports and Setup

```python
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, '/home/22097845d/Depth-Anything-3/src')
from depth_anything_3.bench.datasets.eth3d import ETH3D
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("explore-attn/outputs", exist_ok=True)
print(f"Using device: {device}")
```

---

## Cell 2: Load Model

```python
print("Loading DA3 Giant model...")
model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
model = model.to(device=device)
model.eval()
print("Loaded!")
```

---

## Cell 3: Load ETH3D Playground Scene (8 Frames)

```python
dataset = ETH3D()
scene = dataset.get_data("playground")

num_frames = 8
idxs = list(range(num_frames))
image_paths = [scene.image_files[i] for i in idxs]
extrinsics = scene.extrinsics[idxs]
intrinsics = scene.intrinsics[idxs]

images = [Image.open(p).convert("RGB") for p in image_paths]
print(f"Loaded {len(images)} frames, size={images[0].size}")
```

---

## Cell 4: Preprocess Images

```python
target_size = (518, 518)
images_resized = [img.resize(target_size, Image.BILINEAR) for img in images]

images_np = [np.array(img) for img in images_resized]
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

images_norm = []
for img in images_np:
    f = img.astype(np.float32) / 255.0
    img_norm = (f - mean) / std
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1)
    images_norm.append(img_t)

images_tensor = torch.stack(images_norm, dim=0).to(device)
images_batch = images_tensor.unsqueeze(0).float()

extrinsics_tensor = torch.from_numpy(extrinsics.astype(np.float32)).to(device).unsqueeze(0)
intrinsics_tensor = torch.from_numpy(intrinsics.astype(np.float32)).to(device).unsqueeze(0)

print(f"Batch: {images_batch.shape}")
```

---

## Cell 5: Forward With Attention (Layer 38 Only)

```python
with torch.no_grad():
    _, attn_weights = model.model.forward_with_attention(
        images_batch,
        extrinsics=extrinsics_tensor,
        intrinsics=intrinsics_tensor,
        attn_layers=[38],
        ref_view_strategy="first",
    )

attn_38 = attn_weights[38]  # shape: (8 views, heads, 1370, 1370)
print(f"Layer 38 attention shape: {attn_38.shape}")
```

---

## Cell 6: Query Selection and Helpers

```python
grid_size = 37
patch_size = 14
image_size = 518
patch_idx = 684  # only used if query_mode="patch"

query_mode = "cls"   # "cls" for CLS query (DINO style), "patch" for a chosen patch
top_mass = 0.9       # keep top x% mass (set to 1.0 to disable masking)

def keep_top_mass(vec, mass=0.9):
    """
    Keep smallest set of entries whose sum >= mass; zero the rest.
    Guarantees keeping at least the top element and handles zero-sum safely.
    """
    if mass >= 1.0:
        return vec
    flat = vec.copy()
    total = flat.sum()
    if total <= 0:
        return flat  # all zeros, nothing to keep
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    cutoff_idx = np.searchsorted(cumsum, mass * total, side="left")
    keep_count = min(len(idx), max(1, cutoff_idx + 1))  # keep at least top-1
    keep = idx[:keep_count]
    mask = np.zeros_like(flat)
    mask[keep] = 1.0
    return flat * mask

def head_query_map(attn_mat, smooth_sigma=2.0, query_mode="cls", top_mass=1.0):
    """
    attn_mat: (1370, 1370) for one head, one view.
    - Select query row: CLS if query_mode="cls"; patch_idx+1 if query_mode="patch".
    - Drop CLS column.
    - Zero self for patch queries.
    - Optional top-mass mask (DINO-like) keeps strongest mass.
    - Normalize, reshape to grid, bilinear upsample, smooth.
    """
    attn = attn_mat.copy()
    if query_mode == "cls":
        attn_vec = attn[0]  # CLS query
    else:
        attn_vec = attn[patch_idx + 1]  # specific patch query

    attn_vec = attn_vec[1:]  # drop CLS column
    if query_mode != "cls":
        attn_vec[patch_idx] = 0.0  # remove self for patch query

    attn_vec = keep_top_mass(attn_vec, mass=top_mass)
    attn_vec = attn_vec / (attn_vec.sum() + 1e-8)

    attn_grid = attn_vec.reshape(1, 1, grid_size, grid_size)
    attn_grid = torch.from_numpy(attn_grid).float()
    heatmap = torch.nn.functional.interpolate(
        attn_grid, size=(image_size, image_size), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    if smooth_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)
    return heatmap

def overlay_heatmap(image_np, heatmap, alpha=0.6, cmap="jet"):
    if isinstance(image_np, Image.Image):
        image_np = np.array(image_np)
    image_np = image_np.astype(np.float32)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_color = plt.get_cmap(cmap)(heatmap_norm)[..., :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    blended = (1 - alpha) * image_np + alpha * heatmap_color
    return blended.astype(np.uint8)
```

---

## Cell 7: Per-Head Attention Maps (View 0)

```python
num_heads = attn_38.shape[1]
heatmaps = []
for h in range(num_heads):
    attn_mat = attn_38[0, h].cpu().numpy()
    heatmaps.append(head_query_map(attn_mat, query_mode=query_mode, top_mass=top_mass))

# Heatmaps grid
cols = 4
rows = int(np.ceil(num_heads / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()
for h in range(num_heads):
    ax = axes[h]
    im = ax.imshow(heatmaps[h], cmap="jet")
    ax.set_title(f"Head {h}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for j in range(num_heads, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("explore-attn/outputs/layer38_view0_heads_query_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# Overlays grid
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()
for h in range(num_heads):
    ax = axes[h]
    blended = overlay_heatmap(images_resized[0], heatmaps[h])
    ax.imshow(blended)
    ax.set_title(f"Head {h} Overlay")
    ax.axis("off")
for j in range(num_heads, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("explore-attn/outputs/layer38_view0_heads_query_overlays.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Notes
- `query_mode="cls"` gives the classic CLS query visualization; set to `"patch"` and adjust `patch_idx` to target a specific patch.
- Set `top_mass < 1` (e.g., 0.9) to keep only the strongest mass (DINO-like); set to 1.0 to disable masking.
- CLS column is dropped; self is zeroed for patch queries to reduce diagonal dominance.
- Adjust `smooth_sigma` in `head_query_map` for sharper or smoother maps.

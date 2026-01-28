# Global Attention (Layer 39) — Mean Over Heads, CLS or Patch Query

This notebook visualizes **global attention (layer 39)**:
- Use a single **query**: either **CLS** (DINO-style) or a specific **patch** in view 0.
- Average **over all heads** first, then visualize the query’s attention to each view.
- Optional **top-mass mask** (keep strongest x% of attention) for a sparse DINO look.
- Heatmaps and overlays shown for **views 1–7** (cross-view), with view 0 as the query origin.

Outputs go to `explore-attn/outputs/`.

---

## Cell 1: Imports and Setup

```python
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

print(f"Batch: {images_batch.shape}")
```

---

## Cell 5: Forward With Attention (Layer 39)

```python
with torch.no_grad():
    _, attn_weights = model.model.forward_with_attention(
        images_batch,
        attn_layers=[39],
        ref_view_strategy="first",
    )

attn_39 = attn_weights[39]  # shape: (1, heads, 10960, 10960) for 8 views
print(f"Layer 39 attention shape: {attn_39.shape}")
```

---

## Cell 6: Query Selection and Helpers

```python
grid_size = 37
patch_size = 14
image_size = 518
tokens_per_view = 1 + grid_size * grid_size  # 1370
patch_idx = 684  # only used if query_mode="patch" (center patch)

query_mode = "cls"   # "cls" for CLS query (DINO style), "patch" for a chosen patch in view 0
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
        return flat
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    cutoff_idx = np.searchsorted(cumsum, mass * total, side="left")
    keep_count = min(len(idx), max(1, cutoff_idx + 1))
    keep = idx[:keep_count]
    mask = np.zeros_like(flat)
    mask[keep] = 1.0
    return flat * mask

def view_heatmap(attn_vec_full, view_idx, smooth_sigma=2.0, query_mode="cls", top_mass=1.0):
    """
    attn_vec_full: (8*1370,) attention from the query to all tokens across views.
    Extracts one view slice, drops CLS, zeroes self for patch queries in view 0,
    applies top-mass mask, normalizes, upsamples, smooths.
    """
    start = view_idx * tokens_per_view
    end = (view_idx + 1) * tokens_per_view
    attn_vec = attn_vec_full[start:end]  # (1370,)
    attn_vec = attn_vec[1:]  # drop CLS
    if query_mode != "cls" and view_idx == 0:
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

def draw_query_box(ax, idx, color="white", lw=2):
    r, c = divmod(idx, grid_size)
    x = c * patch_size
    y = r * patch_size
    rect = patches.Rectangle((x, y), patch_size, patch_size,
                             linewidth=lw, edgecolor=color, facecolor="none", alpha=0.9)
    ax.add_patch(rect)
```

---

## Cell 7: Mean Over Heads → Per-View Heatmaps and Overlays (Views 0–7)

```python
# Mean over heads first
attn_mean = attn_39.mean(dim=1)[0]  # (10960, 10960)

# Select query token (view 0 CLS or patch)
if query_mode == "cls":
    query_idx = 0  # CLS of view 0
else:
    query_idx = 1 + patch_idx  # patch in view 0

attn_query = attn_mean[query_idx, :].cpu().numpy()

# Build heatmaps for all views (0–7)
heatmaps_8 = {}
for v in range(0, 8):
    heatmaps_8[v] = view_heatmap(attn_query, v, query_mode=query_mode, top_mass=top_mass)

# Extract even-index subset (0,2,4,6) from the 8-view run for fair comparison
subset_views = [0, 2, 4, 6]
attn_query_even = np.concatenate(
    [attn_query[v * tokens_per_view : (v + 1) * tokens_per_view] for v in subset_views], axis=0
)
heatmaps_8_even = {}
for local_idx, orig_v in enumerate(subset_views):
    heatmaps_8_even[orig_v] = view_heatmap(attn_query_even, local_idx, query_mode=query_mode, top_mass=top_mass)

# Plot heatmaps for views 0–7
cols = 4
rows = 2
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()
for i, v in enumerate(range(0, 8)):
    ax = axes[i]
    im = ax.imshow(heatmaps_8[v], cmap="jet")
    ax.set_title(f"View {v}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for j in range(8, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("explore-attn/outputs/layer39_global_views0_7_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# Overlays for views 0–7
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()
for i, v in enumerate(range(0, 8)):
    ax = axes[i]
    blended = overlay_heatmap(images_resized[v], heatmaps_8[v])
    ax.imshow(blended)
    if query_mode != "cls" and v == 0:
        draw_query_box(ax, patch_idx)
    ax.set_title(f"View {v} Overlay")
    ax.axis("off")
for j in range(8, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("explore-attn/outputs/layer39_global_views0_7_overlays.png", dpi=150, bbox_inches="tight")
plt.show()

# Save objects for later comparison, then clear large tensors
heatmaps_8_views = heatmaps_8  # keep for later comparison
heatmaps_8_even_views = heatmaps_8_even
attn_query_cpu = attn_query.copy()
attn_query_even_cpu = attn_query_even.copy()
del attn_weights, attn_39, attn_mean, images_batch, images_tensor
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

---

## Cell 8: Re-run with Views [0, 2, 4, 6] Only (Image-Only Forward)

```python
# Select subset of views (even indices)
subset_views = [0, 2, 4, 6]
images_subset = [images_resized[i] for i in subset_views]

# Prepare tensor (image-only mode: no extrinsics/intrinsics)
images_subset_tensor = torch.stack(
    [torch.from_numpy(((np.array(img).astype(np.float32) / 255.0 - mean) / std)).permute(2,0,1)
     for img in images_subset],
    dim=0
).to(device)
images_subset_batch = images_subset_tensor.unsqueeze(0).float()

with torch.no_grad():
    _, attn_weights4 = model.model.forward_with_attention(
        images_subset_batch,
        attn_layers=[39],
        ref_view_strategy="first",
    )

attn_39_4 = attn_weights4[39]  # shape: (1, heads, 4*1370, 4*1370)
attn_mean_4 = attn_39_4.mean(dim=1)[0]  # (5480, 5480)

if query_mode == "cls":
    query_idx_4 = 0  # CLS of view 0
else:
    query_idx_4 = 1 + patch_idx  # patch in view 0

attn_query_4 = attn_mean_4[query_idx_4, :].cpu().numpy()
attn_query_4_cpu = attn_query_4.copy()

# Build heatmaps for subset views (indices 0..3 correspond to subset_views)
heatmaps_4 = {}
for local_v, orig_v in enumerate(subset_views):
    heatmaps_4[orig_v] = view_heatmap(attn_query_4, local_v, query_mode=query_mode, top_mass=top_mass)

# Plot heatmaps for subset views
cols = 4
rows = 1
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()
for i, orig_v in enumerate(subset_views):
    ax = axes[i]
    im = ax.imshow(heatmaps_4[orig_v], cmap="jet")
    ax.set_title(f"View {orig_v} (4-view)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("explore-attn/outputs/layer39_global_views_subset_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# Overlays for subset views
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()
for i, orig_v in enumerate(subset_views):
    ax = axes[i]
    blended = overlay_heatmap(images_subset[i], heatmaps_4[orig_v])
    ax.imshow(blended)
    if query_mode != "cls" and orig_v == 0:
        draw_query_box(ax, patch_idx)
    ax.set_title(f"View {orig_v} Overlay (4-view)")
    ax.axis("off")
plt.tight_layout()
plt.savefig("explore-attn/outputs/layer39_global_views_subset_overlays.png", dpi=150, bbox_inches="tight")
plt.show()

# Clear
del attn_weights4, attn_39_4, attn_mean_4, attn_query_4
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

---

## Cell 9: Compare Heatmaps for Views 0, 2, 4, 6 (8-view full, 8-view even-only, 4-view) with Shared Color Scale

```python
cols = 4
rows = 3
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.reshape(rows, cols)

# Shared color scale across all three rows
all_vals = []
for orig_v in subset_views:
    all_vals.append(heatmaps_8_views[orig_v])
    all_vals.append(heatmaps_8_even_views[orig_v])
    all_vals.append(heatmaps_4[orig_v])
all_vals = np.stack(all_vals, axis=0)
vmin = all_vals.min()
vmax = all_vals.max()

# Row 0: from 8-view run (original per-view maps)
for i, orig_v in enumerate(subset_views):
    ax = axes[0, i]
    im = ax.imshow(heatmaps_8_views[orig_v], cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(f"8-view: View {orig_v}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 1: 8-view run, even-only renorm (sliced to views 0,2,4,6)
for i, orig_v in enumerate(subset_views):
    ax = axes[1, i]
    im = ax.imshow(heatmaps_8_even_views[orig_v], cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(f"8-view-even: View {orig_v}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 2: from 4-view run
for i, orig_v in enumerate(subset_views):
    ax = axes[2, i]
    im = ax.imshow(heatmaps_4[orig_v], cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(f"4-view: View {orig_v}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("explore-attn/outputs/layer39_global_views_comparison_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Cell 10: Numeric Metrics (8-view vs 8-view-even vs 4-view) on CPU

```python
def norm_attn_vec(attn_vec, drop_cls=True, zero_self=False, self_idx=None):
    v = attn_vec[1:] if drop_cls else attn_vec.copy()
    if zero_self and self_idx is not None and 0 <= self_idx < v.shape[0]:
        v[self_idx] = 0.0
    v = v / (v.sum() + 1e-8)
    return v

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def l1(a, b):
    return float(np.abs(a - b).sum())

def kl(a, b):
    return float((a * (np.log(a + 1e-8) - np.log(b + 1e-8))).sum())

# Collect per-view vectors (before heatmap) for 8-view full, 8-view-even, 4-view
attn_vecs_8 = {}
attn_vecs_8even = {}
attn_vecs_4 = {}

for v in subset_views:  # [0,2,4,6]
    attn_vecs_8[v] = attn_query_cpu[v * tokens_per_view : (v + 1) * tokens_per_view].copy()
    # even-slice is already concatenated; mapping: local 0->orig0,1->orig2,2->orig4,3->orig6
    local_idx = subset_views.index(v)
    attn_vecs_8even[v] = attn_query_even_cpu[local_idx * tokens_per_view : (local_idx + 1) * tokens_per_view].copy()
    attn_vecs_4[v] = attn_query_4_cpu[local_idx * tokens_per_view : (local_idx + 1) * tokens_per_view].copy()

metrics = {}
for v in subset_views:
    zero_self = (query_mode != "cls")
    drop_cls = (query_mode != "cls")  # keep CLS target when CLS is the query
    self_idx = patch_idx if zero_self else None
    a = norm_attn_vec(attn_vecs_8[v], drop_cls=drop_cls, zero_self=zero_self, self_idx=self_idx)
    b = norm_attn_vec(attn_vecs_8even[v], drop_cls=drop_cls, zero_self=zero_self, self_idx=self_idx)
    c = norm_attn_vec(attn_vecs_4[v], drop_cls=drop_cls, zero_self=zero_self, self_idx=self_idx)
    metrics[v] = {
        "8v_vs_8even_cos": cosine(a, b),
        "8v_vs_4v_cos": cosine(a, c),
        "8even_vs_4v_cos": cosine(b, c),
        "8v_vs_4v_l1": l1(a, c),
        "8v_vs_4v_kl": kl(a, c),
    }

print("Per-view metrics (views 0,2,4,6):")
for v in subset_views:
    print(f"View {v}: {metrics[v]}")

# Free intermediate numpy arrays if desired
del attn_vecs_8, attn_vecs_8even, attn_vecs_4
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

---

## Notes
- `query_mode="cls"` gives CLS query; `"patch"` plus `patch_idx` targets a specific patch and draws the query box on view 0 overlays.
- `top_mass < 1` (e.g., 0.9) produces sparse DINO-like maps; 1.0 disables masking.
- Cell 7: 8-view forward in **image-only mode** (no extrinsics/intrinsics), views 0–7; also slices even views to build a renormalized 8→4 subset.
- Cell 8: 4-view forward on views [0, 2, 4, 6] in **image-only mode**.
- Cell 9: Comparison for views 0, 2, 4, 6 with three rows: (a) 8-view per-view maps, (b) 8-view even-only renormalized, (c) 4-view run, using shared color scale.
- Cell 10: Numeric metrics (cosine, L1, KL) across 8-view, 8→4 even slice, and 4-view, computed on CPU to avoid GPU OOM.

# ETH3D Dataset Reference for Depth-Anything-3

## 1. Disk Layout

**Root:** `/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset/eth3d/`

```
eth3d/
├── courtyard/
├── delivery_area/
├── electro/
├── facade/
├── kicker/
├── office/
├── pipes/
├── playground/
├── relief/
├── relief_2/
└── terrains/
```

**Per-scene structure:**
```
{scene}/
├── combined_mesh.ply                          # GT laser-scanned mesh (PLY)
├── dslr_calibration_jpg/
│   ├── cameras.txt                            # COLMAP intrinsics (4 cameras)
│   ├── images.txt                             # COLMAP poses (2 lines per image)
│   └── points3D.txt                           # SfM 3D points (unused by code)
├── images/dslr_images/
│   ├── DSC_XXXX.JPG                           # RGB images (6048x4032 JPEG)
│   └── DSC_XXXX.JPG.jpg                       # Thumbnails (ignored by code)
├── ground_truth_depth/dslr_images/
│   └── DSC_XXXX.JPG                           # GT depth (binary float32, NOT JPEG despite extension)
└── masks_for_images/dslr_images/
    └── DSC_XXXX.png                           # GT masks (PNG uint8: 0=valid, 1=occluded)
```

**Per-scene image counts (after filtering):**

| Scene | Images (JPG) | GT Depth | Masks | Filtered Out |
|-------|-------------|----------|-------|--------------|
| courtyard | 38 | 38 | 38 | 0 |
| delivery_area | 40 | 44 | 42 | 4 |
| electro | 39 | 45 | 40 | 6 |
| facade | 76 | 76 | 79 | 0 |
| kicker | 31 | 31 | 33 | 0 |
| office | 26 | 26 | 26 | 0 |
| pipes | 14 | 14 | 14 | 0 |
| playground | 32 | 38 | 40 | 6 |
| relief | 19 | 31 | 30 | 12 |
| relief_2 | 20 | 31 | 31 | 11 |
| terrains | 42 | 42 | 42 | 0 |

Note: "Images" column = usable images after filtering. Raw JPG count on disk is higher for filtered scenes.

---

## 2. Constants

**Source:** `src/depth_anything_3/utils/constants.py` (lines 114-158)

```python
ETH3D_EVAL_DATA_ROOT = "/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset/eth3d"

ETH3D_SCENES = [
    "courtyard", "electro", "kicker", "pipes", "relief",
    # "terrace",       # Excluded: known issues
    "delivery_area", "facade",
    # "meadow",        # Excluded: known issues
    "office", "playground", "relief_2", "terrains",
]

ETH3D_FILTER_KEYS = {
    "delivery_area": ["711.JPG", "712.JPG", "713.JPG", "714.JPG"],
    "electro": ["9289.JPG", "9290.JPG", "9291.JPG", "9292.JPG", "9293.JPG", "9298.JPG"],
    "playground": ["587.JPG", "588.JPG", "589.JPG", "590.JPG", "591.JPG", "592.JPG"],
    "relief": [
        "427.JPG", "428.JPG", "429.JPG", "430.JPG", "431.JPG", "432.JPG",
        "433.JPG", "434.JPG", "435.JPG", "436.JPG", "437.JPG", "438.JPG",
    ],
    "relief_2": [
        "458.JPG", "459.JPG", "460.JPG", "461.JPG", "462.JPG", "463.JPG",
        "464.JPG", "465.JPG", "466.JPG", "467.JPG", "468.JPG",
    ],
}

# TSDF fusion hyperparameters
ETH3D_VOXEL_LENGTH = 4.0 / 512.0 * 5   # = 0.0390625 m
ETH3D_SDF_TRUNC    = 0.04 * 5           # = 0.2 m
ETH3D_MAX_DEPTH    = 100000.0           # effectively no truncation

# Point cloud sampling
ETH3D_SAMPLING_NUMBER = 1_000_000

# 3D reconstruction evaluation
ETH3D_EVAL_THRESHOLD = 0.05 * 5         # = 0.25 m (precision/recall threshold)
ETH3D_DOWN_SAMPLE    = 4.0 / 512.0 * 5  # = 0.0390625 m (voxel downsampling)
```

---

## 3. ETH3D Dataset Class

**Source:** `src/depth_anything_3/bench/datasets/eth3d.py`

**Registration:**
```python
@MV_REGISTRY.register(name="eth3d")
@MONO_REGISTRY.register(name="eth3d")
class ETH3D(Dataset):
```

### 3.1 `get_data(scene)` Return Format

Returns `addict.Dict` with:

| Key | Type | Shape | Dtype | Description |
|-----|------|-------|-------|-------------|
| `image_files` | `List[str]` | `[N]` | — | Absolute paths to JPG images |
| `extrinsics` | `np.ndarray` | `[N, 4, 4]` | float32 | World-to-camera transforms |
| `intrinsics` | `np.ndarray` | `[N, 3, 3]` | float32 | Camera intrinsic matrices |
| `aux.gt_mesh_path` | `str` | — | — | Path to `combined_mesh.ply` |
| `aux.heights` | `List[float]` | `[N]` | — | Original image heights (4032.0) |
| `aux.widths` | `List[float]` | `[N]` | — | Original image widths (6048.0) |

**Example values (courtyard, frame 0):**
```python
image_files[0] = ".../eth3d/courtyard/images/dslr_images/DSC_0307.JPG"
extrinsics[0] = [[-0.768, -0.638, -0.054,  1.802],
                 [ 0.256, -0.229, -0.939, -0.821],
                 [ 0.587, -0.736,  0.339, -7.083],
                 [ 0.   ,  0.   ,  0.   ,  1.   ]]   # 4x4 float32
intrinsics[0] = [[3409.58,    0.  , 3036.34],
                 [   0.  , 3409.44, 2013.30],
                 [   0.  ,    0.  ,    1.  ]]          # 3x3 float32
aux.heights[0] = 4032.0
aux.widths[0]  = 6048.0
```

**Caching:** Results are cached in `self._scene_cache` — subsequent calls return the same object.

### 3.2 Calibration File Parsing

**cameras.txt** (COLMAP format, skip first 3 header lines):
```
# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
0 THIN_PRISM_FISHEYE 6048 4032 3411.42 3410.02 3041.29 2014.07 0.21047 ...
```
Parsed fields: `camera_id → {width, height, fx, fy, cx, cy}` (distortion params ignored).

**images.txt** (COLMAP format, skip first 4 header lines, every other line):
```
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
22 -0.292265 -0.174238 0.547904 -0.764214 1.80195 -0.821371 -7.08304 1 dslr_images/DSC_0307.JPG
<points2D line — skipped>
```
Parsed: quaternion `[QW, QX, QY, QZ]` → rotation via `quat2rotmat()`, translation `[TX, TY, TZ]` → 4x4 extrinsic.

**Filtering:** Images whose filename ends with any string in `ETH3D_FILTER_KEYS[scene]` are skipped.

### 3.3 GT Mask Loading (`_load_gt_mask`)

```python
# Mask path:  {root}/{scene}/masks_for_images/dslr_images/DSC_XXXX.png
# Depth path: {root}/{scene}/ground_truth_depth/dslr_images/DSC_XXXX.JPG
```

- GT depth: `np.fromfile(path, dtype=np.float32).reshape(H, W)` — raw binary, NOT JPEG
- GT mask: `cv2.imread(path, IMREAD_GRAYSCALE)` — uint8, value 1 = occluded/invalid
- Combined zero_mask: `valid = NOT(mask==1) AND NOT(depth==0 OR depth==inf)`
- Returns boolean mask where `True` = valid region to keep

### 3.4 `fuse3d(scene, result_path, fuse_path, mode)`

Pipeline:
1. Load GT meta (if exists) or full dataset GT
2. Load original RGB images at original size (6048x4032)
3. Call `_prep_unposed()` or `_prep_posed()`:
   - Resize predicted depth to original size (`cv2.INTER_NEAREST`)
   - Apply GT mask BEFORE scale alignment
   - Umeyama scale alignment (`random_state=42`, `ransac=True`)
   - Adjust intrinsics: `fx *= orig_w/model_w`, `fy *= orig_h/model_h`
4. TSDF fusion → extract mesh → sample 1M points → save PLY

**`_prep_unposed`:** Uses predicted extrinsics (aligned via Umeyama), predicted intrinsics (rescaled).
**`_prep_posed`:** Uses GT extrinsics and GT intrinsics, only depth is from prediction (scaled).

### 3.5 `eval3d(scene, fuse_path)` → Metrics

Loads GT mesh (`combined_mesh.ply`), samples 1M points. Loads predicted point cloud. Computes:

| Metric | Description |
|--------|-------------|
| `acc` | Mean distance from predicted points to GT surface |
| `comp` | Mean distance from GT points to predicted surface |
| `overall` | `(acc + comp) / 2` |
| `precision` | Fraction of predicted points within 0.25m of GT |
| `recall` | Fraction of GT points within 0.25m of predicted |
| `fscore` | `2 * precision * recall / (precision + recall)` |

---

## 4. Evaluator Integration

**Source:** `src/depth_anything_3/bench/evaluator.py`

### 4.1 Frame Sampling (`_sample_frames`)

Triggered when `num_frames > max_frames` (default 100):
```python
random.seed(42)
indices = list(range(num_frames))
random.shuffle(indices)
sampled_indices = sorted(indices[:max_frames])
```

**Seed-42 sampling with max_frames=8 per scene:**

| Scene | Total | Sampled 8 Indices |
|-------|-------|-------------------|
| courtyard | 38 | [5, 10, 11, 12, 29, 32, 33, 34] |
| electro | 39 | [3, 10, 11, 23, 28, 30, 33, 35] |
| kicker | 31 | [5, 6, 10, 12, 14, 19, 24, 27] |
| pipes | 14 | [2, 5, 6, 7, 8, 11, 12, 13] |
| relief | 19 | [4, 5, 6, 9, 13, 14, 15, 17] |
| delivery_area | 40 | [3, 9, 10, 11, 24, 25, 30, 36] |
| facade | 76 | [8, 15, 30, 33, 36, 46, 47, 49] |
| office | 26 | [5, 6, 9, 10, 12, 16, 19, 25] |
| playground | 32 | [5, 6, 10, 11, 15, 22, 25, 26] |
| relief_2 | 20 | [4, 5, 9, 13, 14, 15, 18, 19] |
| terrains | 42 | [3, 9, 10, 18, 25, 26, 37, 39] |

### 4.2 Output Directory Structure

```
work_dir/
├── model_results/
│   └── eth3d/
│       └── {scene}/
│           ├── unposed/                          # For modes: pose, recon_unposed
│           │   └── exports/
│           │       ├── gt_meta.npz               # Sampled GT
│           │       ├── mini_npz/results.npz      # Predictions
│           │       └── fuse/pcd.ply              # Fused point cloud
│           └── posed/                            # For modes: recon_posed
│               └── exports/
│                   ├── gt_meta.npz
│                   ├── mini_npz/results.npz
│                   └── fuse/pcd.ply
└── metric_results/
    ├── eth3d_pose.json
    ├── eth3d_recon_unposed.json
    └── eth3d_recon_posed.json
```

### 4.3 `gt_meta.npz` Format

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `extrinsics` | `[N, 4, 4]` | float32 | GT world-to-camera |
| `intrinsics` | `[N, 3, 3]` | float32 | GT camera intrinsics |
| `image_files` | `[N]` | object (str) | Absolute image paths |

### 4.4 `results.npz` Format (mini_npz)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `depth` | `[N, H_model, W_model]` | float32 | Predicted depth maps (model resolution) |
| `extrinsics` | `[N, 3, 4]` | float32 | Predicted camera poses |
| `intrinsics` | `[N, 3, 3]` | float32 | Predicted camera intrinsics |
| `conf` | `[N, H_model, W_model]` | float32 | Confidence (optional) |

### 4.5 Pose Evaluation

For each scene:
1. Load predicted extrinsics from `results.npz`
2. Load GT extrinsics from `gt_meta.npz`
3. Align both to first camera frame
4. Compute relative pose errors for all frame pairs
5. Calculate AUC at thresholds: 3°, 5°, 15°, 30°

Output metrics: `auc03`, `auc05`, `auc15`, `auc30`

### 4.6 Reconstruction Evaluation

1. Parallel TSDF fusion (4 workers) → `pcd.ply`
2. Sequential eval: nearest-neighbor distances between pred and GT point clouds
3. Output metrics: `acc`, `comp`, `overall`, `precision`, `recall`, `fscore`

---

## 5. BenchmarkDistillDataset (Training)

**Source:** `src/depth_anything_3/distillation/benchmark_dataset.py`

### 5.1 Configuration

```python
BenchmarkDistillDataset(
    dataset_name='eth3d',
    num_views=8,                    # Teacher views
    image_size=(518, 518),          # Resize target
    student_indices=[0, 2, 4, 6],   # Even-indexed from teacher
    augment=True,
    samples_per_scene=2,            # Repeats per scene
    seed=42,
    seeds_list=None,                # Optional: explicit seed per sample
    first_frame_ref=False,          # Force first frame as reference
)
```

### 5.2 Frame Sampling

```python
# Per-sample seed:
sample_seed = seeds_list[sample_idx]   # if seeds_list provided
sample_seed = seed + idx               # otherwise (idx = scene_idx * samples_per_scene + sample_idx)

rng = random.Random(sample_seed)
indices = rng.sample(range(num_available), num_views)  # random without replacement
indices = sorted(indices)
# if first_frame_ref: indices = [indices[0]] + indices[1:]
```

### 5.3 Image Preprocessing

```python
img = cv2.imread(path)                              # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # RGB
img = cv2.resize(img, (518, 518))                    # Resize
img = img.astype(np.float32) / 255.0                 # [0, 1]
# Augmentation (if enabled):
#   brightness jitter: img *= uniform(0.8, 1.2) with 50% prob
#   horizontal flip: consistent across all views, 50% prob
images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)  # [N, 3, H, W]
images = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images)
```

### 5.4 Output Format (`__getitem__`)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `teacher_images` | `[8, 3, 518, 518]` | float32 | ImageNet-normalized |
| `student_images` | `[4, 3, 518, 518]` | float32 | Subset at indices [0,2,4,6] |
| `scene_id` | `str` | — | e.g. `"courtyard_sample0"` |
| `student_frame_indices` | `[4]` | int64 | `[0, 2, 4, 6]` |

---

## 6. benchmark_lora.py (Evaluation Script)

**Source:** `scripts/benchmark_lora.py`

### 6.1 CLI Arguments

```
--lora_path       Path to LoRA weights (default: checkpoints/distill/best_lora.pt)
--base_model      Base model name (default: depth-anything/DA3-GIANT)
--lora_rank       LoRA rank (default: 16)
--lora_alpha      LoRA alpha (default: 16.0)
--work_dir        Output directory (default: ./workspace/evaluation_lora)
--datasets        Datasets to evaluate (default: [eth3d])
--modes           Evaluation modes (default: [pose, recon_unposed])
--max_frames      Max frames per scene (default: 100)
--scenes          Specific scenes to evaluate (default: None = all)
--eval_only       Skip inference, only run evaluation
--print_only      Only print saved metrics
```

### 6.2 LoRADepthAnything3 Wrapper

```python
class LoRADepthAnything3:
    def __init__(self, base_model, lora_path, lora_rank=16, lora_alpha=16.0):
        self.student = StudentModel(model_name=base_model, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.student.load_lora_weights(lora_path)
        self.student.eval()

    def inference(self, image, **kwargs):
        return self.student.da3.inference(image=image, **kwargs)
```

### 6.3 Execution Flow

```python
evaluator = Evaluator(work_dir=..., datas=["eth3d"], modes=["pose", "recon_unposed"], max_frames=8)
api = LoRADepthAnything3(base_model=..., lora_path=...)
evaluator.infer(api)       # inference for all scenes
metrics = evaluator.eval() # pose + reconstruction evaluation
evaluator.print_metrics(metrics)
```

---

## 7. Model API for ETH3D Inference

**Source:** `src/depth_anything_3/api.py`

### 7.1 Standard Inference

```python
api = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1").to("cuda")
prediction = api.inference(
    image=["path1.jpg", "path2.jpg", ...],  # List of image paths
    extrinsics=None,                         # None = unposed mode
    intrinsics=None,
    ref_view_strategy="first",               # or "saddle_balanced"
    export_dir="./output",                   # Optional: save results
    export_format="mini_npz",                # saves to exports/mini_npz/results.npz
)
# prediction.depth:      np.ndarray [N, H, W]
# prediction.extrinsics: np.ndarray [N, 3, 4] or [N, 4, 4]
# prediction.intrinsics: np.ndarray [N, 3, 3]
# prediction.conf:       np.ndarray [N, H, W]
```

Note: `export_to_mini_npz` is decorated with `@async_call` (runs in background thread). If you need synchronous saving, save manually via `np.savez_compressed`.

### 7.2 Split Encoder/Decoder (for feature extraction experiments)

```python
# Source: src/depth_anything_3/model/da3.py

# Step 1: Encoder only
feats, aux_feats, H, W = api.model.forward_backbone_only(
    imgs,                          # [B, N, 3, H, W] tensor on device
    extrinsics=None,
    intrinsics=None,
    ref_view_strategy="first",
)
# feats: list of (features, cam_tokens) tuples per output layer
#   features:   [B, N, num_patches, feat_dim]  (feat_dim=3072 when cat_token=True)
#   cam_tokens: [B, N, feat_dim]

# Step 2: Extract subset of frames
reduced_feats = [(f[:, [0,2,4,6]], c[:, [0,2,4,6]]) for f, c in feats]

# Step 3: Decoder only
output = api.model.forward_head_only(
    reduced_feats, H, W,
    process_camera=True,
    process_sky=True,
)
# output.depth:      [B, N, 1, H, W]
# output.extrinsics: [B, N, 3, 4]
# output.intrinsics: [B, N, 3, 3]

# Step 4: Convert to Prediction
prediction = api._convert_to_prediction(output)
```

### 7.3 Existing 8→4 Extraction Model

**Source:** `src/depth_anything_3/models/da3_8to4_extraction.py`

```python
class DA3_8to4_Extraction(DepthAnything3):
    STUDENT_FRAME_INDICES = [0, 2, 4, 6]
    # forward(): 8 frames → backbone → extract [0,2,4,6] features → head → 4-frame output
```

---

## 8. Existing Experiment Framework

**Source:** `src/depth_anything_3/bench/experiments/frame_extraction_experiment.py`

| Class | Description |
|-------|-------------|
| `Exp1_8FrameBaseline` | 8 frames → full pipeline → benchmark all 8 |
| `Exp2_4FrameDirect` | Select [0,2,4,6] → full pipeline → benchmark 4 |
| `Exp3_8FrameExtract4` | 8 frames → encoder → extract [0,2,4,6] features → decoder → benchmark 4 |
| `Exp4_SpikeDimAblation` | Ablation on camera token spike dimensions |
| `Exp5_TokenReplacement` | Local token replacement analysis |

**Base class:** `src/depth_anything_3/bench/experiments/base_experiment.py`
```python
class BaseFrameExperiment(ABC):
    def __init__(self, model, ref_view_strategy="first"):
        self.model = model
        self.net = model.model  # DepthAnything3Net
    @abstractmethod
    def run_inference(self, images, extrinsics=None, intrinsics=None) -> AdictDict: ...
    def get_output_frames(self) -> List[int]: ...  # None = all frames
    def convert_output_to_numpy(self, output) -> Dict[str, np.ndarray]: ...
```

---

## 9. MV_REGISTRY System

**Source:** `src/depth_anything_3/bench/registries.py`

```python
MV_REGISTRY.register(name="eth3d")(ETH3D)
# Usage:
dataset_cls = MV_REGISTRY.get("eth3d")
dataset = dataset_cls()  # ETH3D instance
```

Available datasets: `eth3d`, `dtu`, `dtu64`, `7scenes`, `scannetpp`, `hiroom`

---

## 10. Quick Reference: Running ETH3D Benchmark

**Baseline (no LoRA):**
```bash
python -m depth_anything_3.bench.evaluator \
    model.path=depth-anything/DA3-GIANT-1.1 \
    eval.datasets=[eth3d] \
    eval.modes=[pose,recon_unposed] \
    eval.max_frames=8 \
    workspace.work_dir=./workspace/eval_eth3d
```

**With LoRA:**
```bash
python scripts/benchmark_lora.py \
    --lora_path ./checkpoints/distill_eth3d/.../epoch_1_lora.pt \
    --base_model depth-anything/DA3-GIANT-1.1 \
    --datasets eth3d \
    --modes pose recon_unposed \
    --max_frames 8 \
    --work_dir ./workspace/eval_eth3d_lora
```

**Custom experiment script (e.g., 8v vs 4v):**
```bash
python scripts/run_8v4v_extraction_experiment.py \
    --seed 42 \
    --work_dir ./workspace/extraction_exp \
    --experiments 8v_all 8v_extract_result 8v_extract_feat 4v_all
```

# Comparison Results - New Distillation Settings

## Generated Images

Successfully generated **3 comparison images** for completed datasets:

1. ✅ `eth3d_comparison.png` (157 KB)
2. ✅ `7scenes_comparison.png` (157 KB)
3. ✅ `scannetpp_comparison.png` (159 KB)

**Note:** hiroom is still running (8-view benchmark in progress)

## Directories Used

- **Baseline**: `checkpoints/baseline/20260117_143036`
- **Distilled (New Settings)**: `checkpoints/sequential/20260117_164313`
- **Output**: `comparison_results_new/`

## New Distillation Settings

These results use the updated distillation configuration:

### Dataset Changes:
- ✅ **All scenes for training** (no validation split)
- ✅ **2 samples per scene** (doubled dataset size)

### Loss Function:
- ✅ **Normalized local token loss** + **Normalized robust global token loss**
- ✅ **NO cosine similarity** components

### Comparison Tables:
- ✅ **Removed Accuracy and Completeness** metrics
- ✅ Shows only **Overall** and **F-score**

## What Each Image Contains

Each image shows **3 comprehensive tables**:

### Table 1: Pose Estimation
- **Metrics**: AUC@30°, AUC@15°, AUC@5°, AUC@3°
- **Columns**:
  - Baseline 4-view
  - Distilled 4-view
  - Δ 4-view (improvement/decrease)
  - Baseline 8-view
  - Distilled 8-view
  - Δ 8-view (improvement/decrease)
- **Higher is better** for all metrics

### Table 2: Reconstruction Unposed
- **Metrics**: Overall, F-score (Accuracy and Completeness removed)
- **Columns**: Same as above
- **Lower is better** for Overall
- **Higher is better** for F-score

### Table 3: Reconstruction Posed
- **Metrics**: Overall, F-score (Accuracy and Completeness removed)
- **Columns**: Same as above
- **Lower is better** for Overall
- **Higher is better** for F-score

## Color Coding

- 🟢 **Green cells**: Improved performance
- 🔴 **Red cells**: Decreased performance
- ⚪ **White cells**: No significant change

## Dataset Names

Each image has the **dataset name prominently displayed** in the main title:
- "ETH3D - Baseline vs Distilled Model Comparison"
- "7SCENES - Baseline vs Distilled Model Comparison"
- "SCANNETPP - Baseline vs Distilled Model Comparison"

## How to View

```bash
# View all images
ls comparison_results_new/*_comparison.png

# Open images (Linux)
xdg-open comparison_results_new/eth3d_comparison.png
xdg-open comparison_results_new/7scenes_comparison.png
xdg-open comparison_results_new/scannetpp_comparison.png

# Or open all at once
xdg-open comparison_results_new/*_comparison.png
```

## Interpreting Results

### Green Δ (Improvement)
- **Pose**: Distilled model has higher AUC (better)
- **Recon (Overall)**: Distilled model has lower value (better)
- **Recon (F-score)**: Distilled model has higher F-score (better)

### Red Δ (Decrease)
- **Pose**: Distilled model has lower AUC (worse)
- **Recon (Overall)**: Distilled model has higher value (worse)
- **Recon (F-score)**: Distilled model has lower F-score (worse)

## Comparison with Previous Results

### Previous Settings (comparison_results/):
- 80% scenes for training, 20% for validation
- 1 sample per scene
- Loss: Normalized local robust + cosine components
- Tables showed: Accuracy, Completeness, Overall, F-score

### New Settings (comparison_results_new/):
- 100% scenes for training, no validation
- 2 samples per scene (2x dataset size)
- Loss: Normalized local + normalized robust global (NO cosine)
- Tables show: Overall, F-score only

## Next Steps

1. **Wait for hiroom to complete** (8-view benchmark still running)
2. **Regenerate with hiroom** once it finishes:
   ```bash
   python scripts/generate_dataset_comparisons.py \
       --baseline_dir checkpoints/baseline/20260117_143036 \
       --distilled_dir checkpoints/sequential/20260117_164313 \
       --output_dir comparison_results_new \
       --datasets eth3d 7scenes scannetpp hiroom
   ```
3. **Compare with previous results** to see impact of new settings
4. **Analyze which metrics improved** with the new distillation approach

## Training Status

- ✅ **eth3d**: Complete (4-view + 8-view benchmarks done)
- ✅ **7scenes**: Complete (4-view + 8-view benchmarks done)
- ✅ **scannetpp**: Complete (4-view + 8-view benchmarks done)
- ⏳ **hiroom**: In progress (4-view done, 8-view running)
- ⏳ **dtu**: Not started

## Summary

✅ **3 images generated** (hiroom pending)
✅ **Dataset names clearly visible** in titles
✅ **3 tables per image** (Pose, Recon Unposed, Recon Posed)
✅ **Both 4 and 8 views** compared
✅ **Color-coded improvements/decreases**
✅ **Simplified metrics** (removed Accuracy/Completeness)
✅ **High resolution** (150 DPI)

All comparison images are ready for analysis! 📊

## Distillation Configuration

```bash
# Training command used:
python scripts/train_and_benchmark_sequential.py \
    --output_base_dir ./checkpoints/sequential \
    --local_global_norm_robust \
    --epochs 1 \
    --batch_size 2
```

**Key differences from baseline:**
- All scenes used for training (no validation split)
- 2 samples per scene for data augmentation
- Normalized local + normalized robust global loss
- No cosine similarity components

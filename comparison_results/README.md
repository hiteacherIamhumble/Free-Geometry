# Comparison Results Summary

## Generated Images

Successfully generated **4 comparison images** (DTU excluded as requested):

1. ✅ `eth3d_comparison.png` (159 KB)
2. ✅ `7scenes_comparison.png` (159 KB)
3. ✅ `scannetpp_comparison.png` (159 KB)
4. ✅ `hiroom_comparison.png` (158 KB)

## Location

All images saved to: `comparison_results/`

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
- **Metrics**: Accuracy, Completeness, Overall, F-score
- **Columns**: Same as above
- **Lower is better** for Accuracy, Completeness, Overall
- **Higher is better** for F-score

### Table 3: Reconstruction Posed
- **Metrics**: Accuracy, Completeness, Overall, F-score
- **Columns**: Same as above
- **Lower is better** for Accuracy, Completeness, Overall
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
- "HIROOM - Baseline vs Distilled Model Comparison"

## Directories Used

- **Baseline**: `checkpoints/baseline/20260117_143036`
- **Distilled**: `checkpoints/sequential/20260117_124614`
- **Output**: `comparison_results/`

## How to View

```bash
# View all images
ls comparison_results/*_comparison.png

# Open images (Linux)
xdg-open comparison_results/eth3d_comparison.png
xdg-open comparison_results/7scenes_comparison.png
xdg-open comparison_results/scannetpp_comparison.png
xdg-open comparison_results/hiroom_comparison.png

# Or open all at once
xdg-open comparison_results/*_comparison.png
```

## Interpreting Results

### Green Δ (Improvement)
- **Pose**: Distilled model has higher AUC (better)
- **Recon (Acc/Comp/Overall)**: Distilled model has lower values (better)
- **Recon (F-score)**: Distilled model has higher F-score (better)

### Red Δ (Decrease)
- **Pose**: Distilled model has lower AUC (worse)
- **Recon (Acc/Comp/Overall)**: Distilled model has higher values (worse)
- **Recon (F-score)**: Distilled model has lower F-score (worse)

## Next Steps

1. **Review the images** to see where distillation helped or hurt
2. **Identify patterns** across datasets
3. **Compare 4-view vs 8-view** performance
4. **Analyze which metrics improved most**

## Regenerate if Needed

To regenerate with different settings:

```bash
# Include DTU
python scripts/generate_dataset_comparisons.py \
    --baseline_dir checkpoints/baseline/20260117_143036 \
    --distilled_dir checkpoints/sequential/20260117_124614 \
    --output_dir comparison_results \
    --datasets eth3d 7scenes scannetpp hiroom dtu

# Only specific datasets
python scripts/generate_dataset_comparisons.py \
    --baseline_dir checkpoints/baseline/20260117_143036 \
    --distilled_dir checkpoints/sequential/20260117_124614 \
    --output_dir comparison_results \
    --datasets eth3d scannetpp
```

## Summary

✅ **4 images generated** (DTU excluded)
✅ **Dataset names clearly visible** in titles
✅ **3 tables per image** (Pose, Recon Unposed, Recon Posed)
✅ **Both 4 and 8 views** compared
✅ **Color-coded improvements/decreases**
✅ **High resolution** (150 DPI)

All comparison images are ready for analysis! 📊

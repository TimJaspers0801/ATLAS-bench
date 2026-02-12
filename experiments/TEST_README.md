# ATLAS Testing Scripts

This directory contains scripts for testing trained ATLAS models.

## Files

- **test_atlas.sh** - SLURM cluster script for batch testing all models
- **test_atlas_local.sh** - Local script for testing on your machine

## Usage

### 1. Cluster Testing (SLURM + Apptainer)

Test all models in one batch job:

```bash
cd experiments
sbatch test_atlas.sh
```

The script will:
- Test all models defined in the MODELS array
- Save individual results to `test_results/`
- Generate a summary CSV with all metrics

### 2. Local Testing

**Test a single model:**

```bash
cd experiments
chmod +x test_atlas_local.sh
./test_atlas_local.sh lh-dinov3-vitl-256-surgenet2m outputs/lh_dinov3_vitl_256_surgenet2m_atlas_seed0/best_model.pth
```

**Test all models:**

```bash
./test_atlas_local.sh
```

### 3. Direct Python Testing

Test any model directly:

```bash
python test_atlas.py \
    --model lh-dinov3-vitl-256-surgenet2m \
    --checkpoint outputs/lh_dinov3_vitl_256_surgenet2m_atlas_seed0/best_model.pth \
    --data_path atlas.zip \
    --output test_results/my_test.json
```

## Configuration

### Models Tested

The scripts test these model categories:

1. **DINOv2 Pretrained** (no checkpoint needed)
   - lh-vit-s-dinov2, lh-vit-b-dinov2, lh-vit-l-dinov2

2. **DINOv3 Pretrained** (no checkpoint needed)
   - lh-vit-b-dinov3, lh-vit-l-dinov3

3. **DINOv1/v2/v3 SurgeNet2M** (checkpoint required)
   - lh-dinov1-vitb-224-surgenet2m
   - lh-dinov2-vitb-336-surgenet2m
   - lh-dinov3-vitb-256-surgenet2m
   - lh-dinov3-vitl-256-surgenet2m

4. **SurgeNet Baselines** (checkpoint required)
   - surgenet-pvtv2-b2
   - surgenet-convnextv2-tiny
   - surgenet-caformer-s18

5. **VideoMT** (checkpoint required, batch_size=1)
   - videomt (seeds: 0, 1, 2)
   - Note: Uses online processing with temporal memory

### Expected Directory Structure

```
atlas-bench/
├── test_atlas.py
├── experiments/
│   ├── test_atlas.sh
│   └── test_atlas_local.sh
├── outputs/                    # Training outputs
│   ├── lh_dinov3_vitl_256_surgenet2m_atlas_seed0/
│   │   └── best_model.pth
│   ├── pvtv2_atlas_seed0/
│   │   └── best_model.pth
│   └── ...
└── test_results/              # Generated test results
    ├── summary.csv
    └── *.json
```

## Output Format

### Individual Results (JSON)

Each test generates a JSON file with:

```json
{
  "model": "lh-dinov3-vitl-256-surgenet2m",
  "checkpoint": "outputs/.../best_model.pth",
  "metrics": {
    "mIoU": 0.7234,
    "Dice": 0.8012,
    "AP": 0.6543,
    "AP50": 0.8123,
    "AP75": 0.7012
  }
}
```

### Summary (CSV)

Aggregated results in `test_results/summary.csv`:

```csv
experiment,model,mIoU,Dice,AP,AP50,AP75
lh_dinov3_vitl_256_surgenet2m_atlas_seed0,lh-dinov3-vitl-256-surgenet2m,0.7234,0.8012,0.6543,0.8123,0.7012
...
```

## Customization

### Add New Models

Edit the `MODELS` array in the script:

```bash
MODELS=(
    ...
    "your-model-name|best_model.pth|experiment_name|0"
)
```

Format: `"model_name|checkpoint_pattern|experiment_pattern|seed|batch_size"`
- Use `None` for checkpoint_pattern if pretrained model
- batch_size is optional and defaults to 32 (use 1 for VideoMT)

### Change Batch Size

Edit the `BATCH_SIZE` variable (default: 32)

### Test Multiple Seeds

Add multiple entries with different seeds:

```bash
MODELS=(
    "lh-dinov3-vitl-256-surgenet2m|best_model.pth|lh_dinov3_vitl_256_surgenet2m_atlas|0|32"
    "lh-dinov3-vitl-256-surgenet2m|best_model.pth|lh_dinov3_vitl_256_surgenet2m_atlas|1|32"
    "lh-dinov3-vitl-256-surgenet2m|best_model.pth|lh_dinov3_vitl_256_surgenet2m_atlas|2|32"
)
```

For VideoMT (requires batch_size=1):

```bash
MODELS=(
    "videomt|best_model.pth|videomt_atlas|0|1"
    "videomt|best_model.pth|videomt_atlas|1|1"
    "videomt|best_model.pth|videomt_atlas|2|1"
)
```

## Troubleshooting

**Checkpoint not found:**
- Verify the experiment completed training
- Check the output directory path
- Confirm checkpoint file name (usually `best_model.pth`)

**CUDA out of memory:**
- Reduce `BATCH_SIZE` (try 16 or 8)
- Test large models separately

**Missing data file:**
- Update `DATA_ZIP` path in the script
- Ensure atlas.zip is accessible

## VideoMT Testing

VideoMT is a video-based model with online processing that maintains temporal state across frames.
**It requires batch_size=1** to properly track temporal memory.

VideoMT is already included in the test scripts with all 3 seeds:
- videomt_atlas_seed0
- videomt_atlas_seed1  
- videomt_atlas_seed2

To test VideoMT manually:

```bash
python test_atlas.py \
    --model videomt \
    --checkpoint outputs/videomt_atlas_seed0/best_model.pth \
    --data_path atlas.zip \
    --batch_size 1
```

VideoMT processes frames sequentially with temporal memory.

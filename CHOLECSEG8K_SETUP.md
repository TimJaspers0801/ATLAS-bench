# CholecSeg8k Training Setup - Configuration Summary

## Overview
This document outlines the setup for training deep learning models on the CholecSeg8k dataset using the atlas-bench framework.

## Dataset Structure

The CholecSeg8k dataset is expected to be in a zip file with the following structure:

```
cholecseg8k.zip
├── cholecseg8k/
    ├── Train/
    │   ├── frames/          (PNG images: video{XX}_nb_frame_{framenum}_endo.png)
    │   └── masks/           (PNG masks)
    ├── Validation/
    │   ├── frames/
    │   └── masks/
    └── Test/
        ├── frames/
        └── masks/
```

## Classes

The CholecSeg8k dataset contains 9 classes:

| Class ID | Name |
|----------|------|
| 0 | Black Background |
| 1 | Abdominal Wall |
| 2 | Liver |
| 3 | Gastrointestinal Tract |
| 4 | Fat |
| 5 | Grasper |
| 6 | Connective Tissue |
| 7 | L-hook Electrocautery |
| 8 | Gallbladder |

## Created/Modified Files

### 1. New Dataset Class
- **File**: `datasets/cholecseg8k.py`
- **Purpose**: Implements the CholecSeg8kDataset class compatible with the atlas-bench framework
- **Features**:
  - Reads from zip files with the specified structure
  - Treats all frames as part of a single clip
  - Supports train/val/test splits
  - Applies automatic class mapping
  - Compatible with frame percentage sampling and first-frame-only modes

### 2. Modified Training Script
- **File**: `train_frame_level_cholecseg8k.py`
- **Changes**:
  - Updated imports to use CholecSeg8kDataset instead of AtlasDataset
  - Changed WandB project to "CHOLECSEG8K-BENCH"
  - Set default num_classes to 9
  - All other training logic remains the same

### 3. Experiment Scripts
Created the following experiment scripts in `experiments/` directory:

1. `run_surgenet_exp_cholecseg8k.sh` - SurgeNet-based models (pvtv2, convnextv2, caformer)
2. `run_endofm_exp_cholecseg8k.sh` - EndoFM model
3. `run_endovit_exp_cholecseg8k.sh` - EndoViT model
4. `run_gastronet5m_exp_cholecseg8k.sh` - GastroNet5M model
5. `run_vit_lh_exp_cholecseg8k.sh` - Vision Transformer models (DINOv2 and DINOv3)
6. `run_dinov1_vitb_224_surgenet2m_exp_cholecseg8k.sh` - DINOv1 ViT-B 224
7. `run_dinov2_vitb_336_surgenet2m_exp_cholecseg8k.sh` - DINOv2 ViT-B 336
8. `run_dinov3_vitl_256_surgenet2m_exp_cholecseg8k.sh` - DINOv3 ViT-L 256

All experiment scripts:
- Reference the CholecSeg8k dataset zip file (expected at `/gpfs/work5/0/tesr0602/Tim/datasets/cholecseg8k/cholecseg8k.zip`)
- Use 9 classes for training
- Run 3 seeds (0, 1, 2) for reproducibility
- Log results to WandB under "CHOLECSEG8K-BENCH" project
- Save best models and final evaluation metrics

## Usage Instructions

### Local Testing
```bash
python train_frame_level_cholecseg8k.py \
  --data_path /path/to/cholecseg8k.zip \
  --model pvtv2 \
  --num_classes 9 \
  --epochs 10 \
  --batch_size 32 \
  --img_size 256 \
  --experiment_name test_experiment
```

### HPC Submission
```bash
sbatch experiments/run_surgenet_exp_cholecseg8k.sh
sbatch experiments/run_vit_lh_exp_cholecseg8k.sh
# ... and so on
```

## Key Configuration Parameters

- **Number of classes**: 9
- **Default image size**: 256 (automatically adjusted for some models)
- **WandB project**: CHOLECSEG8K-BENCH
- **Loss function**: CrossEntropyLoss with ignore_index=255 (for background)
- **Background masking**: Enabled (class 0 mapped to 255 for loss computation)

## Notes

1. **Normalization**: The CholecSeg8kDataset uses standard ImageNet normalization stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). These can be updated based on actual dataset statistics.

2. **Data Path**: Update the DATA_ZIP path in experiment scripts to match your HPC setup.

3. **Frame Handling**: All frames in the dataset are treated as part of a single clip for compatibility with the evaluation framework. This means metrics are computed across all frames.

4. **Reproducibility**: All experiments use fixed seeds for reproducibility. Run with seeds 0, 1, 2 for statistical significance.

## Verification Checklist

- ✅ CholecSeg8kDataset class created in `datasets/cholecseg8k.py`
- ✅ train_frame_level_cholecseg8k.py modified and cleaned of ATLAS references
- ✅ All experiment .sh files created with _cholecseg8k extension
- ✅ Dataset structure documentation
- ✅ Default parameters set correctly (num_classes=9, data path)

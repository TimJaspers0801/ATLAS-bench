<div align="center">

# ATLAS-Bench

**Benchmark for Surgical Anatomy Segmentation on ATLAS-120k**

[![Dataset](https://img.shields.io/badge/🤗_Dataset-HuggingFace-ffcc00?style=flat)](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

</div>

---

ATLAS-Bench evaluates semantic segmentation models on the [ATLAS-120k](https://huggingface.co/TimJaspersTue/datasets/atlas-120k) surgical anatomy dataset — 120k annotated frames from 100 surgical videos across 14 procedures and 42 anatomical classes.

Scope note: in this repository, SOTA baseline and foundation models are trained and evaluated. ATLAS and EOMT models are trained in [ATLAS-model](https://github.com/rlpddejong/ATLAS-model) and are evaluated here.

This repository is part of the broader ATLAS project:

| Repository | Description |
|---|---|
| [ATLAS-120k](https://github.com/TimJaspers0801/ATLAS) | Dataset download and processing scripts |
| [ATLAS-interactive](https://github.com/rlpddejong/ATLAS-Interactive) | Annotation platform and interactive segmentation tools |
| [ATLAS-model](https://github.com/rlpddejong/ATLAS-model) | ATLAS model implementation and training code |
| **This repo** | Training/evaluation benchmark for SOTA baselines; evaluation for ATLAS and EOMT models |
| [SurgeNetDINO](https://github.com/rlpddejong/SurgeNetDINO) | Pretrained DINOv1/v2/v3 surgical foundation backbones |
| [SurgeNet](https://github.com/TimJaspers0801/SurgeNet) | SurgeNet pretraining dataset |

## Models

The benchmark covers three categories of model:

**Image-based models with linear segmentation heads**

| Model | Key | Input size |
|---|---|---|
| DINOv1 ViT-S/B (ImageNet) | `lh-vit-s/b-dinov1` | 224 |
| DINOv2 ViT-S/B/L (ImageNet) | `lh-vit-s/b/l-dinov2` | 518 |
| DINOv3 ViT-S/B/L (ImageNet) | `lh-vit-s/b/l-dinov3` | 256 |
| DINOv1 ViT-B (SurgeNet2M) | `lh-dinov1-vitb-224-surgenet2m` | 224 |
| DINOv2 ViT-B (SurgeNet2M) | `lh-dinov2-vitb-336-surgenet2m` | 336 |
| DINOv3 ViT-B/L (SurgeNet2M) | `lh-dinov3-vitb/l-256-surgenet2m` | 256 |

**CNN / hybrid baselines (SurgeNet decoder)**

| Model | Key | Input size |
|---|---|---|
| PVTv2-B2 | `surgenet-pvtv2-b2` | 256 |
| ConvNeXtv2-Tiny | `surgenet-convnextv2-tiny` | 256 |
| CAFormer-S18 | `surgenet-caformer-s18` | 256 |

**Surgical foundation models**

| Model | Key | Input size |
|---|---|---|
| EndoFM | `endofm` | 256 |
| EndoViT | `endovit` | 256 |
| GastroNet5M | `gastronet5m` | 336 |
| SAM2-UNet | `sam2unet` | 256 |
| SAM3-UNet | `sam3unet` | 336 |

**ATLAS models**

| Model | Key | Input size |
|---|---|---|
| ATLAS-ViT-B DINOv1 | `atlas_vitb_dinov1` | 224 |
| ATLAS-ViT-B DINOv2 | `atlas_vitb_dinov2` | 336 |
| ATLAS-ViT-S DINOv3 | `atlas_vits_dinov3` | 256 |
| ATLAS-ViT-B-DINOv3 | `atlas_vitb_dinov3` | 256 |
| ATLAS-ViT-L-DINOv3 | `atlas_vitl_dinov3` | 256 |

## Setup

```bash
git clone https://github.com/TimJaspers0801/ATLAS-bench-public
cd ATLAS-bench-public
pip install torch torchvision timm transformers wandb tqdm numpy opencv-python pycocotools
```

SurgeNet2M backbone weights go in `weights/`:

```
weights/
  DINOv1-vitb-224-SurgNet2M.pth
  DINOv2-vitb-336-surgenet2M.pth
  DINOv3-vitb-256-surgenet2M.pth
  DINOv3-vitl-256-surgenet2M.pth
```

The ATLAS-120k dataset (annotations + extracted frames) should be packaged as a zip and placed at the path you pass to `--data_path`. See the [ATLAS-120k repository]() for download and processing instructions.

## Training

This training script is intended for the SOTA baseline and foundation models in this repository. ATLAS/EOMT training is handled in [ATLAS-model](https://github.com/rlpddejong/ATLAS-model).

```bash
python train_atlas120k.py \
    --model lh-dinov3-vitl-256-surgenet2m \
    --data_path atlas120k.zip \
    --num_classes 30 \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --img_size 256 \
    --experiment_name my_run \
    --output_dir outputs/
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | — | Model key (see tables above) |
| `--data_path` | — | Path to ATLAS-120k zip |
| `--num_classes` | 46 | Number of foreground classes (use 30 for the training taxonomy) |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--img_size` | 256 | Input resolution |
| `--experiment_name` | — | Run name (used for output directory and W&B) |
| `--seed` | 42 | Random seed |
| `--frame_percentage` | 100 | Fraction of training frames to use (1–100) |

Checkpoints and logs are written to `outputs/<experiment_name>/`. [Weights & Biases](https://wandb.ai) is used for experiment tracking.

## Evaluation

This repository supports evaluation for all listed models, including ATLAS and EOMT checkpoints trained in [ATLAS-model](https://github.com/rlpddejong/ATLAS-model).

```bash
python test_atlas120k.py \
    --model lh-dinov3-vitl-256-surgenet2m \
    --checkpoint outputs/my_run/best_model.pth \
    --data_path atlas120k.zip \
    --num_classes 30
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | — | Model key |
| `--checkpoint` | None | Path to trained checkpoint (omit for zero-shot) |
| `--data_path` | — | Path to ATLAS-120k zip |
| `--num_classes` | 47 | Number of foreground classes |
| `--batch_size` | 32 | Evaluation batch size |
| `--img_size` | 256 | Input resolution (auto-inferred from model name if omitted) |
| `--output` | None | Path to save results as JSON |
| `--visualize_samples` | 25 | Number of random frames to visualise (0 to disable) |

Results are printed to stdout and optionally saved as JSON:

```json
{
  "model": "lh-dinov3-vitl-256-surgenet2m",
  "metrics": {
    "mIoU": 0.42,
    "Dice": 0.55,
    "AP": 0.38,
    "AP50": 0.61,
    "AP75": 0.40,
    "mVC_12": 0.81,
    "mVC_24": 0.78
  }
}
```

## Efficiency Benchmarking

Measure FPS, parameter count and GFLOPs:

```bash
python benchmark_models.py \
    --model lh-dinov3-vitl-256-surgenet2m \
    --checkpoint outputs/my_run/best_model.pth \
    --img_size 256 \
    --num_classes 30
```

## Repository Structure

```
train_atlas120k.py          — frame-level training script
test_atlas120k.py           — evaluation script
benchmark_models.py         — FPS / params / GFLOPs benchmarking
utils.py                    — colour palette, checkpoint utilities

datasets/
  atlas.py                  — AtlasDataset (zip-based loader)
  class_mapping.py          — class ID remapping utilities

models/
  load_models.py            — model factory (all supported models)
  atlas/                    — ATLAS temporal model
  eomt/                     — EOMT and shared ViT backbone
  surgenet/                 — PVTv2, ConvNeXtv2, CAFormer
  decoders/                 — linear and ViT-based segmentation heads
  GastroNet5M/              — GastroNet5M ViT
  EndoFM/                   — EndoFM TransUNet
  EndoViT/                  — EndoViT fine-tuning wrapper
  SAM2UNet/                 — SAM2-UNet
  SAM3-UNet/                — SAM3-UNet

evaluation/
  metrics.py                — IoU, Dice, AP (COCO), mVC
  dataset_evaluation.py     — evaluate_model() and evaluate_atlas_temporal()
  visual_logging.py         — mask overlay visualisation
```

## Citation

If you use ATLAS-120k in your research, please cite:

```bibtex
@misc{dejong2026atlas,
      title={Surgical Anatomy Recognition with Context Learning using Foundation Representations}, 
      author={Ronald L. P. D. de Jong and Tim J. M. Jaspers and Raf A. H. Vervoort and Aron F. H. A. Bakker and Yiping Li and Jip L. Tolenaar and Jelle P. Ruurda and Willem M. Brinkman and Josien P. W. Pluim and Marcel Breeuwer and Daan de Geus and Fons van der Sommen},
      year={2026},
      eprint={2606.22124},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.22124}, 
}
```

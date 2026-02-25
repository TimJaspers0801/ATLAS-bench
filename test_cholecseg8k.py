"""
Test script for benchmarking ATLAS-trained models on the CholecSeg8K dataset.

This script evaluates ATLAS models on CholecSeg8K by:
1. Combining all three splits (train/val/test) as one unified test set
2. Mapping ATLAS's 30 output classes to CholecSeg8K's 9 classes
3. Computing metrics on the mapped output

Supports:
- Image-based models (DINOv2, DINOv3, SurgeNet variants, EOMT)
- Video-based models (VideoMT with online processing)
- Frame-by-frame evaluation on combined splits

Usage:
    python test_cholecseg8k.py --model lh-dinov3-vitl-256-surgenet2m \\
                               --checkpoint weights/DINOv3-vitl-256-surgenet2m.pth \\
                               --data_path cholecseg8k.zip
    
    python test_cholecseg8k.py --model atlas_vitl_dinov3_tracking \\
                               --checkpoint checkpoints/atlas_vitl_dinov3_tracking.pth \\
                               --data_path cholecseg8k.zip
"""

import argparse
import random
import torch
import os
import json
from tqdm import tqdm
from collections import defaultdict
import cv2

from torch.utils.data import DataLoader, ConcatDataset
from datasets.cholecseg8k import CholecSeg8kDataset
import torchvision.transforms.v2 as T
from torch import nn
import torch.nn.functional as F

from utils import load_checkpoint, color_palette
from models.load_models import (
    load_lh_vit_s_dinov1, load_lh_vit_b_dinov1, 
    load_lh_vit_s_dinov2, load_lh_vit_b_dinov2, load_lh_vit_l_dinov2,
    load_lh_vit_s_dinov3, load_lh_vit_b_dinov3, load_lh_vit_l_dinov3,
    load_surgenet_caformer_s18, load_surgenet_convnextv2_tiny, load_surgenet_pvtv2_b2,
    load_endofm, load_endovit, load_lh_gastronet5m,
    load_lh_dinov1_vitb_224_surgenet2m, load_lh_dinov2_vitb_336_surgenet2m,
    load_lh_dinov3_vitb_256_surgenet2m, load_lh_dinov3_vitl_256_surgenet2m
)
from evaluation.dataset_evaluation import evaluate_model
from evaluation.visual_logging import apply_mask_overlay, denormalize
import numpy as np


# ===========================
# Class Mapping: ATLAS 30 classes -> CholecSeg8K 9 classes
# ===========================
# This mapping defines how ATLAS's 30 output classes are mapped to CholecSeg8K's 9 classes.
# EVALUATION FOCUS: Only 7 classes are evaluated (excluding Connective Tissue class 6)
#
# ATLAS classes (30):
#   0: Background
#   1: Tools/camera
#   2: Vein
#   3: Artery
#   4: Nerve
#   5: Small intestine
#   6: Colon/rectum
#   7: Abdominal wall
#   8: Diaphragm
#   9: Fat
#   10: Liver
#   11: Bile/lymph Duct
#   12: Gallbladder
#   13: Hepatic ligament
#   14: Cystic plate
#   15: Stomach
#   16: Spleen
#   17-29: Other (reproductive/urinary structures - not relevant for cholecystectomy)
#
# CholecSeg8K classes (9):
#   0: Black Background
#   1: Abdominal Wall
#   2: Liver
#   3: Gastrointestinal Tract
#   4: Fat
#   5: Grasper (now includes L-hook Electrocautery)
#   6: Connective Tissue (EXCLUDED FROM EVALUATION)
#   7: L-hook Electrocautery -> mapped to class 5 (Grasper)
#   8: Gallbladder
#
# Mapping Strategy:
#   - Background structures -> 0 (Background)
#   - Tools (ATLAS 1) -> 5 (Grasper) - surgical tools
#   - Vessels, Ducts, Nerves, Ligaments -> 0 (Background) - excluded from evaluation
#   - GI tract structures -> 3 (Gastrointestinal Tract)
#   - Body structures -> appropriate direct mapping
#   - Non-cholecystectomy organs -> 0 (Background)

ATLAS_TO_CHOLECSEG8K_MAPPING = {
    0:  0,   # Background -> Background
    1:  5,   # Tools/camera -> Grasper (surgical tools)
    2:  0,   # Vein -> Background (excluded - was Connective Tissue)
    3:  0,   # Artery -> Background (excluded - was Connective Tissue)
    4:  0,   # Nerve -> Background (excluded - was Connective Tissue)
    5:  3,   # Small intestine -> Gastrointestinal Tract
    6:  3,   # Colon/rectum -> Gastrointestinal Tract
    7:  1,   # Abdominal wall -> Abdominal Wall (direct mapping)
    8:  0,   # Diaphragm -> Background (not relevant for cholecystectomy)
    9:  4,   # Fat -> Fat (direct mapping)
    10: 2,   # Liver -> Liver (direct mapping)
    11: 0,   # Bile/lymph Duct -> Background (excluded - was Connective Tissue)
    12: 8,   # Gallbladder -> Gallbladder (direct mapping - primary organ)
    13: 0,   # Hepatic ligament -> Background (excluded - was Connective Tissue)
    14: 0,   # Cystic plate -> Background (excluded - was Connective Tissue)
    15: 3,   # Stomach -> Gastrointestinal Tract
    16: 0,   # Spleen -> Background (not directly involved in cholecystectomy)
    17: 0,   # Uterus -> Background (not relevant for cholecystectomy)
    18: 0,   # Ovary -> Background
    19: 0,   # Oviduct -> Background
    20: 0,   # Prostate -> Background
    21: 0,   # Urethra -> Background
    22: 0,   # Ligated plexus -> Background
    23: 0,   # Seminal vesicles -> Background
    24: 0,   # Non anatomical -> Background
    25: 0,   # Bladder -> Background (not directly relevant)
    26: 0,   # Lung -> Background (not relevant)
    27: 0,   # Airway -> Background
    28: 0,   # Esophagus -> Background
    29: 0,   # Pericardium -> Background
}

CLASS_MAPPING_EXPLANATION = """
ATLAS 30 Classes -> CholecSeg8K Classes Mapping for Evaluation
=========================================================

EVALUATION CLASSES (7 classes only):

  ✓ Class 0 (Background) - Direct mapping from ATLAS
  ✓ Class 1 (Abdominal Wall) - Direct mapping from ATLAS class 7
  ✓ Class 2 (Liver) - Direct mapping from ATLAS class 10
  ✓ Class 3 (Gastrointestinal Tract) - Unified from ATLAS classes 5,6,15 (intestine/colon/stomach)
  ✓ Class 4 (Fat) - Direct mapping from ATLAS class 9
  ✓ Class 5 (Grasper/L-hook) - ATLAS class 1 (Tools) + CholecSeg8K class 7 (L-hook)
    Both surgical tools consolidated into single class
  ✓ Class 8 (Gallbladder) - Direct mapping from ATLAS class 12

EXCLUDED FROM EVALUATION:

  ✗ Class 6 (Connective Tissue) - This class is NOT evaluated
    ATLAS structures like vessels (2,3), nerves (4), ducts (11), ligaments (13,14)
    are mapped to Background (0) instead of being grouped into Connective Tissue
    
RATIONALE FOR EXCLUSIONS:

  - Connective Tissue (class 6): Fine anatomical structures without clear clinical relevance
    in cholecystectomy segmentation; focus is on surgical anatomy and tools
  - L-hook Electrocautery remapping: Consolidated with Grasper (class 5) as both are
    surgical tools used during the procedure
  - Non-cholecystectomy organs: Structures like spleen, diaphragm, reproductive organs
    are mapped to background as they're not relevant to cholecystectomy

MAPPING SUMMARY FOR EVALUATION:
  - Evaluated classes: 0, 1, 2, 3, 4, 5, 8 (7 classes total)
  - Excluded classes: 6
  - Unmapped: Class 7 (L-hook) in CholecSeg8K is remapped to class 5 (Grasper)

Note: This focused mapping and evaluation ensures fair assessment of ATLAS models
on CholecSeg8K while concentrating on clinically relevant anatomical structures.
"""

print(CLASS_MAPPING_EXPLANATION)


# Model registry
MODEL_REGISTRY = {
    # DINOv1 variants
    "lh-vit-s-dinov1": load_lh_vit_s_dinov1,
    "lh-vit-b-dinov1": load_lh_vit_b_dinov1,

    # DINOv2 variants
    "lh-vit-s-dinov2": load_lh_vit_s_dinov2,
    "lh-vit-b-dinov2": load_lh_vit_b_dinov2,
    "lh-vit-l-dinov2": load_lh_vit_l_dinov2,
    
    # DINOv3 variants
    "lh-vit-s-dinov3": load_lh_vit_s_dinov3,
    "lh-vit-b-dinov3": load_lh_vit_b_dinov3,
    "lh-vit-l-dinov3": load_lh_vit_l_dinov3,
    
    # DINOv1 SurgeNet2M
    "lh-dinov1-vitb-224-surgenet2m": load_lh_dinov1_vitb_224_surgenet2m,
    
    # DINOv2 SurgeNet2M
    "lh-dinov2-vitb-336-surgenet2m": load_lh_dinov2_vitb_336_surgenet2m,
    
    # DINOv3 SurgeNet2M
    "lh-dinov3-vitb-256-surgenet2m": load_lh_dinov3_vitb_256_surgenet2m,
    "lh-dinov3-vitl-256-surgenet2m": load_lh_dinov3_vitl_256_surgenet2m,
    
    # SurgeNet baselines
    "surgenet-caformer-s18": load_surgenet_caformer_s18,
    "surgenet-convnextv2-tiny": load_surgenet_convnextv2_tiny,
    "surgenet-pvtv2-b2": load_surgenet_pvtv2_b2,
    
    # Other vision models
    "endofm": load_endofm,
    "endovit": load_endovit,
    "gastronet5m": load_lh_gastronet5m,
}

# Image size mapping
IMAGE_SIZE_MAP = {
    "224": 224,
    "256": 256,
    "336": 336,
    "518": 518,
    "dinov2": 224,
    "dinov3": 256,
    "gastronet": 336,
    'eomt': 256,
}


def get_image_size(model_name: str) -> int:
    """Infer image size from model name."""
    # Special case: eomt_dinov2 models use 518x518
    if "eomt" in model_name and "dinov2" in model_name:
        return 518
    
    # Check for 518 in model name (explicit size specification)
    if "518" in model_name:
        return 518
    
    for key, size in IMAGE_SIZE_MAP.items():
        if key in model_name:
            return size
    return 256  # default


def map_atlas_to_cholecseg8k(pred_mask):
    """
    Map ATLAS predictions (30 classes) to CholecSeg8K classes (9 classes).
    
    Args:
        pred_mask: Tensor of shape (B, H, W) or (H, W) with class indices
    
    Returns:
        Mapped tensor with CholecSeg8K class indices
    """
    device = pred_mask.device
    mapped = torch.zeros_like(pred_mask)
    
    for atlas_class, cholecseg_class in ATLAS_TO_CHOLECSEG8K_MAPPING.items():
        mapped[pred_mask == atlas_class] = cholecseg_class
    
    return mapped


def load_model(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    if model_name.startswith("atlas"):
        return load_atlas(model_name, checkpoint_path, num_classes, device)
    elif model_name.startswith("eomt"):
        return load_eomt(model_name, checkpoint_path, num_classes, device)
    elif model_name in MODEL_REGISTRY:
        # Load from registry
        loader = MODEL_REGISTRY[model_name]
        model = loader(num_classes)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path)
        
        return model.to(device)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available models:\n"
            f"  Image-based: {', '.join(MODEL_REGISTRY.keys())}\n"
            f"  Video-based: videomt\n"
            f"  EOMT variants: eomt_vits_dinov2, eomt_vitb_dinov2, eomt_vitl_dinov2, "
            f"eomt_vits_dinov3, eomt_vitb_dinov3, eomt_vitl_dinov3\n"
            f"  ATLAS (temporal): atlas_vitl_dinov3"
        )


def load_eomt(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    """Load EOMT model variants."""
    from models.eomt.eomt import (
        eomt_vits_dinov2, eomt_vitb_dinov2, eomt_vitl_dinov2,
        eomt_vits_dinov3, eomt_vitb_dinov3, eomt_vitl_dinov3
    )
    
    # Map model names to loader functions
    eomt_loaders = {
        "eomt_vits_dinov2": eomt_vits_dinov2,
        "eomt_vitb_dinov2": eomt_vitb_dinov2,
        "eomt_vitl_dinov2": eomt_vitl_dinov2,
        "eomt_vits_dinov3": eomt_vits_dinov3,
        "eomt_vitb_dinov3": eomt_vitb_dinov3,
        "eomt_vitl_dinov3": eomt_vitl_dinov3,
    }
    
    if model_name not in eomt_loaders:
        raise ValueError(
            f"Unknown EOMT variant: {model_name}. Available: {', '.join(eomt_loaders.keys())}"
        )
    
    print(f"Loading EOMT model: {model_name}")
    model = eomt_loaders[model_name](num_classes=num_classes)
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading EOMT checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # First pass: understand what format the checkpoint is in
        has_timm_keys = any("patch_embed" in k or "blocks" in k for k in state_dict.keys() if k.startswith("network.encoder.backbone"))
        has_hf_keys = any("embeddings" in k or "layer" in k for k in state_dict.keys() if k.startswith("network.encoder.backbone"))
        
        # Transform keys
        new_state_dict = {}
        for key, value in state_dict.items():
            # Skip criterion keys (training-only)
            if key.startswith("criterion."):
                continue
            
            # Strip "network." prefix from Lightning wrapper
            new_key = key[8:] if key.startswith("network.") else key  # len("network.") = 8
            
            # If checkpoint has TIMM format but we need HF format, convert
            if has_timm_keys and not has_hf_keys:
                new_key = new_key.replace("encoder.backbone.patch_embed.", "encoder.backbone.embeddings.")
                new_key = new_key.replace("encoder.backbone.blocks.", "encoder.backbone.layer.")
            
            new_state_dict[new_key] = value
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
        
    return model.to(device)


def load_atlas(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    """Load ATLAS model with temporal capabilities."""
    from models.atlas.atlas import atlas_vitl_dinov3, atlas_vitb_dinov3, atlas_vits_dinov3, atlas_vitl_dinov3_tracking
    
    # Map model names to loader functions
    atlas_loaders = {
        "atlas_vitl_dinov3": atlas_vitl_dinov3,
        "atlas_vitl_dinov3_tracking": atlas_vitl_dinov3_tracking,
        "atlas_vitb_dinov3": atlas_vitb_dinov3,
        "atlas_vits_dinov3": atlas_vits_dinov3,
    }
    
    if model_name not in atlas_loaders:
        raise ValueError(
            f"Unknown ATLAS variant: {model_name}. Available: {', '.join(atlas_loaders.keys())}"
        )
    
    print(f"Loading ATLAS model: {model_name}")
    model = atlas_loaders[model_name](num_classes=num_classes)
    
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"Loading ATLAS checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except RuntimeError as e:
            if "PytorchStreamReader failed" in str(e) or "zip archive" in str(e):
                file_size = os.path.getsize(checkpoint_path)
                raise RuntimeError(
                    f"Checkpoint file is corrupted or invalid: {checkpoint_path}\n"
                    f"File size: {file_size} bytes\n"
                    f"This typically means:\n"
                    f"  - The file download was incomplete\n"
                    f"  - The file was corrupted during transfer\n"
                    f"  - The file is not a valid PyTorch checkpoint\n"
                    f"Original error: {e}"
                ) from e
            raise
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Strip "network." prefix from Lightning wrapper if present
        new_state_dict = {}
        for key, value in state_dict.items():
            # Skip criterion keys (training-only)
            if key.startswith("criterion."):
                continue
            
            # Strip "network." prefix if present
            new_key = key[8:] if key.startswith("network.") else key  # len("network.") = 8
            new_state_dict[new_key] = value
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
        
    return model.to(device)


def evaluate_cholecseg8k_temporal(model, test_loader, device, num_classes, use_query_propagation=True):
    """Evaluate ATLAS temporal model using class mapping."""
    from evaluation.dataset_evaluation import evaluate_atlas_temporal as _evaluate_atlas_temporal
    
    # Get raw ATLAS metrics (on 30 classes)
    raw_metrics = _evaluate_atlas_temporal(model, test_loader, device, num_classes, use_query_propagation)
    
    # The metrics need remapping - this is handled in the main evaluation loop below
    return raw_metrics


def save_random_visualizations(
    model,
    dataloader,
    device,
    output_dir,
    num_samples,
    img_size,
    is_videomt,
    is_atlas,
    seed,
    mask_background=False,
    apply_class_mapping=True,
):
    if num_samples <= 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    mean = torch.tensor(dataloader.dataset.datasets[0].mean).view(3, 1, 1) if hasattr(dataloader.dataset, 'datasets') else torch.tensor(dataloader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(dataloader.dataset.datasets[0].std).view(3, 1, 1) if hasattr(dataloader.dataset, 'datasets') else torch.tensor(dataloader.dataset.std).view(3, 1, 1)

    with torch.no_grad():
        current_video = None
        current_clip = None
        prev_query_embed = None
        sample_count = 0
        
        for batch in tqdm(dataloader, desc="Collecting visualizations"):
            if sample_count >= num_samples:
                break
                
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            
            # Apply class mapping to ground truth
            if apply_class_mapping:
                gt_masks = map_atlas_to_cholecseg8k(gt_masks)
            
            # Get clip info
            procedure = batch["procedure"][0] if isinstance(batch["procedure"], list) else batch["procedure"]
            video = batch["video"][0] if isinstance(batch["video"], list) else batch["video"]
            clip_id = batch["clip"][0] if isinstance(batch["clip"], list) else batch["clip"]
            clip_key = f"{procedure}/{video}/{clip_id}"
            
            # Reset queries when entering a new clip (for ATLAS)
            if is_atlas and current_clip != clip_key:
                current_clip = clip_key
                prev_query_embed = None

            if is_videomt:
                batch_video = f"{batch['procedure'][0]}/{batch['video'][0]}"
                if batch_video != current_video:
                    model.reset_memory()
                    current_video = batch_video

                images_denorm = images.cpu() * std + mean
                images_denorm = images_denorm.clamp(0.0, 1.0)
                images_denorm = images_denorm.to(device)

                outputs_list = []
                for i in range(images.shape[0]):
                    frame = images_denorm[i:i+1]
                    outputs_list.append(model.forward_frame(frame))

                pred_logits = torch.cat([o['pred_logits'] for o in outputs_list], dim=0)
                pred_masks = torch.cat([o['pred_masks'] for o in outputs_list], dim=0)

                mask_cls = F.softmax(pred_logits, dim=-1)[:, :, :-1]
                pred_masks_prob = torch.sigmoid(pred_masks)
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, pred_masks_prob)
                preds = semseg.argmax(dim=1)
            elif is_atlas:
                # ATLAS model with query propagation
                outputs = model(
                    images,
                    prev_query_embed=prev_query_embed,
                    return_query_embedding=True,
                )
                
                # Unpack outputs
                mask_logits_per_block, class_logits_per_block, procedure_logits_per_block, query_embed = outputs
                prev_query_embed = query_embed  # Store for next frame
                
                # Get predictions from final block
                mask_logits = mask_logits_per_block[-1]  # (B, num_q, H, W)
                class_logits = class_logits_per_block[-1]  # (B, num_q, num_classes+1)
                
                # Convert query-level predictions to per-pixel logits
                per_pixel_logits = torch.einsum(
                    "bqhw, bqc -> bchw",
                    mask_logits.sigmoid(),
                    class_logits.softmax(dim=-1)[..., :-1]
                )
                
                # Resize logits to match GT size
                if per_pixel_logits.shape[-2:] != gt_masks.shape[-2:]:
                    per_pixel_logits = torch.nn.functional.interpolate(
                        per_pixel_logits,
                        size=gt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                preds = torch.argmax(per_pixel_logits, dim=1)
            else:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            # Apply class mapping to predictions
            if apply_class_mapping:
                preds = map_atlas_to_cholecseg8k(preds)

            for i in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                    
                img = images[i].cpu()
                gt_mask = gt_masks[i].cpu()
                pred_mask = preds[i].cpu()

                img_np = denormalize(img, mean, std)
                gt_np = gt_mask.squeeze().numpy().astype(np.int32)
                pred_np = pred_mask.squeeze().numpy().astype(np.int32)

                if pred_np.shape != gt_np.shape:
                    pred_np = cv2.resize(
                        pred_np.astype(np.uint8),
                        (gt_np.shape[1], gt_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.int32)
                
                # Mask background predictions if requested
                if mask_background:
                    pred_np[gt_np == 0] = 0

                gt_overlay = apply_mask_overlay(img_np, gt_np, color_palette)
                pred_overlay = apply_mask_overlay(img_np, pred_np, color_palette)

                row = np.concatenate([img_np, gt_overlay, pred_overlay], axis=1)
                if row.shape[0] != img_size:
                    row = cv2.resize(row, (img_size * 3, img_size), interpolation=cv2.INTER_AREA)

                out_path = os.path.join(output_dir, f"sample_{sample_count:03d}.png")
                cv2.imwrite(out_path, row[:, :, ::-1])
                sample_count += 1


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Infer image size
    img_size = args.img_size if args.img_size != 256 else get_image_size(args.model)
    print(f"Image size: {img_size}")
    
    # Create test dataset by combining all splits
    test_transform = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
    ])
    
    # Determine normalization type based on model
    # DINO models (v1, v2, v3) use ImageNet normalization
    if any(x in args.model.lower() for x in ['dinov1', 'dinov2', 'dinov3', 'vit']):
        normalization_type = "surgical"
    else:
        normalization_type = "surgical"
    
    # Load all three splits and combine them
    print("\nLoading CholecSeg8K dataset splits...")
    train_dataset = CholecSeg8kDataset(
        zip_path=args.data_path,
        split="train",
        transform=test_transform,
        frame_percentage=args.test_percentage,
        seed=args.seed,
        normalization_type=normalization_type,
    )
    val_dataset = CholecSeg8kDataset(
        zip_path=args.data_path,
        split="val",
        transform=test_transform,
        frame_percentage=args.test_percentage,
        seed=args.seed,
        normalization_type=normalization_type,
    )
    test_dataset = CholecSeg8kDataset(
        zip_path=args.data_path,
        split="test",
        transform=test_transform,
        frame_percentage=args.test_percentage,
        seed=args.seed,
        normalization_type=normalization_type,
    )
    
    # Combine all splits into one test set
    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    print(f"Combined dataset size: {len(combined_dataset)} frames")
    print(f"  - Train: {len(train_dataset)}")
    print(f"  - Val: {len(val_dataset)}")
    print(f"  - Test: {len(test_dataset)}")
    
    test_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: maintain order for clips
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Load model (trained for 30 ATLAS classes)
    print(f"\nLoading model: {args.model}")
    model = load_model(
        args.model,
        args.checkpoint,
        args.num_classes,  # Use ATLAS's 30 classes
        device
    )
    
    # For ATLAS models, enforce batch_size=1 for proper temporal processing
    if args.model.startswith("atlas"):
        if args.batch_size != 1:
            print(f"⚠️  Warning: ATLAS models require batch_size=1 for temporal query propagation.")
            print(f"   Overriding batch_size from {args.batch_size} to 1")
            args.batch_size = 1
            # Recreate dataloader with batch_size=1
            test_loader = DataLoader(
                combined_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
    
    # Evaluate with class mapping
    print(f"\nEvaluating on CholecSeg8K (all splits combined, {len(combined_dataset)} frames)...")
    print(f"Class mapping pipeline:")
    print(f"  1. ATLAS 30 classes → CholecSeg8K 9 classes")
    print(f"  2. L-hook Electrocautery (GT class 7) → Grasper (class 5)")
    print(f"  3. Exclude Connective Tissue (class 6) from evaluation")
    print(f"  4. Evaluate on 7 classes: 0, 1, 2, 3, 4, 5, 8")

    # Custom evaluation with class mapping
    metrics = evaluate_model_with_mapping(
        model,
        test_loader,
        device,
        args.num_classes,  # ATLAS has 30 classes
        args.num_cholecseg8k_classes,  # CholecSeg8K has 9 classes
        is_atlas=args.model.startswith("atlas"),
    )

    # Save visualizations
    if args.visualize_samples > 0:
        output_dir = os.path.join(args.visualize_dir, args.model)
        # Determine if we should mask background (all models except VideoMT)
        mask_bg = not (args.model == "videomt")
        save_random_visualizations(
            model=model,
            dataloader=test_loader,
            device=device,
            output_dir=output_dir,
            num_samples=args.visualize_samples,
            img_size=img_size,
            is_videomt=(args.model == "videomt"),
            is_atlas=args.model.startswith("atlas"),
            seed=args.seed,
            mask_background=mask_bg,
            apply_class_mapping=True,
        )
        print(f"\nSaved {args.visualize_samples} visualizations to: {output_dir}")
    
    # Save results if requested
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        results = {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "dataset": "CholecSeg8K (combined train/val/test splits)",
            "num_frames": len(combined_dataset),
            "class_mapping": "ATLAS 30 classes -> CholecSeg8K 9 classes (with L-hook -> Grasper consolidation)",
            "evaluation_classes": "0 (Background), 1 (Abdominal Wall), 2 (Liver), 3 (GI Tract), 4 (Fat), 5 (Grasper+L-hook), 8 (Gallbladder)",
            "excluded_classes": "6 (Connective Tissue)",
            "metrics": {
                "mIoU": float(metrics["mIoU"]),
                "Dice": float(metrics["Dice"]),
                "AP": float(metrics["AP"]),
                "AP50": float(metrics["AP50"]),
                "AP75": float(metrics["AP75"]),
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return metrics


def apply_evaluation_mapping(pred_masks, gt_masks):
    """
    Apply evaluation-specific remapping:
    1. Map and consolidate L-hook Electrocautery (GT class 7) to Grasper (class 5)
    2. Exclude Connective Tissue (class 6) by remapping to background (0)
    """
    gt_masks = gt_masks.clone()
    pred_masks = pred_masks.clone()
    
    # Map CholecSeg8K L-hook (7) to Grasper (5) in ground truth
    gt_masks[gt_masks == 7] = 5
    
    # Remap any class 6 (Connective Tissue) in predictions to background
    pred_masks[pred_masks == 6] = 0
    
    # Remap any class 6 in ground truth to background as well
    gt_masks[gt_masks == 6] = 0
    
    return pred_masks, gt_masks


def evaluate_model_with_mapping(model, test_loader, device, num_atlas_classes, num_cholecseg8k_classes, is_atlas=False):
    """
    Evaluate model with class mapping from ATLAS to CholecSeg8K.
    Excludes Connective Tissue class (6) from evaluation.
    """
    from evaluation.metrics import compute_metrics
    
    all_preds = []
    all_gts = []
    
    model.eval()
    
    with torch.no_grad():
        prev_query_embed = None
        current_clip = None
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            
            # Apply class mapping to ground truth (ATLAS 30 -> CholecSeg8K)
            gt_masks = map_atlas_to_cholecseg8k(gt_masks)
            
            if is_atlas:
                # ATLAS with temporal query propagation
                procedure = batch.get("procedure")
                video = batch.get("video")
                clip_id = batch.get("clip")
                
                if procedure is not None:
                    clip_key = f"{procedure[0] if isinstance(procedure, list) else procedure}/"
                    clip_key += f"{video[0] if isinstance(video, list) else video}/"
                    clip_key += f"{clip_id[0] if isinstance(clip_id, list) else clip_id}"
                    
                    if current_clip != clip_key:
                        current_clip = clip_key
                        prev_query_embed = None
                
                # Get ATLAS predictions (30 classes)
                outputs = model(
                    images,
                    prev_query_embed=prev_query_embed,
                    return_query_embedding=True,
                )
                
                # Unpack outputs
                mask_logits_per_block, class_logits_per_block, procedure_logits_per_block, query_embed = outputs
                prev_query_embed = query_embed
                
                # Get predictions from final block
                mask_logits = mask_logits_per_block[-1]
                class_logits = class_logits_per_block[-1]
                
                # Convert to per-pixel logits
                per_pixel_logits = torch.einsum(
                    "bqhw, bqc -> bchw",
                    mask_logits.sigmoid(),
                    class_logits.softmax(dim=-1)[..., :-1]
                )
                
                # Resize to match GT
                if per_pixel_logits.shape[-2:] != gt_masks.shape[-2:]:
                    per_pixel_logits = F.interpolate(
                        per_pixel_logits,
                        size=gt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                preds = torch.argmax(per_pixel_logits, dim=1)
            else:
                # Standard forward pass
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            # Apply class mapping to predictions (ATLAS 30 -> CholecSeg8K)
            preds = map_atlas_to_cholecseg8k(preds)
            
            # Apply evaluation-specific remapping (consolidate L-hook, exclude Connective Tissue)
            preds, gt_masks = apply_evaluation_mapping(preds, gt_masks)
            
            # Collect predictions and ground truth
            all_preds.append(preds.cpu().numpy())
            all_gts.append(gt_masks.cpu().numpy())
    
    # Concatenate all predictions and ground truths
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    # Compute metrics on CholecSeg8K (9 classes total, but class 6 is excluded)
    # We still use 9 classes so the class indices remain consistent
    metrics = compute_metrics(all_gts, all_preds, num_classes=num_cholecseg8k_classes)
    
    print(f"\nMetrics (CholecSeg8K - Evaluation Classes):")
    print(f"  Classes: 0 (Background), 1 (Abdominal Wall), 2 (Liver), 3 (GI Tract),")
    print(f"           4 (Fat), 5 (Grasper/L-hook), 8 (Gallbladder)")
    print(f"  Excluded: Class 6 (Connective Tissue)")
    print(f"  L-hook Electrocautery consolidated into Grasper (class 5)")
    print(f"\n  mIoU: {metrics['mIoU']:.4f}")
    print(f"  Dice: {metrics['Dice']:.4f}")
    print(f"  AP: {metrics['AP']:.4f}")
    print(f"  AP50: {metrics['AP50']:.4f}")
    print(f"  AP75: {metrics['AP75']:.4f}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ATLAS models on CholecSeg8K dataset")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (see MODEL_REGISTRY or use 'videomt')"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CholecSeg8K dataset zip file"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=30,
        help="Number of classes in ATLAS model (30)"
    )
    parser.add_argument(
        "--num_cholecseg8k_classes",
        type=int,
        default=9,
        help="Number of CholecSeg8K classes (9)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size (automatically inferred from model if not specified)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--test_percentage",
        type=int,
        default=100,
        help="Percentage of test frames to evaluate (1-100)"
    )
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=25,
        help="Number of random samples to visualize (0 to disable)"
    )
    parser.add_argument(
        "--visualize_dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save visualization images"
    )
    
    args = parser.parse_args()
    metrics = main(args)

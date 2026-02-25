"""
Test script for benchmarking ATLAS-trained models on the CholecSeg8K dataset.

This script evaluates ATLAS models on CholecSeg8K by:
1. Loading the full dataset (no predefined splits)
2. Mapping ATLAS's 30 output classes to CholecSeg8K's 13 classes
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

from torch.utils.data import DataLoader
from datasets.cholecseg8k import CholecSeg8kDataset, CHOLECSEG8K_CLASS_NAMES
import torchvision.transforms.v2 as T
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
from evaluation.visual_logging import apply_mask_overlay, denormalize
import numpy as np


# ===========================
# Class Mapping: ATLAS 30 classes -> CholecSeg8K 13 classes
# ===========================
# This mapping defines how ATLAS's 30 output classes are mapped to CholecSeg8K's 13 classes.
# EVALUATION FOCUS: All foreground classes (1-12) are evaluated (background excluded)
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
# CholecSeg8K classes (13):
#   0: Black Background
#   1: Abdominal Wall
#   2: Liver
#   3: Gastrointestinal Tract
#   4: Fat
#   5: Grasper
#   6: Connective Tissue
#   7: Blood
#   8: Cystic Duct
#   9: L-hook Electrocautery
#   10: Gallbladder
#   11: Hepatic Vein
#   12: Liver Ligament
#
# Mapping Strategy:
#   - Background structures -> 0 (Background)
#   - Tools (ATLAS 1) -> 5 (Grasper)
#   - Vein -> 11 (Hepatic Vein)
#   - Artery -> 7 (Blood)
#   - Nerve -> 6 (Connective Tissue)
#   - Bile/lymph Duct -> 8 (Cystic Duct)
#   - Hepatic ligament/Cystic plate -> 12 (Liver Ligament)
#   - GI tract structures -> 3 (Gastrointestinal Tract)
#   - Non-cholecystectomy organs -> 0 (Background)

ATLAS_TO_CHOLECSEG8K_MAPPING = {
    0:  0,   # Background -> Background
    1:  5,   # Tools/camera -> Grasper (surgical tools)
    2:  11,  # Vein -> Hepatic Vein
    3:  7,   # Artery -> Blood
    4:  6,   # Nerve -> Connective Tissue
    5:  3,   # Small intestine -> Gastrointestinal Tract
    6:  3,   # Colon/rectum -> Gastrointestinal Tract
    7:  1,   # Abdominal wall -> Abdominal Wall (direct mapping)
    8:  0,   # Diaphragm -> Background (not relevant for cholecystectomy)
    9:  4,   # Fat -> Fat (direct mapping)
    10: 2,   # Liver -> Liver (direct mapping)
    11: 8,   # Bile/lymph Duct -> Cystic Duct
    12: 10,  # Gallbladder -> Gallbladder (direct mapping - primary organ)
    13: 12,  # Hepatic ligament -> Liver Ligament
    14: 12,  # Cystic plate -> Liver Ligament
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
ATLAS 30 Classes -> CholecSeg8K Classes Mapping
===============================================

Key Mappings:
  - Tools/camera -> Grasper (class 5)
  - Vein -> Hepatic Vein (class 11)
  - Artery -> Blood (class 7)
  - Nerve -> Connective Tissue (class 6)
  - Bile/lymph Duct -> Cystic Duct (class 8)
  - Hepatic ligament/Cystic plate -> Liver Ligament (class 12)
  - Gallbladder -> Gallbladder (class 10)
  - GI tract structures -> Gastrointestinal Tract (class 3)
  - Non-cholecystectomy organs -> Background (class 0)
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
    Map ATLAS predictions (30 classes) to CholecSeg8K classes (13 classes).
    
    Args:
        pred_mask: Tensor of shape (B, H, W) or (H, W) with class indices
    
    Returns:
        Mapped tensor with CholecSeg8K class indices
    """
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


def print_class_occurrences(dataloader, num_classes, class_names):
    """Print per-class pixel occurrences across the dataset."""
    counts = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Counting class occurrences"):
            masks = batch["mask"]
            if masks.ndim > 3:
                masks = masks.squeeze(1)
            flat = masks.view(-1).to(torch.int64)
            counts += torch.bincount(flat, minlength=num_classes).cpu().numpy()

    print("\nClass occurrences (pixel counts):")
    print("=" * 60)
    for class_id in range(num_classes):
        name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"{class_id:>2} | {name:<22} | {counts[class_id]}")
    print("=" * 60)


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
    
    # Create test dataset (no predefined splits)
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
    
    # Load dataset (no predefined splits)
    print("\nLoading CholecSeg8K dataset...")
    dataset = CholecSeg8kDataset(
        zip_path=args.data_path,
        transform=test_transform,
        frame_percentage=args.test_percentage,
        seed=args.seed,
        normalization_type=normalization_type,
    )
    
    print(f"Dataset size: {len(dataset)} frames")
    
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: maintain order for clips
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Print class occurrences before evaluation
    print_class_occurrences(test_loader, args.num_cholecseg8k_classes, CHOLECSEG8K_CLASS_NAMES)
    
    # Load model with ATLAS's original 30 classes (checkpoint was trained on ATLAS)
    # Class mapping to CholecSeg8K's 13 classes happens during inference
    print(f"\nLoading model: {args.model}")
    print(f"Model classes: {args.num_classes} (ATLAS)")
    print(f"Target classes: {args.num_cholecseg8k_classes} (CholecSeg8K)")
    model = load_model(
        args.model,
        args.checkpoint,
        args.num_classes,  # ATLAS model was trained with 30 classes
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
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
    
    # Evaluate with class mapping
    print(f"\nEvaluating on CholecSeg8K ({len(dataset)} frames)...")
    print("Class mapping pipeline:")
    print("  1. ATLAS 30 classes → CholecSeg8K 13 classes")
    print("  2. Evaluate on classes 1-6, 8-11 (background, blood, liver ligament excluded)")

    # Custom evaluation with class mapping
    metrics = evaluate_model_with_mapping(
        model,
        test_loader,
        device,
        args.num_classes,  # ATLAS has 30 classes
        args.num_cholecseg8k_classes,  # CholecSeg8K has 13 classes
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
            "dataset": "CholecSeg8K (full dataset)",
            "num_frames": len(dataset),
            "class_mapping": "ATLAS 30 classes -> CholecSeg8K 13 classes",
            "evaluation_classes": "1-6, 8-11 (foreground classes)",
            "excluded_classes": "0 (Background), 7 (Blood), 12 (Liver Ligament)",
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
    Apply evaluation-specific remapping.

    Currently, this is a no-op and simply returns the inputs unchanged.
    """
    return pred_masks, gt_masks


def evaluate_model_with_mapping(model, test_loader, device, num_atlas_classes, num_cholecseg8k_classes, is_atlas=False):
    """
    Evaluate model with class mapping from ATLAS to CholecSeg8K.
    Evaluates foreground classes (1-6, 8-11) and excludes background (0), blood (7), and liver ligament (12).
    """
    from evaluation.metrics import compute_class_metrics, SegmentationAPEvaluator
    
    model.eval()
    
    ignore_index = 255
    
    # Track scores per class across the entire dataset
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)
    
    # AP tracking
    clip_ap = {}
    current_clip_id = None
    ap_evaluator = None
    
    with torch.no_grad():
        prev_query_embed = None
        current_atlas_clip = None
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            
            if is_atlas:
                # ATLAS with temporal query propagation
                procedure_field = batch.get("procedure")
                video_field = batch.get("video")
                clip_field = batch.get("clip")
                
                if procedure_field is not None:
                    if isinstance(procedure_field, (list, tuple)):
                        procedure = procedure_field[0]
                        video = video_field[0]
                        clip_id = clip_field[0]
                    else:
                        procedure = procedure_field
                        video = video_field
                        clip_id = clip_field
                    
                    clip_key = f"{procedure}/{video}/{clip_id}"
                    
                    if current_atlas_clip != clip_key:
                        current_atlas_clip = clip_key
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
                
                # Convert to per-pixel logits (30-class output)
                probs = torch.einsum(
                    "bqhw, bqc -> bchw",
                    mask_logits.sigmoid(),
                    class_logits.softmax(dim=-1)[..., :-1]
                )
                
                # Resize to match GT
                if probs.shape[-2:] != gt_masks.shape[-2:]:
                    probs = F.interpolate(
                        probs,
                        size=gt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                preds = torch.argmax(probs, dim=1, keepdim=True)
            else:
                # Standard forward pass
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Resize to match GT
                if probs.shape[-2:] != gt_masks.shape[-2:]:
                    probs = F.interpolate(
                        probs,
                        size=gt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                preds = torch.argmax(probs, dim=1, keepdim=True)
            
            # Apply class mapping to predictions (ATLAS 30 -> CholecSeg8K 13)
            # Squeeze preds to (B, H, W) for mapping
            preds_2d = preds.squeeze(1)
            preds_mapped = map_atlas_to_cholecseg8k(preds_2d)
            gt_masks_mapped = map_atlas_to_cholecseg8k(gt_masks)
            
            # Apply evaluation-specific remapping (currently no-op)
            preds_mapped, gt_masks_mapped = apply_evaluation_mapping(preds_mapped, gt_masks_mapped)
            
            # Compute metrics for each image in batch
            # Evaluate foreground classes (exclude background 0, blood 7, liver ligament 12)
            classes_to_eval = [c for c in range(1, num_cholecseg8k_classes) if c not in [7, 12]]
            
            for i in range(len(images)):
                # --- AP Handling ---
                # Get clip info for AP tracking (handle both list and single value cases)
                procedure_field = batch.get("procedure")
                video_field = batch.get("video")
                clip_field = batch.get("clip")
                
                if isinstance(procedure_field, (list, tuple)):
                    procedure = procedure_field[i]
                    video = video_field[i]
                    clip = clip_field[i]
                else:
                    procedure = procedure_field
                    video = video_field
                    clip = clip_field
                
                clip_id_str = f"{procedure}/{video}/{clip}"
                
                if current_clip_id != clip_id_str:
                    if ap_evaluator is not None:
                        clip_ap[current_clip_id] = ap_evaluator.evaluate()
                    current_clip_id = clip_id_str
                    ap_evaluator = SegmentationAPEvaluator()
                
                pred_np = preds_mapped[i].cpu().numpy()
                gt_np = gt_masks_mapped[i].cpu().numpy()
                
                # Ensure 2D arrays (squeeze any extra dimensions)
                if pred_np.ndim > 2:
                    pred_np = pred_np.squeeze()
                if gt_np.ndim > 2:
                    gt_np = gt_np.squeeze()
                
                # --- Metric Collection ---
                for c in classes_to_eval:
                    iou_c, dice_c = compute_class_metrics(pred_np, gt_np, c, ignore_index=ignore_index)
                    
                    if iou_c is not None:
                        class_ious[c].append(iou_c)
                        class_dices[c].append(dice_c)
                
                # Binary AP logic
                gt_binary = (gt_np > 0).astype(np.uint8)
                pred_binary = (pred_np > 0).astype(np.uint8)
                
                # Compute confidence score (we can't use probs directly since we mapped classes)
                # Use a simple approach: count of foreground pixels as proxy
                if pred_binary.sum() > 0:
                    score = 0.9  # Default high confidence since we're using argmax
                else:
                    score = 0.1  # Low confidence for no predictions
                
                ap_evaluator.add_frame(gt_binary, pred_binary, score)
        
        # Final AP evaluation
        if ap_evaluator is not None:
            clip_ap[current_clip_id] = ap_evaluator.evaluate()
    
    # --- FINAL DATASET AGGREGATION ---
    final_per_class_iou = {}
    final_per_class_dice = {}
    
    for c in classes_to_eval:
        if len(class_ious[c]) > 0:
            final_per_class_iou[c] = np.mean(class_ious[c])
            final_per_class_dice[c] = np.mean(class_dices[c])
        else:
            final_per_class_iou[c] = 0.0
            final_per_class_dice[c] = 0.0
    
    # The Final Mean Scores
    mIoU = np.mean(list(final_per_class_iou.values()))
    mDice = np.mean(list(final_per_class_dice.values()))
    
    # AP Metrics
    AP_total = np.mean([v["AP"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_50 = np.mean([v["AP50"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_75 = np.mean([v["AP75"] for v in clip_ap.values()]) if clip_ap else 0.0
    
    print(f"\nMetrics (CholecSeg8K - Evaluation Classes):")
    print("=" * 60)
    print(f"{'Class ID':<15} | {'IoU':<10} | {'Dice':<10}")
    print("-" * 60)
    class_names = {i: CHOLECSEG8K_CLASS_NAMES[i] for i in classes_to_eval}
    for c in classes_to_eval:
        class_name = class_names.get(c, f"Class {c}")
        print(f"{class_name:<15} | {final_per_class_iou[c]:.4f}     | {final_per_class_dice[c]:.4f}")
    print("-" * 60)
    print(f"{'OVERALL':<15} | {mIoU:.4f}     | {mDice:.4f}")
    print(f"{'mAP':<15} | {AP_total:.4f}")
    print("=" * 60)
    print("Note: Background (0), Blood (7), and Liver Ligament (12) excluded from evaluation")
    print()
    
    return {
        "mIoU": mIoU,
        "Dice": mDice,
        "AP": AP_total,
        "AP50": AP_50,
        "AP75": AP_75,
        "per_class_iou": final_per_class_iou,
        "per_class_dice": final_per_class_dice
    }


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
        help="Number of classes in ATLAS model output (30 for ATLAS-trained models)"
    )
    parser.add_argument(
        "--num_cholecseg8k_classes",
        type=int,
        default=13,
        help="Number of CholecSeg8K classes (13)"
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

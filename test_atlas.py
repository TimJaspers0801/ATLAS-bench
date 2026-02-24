"""
Test script for benchmarking all models on the ATLAS dataset.

Supports:
- Image-based models (DINOv2, DINOv3, SurgeNet variants, EOMT)
- Video-based models (VideoMT with online processing)
- Frame-by-frame evaluation on test set

Usage:
    python test_atlas.py --model lh-dinov3-vitl-256-surgenet2m \\
                         --checkpoint weights/DINOv3-vitl-256-surgenet2m.pth \\
                         --data_path atlas.zip
    
    python test_atlas.py --model videomt \\
                         --checkpoint checkpoints/videomt.pth \\
                         --data_path atlas.zip
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
from datasets.atlas import AtlasDataset
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
    "dinov2": 224,
    "dinov3": 256,
    "gastronet": 336,
    'videomt': 1280,
    'eomt': 256,
}


def get_image_size(model_name: str) -> int:
    """Infer image size from model name."""
    for key, size in IMAGE_SIZE_MAP.items():
        if key in model_name:
            return size
    return 256  # default


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
    from models.atlas.atlas import atlas_vitl_dinov3
    
    # Map model names to loader functions
    atlas_loaders = {
        "atlas_vitl_dinov3": atlas_vitl_dinov3,
    }
    
    if model_name not in atlas_loaders:
        raise ValueError(
            f"Unknown ATLAS variant: {model_name}. Available: {', '.join(atlas_loaders.keys())}"
        )
    
    print(f"Loading ATLAS model: {model_name}")
    model = atlas_loaders[model_name](num_classes=num_classes)
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading ATLAS checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
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


def evaluate_atlas_temporal(model, test_loader, device, num_classes, use_query_propagation=True):
    """Wrapper function that calls evaluate_atlas_temporal from evaluation module."""
    from evaluation.dataset_evaluation import evaluate_atlas_temporal as _evaluate_atlas_temporal
    return _evaluate_atlas_temporal(model, test_loader, device, num_classes, use_query_propagation)



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
):
    if num_samples <= 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    mean = torch.tensor(dataloader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(dataloader.dataset.std).view(3, 1, 1)

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
                
                # Mask background predictions if requested (makes visualization cleaner)
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
    
    # Create test dataset
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
    
    test_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="test",
        transform=test_transform,
        frame_percentage=args.test_percentage,
        seed=args.seed,
        normalization_type=normalization_type,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: maintain order for clips
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_model(
        args.model,
        args.checkpoint,
        args.num_classes,
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
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
    
    # Evaluate
    print(f"\nEvaluating on test set...")

    if args.model.startswith("atlas"):
        # ATLAS models use temporal evaluation
        metrics = evaluate_atlas_temporal(model, test_loader, device, args.num_classes)
    elif args.model.startswith("eomt"):
        # EOMT models were trained without background class (0), all classes shifted down by 1
        metrics = evaluate_model(model, test_loader, device, args.num_classes)
    else:
        # Other models were also trained ignoring background
        metrics = evaluate_model(model, test_loader, device, args.num_classes)

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
        )
        print(f"\nSaved {args.visualize_samples} visualizations to: {output_dir}")
    
    # Save results if requested
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        results = {
            "model": args.model,
            "checkpoint": args.checkpoint,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ATLAS models")
    
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
        help="Path to ATLAS dataset zip file"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=47,
        help="Number of classes in the dataset"
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
    parser.add_argument(
        "--videomt_window",
        type=int,
        default=4,
        help="Window size for VideoMT clip processing to limit memory usage"
    )
    
    args = parser.parse_args()
    metrics = main(args)

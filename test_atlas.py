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

from torch.utils.data import DataLoader
from datasets.atlas import AtlasDataset
import torchvision.transforms.v2 as T
from torch import nn
import torch.nn.functional as F

from utils import load_checkpoint
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
}


def get_image_size(model_name: str) -> int:
    """Infer image size from model name."""
    for key, size in IMAGE_SIZE_MAP.items():
        if key in model_name:
            return size
    return 256  # default


def load_model(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    """Load a model from the registry with optional checkpoint."""
    if model_name == "videomt":
        return load_videomt(checkpoint_path, num_classes, device)
    elif model_name == "eomt":
        return load_eomt(checkpoint_path, num_classes, device)
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
            f"  Video-based: videomt, eomt"
        )


def load_videomt(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load VideoMT model for online video processing."""
    from models.videomt.videomt_standalone import VideoMT
    
    # Initialize with training configuration from ATLAS config
    model = VideoMT(
        img_size=1280,
        num_classes=124,
        num_queries=200,
        task='vss',
        model_name='vit_large_patch14_dinov2.lvd142m',
    )
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading VideoMT checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Handle key name mismatches between Detectron2 version and standalone
        key_mappings = {
            'q.weight': 'query_embedding.weight',  # Detectron2 uses 'q', standalone uses 'query_embedding'
        }
        
        for old_key, new_key in key_mappings.items():
            if old_key in state_dict and new_key not in state_dict:
                print(f"  Remapping key: {old_key} → {new_key}")
                state_dict[new_key] = state_dict.pop(old_key)
        
        # Handle pos_embed shape mismatch: checkpoint was trained without class token
        # Checkpoint: [1, 6400, 1024] (no class token), Model: [1, 6401, 1024] (with class token)
        if 'encoder.backbone.pos_embed' in state_dict:
            checkpoint_pos = state_dict['encoder.backbone.pos_embed']
            model_pos = model.encoder.backbone.pos_embed
            
            if checkpoint_pos.shape != model_pos.shape:
                print(f"⚠ Pos_embed shape mismatch - checkpoint trained without class token:")
                print(f"  Checkpoint: {checkpoint_pos.shape}")
                print(f"  Model:      {model_pos.shape}")
                
                # Add position embedding for class token (prepend zeros)
                B, N, D = checkpoint_pos.shape
                class_pos_embed = torch.zeros(B, 1, D)  # Initialize class token pos_embed to zero
                expanded_pos = torch.cat([class_pos_embed, checkpoint_pos], dim=1)  # Prepend to match model
                
                state_dict['encoder.backbone.pos_embed'] = expanded_pos
                print(f"  Expanded to: {expanded_pos.shape} (added zero-initialized class token position)")
        
        # Remove keys that are not part of the model
        keys_to_remove = ['criterion.empty_weight', 'attn_mask_probs', 'encoder.backbone.reg_token']
        for key in keys_to_remove:
            if key in state_dict:
                print(f"  Removing non-model key: {key}")
                del state_dict[key]
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if not missing_keys and not unexpected_keys:
            print("✓ All keys loaded successfully")
        else:
            if missing_keys:
                print(f"⚠ Missing keys ({len(missing_keys)}):")
                # pixel_mean/pixel_std are buffers that will be initialized by register_buffer
                important_missing = [k for k in missing_keys if 'pixel_' not in k]
                if important_missing:
                    print(f"  Important missing keys:")
                    for key in important_missing[:5]:
                        print(f"    - {key}")
                    if len(important_missing) > 5:
                        print(f"    ... and {len(important_missing) - 5} more")
                else:
                    print(f"  (Only pixel normalization missing, which is acceptable)")
            if unexpected_keys:
                print(f"⚠ Unexpected keys ({len(unexpected_keys)}):")
                for key in unexpected_keys:
                    print(f"  - {key}")
    
    return model.to(device)


def load_eomt(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load EOMT model."""
    from models.eomt.eomt import EOMT
    from models.decoders.vit import ViTSegmenter
    
    vit_backbone = EOMT(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    )
    
    model = ViTSegmenter(
        vit_model=vit_backbone,
        decoder_name="linear",
        num_classes=num_classes,
    )
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading EOMT checkpoint: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path)
    
    return model.to(device)


def evaluate_videomt(model, test_loader, device, num_classes):
    """
    Evaluate VideoMT in online fashion, processing frames sequentially
    within each clip while maintaining temporal state.
    """
    model.eval()
    
    # Track scores per class
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)
    
    # AP tracking
    from evaluation.metrics import SegmentationAPEvaluator
    clip_ap = {}
    current_clip = None
    ap_evaluator = None
    
    with torch.no_grad():
        current_video = None
        debug_count = 0
        frame_count = 0
        
        print(f"Starting evaluation on {len(test_loader)} frames...")
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating VideoMT")):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            
            # Check if we need to reset memory for a new video
            batch_video = f"{batch['procedure'][0]}/{batch['video'][0]}"
            if batch_video != current_video:
                model.reset_memory()
                current_video = batch_video
                if batch_idx % 500 == 0 or debug_count < 5:
                    print(f"\n[Progress] Processing video: {batch_video}, Frame batch: {batch_idx}")
            
            # Process frames online
            B = images.shape[0]
            outputs_list = []
            
            for i in range(B):
                frame = images[i:i+1]  # (1, 3, H, W)
                
                # Online forward pass
                output = model.forward_frame(frame)
                outputs_list.append(output)
                    
            debug_count += 1
            
            # Combine outputs from batch
            pred_logits = torch.cat([o['pred_logits'] for o in outputs_list], dim=0)  # (B, Q, C+1)
            pred_masks = torch.cat([o['pred_masks'] for o in outputs_list], dim=0)      # (B, Q, H, W)
            
            # VSS Inference: Weight all query masks by their class probabilities
            # Get class probabilities (exclude background/void class which is the last one)
            mask_cls = F.softmax(pred_logits, dim=-1)[:, :, :-1]  # (B, Q, C)
            
            # Apply sigmoid to masks to convert logits to [0, 1]
            pred_masks_prob = torch.sigmoid(pred_masks)  # (B, Q, H, W)
            
            # Use einsum to aggregate: weight each mask by class probability
            # Result: for each class, compute weighted sum of all masks
            # einsum("bqc,bqhw->bchw", mask_cls, pred_masks_prob) aggregates masks weighted by class confidence
            semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, pred_masks_prob)  # (B, C, H, W)
            
            # Take argmax over classes (dimension 1) to get predicted class for each pixel
            sem_mask = semseg.argmax(dim=1)  # (B, H, W) - predicted class index (0 to C-1)
            
            # Debug: Show aggregation info
            if debug_count < 5:
                unique_preds = torch.unique(sem_mask[0])
                unique_gt = torch.unique(gt_masks[0])
                print(f"\n[Batch {batch_idx}] Pred classes: {unique_preds.tolist()}, GT classes: {unique_gt.tolist()}")
                
            debug_count += 1
            
            # Report progress every 1000 frames
            frame_count += B
            if frame_count % 1000 == 0:
                print(f"\n[Progress] Processed {frame_count}/{len(test_loader)} frames")
                
            for i in range(B):
                # Get GT mask and ensure 2D shape
                gt_mask = gt_masks[i].squeeze()  # Remove any extra dimensions
                
                # Get predicted segmentation (semantic mask)
                pred_seg = sem_mask[i]  # (H, W) - class indices from 0 to C-1
                
                # Interpolate to GT size if needed
                if pred_seg.shape != gt_mask.shape:
                    pred_seg = pred_seg.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                    pred_seg = F.interpolate(
                        pred_seg,
                        size=gt_mask.shape,
                        mode='nearest'
                    ).squeeze().long()  # (H, W)
                
                # Debug info
                if debug_count <= 5 and i == 0:
                    print(f"  → Pred shape: {pred_seg.shape}, unique: {torch.unique(pred_seg).tolist()}")
                    print(f"  → GT shape: {gt_mask.shape}, unique: {torch.unique(gt_mask).tolist()}")
                
                # Convert to numpy for metrics computation
                pred_np = pred_seg.cpu().numpy()
                gt_np = gt_mask.cpu().numpy()
                
                # AP Handling
                clip_id = f"{batch['procedure'][i]}/{batch['video'][i]}/{batch['clip'][i]}"
                if current_clip != clip_id:
                    if ap_evaluator is not None:
                        clip_ap[current_clip] = ap_evaluator.evaluate()
                    current_clip = clip_id
                    ap_evaluator = SegmentationAPEvaluator()
                
                # Binary segmentation for AP: foreground vs background
                gt_binary = (gt_np > 0).astype(np.uint8)
                pred_binary = (pred_np > 0).astype(np.uint8)
                
                # Use max confidence from semseg for score
                max_score = semseg[i].max().item()
                ap_evaluator.add_frame(gt_binary, pred_binary, max_score)
                
                # Compute per-class metrics
                from evaluation.metrics import compute_class_metrics
                classes_to_eval = range(1, num_classes+1)
                for c in classes_to_eval:
                    iou_c, dice_c = compute_class_metrics(pred_np, gt_np, c)
                    if iou_c is not None:
                        class_ious[c].append(iou_c)
                        class_dices[c].append(dice_c)
    
    if ap_evaluator is not None:
        clip_ap[current_clip] = ap_evaluator.evaluate()
    
    print(f"\n✓ Evaluation complete! Processed {frame_count} frames total")
    
    # Aggregate metrics
    from evaluation.metrics import compute_class_metrics
    classes_to_eval = range(1, num_classes+1)
    
    final_per_class_iou = {}
    final_per_class_dice = {}
    
    for c in classes_to_eval:
        if len(class_ious[c]) > 0:
            final_per_class_iou[c] = np.mean(class_ious[c])
            final_per_class_dice[c] = np.mean(class_dices[c])
        else:
            final_per_class_iou[c] = 0.0
            final_per_class_dice[c] = 0.0
    
    mIoU = np.mean(list(final_per_class_iou.values()))
    mDice = np.mean(list(final_per_class_dice.values()))
    
    AP_total = np.mean([v["AP"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_50 = np.mean([v["AP50"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_75 = np.mean([v["AP75"] for v in clip_ap.values()]) if clip_ap else 0.0
    
    # Print report
    print("\n" + "=" * 40)
    print(f"{'Class ID':<10} | {'IoU':<10} | {'Dice':<10}")
    print("-" * 40)
    for c in classes_to_eval:
        print(f"Class {c:<4} | {final_per_class_iou[c]:.4f}     | {final_per_class_dice[c]:.4f}")
    print("-" * 40)
    print(f"{'OVERALL':<10} | {mIoU:.4f}     | {mDice:.4f}")
    print(f"{'mAP':<10} | {AP_total:.4f}")
    print("=" * 40 + "\n")
    
    return {
        "mIoU": mIoU,
        "Dice": mDice,
        "AP": AP_total,
        "AP50": AP_50,
        "AP75": AP_75,
        "per_class_iou": final_per_class_iou,
        "per_class_dice": final_per_class_dice
    }


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
    
    test_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="test",
        transform=test_transform,
        seed=args.seed,
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
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    if args.model == "videomt":
        metrics = evaluate_videomt(model, test_loader, device, args.num_classes)
    else:
        metrics = evaluate_model(model, test_loader, device, args.num_classes)
    
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
    
    args = parser.parse_args()
    metrics = main(args)

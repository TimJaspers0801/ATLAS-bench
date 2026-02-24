"""
Quick debugging script for SAM evaluation.
Run a single clip with debug output to diagnose the Dice=0 issue.
"""

import argparse
import random
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.v2 as T

from torch.utils.data import DataLoader
from datasets.atlas import AtlasDataset
from eval_sam import (
    generate_clicks_per_class, 
    process_clip_sam2, 
    process_clip_sam3,
    load_sam2_model,
    load_sam3_model,
    SAM2_MODELS,
    SAM3_MODELS
)
from evaluation.metrics import compute_class_metrics


def debug_single_clip(args):
    """Debug a single clip to understand the issue."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine model type
    is_sam3 = args.model in SAM3_MODELS
    
    # Load model
    if is_sam3:
        model, processor = load_sam3_model(args.model, device)
    else:
        model, processor = load_sam2_model(args.model, device)
    
    # Create test dataset
    test_transform = T.Compose([
        T.Resize(args.img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.img_size),
    ])
    
    test_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="test",
        transform=test_transform,
        frame_percentage=100,
        seed=args.seed,
        normalization_type="none",
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"\nTest set: {len(test_dataset)} frames")
    
    # Get first clip
    mean = torch.tensor(test_loader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(test_loader.dataset.std).view(3, 1, 1)
    
    clip_frames = []
    clip_gts = []
    clip_class_clicks = None
    current_clip = None
    sample_count = 0
    
    print("\n" + "=" * 80)
    print("DEBUGGING FIRST CLIP")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if sample_count >= 3:  # Only take first 3 frames of a clip for speed
                break
            
            images = batch["image"]
            gt_masks = batch["mask"].to(device)
            clip_id = f"{batch['procedure'][0]}/{batch['video'][0]}/{batch['clip'][0]}"
            
            # Initialize new clip
            if current_clip is None:
                current_clip = clip_id
                print(f"\nClip ID: {clip_id}")
            
            # Process completed clip
            if clip_id != current_clip:
                print(f"\nNew clip {clip_id}, processing previous clip...")
                break
            
            # Denormalize and convert to PIL
            img_tensor = images[0]
            img_denorm = img_tensor * std + mean
            img_denorm = img_denorm.clamp(0.0, 1.0)
            img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            gt_np = gt_masks[0].squeeze().cpu().numpy().astype(np.int32)
            
            clip_frames.append(img_pil)
            clip_gts.append(gt_np)
            
            print(f"\nFrame {sample_count}:")
            print(f"  Image size: {img_pil.size}")
            print(f"  GT shape: {gt_np.shape}")
            print(f"  GT unique classes: {np.unique(gt_np)}")
            print(f"  GT class counts: {[(c, (gt_np == c).sum()) for c in np.unique(gt_np)]}")
            
            # Generate clicks from first frame only
            if clip_class_clicks is None:
                clip_class_clicks = generate_clicks_per_class(
                    gt_np,
                    num_clicks_per_class=args.num_clicks,
                    seed=args.seed
                )
                print(f"\nGenerated clicks: {clip_class_clicks}")
            
            sample_count += 1
    
    # Now process this clip with SAM
    print("\n" + "=" * 80)
    print("PROCESSING WITH SAM MODEL")
    print("=" * 80)
    
    if is_sam3:
        predictions = process_clip_sam3(
            model, processor, clip_frames, clip_class_clicks, device, debug=True
        )
    else:
        predictions = process_clip_sam2(
            model, processor, clip_frames, clip_class_clicks, device, debug=True
        )
    
    # Compute metrics
    print("\n" + "=" * 80)
    print("METRIC COMPUTATION")
    print("=" * 80)
    
    classes_to_eval = range(1, args.num_classes + 1)
    
    for frame_idx, (pred_mask, gt_mask) in enumerate(zip(predictions, clip_gts)):
        print(f"\nFrame {frame_idx}:")
        print(f"  Prediction shape: {pred_mask.shape}")
        print(f"  Prediction unique classes: {np.unique(pred_mask)}")
        print(f"  Prediction class counts: {[(c, (pred_mask == c).sum()) for c in np.unique(pred_mask)]}")
        print(f"  GT unique classes: {np.unique(gt_mask)}")
        
        # Check union/intersection
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        print(f"  Pred pixels (>0): {pred_binary.sum()}")
        print(f"  GT pixels (>0): {gt_binary.sum()}")
        print(f"  Intersection: {intersection}")
        print(f"  Union: {union}")
        
        if union > 0:
            overall_iou = intersection / union
            print(f"  Overall IoU: {overall_iou:.4f}")
        
        # Per-class metrics
        print(f"\n  Per-class metrics:")
        for c in np.unique(gt_mask):
            if c not in classes_to_eval:
                continue
            iou_c, dice_c = compute_class_metrics(pred_mask, gt_mask, c, ignore_index=255)
            if iou_c is not None:
                print(f"    Class {c}: IoU={iou_c:.4f}, Dice={dice_c:.4f}")
            else:
                print(f"    Class {c}: Not in GT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug SAM evaluation on single clip")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SAM2_MODELS.keys()) + list(SAM3_MODELS.keys()),
        help="SAM model variant to evaluate"
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
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--num_clicks",
        type=int,
        default=3,
        help="Number of clicks per class"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1024,
        help="Image size for SAM models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    
    args = parser.parse_args()
    debug_single_clip(args)

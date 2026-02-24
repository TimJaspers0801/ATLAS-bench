"""
Evaluation script for SAM2 and SAM3 models on ATLAS dataset.

This script:
1. Generates random clicks from ground truth masks on the first frame of each clip
2. Propagates these clicks through subsequent frames using SAM's video tracking
3. Evaluates segmentation performance using the same metrics as test_atlas.py

Usage:
    python eval_sam.py --model sam2-hiera-large --data_path atlas.zip
    python eval_sam.py --model sam3-base --data_path atlas.zip --num_clicks 5
"""

import argparse
import random
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import cv2
from pathlib import Path

from torch.utils.data import DataLoader
from datasets.atlas import AtlasDataset
import torchvision.transforms.v2 as T
from PIL import Image

from evaluation.metrics import compute_class_metrics, SegmentationAPEvaluator
from evaluation.visual_logging import apply_mask_overlay, denormalize
from utils import color_palette


# SAM2 model variants
SAM2_MODELS = {
    "sam2-hiera-tiny": "facebook/sam2-hiera-tiny",
    "sam2-hiera-small": "facebook/sam2-hiera-small",
    "sam2-hiera-base-plus": "facebook/sam2-hiera-base-plus",
    "sam2-hiera-large": "facebook/sam2-hiera-large",
}

# SAM3 model variants
SAM3_MODELS = {
    "sam3-tiny": "facebook/sam3-tiny",
    "sam3-small": "facebook/sam3-small",
    "sam3-base": "facebook/sam3-base",
    "sam3-large": "facebook/sam3-large",
}


def load_sam2_model(model_name: str, device: torch.device):
    """Load SAM2 model from HuggingFace."""
    from transformers import AutoProcessor, AutoModelForMaskGeneration
    
    model_id = SAM2_MODELS[model_name]
    print(f"Loading SAM2 model: {model_id}")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMaskGeneration.from_pretrained(model_id).to(device)
    model.eval()
    
    return model, processor


def load_sam3_model(model_name: str, device: torch.device):
    """Load SAM3 model from HuggingFace."""
    from transformers import AutoProcessor, AutoModelForMaskGeneration
    
    model_id = SAM3_MODELS[model_name]
    print(f"Loading SAM3 model: {model_id}")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMaskGeneration.from_pretrained(model_id).to(device)
    model.eval()
    
    return model, processor


def generate_clicks_from_mask(mask, num_clicks=3, seed=42):
    """
    Generate positive click coordinates from a binary mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        num_clicks: Number of positive clicks to generate
        seed: Random seed for reproducibility
        
    Returns:
        clicks: List of (x, y) coordinates
    """
    rng = random.Random(seed)
    
    # Get all positive pixel coordinates
    positive_coords = np.argwhere(mask > 0)  # Returns (y, x) format
    
    if len(positive_coords) == 0:
        return []
    
    # Sample random points from positive pixels
    n_samples = min(num_clicks, len(positive_coords))
    sampled_indices = rng.sample(range(len(positive_coords)), n_samples)
    
    # Convert to (x, y) format expected by SAM
    clicks = [(int(positive_coords[i, 1]), int(positive_coords[i, 0])) 
              for i in sampled_indices]
    
    return clicks


def generate_clicks_per_class(mask, num_clicks_per_class=2, seed=42):
    """
    Generate positive clicks for each class present in the mask.
    
    Args:
        mask: Segmentation mask (H, W) with class indices
        num_clicks_per_class: Number of clicks per class
        seed: Random seed
        
    Returns:
        class_clicks: Dict mapping class_id to list of (x, y) coordinates
    """
    rng = random.Random(seed)
    class_clicks = {}
    
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background
    
    for class_id in unique_classes:
        binary_mask = (mask == class_id).astype(np.uint8)
        clicks = generate_clicks_from_mask(binary_mask, num_clicks_per_class, seed)
        if clicks:
            class_clicks[int(class_id)] = clicks
    
    return class_clicks


def process_clip_sam2(model, processor, clip_frames, class_clicks, device, debug=False):
    """
    Process a video clip with SAM2 using click prompts from first frame.
    
    Args:
        model: SAM2 model
        processor: SAM2 processor
        clip_frames: List of PIL Images
        class_clicks: Dict mapping class_id to click coordinates
        device: Torch device
        debug: Print debug information
        
    Returns:
        predictions: List of predicted masks (H, W) for each frame
    """
    if not clip_frames:
        return []
    
    # Get target dimensions from PIL image (width, height)
    w, h = clip_frames[0].size
    
    if not class_clicks:
        if debug:
            print("WARNING: No class clicks provided!")
        return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
    
    predictions = []
    
    # Organize clicks by object (class) for proper 4-level nesting
    objects_points = []  # Will contain one list of points per object/class
    class_id_list = []  # Track which class each object corresponds to
    
    if debug:
        print(f"Class clicks: {class_clicks}")
    
    for class_id, clicks in sorted(class_clicks.items()):
        if clicks:  # Only add if there are actual clicks
            # Convert clicks to proper format: list of [x, y] pairs
            points_for_object = [[x, y] for x, y in clicks]
            objects_points.append(points_for_object)
            class_id_list.append(class_id)
            if debug:
                print(f"  Class {class_id}: {len(clicks)} clicks")
    
    # If no valid points after filtering, return empty masks
    if not objects_points:
        if debug:
            print("WARNING: No valid points after filtering!")
        return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
    
    # Process all frames with prompts from the first frame
    try:
        with torch.no_grad():
            # For SAM2 video, we need to process one frame at a time
            # because the processor doesn't handle None properly in batch mode
            
            # First, process the first frame with clicks
            inputs = processor(
                [clip_frames[0]],  # Single frame as list
                input_points=[objects_points],  # Points for first frame
                return_tensors="pt"
            ).to(device)
            
            if debug:
                print(f"Processor input keys: {inputs.keys()}")
                for key, val in inputs.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            
            outputs = model(**inputs)
            
            if debug:
                print(f"Model output keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)}")
            
            # Get masks
            if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
                all_masks = outputs.pred_masks.sigmoid().cpu().numpy()
                if debug:
                    print(f"Using pred_masks: shape={all_masks.shape}, min={all_masks.min():.3f}, max={all_masks.max():.3f}")
            elif hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
                all_masks = torch.sigmoid(outputs.mask_logits).cpu().numpy()
                if debug:
                    print(f"Using mask_logits: shape={all_masks.shape}, min={all_masks.min():.3f}, max={all_masks.max():.3f}")
            else:
                if debug:
                    print("ERROR: No pred_masks or mask_logits in outputs!")
                    print(f"Available outputs: {outputs}")
                return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
            
            
            # Extract first frame masks
            if all_masks.ndim == 5:
                # (batch, num_frames, num_masks, H, W)
                frame_masks = all_masks[0, 0]  # (num_masks, H, W)
            elif all_masks.ndim == 4:
                # (batch, num_masks, H, W)
                frame_masks = all_masks[0]  # (num_masks, H, W)
            elif all_masks.ndim == 3:
                # (num_masks, H, W)
                frame_masks = all_masks
            else:
                if debug:
                    print(f"ERROR: Unexpected mask shape: {all_masks.shape}")
                return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
            
            if debug:
                print(f"Frame masks shape: {frame_masks.shape}")
                print(f"Frame masks value range: min={frame_masks.min():.3f}, max={frame_masks.max():.3f}")
            
            # Create first frame prediction
            combined_mask = np.zeros((h, w), dtype=np.int32)
            num_masks_found = 0
            
            for i, class_id in enumerate(class_id_list):
                if i < len(frame_masks):
                    mask = frame_masks[i]
                    # Resize mask to match target size if needed
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    num_pixels = mask_binary.sum()
                    if debug:
                        print(f"  Class {class_id}: mask shape={mask.shape}, pixels after threshold={num_pixels}")
                    if num_pixels > 0:
                        combined_mask[mask_binary > 0] = class_id
                        num_masks_found += 1
            
            if debug:
                unique_vals = np.unique(combined_mask)
                print(f"Combined first frame mask: unique values={unique_vals}, total pixels={(combined_mask > 0).sum()}")
            
            predictions.append(combined_mask)
            
            # For remaining frames, process without points (propagation)
            for frame_idx in range(1, len(clip_frames)):
                if debug:
                    print(f"Processing frame {frame_idx}...")
                
                inputs = processor(
                    [clip_frames[frame_idx]],
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(**inputs)
                
                # Get masks
                if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
                    all_masks = outputs.pred_masks.sigmoid().cpu().numpy()
                elif hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
                    all_masks = torch.sigmoid(outputs.mask_logits).cpu().numpy()
                else:
                    if debug:
                        print(f"  WARNING: No masks returned for frame {frame_idx}")
                    combined_mask = np.zeros((h, w), dtype=np.int32)
                    predictions.append(combined_mask)
                    continue
                
                # Extract frame masks
                if all_masks.ndim == 5:
                    frame_masks = all_masks[0, 0]
                elif all_masks.ndim == 4:
                    frame_masks = all_masks[0]
                elif all_masks.ndim == 3:
                    frame_masks = all_masks
                else:
                    combined_mask = np.zeros((h, w), dtype=np.int32)
                    predictions.append(combined_mask)
                    continue
                
                # Create frame prediction
                combined_mask = np.zeros((h, w), dtype=np.int32)
                for i, class_id in enumerate(class_id_list):
                    if i < len(frame_masks):
                        mask = frame_masks[i]
                        # Resize mask to match target size if needed
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        mask_binary = (mask > 0.5).astype(np.uint8)
                        combined_mask[mask_binary > 0] = class_id
                predictions.append(combined_mask)
    
    except Exception as e:
        print(f"ERROR processing clip with SAM2: {e}")
        import traceback
        traceback.print_exc()
        # Return empty masks for all frames if there's an error
        return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
    
    return predictions


def process_clip_sam3(model, processor, clip_frames, class_clicks, device, debug=False):
    """
    Process a video clip with SAM3 using click prompts from first frame.
    
    Args:
        model: SAM3 model
        processor: SAM3 processor
        clip_frames: List of PIL Images
        class_clicks: Dict mapping class_id to click coordinates
        device: Torch device
        debug: Print debug information
        
    Returns:
        predictions: List of predicted masks (H, W) for each frame
    """
    if not clip_frames:
        return []
    
    # Get target dimensions from PIL image (width, height)
    w, h = clip_frames[0].size
    
    if not class_clicks:
        if debug:
            print("WARNING: No class clicks provided!")
        return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
    
    predictions = []
    
    # Organize clicks by object (class) for proper 4-level nesting
    objects_points = []  # Will contain one list of points per object/class
    class_id_list = []  # Track which class each object corresponds to
    
    if debug:
        print(f"Class clicks: {class_clicks}")
    
    for class_id, clicks in sorted(class_clicks.items()):
        if clicks:  # Only add if there are actual clicks
            # Convert clicks to proper format: list of [x, y] pairs
            points_for_object = [[x, y] for x, y in clicks]
            objects_points.append(points_for_object)
            class_id_list.append(class_id)
            if debug:
                print(f"  Class {class_id}: {len(clicks)} clicks")
    
    # If no valid points after filtering, return empty masks
    if not objects_points:
        if debug:
            print("WARNING: No valid points after filtering!")
        return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
    
    try:
        with torch.no_grad():
            # SAM3 processes video frames together
            # Input points: [image_level][object_level][point_level][coordinates]
            if debug:
                print(f"Calling processor with {len(clip_frames)} frames and {len(objects_points)} objects")
            
            inputs = processor(
                images=clip_frames,
                input_points=[objects_points],  # Wrap in image-level list
                return_tensors="pt"
            ).to(device)
            
            if debug:
                print(f"Processor input keys: {inputs.keys()}")
                for key, val in inputs.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            
            outputs = model(**inputs)
            
            if debug:
                print(f"Model output keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)}")
            
            # Get predicted masks for all frames
            if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
                pred_masks = outputs.pred_masks.sigmoid().cpu().numpy()
                if debug:
                    print(f"Using pred_masks: shape={pred_masks.shape}, min={pred_masks.min():.3f}, max={pred_masks.max():.3f}")
            elif hasattr(outputs, 'mask_logits') and outputs.mask_logits is not None:
                pred_masks = torch.sigmoid(outputs.mask_logits).cpu().numpy()
                if debug:
                    print(f"Using mask_logits: shape={pred_masks.shape}, min={pred_masks.min():.3f}, max={pred_masks.max():.3f}")
            else:
                if debug:
                    print("ERROR: No pred_masks or mask_logits in outputs!")
                return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
            
            # Handle different mask shapes
            if pred_masks.ndim == 4:
                # (B, num_objects, H, W) - normal case
                num_frames = pred_masks.shape[0]
                if debug:
                    print(f"Mask shape is 4D: (batch={pred_masks.shape[0]}, objects={pred_masks.shape[1]}, H={pred_masks.shape[2]}, W={pred_masks.shape[3]})")
            elif pred_masks.ndim == 3:
                # (num_objects, H, W) - might be single frame or different format
                num_frames = 1
                pred_masks = pred_masks[np.newaxis, ...]  # Add batch dimension
                if debug:
                    print(f"Mask shape is 3D, treating as single frame")
            else:
                if debug:
                    print(f"ERROR: Unexpected mask shape: {pred_masks.shape}")
                return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
            
            for frame_idx in range(len(clip_frames)):
                if frame_idx < len(pred_masks):
                    masks = pred_masks[frame_idx]  # (num_objects, H, W)
                else:
                    masks = np.zeros((len(class_id_list), h, w))
                    if debug:
                        print(f"  Frame {frame_idx}: No masks in pred_masks, using zeros")
                
                combined_mask = np.zeros((h, w), dtype=np.int32)
                
                for i, class_id in enumerate(class_id_list):
                    if i < len(masks):
                        mask = masks[i]
                        # Resize mask to match target size if needed
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        mask_binary = (mask > 0.5).astype(np.uint8)
                        combined_mask[mask_binary > 0] = class_id
                
                predictions.append(combined_mask)
    
    except Exception as e:
        print(f"ERROR processing clip with SAM3: {e}")
        import traceback
        traceback.print_exc()
        return [np.zeros((h, w), dtype=np.int32) for _ in clip_frames]
    
    return predictions


def evaluate_sam(model, processor, test_loader, device, num_classes, num_clicks_per_class, 
                 is_sam3=False, seed=42, debug=False):
    """
    Evaluate SAM model on ATLAS test set.
    
    Args:
        model: SAM model
        processor: SAM processor
        test_loader: DataLoader for test set
        device: Torch device
        num_classes: Number of classes (including background)
        num_clicks_per_class: Number of clicks to generate per class
        is_sam3: Whether using SAM3 (vs SAM2)
        seed: Random seed
        debug: Print debug information
    """
    print(f"\nEvaluating SAM{'3' if is_sam3 else '2'} model...")
    if debug:
        print("DEBUG MODE ENABLED")
    
    # Track metrics per class
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)
    
    # AP tracking per clip
    clip_ap = {}
    
    # Process clips
    current_clip = None
    clip_frames = []
    clip_gts = []
    clip_class_clicks = None
    
    mean = torch.tensor(test_loader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(test_loader.dataset.std).view(3, 1, 1)
    
    def process_clip(clip_id, frames_pil, gts, class_clicks):
        """Process a complete clip and compute metrics."""
        ap_evaluator = SegmentationAPEvaluator()
        
        # Generate predictions for the clip
        if debug:
            print(f"\nProcessing clip: {clip_id}")
        
        if is_sam3:
            predictions = process_clip_sam3(model, processor, frames_pil, class_clicks, device, debug=debug)
        else:
            predictions = process_clip_sam2(model, processor, frames_pil, class_clicks, device, debug=debug)
        
        # Evaluate each frame
        classes_to_eval = range(1, num_classes + 1)
        
        if debug:
            print(f"Processing {len(predictions)} predictions")
        
        for frame_idx, (pred_mask, gt_mask) in enumerate(zip(predictions, gts)):
            gt_np = gt_mask.squeeze().cpu().numpy().astype(np.int32)
            
            if debug and frame_idx == 0:
                unique_gt = np.unique(gt_np)
                unique_pred = np.unique(pred_mask)
                print(f"  Frame 0: GT classes={unique_gt}, Pred classes={unique_pred}")
            
            # Resize prediction to match GT if needed
            if pred_mask.shape != gt_np.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_np.shape[1], gt_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
            
            # Compute metrics per class
            for c in classes_to_eval:
                iou_c, dice_c = compute_class_metrics(pred_mask, gt_np, c, ignore_index=255)
                if iou_c is not None:
                    class_ious[c].append(iou_c)
                    class_dices[c].append(dice_c)
            
            # AP computation
            gt_binary = (gt_np > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            max_score = 1.0  # SAM outputs are binary, use constant confidence
            ap_evaluator.add_frame(gt_binary, pred_binary, max_score)
        
        clip_ap[clip_id] = ap_evaluator.evaluate()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"]
            gt_masks = batch["mask"].to(device)
            
            for i in range(images.shape[0]):
                clip_id = f"{batch['procedure'][i]}/{batch['video'][i]}/{batch['clip'][i]}"
                
                # Initialize new clip
                if current_clip is None:
                    current_clip = clip_id
                
                # Process completed clip
                if clip_id != current_clip:
                    process_clip(current_clip, clip_frames, clip_gts, clip_class_clicks)
                    clip_frames = []
                    clip_gts = []
                    clip_class_clicks = None
                    current_clip = clip_id
                
                # Denormalize and convert to PIL
                img_tensor = images[i]
                img_denorm = img_tensor * std + mean
                img_denorm = img_denorm.clamp(0.0, 1.0)
                img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                clip_frames.append(img_pil)
                clip_gts.append(gt_masks[i])
                
                # Generate clicks from first frame of clip
                if clip_class_clicks is None:
                    first_gt = gt_masks[i].squeeze().cpu().numpy().astype(np.int32)
                    clip_class_clicks = generate_clicks_per_class(
                        first_gt, 
                        num_clicks_per_class=num_clicks_per_class,
                        seed=seed
                    )
        
        # Process final clip
        if clip_frames:
            process_clip(current_clip, clip_frames, clip_gts, clip_class_clicks)
    
    # Aggregate metrics
    classes_to_eval = range(1, num_classes + 1)
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
    
    # Print results
    print("\n" + "=" * 40)
    print("SAM Evaluation Results")
    print("=" * 40)
    print(f"{'Class ID':<10} | {'IoU':<10} | {'Dice':<10}")
    print("-" * 40)
    for c in classes_to_eval:
        print(f"Class {c:<4} | {final_per_class_iou[c]:.4f}     | {final_per_class_dice[c]:.4f}")
    print("-" * 40)
    print(f"{'OVERALL':<10} | {mIoU:.4f}     | {mDice:.4f}")
    print(f"{'mAP':<10} | {AP_total:.4f}")
    print(f"{'AP@50':<10} | {AP_50:.4f}")
    print(f"{'AP@75':<10} | {AP_75:.4f}")
    print("=" * 40 + "\n")
    
    return {
        "mIoU": mIoU,
        "Dice": mDice,
        "AP": AP_total,
        "AP50": AP_50,
        "AP75": AP_75,
        "per_class_iou": final_per_class_iou,
        "per_class_dice": final_per_class_dice,
    }


def save_visualizations(model, processor, test_loader, device, output_dir, num_samples, 
                       num_clicks_per_class, is_sam3, seed):
    """Save random visualization samples."""
    if num_samples <= 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    samples_saved = 0
    
    mean = torch.tensor(test_loader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(test_loader.dataset.std).view(3, 1, 1)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting visualizations"):
            if samples_saved >= num_samples:
                break
            
            images = batch["image"]
            gt_masks = batch["mask"].to(device)
            
            # Process only first frame with GT clicks
            for i in range(min(images.shape[0], num_samples - samples_saved)):
                # Denormalize
                img_tensor = images[i]
                img_denorm = img_tensor * std + mean
                img_denorm = img_denorm.clamp(0.0, 1.0)
                img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                # Get GT
                gt_np = gt_masks[i].squeeze().cpu().numpy().astype(np.int32)
                
                # Generate clicks
                class_clicks = generate_clicks_per_class(
                    gt_np, 
                    num_clicks_per_class=num_clicks_per_class,
                    seed=seed + samples_saved
                )
                
                # Generate prediction
                if is_sam3:
                    predictions = process_clip_sam3(model, processor, [img_pil], class_clicks, device)
                else:
                    predictions = process_clip_sam2(model, processor, [img_pil], class_clicks, device)
                
                pred_np = predictions[0]
                
                # Create visualization
                gt_overlay = apply_mask_overlay(img_np, gt_np, color_palette)
                pred_overlay = apply_mask_overlay(img_np, pred_np, color_palette)
                
                # Mark clicks on image
                img_with_clicks = img_np.copy()
                for clicks in class_clicks.values():
                    for x, y in clicks:
                        cv2.circle(img_with_clicks, (x, y), 5, (0, 255, 0), -1)
                
                row = np.concatenate([img_with_clicks, gt_overlay, pred_overlay], axis=1)
                
                out_path = os.path.join(output_dir, f"sample_{samples_saved:02d}.png")
                cv2.imwrite(out_path, row[:, :, ::-1])
                
                samples_saved += 1


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
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
    # SAM models need raw images, minimal preprocessing
    test_transform = T.Compose([
        T.Resize(args.img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.img_size),
    ])
    
    test_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="test",
        transform=test_transform,
        frame_percentage=args.test_percentage,
        seed=args.seed,
        normalization_type="none",
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one frame at a time for click generation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Test set: {len(test_dataset)} frames")
    
    # Evaluate
    metrics = evaluate_sam(
        model=model,
        processor=processor,
        test_loader=test_loader,
        device=device,
        num_classes=args.num_classes,
        num_clicks_per_class=args.num_clicks,
        is_sam3=is_sam3,
        seed=args.seed,
        debug=args.debug,
    )
    
    # Save visualizations
    if args.visualize_samples > 0:
        output_dir = os.path.join(args.visualize_dir, args.model)
        save_visualizations(
            model=model,
            processor=processor,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            num_samples=args.visualize_samples,
            num_clicks_per_class=args.num_clicks,
            is_sam3=is_sam3,
            seed=args.seed,
        )
        print(f"\nSaved {args.visualize_samples} visualizations to: {output_dir}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs("test_results", exist_ok=True)
        output_path = f"test_results/{args.model}_results.json"
    
    results = {
        "model": args.model,
        "num_clicks_per_class": args.num_clicks,
        "metrics": metrics,
        "config": vars(args),
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM2/SAM3 on ATLAS dataset")
    
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
        help="Number of classes in dataset (including background)"
    )
    parser.add_argument(
        "--num_clicks",
        type=int,
        default=3,
        help="Number of positive clicks to generate per class"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1024,
        help="Image size for SAM models (default: 1024)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (must be 1 for click-based evaluation)"
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
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON"
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
        help="Number of samples to visualize (0 to disable)"
    )
    parser.add_argument(
        "--visualize_dir",
        type=str,
        default="outputs/sam_visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output to diagnose issues"
    )
    
    args = parser.parse_args()
    
    if args.batch_size != 1:
        print("Warning: batch_size must be 1 for click-based evaluation. Setting to 1.")
        args.batch_size = 1
    
    metrics = main(args)

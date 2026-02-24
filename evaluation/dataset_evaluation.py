import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from .metrics import compute_class_metrics, SegmentationAPEvaluator


def evaluate_model(model, dataloader, device, num_classes):
    """
    Args:
        model: The model to evaluate.
        dataloader: The dataloader for evaluation.
        device: Device to use (cuda or cpu).
        num_classes: Number of classes.
        threshold: Threshold for binary decisions.
    """
    model.eval()
    
    ignore_index = 255  # Standard ignore value used in Cityscapes and other datasets

    # Track scores per class across the entire dataset
    # format: {class_id: [score_img1, score_img2, ...]}
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)

    # We still keep the AP logic as it was
    clip_ap = {}
    current_clip = None
    ap_evaluator = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)
            
            classes_to_eval = range(1, num_classes+1)  # Skip background 0

            for i in range(len(images)):
                # --- AP Handling ---
                clip_id = f"{batch['procedure'][i]}/{batch['video'][i]}/{batch['clip'][i]}"
                if current_clip != clip_id:
                    if ap_evaluator is not None:
                        clip_ap[current_clip] = ap_evaluator.evaluate()
                    current_clip = clip_id
                    ap_evaluator = SegmentationAPEvaluator()

                pred_np = preds[i, 0].cpu().numpy()
                gt_np = gt_masks[i].cpu().numpy().squeeze()
                
                gt_np_masked = gt_np

                # --- Metric Collection (The part you wanted) ---
                for c in classes_to_eval:
                    iou_c, dice_c = compute_class_metrics(pred_np, gt_np_masked, c, ignore_index=ignore_index)

                    if iou_c is not None:
                        class_ious[c].append(iou_c)
                        class_dices[c].append(dice_c)

                # Binary AP logic with improved confidence scoring
                gt_binary = (gt_np > 0).astype(np.uint8)
                pred_binary = (pred_np > 0).astype(np.uint8)
                
                # Compute per-frame confidence score: average probability of predicted class in predicted mask
                # Move to CPU to avoid large GPU tensor allocation
                probs_np = probs[i].cpu().numpy()  # Shape: (num_classes, H, W)
                preds_np = preds[i, 0].cpu().numpy()  # Shape: (H, W) - class index per pixel
                
                # Use advanced indexing to get prob of predicted class for each pixel
                H, W = preds_np.shape
                h_indices, w_indices = np.mgrid[0:H, 0:W]
                pred_class_probs = probs_np[preds_np, h_indices, w_indices]  # Shape: (H, W)
                pred_mask = (pred_np > 0)  # Mask of non-background predictions
                
                if pred_mask.sum() > 0:
                    score = pred_class_probs[pred_mask].mean()
                else:
                    # If no mask predicted, use max probability as fallback
                    score = probs_np.max()
                
                ap_evaluator.add_frame(gt_binary, pred_binary, score)

        if ap_evaluator is not None:
            clip_ap[current_clip] = ap_evaluator.evaluate()

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

    # --- Print Detailed Report ---
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


def evaluate_atlas_temporal(
    model,
    test_loader,
    device,
    num_classes,
    use_query_propagation=True,
):
    """
    Evaluate ATLAS model with temporal processing on a clip basis.
    
    Processes each clip frame-by-frame in an online manner:
    - Query embeddings are propagated between frames within the same clip
    - Query embeddings are reset when moving to a new clip
    - Batch size should be 1 for proper temporal state management
    
    Args:
        model: ATLAS model with temporal capabilities
        test_loader: DataLoader yielding individual frames with clip metadata
        device: torch device
        num_classes: number of segmentation classes
        use_query_propagation: whether to propagate query embeddings between frames
    
    Returns:
        Dictionary of metrics (mIoU, Dice, AP, AP50, AP75, per_class metrics)
    """
    model.eval()
    
    ignore_index = 255
    
    # Track metrics per class
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)
    
    # Track AP per clip
    clip_ap = {}
    current_clip = None
    ap_evaluator = None
    
    # Track query embeddings for propagation
    prev_query_embed = None
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating ATLAS temporal"):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            
            # Get clip information
            procedure = batch["procedure"][0] if isinstance(batch["procedure"], list) else batch["procedure"]
            video = batch["video"][0] if isinstance(batch["video"], list) else batch["video"]
            clip_id = batch["clip"][0] if isinstance(batch["clip"], list) else batch["clip"]
            
            clip_key = f"{procedure}/{video}/{clip_id}"
            
            # Reset query embeddings when entering a new clip
            if current_clip != clip_key:
                if ap_evaluator is not None:
                    clip_ap[current_clip] = ap_evaluator.evaluate()
                current_clip = clip_key
                ap_evaluator = SegmentationAPEvaluator()
                prev_query_embed = None  # Reset queries for new clip
            
            # Forward pass with query propagation
            outputs = model(
                images,
                prev_query_embed=prev_query_embed if use_query_propagation else None,
                return_query_embedding=use_query_propagation,
            )
            
            # Unpack outputs
            if use_query_propagation:
                mask_logits_per_block, class_logits_per_block, procedure_logits_per_block, query_embed = outputs
                prev_query_embed = query_embed  # Store for next frame within same clip
            else:
                mask_logits_per_block, class_logits_per_block, procedure_logits_per_block = outputs
            
            # Get predictions from final block
            mask_logits = mask_logits_per_block[-1]  # (B, num_q, H, W)
            class_logits = class_logits_per_block[-1]  # (B, num_q, num_classes+1)
            
            # Convert query-level predictions to per-pixel predictions
            # Shape: (B, num_classes, H, W)
            per_pixel_logits = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                class_logits.softmax(dim=-1)[..., :-1]  # Exclude background class
            )
            
            # Get predictions
            preds = torch.argmax(per_pixel_logits, dim=1)  # (B, H, W)
            
            # Evaluate each frame in the batch
            classes_to_eval = range(1, num_classes + 1)  # Skip background 0
            
            for b in range(images.shape[0]):
                pred_np = preds[b].cpu().numpy()
                gt_np = gt_masks[b].cpu().numpy().squeeze()
                
                # --- Compute per-class metrics (IoU, Dice) ---
                for c in classes_to_eval:
                    iou_c, dice_c = compute_class_metrics(
                        pred_np, gt_np, c, ignore_index=ignore_index
                    )
                    
                    if iou_c is not None:
                        class_ious[c].append(iou_c)
                        class_dices[c].append(dice_c)
                
                # --- Compute AP metrics (binary) ---
                gt_binary = (gt_np > 0).astype(np.uint8)
                pred_binary = (pred_np > 0).astype(np.uint8)
                
                # Compute per-frame confidence score
                probs_np = class_logits[b].softmax(dim=-1)[:, :-1].max(dim=-1)[0].cpu().numpy()  # (num_q,)
                mask_prob = mask_logits[b].sigmoid().cpu().numpy()  # (num_q, H, W)
                
                # Average confidence across queries and spatial dimensions
                if mask_prob.sum() > 0:
                    score = (mask_prob * probs_np[:, None, None]).sum() / mask_prob.sum()
                else:
                    score = 0.5
                
                ap_evaluator.add_frame(gt_binary, pred_binary, score)
    
    # Finalize last clip's AP
    if ap_evaluator is not None:
        clip_ap[current_clip] = ap_evaluator.evaluate()
    
    # --- Compute final metrics ---
    final_per_class_iou = {}
    final_per_class_dice = {}
    
    for c in classes_to_eval:
        if len(class_ious[c]) > 0:
            final_per_class_iou[c] = np.mean(class_ious[c])
            final_per_class_dice[c] = np.mean(class_dices[c])
        else:
            final_per_class_iou[c] = 0.0
            final_per_class_dice[c] = 0.0
    
    # Compute overall metrics
    mIoU = np.mean(list(final_per_class_iou.values()))
    mDice = np.mean(list(final_per_class_dice.values()))
    
    # AP Metrics
    AP_total = np.mean([v["AP"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_50 = np.mean([v["AP50"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_75 = np.mean([v["AP75"] for v in clip_ap.values()]) if clip_ap else 0.0
    
    # Print detailed report
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
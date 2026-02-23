import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from .metrics import compute_class_metrics, SegmentationAPEvaluator


def evaluate_model(model, dataloader, device, num_classes, threshold=0.5, remap_classes=False, ignore_background=False):
    """
    Args:
        remap_classes: If True, add 1 to all predictions (for models trained without background class).
                      This remaps predictions back to original ATLAS class indices.
        ignore_background: If True, mask out background pixels (GT==0) from evaluation.
                          Fair for models that never learned background (e.g., EOMT).
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
            
            # Remap classes if needed (e.g., for EOMT trained without background class)
            if remap_classes:
                preds = preds + 1  # Shift predictions: 0->1, 1->2, etc.
            
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
                
                # Mask background pixels if requested (e.g., for EOMT)
                # Set GT background pixels to ignore_index so they're excluded from metrics
                if ignore_background:
                    gt_np_masked = gt_np.copy()
                    gt_np_masked[gt_np == 0] = ignore_index
                else:
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
                pred_class_probs = probs[i, preds[i, 0]]  # Get prob of predicted class for each pixel
                pred_mask = (pred_np > 0)  # Mask of non-background predictions
                
                if pred_mask.sum() > 0:
                    score = pred_class_probs[pred_mask].mean().item()
                else:
                    # If no mask predicted, use max probability as fallback
                    score = probs[i].max().item()
                
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
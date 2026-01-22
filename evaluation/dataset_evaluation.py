import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from .metrics import compute_iou, compute_dice, SegmentationAPEvaluator


def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()

    clip_iou = defaultdict(list)
    clip_dice = defaultdict(list)
    clip_ap = {}

    current_clip = None
    ap_evaluator = None

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            # model outputs: [B, 1, H, W] or [B, C, H, W]
            outputs = model(images)

            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).long()
                scores = probs.mean(dim=(2,3))  # global confidence
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True)
                scores = probs.max(dim=1)[0].mean(dim=(1,2))

            for i in range(len(images)):
                clip_id = f"{batch['procedure'][i]}/{batch['video'][i]}/{batch['clip'][i]}"

                # New clip → finalize previous AP
                if current_clip != clip_id:
                    if ap_evaluator is not None:
                        clip_ap[current_clip] = ap_evaluator.evaluate()

                    current_clip = clip_id
                    ap_evaluator = SegmentationAPEvaluator()

                pred = preds[i,0].cpu().numpy()
                gt = gt_masks[i,0].cpu().numpy() if gt_masks.shape[1] == 1 else gt_masks[i].sum(0).cpu().numpy() > 0

                pred = (pred > 0).astype(np.uint8)
                gt = (gt > 0).astype(np.uint8)

                iou = compute_iou(pred, gt)
                dice = compute_dice(pred, gt)

                clip_iou[clip_id].append(iou)
                clip_dice[clip_id].append(dice)

                ap_evaluator.add_frame(gt, pred, scores[i].item())

        # last clip
        if ap_evaluator is not None:
            clip_ap[current_clip] = ap_evaluator.evaluate()

    # Aggregate per clip
    clip_iou_mean = [np.mean(v) for v in clip_iou.values()]
    clip_dice_mean = [np.mean(v) for v in clip_dice.values()]

    AP = [v["AP"] for v in clip_ap.values()]
    AP50 = [v["AP50"] for v in clip_ap.values()]
    AP75 = [v["AP75"] for v in clip_ap.values()]

    print("\n===== Final Evaluation =====")
    print(f"mIoU (clip):   {np.mean(clip_iou_mean):.4f}")
    print(f"Dice (clip):   {np.mean(clip_dice_mean):.4f}")
    print(f"AP:            {np.mean(AP):.4f}")
    print(f"AP50:          {np.mean(AP50):.4f}")
    print(f"AP75:          {np.mean(AP75):.4f}")

    return {
        "mIoU": np.mean(clip_iou_mean),
        "Dice": np.mean(clip_dice_mean),
        "AP": np.mean(AP),
        "AP50": np.mean(AP50),
        "AP75": np.mean(AP75),
    }

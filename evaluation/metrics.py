import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
import os

def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return 2 * intersection / total

def mask_to_rle(binary_mask):
    """
    binary_mask: HxW numpy array (0,1)
    """
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # json serializable
    return rle


class SegmentationAPEvaluator:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.predictions = []
        self.image_id = 0
        self.ann_id = 0

    def add_frame(self, gt_mask, pred_mask, score):
        """
        gt_mask, pred_mask: HxW binary numpy arrays
        score: confidence score for prediction
        """
        h, w = gt_mask.shape
        img_id = self.image_id
        self.image_id += 1

        # image entry
        self.images.append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": f"{img_id}.png"
        })

        # GT annotation
        gt_rle = mask_to_rle(gt_mask)
        area = gt_mask.sum()

        self.annotations.append({
            "id": self.ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": gt_rle,
            "area": float(area),
            "bbox": [0, 0, w, h],
            "iscrowd": 0
        })
        self.ann_id += 1

        # Prediction
        pred_rle = mask_to_rle(pred_mask)

        self.predictions.append({
            "image_id": img_id,
            "category_id": 1,
            "segmentation": pred_rle,
            "score": float(score)
        })

    def evaluate(self):
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": 1, "name": "object"}]
        }
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self.predictions)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
        }

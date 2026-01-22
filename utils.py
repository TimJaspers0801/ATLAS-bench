import torch
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from typing import List, Optional


color_palette = {
    0: (0, 0, 0),  # Background
    1: (255, 255, 255), 2: (0, 0, 255), 3: (255, 0, 0), 4: (255, 255, 0), 5: (0, 255, 0),
    6: (0, 200, 100), 7: (200, 150, 100), 8: (250, 150, 100), 9: (255, 200, 100),
    10: (180, 0, 0), 11: (0, 0, 180), 12: (150, 100, 50), 13: (0, 255, 255),
    14: (0, 200, 255), 15: (0, 100, 255), 16: (255, 150, 50), 17: (255, 220, 200),
    18: (200, 100, 200), 19: (144, 238, 144), 20: (247, 255, 0), 21: (255, 206, 27),
    22: (200, 0, 200), 23: (255, 0, 150), 24: (255, 100, 200), 25: (200, 100, 255),
    26: (150, 0, 100), 27: (255, 200, 255), 28: (150, 100, 75), 29: (200, 0, 150),
    30: (100, 100, 100), 31: (255, 150, 255), 32: (100, 200, 255), 33: (150, 200, 255),
    34: (0, 150, 255), 35: (255, 100, 100), 36: (200, 200, 255), 37: (100, 100, 255),
    38: (0, 255, 150), 39: (255, 255, 100),  40: (150, 150, 150),
    41: (50, 50, 50), 43: (173, 216, 230),
    44: (255, 140, 0), 45: (252, 186, 3), 46: (0, 80, 100)
}

bgr_palette = {k: (v[2], v[1], v[0]) for k, v in color_palette.items()}

def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(checkpoint["model"], strict=False)  # Adjust keys if needed
    print(msg)

def apply_mask(image, mask):
    """Overlay masks emphasizing thicker edges on a per-class basis."""
    color_mask = np.zeros_like(image)
    for class_id, color in bgr_palette.items():
        color_mask[mask == class_id] = color

    mask_area = (mask != 255).astype(np.uint8)
    all_edges = np.zeros_like(mask_area, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)

    for class_id in bgr_palette.keys():
        class_mask = (mask == class_id).astype(np.uint8) * 255
        if class_mask.sum() == 0:
            continue
        edges = cv2.Canny(class_mask, 50, 150)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2).astype(bool)
        all_edges |= edges_dilated

    blended = image.copy()
    inner_area = mask_area.astype(bool) & (~all_edges)
    blended[inner_area] = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)[inner_area]
    blended[all_edges] = cv2.addWeighted(image, 0.1, color_mask, 0.9, 0)[all_edges]

    return blended

def visualize_predictions(images, masks, outputs, wandb, max_samples=12):
    preds = outputs.softmax(dim=1)
    preds = torch.argmax(preds, dim=1)

    # Denormalization constants
    mean = torch.tensor([0.46888983, 0.29536288, 0.28712815], device=images.device).view(3, 1, 1)
    std = torch.tensor([0.24689102, 0.21034359, 0.21188641], device=images.device).view(3, 1, 1)

    def denormalize(tensor):
        return tensor * std + mean

    imgs = []
    gt_overlays = []
    pred_overlays = []

    for idx, (img, gt, pred) in enumerate(zip(images, masks, preds)):
        if idx >= max_samples:
            break

        # Denormalize and convert image to NumPy (HWC, uint8)
        img_denorm = denormalize(img)
        img_clamped = torch.clamp(img_denorm, 0, 1)  # Clamp to [0, 1] range
        img_np = TF.to_pil_image(img_clamped.cpu()).convert("RGB")
        img_np = np.array(img_np)

        gt_np = gt.cpu().numpy().astype(np.uint8)
        pred_np = pred.cpu().numpy().astype(np.uint8)

        imgs.append(img_np)
        gt_overlays.append(apply_mask(img_np, gt_np))
        pred_overlays.append(apply_mask(img_np, pred_np))

    # If fewer than max_samples, pad with blank images
    def pad_list(lst):
        blank_img = np.zeros_like(imgs[0]) if imgs else np.zeros((256, 256, 3), dtype=np.uint8)
        while len(lst) < max_samples:
            lst.append(blank_img)
        return lst

    imgs = pad_list(imgs)
    gt_overlays = pad_list(gt_overlays)
    pred_overlays = pad_list(pred_overlays)

    # Arrange images in 4 columns per row
    cols = 4

    def stack_row(images_row):
        return np.hstack(images_row)

    top_row = stack_row(imgs[:cols])
    middle_row = stack_row(gt_overlays[:cols])
    bottom_row = stack_row(pred_overlays[:cols])

    # Combine rows vertically
    grid_image = np.vstack([top_row, middle_row, bottom_row])

    wandb.log({"predictions_grid": wandb.Image(grid_image)})

def compute_per_class_dice(preds, targets, num_classes):
    per_class_dice = {}

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        if target_inds.sum() == 0:
            continue  # Skip class not present in ground truth

        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()

        dice = 2 * intersection / union if union > 0 else 0.0
        per_class_dice[cls] = dice

    return per_class_dice

class_names_surgenetseg75k  = {
    0: "Background",
    1: "Tools/camera",
    2: "Vein (major)",
    3: "Artery (major)",
    4: "Nerve (major)",
    5: "Small intestine",
    6: "Colon/rectum",
    7: "Abdominal wall",
    8: "Diaphragm",
    9: "Omentum",
    10: "Aorta",
    11: "Vena cava",
    12: "Liver",
    13: "Cystic duct",
    14: "Gallbladder",
    15: "Hepatic vein",
    16: "Hepatic ligament",
    17: "Cystic plate",
    18: "Stomach",
    19: "Ductus choledochus",
    20: "Mesenterium",
    21: "Ductus hepaticus",
    22: "Spleen",
    23: "Uterus",
    24: "Ovary",
    25: "Oviduct",
    26: "Prostate",
    27: "Urethra",
    28: "Ligated plexus",
    29: "Seminal vesicles",
    30: "Catheter",
    31: "Bladder",
    32: "Kidney",
    33: "Lung",
    34: "Airway (bronchus/trachea)",
    35: "Esophagus",
    36: "Pericardium",
    37: "V azygos",
    38: "Thoracic duct",
    39: "Nerves",
    40: "Ureter",
    41: "Non anatomical structures",
    42: "Excluded frames",
    43: "Mesocolon",
    44: "Adrenal Gland",
    45: "Pancreas",
}

class_names_cholecseg8k = {
    0: 'Black Background',
    1: 'Abdominal Wall',
    2: 'Liver',
    3: 'Gastrointestinal Tract',
    4: 'Fat',
    5: 'Grasper',
    6: 'Connective Tissue',
    7: 'L-hook Electrocautery',
    8: 'Gallbladder'
}

# Remapping from SurgeNetSeg-75k→CholecSeg
orig_to_ch8k = {
    0: 0,    # Background
    7: 1,    # Abdominal Wall
    12: 2,   # Liver
    5: 3,    # GI Tract ← Small intestine
    6: 3,    # GI Tract ← Colon/rectum
    18: 3,   # GI Tract ← Stomach
    9: 4,    # Omentum ← Fat
    20: 4,   # Mesenterium ← Fat
    1: 5,    # Grasper ← Tools/Camera
    14: 8    # Gallbladder
}

# helper: map a mask tensor in-place
def remap_cholecseg_mask(mask: torch.Tensor):
    # mask is H×W with values in 0–45
    mapped = torch.zeros_like(mask)
    for orig_lbl, new_lbl in orig_to_ch8k.items():
        mapped[mask == orig_lbl] = new_lbl
    return mapped


def collapse_cholecseg_preds(preds: torch.Tensor) -> torch.Tensor:
    """
    Merge L-hook (7) into Grasper (5) in the model’s predictions.
    Expects preds to be an H×W tensor with values in {0,…,8}.
    """
    collapsed = preds.clone()
    collapsed[preds == 7] = 5
    return collapsed


def compute_mVC_for_clip(preds_list: List[np.ndarray],
                         gts_list: List[np.ndarray],
                         n: int,
                         background_label: int = 0,
                         skip_empty_windows: bool = True) -> Optional[float]:
    """
    Compute VC_n for a single clip (sliding windows).
    Returns clip-level VC_n (mean over windows) or None if the clip
    had no valid windows (all windows had empty gt_common after excluding background).

    preds_list and gts_list should be lists/arrays of 2D integer label maps (H x W),
    in temporal order.
    """
    C = len(preds_list)
    if C < n:
        return None

    window_vcs = []
    total_windows = C - n + 1

    for i in range(total_windows):
        wp = np.stack(preds_list[i:i + n], axis=0)  # (n, H, W)
        wg = np.stack(gts_list[i:i + n], axis=0)    # (n, H, W)

        gt_equal = np.all(wg == wg[0:1], axis=0)    # (H, W)
        gt_label_first = wg[0]                      # (H, W)
        gt_common_nonbg = gt_equal & (gt_label_first != background_label)
        denom = int(gt_common_nonbg.sum())

        if denom == 0:
            if skip_empty_windows:
                continue   # skip window as invalid
            else:
                # Decide how you want to treat empty windows; here we treat as 0
                window_vcs.append(0.0)
                continue

        pred_common = np.all(wp == wp[0:1], axis=0)
        num = int(np.logical_and(gt_common_nonbg, pred_common).sum())

        ratio = float(num) / float(denom)
        window_vcs.append(ratio)

    if len(window_vcs) == 0:
        return None

    return float(np.mean(window_vcs))

def compute_mVC_all_clips(clip_preds, clip_gts, ns=(8, 16),
                          background_label=0, debug=False, max_debug_windows=3):
    """
    clip_preds, clip_gts: dict clip_id -> list of 2D numpy arrays (in temporal order)
    returns dict {n: mVC_n}
    mVC_n computed as mean of per-clip VC_n (skip clips with no valid windows)
    Args:
        background_label (int): label value considered background and excluded from denominator
        debug (bool): if True, print per-clip debug info
    """
    results = {}
    for n in ns:
        clip_vcs = []
        for clip_id in clip_preds.keys():
            preds_list = clip_preds[clip_id]
            gts_list = clip_gts[clip_id]

            # small sanity prints when debug True
            if debug:
                print(f"\n[mVC-all-debug] Processing clip_id='{clip_id}', frames={len(preds_list)} for n={n}")

            vc_clip = compute_mVC_for_clip(preds_list, gts_list, n,
                                           background_label=background_label)
            if vc_clip is not None:
                clip_vcs.append(vc_clip)

        if clip_vcs:
            results[n] = float(np.mean(clip_vcs))
        else:
            results[n] = float('nan')

    return results


def compute_mVC_for_clip_per_class(preds_list: list,
                                   gts_list: list,
                                   n: int,
                                   class_id: int,
                                   background_label: int = 0,
                                   skip_empty_windows: bool = True) -> Optional[float]:
    """
    Compute VC_n for a single clip *for a specific class* (sliding windows).
    Returns clip-level VC_n (mean over windows) or None if the clip
    had no valid windows (all windows had empty gt for this class after excluding background).

    preds_list and gts_list should be lists/arrays of 2D integer label maps (H x W),
    in temporal order.
    """
    C = len(preds_list)
    if C < n:
        return None

    window_vcs = []
    total_windows = C - n + 1

    for i in range(total_windows):
        wp = np.stack(preds_list[i:i + n], axis=0)  # (n, H, W)
        wg = np.stack(gts_list[i:i + n], axis=0)    # (n, H, W)

        # find pixels that have the same GT label across the window
        gt_equal = np.all(wg == wg[0:1], axis=0)    # (H, W)
        gt_label_first = wg[0]                      # (H, W)

        # denom: pixels that are constant across the window and equal to the class_id
        gt_common_class = gt_equal & (gt_label_first == class_id)
        denom = int(gt_common_class.sum())

        if denom == 0:
            if skip_empty_windows:
                continue   # skip window as invalid
            else:
                # Treat empty windows as 0 if requested
                window_vcs.append(0.0)
                continue

        # pred_common: pixels where prediction is constant across the window
        pred_common = np.all(wp == wp[0:1], axis=0)
        num = int(np.logical_and(gt_common_class, pred_common).sum())

        ratio = float(num) / float(denom)
        window_vcs.append(ratio)

    if len(window_vcs) == 0:
        return None

    return float(np.mean(window_vcs))


def compute_mVC_all_clips_per_class(clip_preds: dict,
                                    clip_gts: dict,
                                    ns=(8, 16),
                                    class_ids=None,
                                    background_label=0,
                                    debug=False):
    """
    Compute mVC_n for each class separately.

    Returns dict: {n: {class_id: mVC_n_for_class}}
    mVC_n_for_class computed as mean of per-clip VC_n (skip clips with no valid windows for that class)
    """
    if class_ids is None:
        # auto infer class ids from data
        class_ids = set()
        for clip_id, gts in clip_gts.items():
            for g in gts:
                class_ids.update(np.unique(g).tolist())
        class_ids = sorted(list(class_ids))

    results = {}
    for n in ns:
        results[n] = {}
        for cls in class_ids:
            clip_vcs = []
            for clip_id in clip_preds.keys():
                preds_list = clip_preds[clip_id]
                gts_list = clip_gts[clip_id]

                if debug:
                    print(f"[mVC-per-class-debug] clip={clip_id}, n={n}, class={cls}, frames={len(preds_list)}")

                vc_clip = compute_mVC_for_clip_per_class(preds_list, gts_list, n,
                                                        class_id=cls,
                                                        background_label=background_label)
                if vc_clip is not None:
                    clip_vcs.append(vc_clip)

            if clip_vcs:
                results[n][cls] = float(np.mean(clip_vcs))
            else:
                results[n][cls] = float('nan')

    return results
import numpy as np
import torch
import cv2
from collections import defaultdict



def decode_mask(mask, color_palette):
    """
    Convert integer mask (H, W) → RGB mask (H, W, 3)
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in color_palette.items():
        if isinstance(color, list):
            color = color[0]  # pick first if ambiguous
        rgb[mask == cls] = color

    return rgb
def denormalize(img, mean, std):
    """
    img: Tensor (3, H, W) normalized
    returns: uint8 HWC image
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    img = img * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def apply_mask_overlay(image, mask, palette, ignore_index=255):
    """
    image : uint8 HxWx3 (RGB)
    mask  : HxW (int)
    palette: dict[class_id] -> (R,G,B)
    """

    color_mask = np.zeros_like(image)
    mask = mask.squeeze()

    for class_id, color in palette.items():
        color_mask[mask == class_id] = color

    mask_area = (mask != ignore_index).astype(np.uint8)
    all_edges = np.zeros_like(mask_area, dtype=bool)

    kernel = np.ones((3, 3), np.uint8)

    for class_id in palette.keys():
        class_mask = (mask == class_id).astype(np.uint8) * 255
        if class_mask.sum() == 0:
            continue

        edges = cv2.Canny(class_mask, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=2).astype(bool)
        all_edges |= edges

    blended = image.copy()

    inner_area = mask_area.astype(bool) & (~all_edges)
    blended[inner_area] = cv2.addWeighted(
        image, 0.5, color_mask, 0.5, 0
    )[inner_area]

    blended[all_edges] = cv2.addWeighted(
        image, 0.1, color_mask, 0.9, 0
    )[all_edges]

    return blended


@torch.no_grad()
def collect_visual_grids(model, dataloader, device, palette, mean, std, n_clips=3, frames_per_clip=4):
    model.eval()
    grids = []

    # 1. Just take the first few batches normally instead of searching
    # This is 100x faster than the 'next(enumerate...)' approach
    collected_frames = []
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for i in range(images.shape[0]):
            img = denormalize(images[i].cpu(), mean, std)
            # IMPORTANT: OpenCV is BGR, WandB is RGB. Convert here!
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gt_m = masks[i].cpu().numpy()
            pr_m = preds[i].cpu().numpy()

            print("image shape: ", img.shape)
            print("gt shape and pr shape", gt_m.shape, pr_m.shape)
            print(np.unique(gt_m), np.unique(pr_m))

            gt_ov = apply_mask_overlay(img_rgb, gt_m, palette)
            pr_ov = apply_mask_overlay(img_rgb, pr_m, palette)

            # Optional: Resize here to make upload faster
            # row = np.concatenate([img_rgb, gt_ov, pr_ov], axis=1)
            # row = cv2.resize(row, (0,0), fx=0.5, fy=0.5)

            collected_frames.append(np.concatenate([img_rgb, gt_ov, pr_ov], axis=1))

            if len(collected_frames) >= (n_clips * frames_per_clip):
                break
        if len(collected_frames) >= (n_clips * frames_per_clip):
            break

    # 2. Chunk frames into grids
    for i in range(0, len(collected_frames), frames_per_clip):
        clip_frames = collected_frames[i: i + frames_per_clip]
        grids.append(np.concatenate(clip_frames, axis=0))

    return grids
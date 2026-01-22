import numpy as np
import torch

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

def collect_visual_examples(
    model,
    dataloader,
    device,
    color_palette,
    max_samples=6,
):
    """
    Collect qualitative segmentation examples.

    Returns a list of dicts:
      {
        "image": uint8 HWC,
        "gt": uint8 HWC,
        "pred": uint8 HWC,
        "meta": dict (optional)
      }
    """

    model.eval()
    visuals = []

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for i in range(images.size(0)):
            if len(visuals) >= max_samples:
                return visuals

            img = images[i].cpu()
            gt = masks[i].cpu().numpy()
            pr = preds[i].cpu().numpy()

            # Normalize image to [0, 255]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            visuals.append({
                "image": img,
                "gt": decode_mask(gt, color_palette),
                "pred": decode_mask(pr, color_palette),
            })

    return visuals

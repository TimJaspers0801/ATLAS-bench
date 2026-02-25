"""
Visualize clip predictions for multiple experiments.

This script saves per-clip visualizations at 1 FPS with folders:
  <output_dir>/<clip_id>/images
  <output_dir>/<clip_id>/GT
  <output_dir>/<clip_id>/<experiment_name>
"""

import argparse
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from datasets.atlas import AtlasDataset
from datasets.cholecseg8k import CholecSeg8kDataset
from evaluation.visual_logging import apply_mask_overlay, denormalize
from utils import color_palette, load_checkpoint
from models.load_models import (
    load_lh_vit_s_dinov1, load_lh_vit_b_dinov1,
    load_lh_vit_s_dinov2, load_lh_vit_b_dinov2, load_lh_vit_l_dinov2,
    load_lh_vit_s_dinov3, load_lh_vit_b_dinov3, load_lh_vit_l_dinov3,
    load_surgenet_caformer_s18, load_surgenet_convnextv2_tiny, load_surgenet_pvtv2_b2,
    load_endofm, load_endovit, load_lh_gastronet5m,
    load_lh_dinov1_vitb_224_surgenet2m, load_lh_dinov2_vitb_336_surgenet2m,
    load_lh_dinov3_vitb_256_surgenet2m, load_lh_dinov3_vitl_256_surgenet2m,
    load_sam2unet, load_sam3unet,
)


MODEL_REGISTRY = {
    "lh-vit-s-dinov1": load_lh_vit_s_dinov1,
    "lh-vit-b-dinov1": load_lh_vit_b_dinov1,
    "lh-vit-s-dinov2": load_lh_vit_s_dinov2,
    "lh-vit-b-dinov2": load_lh_vit_b_dinov2,
    "lh-vit-l-dinov2": load_lh_vit_l_dinov2,
    "lh-vit-s-dinov3": load_lh_vit_s_dinov3,
    "lh-vit-b-dinov3": load_lh_vit_b_dinov3,
    "lh-vit-l-dinov3": load_lh_vit_l_dinov3,
    "lh-dinov1-vitb-224-surgenet2m": load_lh_dinov1_vitb_224_surgenet2m,
    "lh-dinov2-vitb-336-surgenet2m": load_lh_dinov2_vitb_336_surgenet2m,
    "lh-dinov3-vitb-256-surgenet2m": load_lh_dinov3_vitb_256_surgenet2m,
    "lh-dinov3-vitl-256-surgenet2m": load_lh_dinov3_vitl_256_surgenet2m,
    "surgenet-caformer-s18": load_surgenet_caformer_s18,
    "surgenet-convnextv2-tiny": load_surgenet_convnextv2_tiny,
    "surgenet-pvtv2-b2": load_surgenet_pvtv2_b2,
    "endofm": load_endofm,
    "endovit": load_endovit,
    "gastronet5m": load_lh_gastronet5m,
    "sam2unet": load_sam2unet,
    "sam3unet": load_sam3unet,
}

IMAGE_SIZE_MAP = {
    "224": 224,
    "256": 256,
    "336": 336,
    "518": 518,
    "dinov2": 518,
    "dinov3": 256,
    "gastronet": 336,
    "eomt": 256,
}

ATLAS_TO_CHOLECSEG8K_MAPPING = {
    0: 0,
    1: 5,
    2: 11,
    3: 7,
    4: 6,
    5: 3,
    6: 3,
    7: 1,
    8: 0,
    9: 4,
    10: 2,
    11: 8,
    12: 10,
    13: 12,
    14: 12,
    15: 3,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
    27: 0,
    28: 0,
    29: 0,
}


def get_image_size(model_name: str) -> int:
    if "atlas" in model_name and "dinov1" in model_name:
        return 224
    if "atlas" in model_name and "dinov2" in model_name:
        return 336
    if "atlas" in model_name and "dinov3" in model_name:
        return 256
    if "eomt" in model_name and "dinov2" in model_name:
        return 518
    if "sam2unet" in model_name:
        return 256
    if "sam3unet" in model_name:
        return 336
    if "518" in model_name:
        return 518
    for key, size in IMAGE_SIZE_MAP.items():
        if key in model_name:
            return size
    return 256


def map_atlas_to_cholecseg8k(pred_mask: torch.Tensor) -> torch.Tensor:
    mapped = torch.zeros_like(pred_mask)
    for atlas_class, cholecseg_class in ATLAS_TO_CHOLECSEG8K_MAPPING.items():
        mapped[pred_mask == atlas_class] = cholecseg_class
    return mapped


def load_eomt(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    from models.eomt.eomt import (
        eomt_vits_dinov2, eomt_vitb_dinov2, eomt_vitl_dinov2,
        eomt_vits_dinov3, eomt_vitb_dinov3, eomt_vitl_dinov3,
    )

    eomt_loaders = {
        "eomt_vits_dinov2": eomt_vits_dinov2,
        "eomt_vitb_dinov2": eomt_vitb_dinov2,
        "eomt_vitl_dinov2": eomt_vitl_dinov2,
        "eomt_vits_dinov3": eomt_vits_dinov3,
        "eomt_vitb_dinov3": eomt_vitb_dinov3,
        "eomt_vitl_dinov3": eomt_vitl_dinov3,
    }

    if model_name not in eomt_loaders:
        raise ValueError(f"Unknown EOMT variant: {model_name}")

    model = eomt_loaders[model_name](num_classes=num_classes)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        has_timm_keys = any(
            "patch_embed" in k or "blocks" in k
            for k in state_dict.keys()
            if k.startswith("network.encoder.backbone")
        )
        has_hf_keys = any(
            "embeddings" in k or "layer" in k
            for k in state_dict.keys()
            if k.startswith("network.encoder.backbone")
        )

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("criterion."):
                continue
            new_key = key[8:] if key.startswith("network.") else key
            if has_timm_keys and not has_hf_keys:
                new_key = new_key.replace("encoder.backbone.patch_embed.", "encoder.backbone.embeddings.")
                new_key = new_key.replace("encoder.backbone.blocks.", "encoder.backbone.layer.")
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)

    return model.to(device)


def load_atlas(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    from models.atlas.atlas import (
        atlas_vitl_dinov3, atlas_vitb_dinov3, atlas_vits_dinov3,
        atlas_vitl_dinov3_tracking, atlas_vitb_dinov2, atlas_vitb_dinov1,
    )

    atlas_loaders = {
        "atlas_vitl_dinov3": atlas_vitl_dinov3,
        "atlas_vitl_dinov3_tracking": atlas_vitl_dinov3_tracking,
        "atlas_vitb_dinov3": atlas_vitb_dinov3,
        "atlas_vits_dinov3": atlas_vits_dinov3,
        "atlas_vitb_dinov2": atlas_vitb_dinov2,
        "atlas_vitb_dinov1": atlas_vitb_dinov1,
    }

    if model_name not in atlas_loaders:
        raise ValueError(f"Unknown ATLAS variant: {model_name}")

    model = atlas_loaders[model_name](num_classes=num_classes)

    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("criterion."):
                continue
            new_key = key[8:] if key.startswith("network.") else key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)

    return model.to(device)


def load_model(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    if model_name.startswith("atlas"):
        return load_atlas(model_name, checkpoint_path, num_classes, device)
    if model_name.startswith("eomt"):
        return load_eomt(model_name, checkpoint_path, num_classes, device)
    if model_name in MODEL_REGISTRY:
        loader = MODEL_REGISTRY[model_name]
        model = loader(num_classes)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            load_checkpoint(model, checkpoint_path)
        return model.to(device)

    raise ValueError(f"Unknown model: {model_name}")


def build_clip_index(dataset, dataset_name: str):
    clip_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        if dataset_name == "atlas":
            clip_id = f"{sample['procedure']}/{sample['video']}/{sample['clip']}"
            filename = sample["filename"]
        else:
            clip_id = sample.get("clip_id") or f"{sample['video']}/{sample['clip']}"
            filename = sample["filename"]
        clip_to_indices[clip_id].append((filename, idx))

    ordered = {}
    for clip_id, items in clip_to_indices.items():
        items.sort(key=lambda x: x[0])
        ordered[clip_id] = [idx for _, idx in items]
    return ordered


def save_clip_visualizations(
    model,
    dataset,
    clip_to_indices,
    clip_ids,
    device,
    output_dir,
    experiment_name,
    fps_in,
    fps_out,
    max_frames_per_clip,
    img_size,
    is_atlas,
    apply_class_mapping,
    mask_background,
):
    stride = max(1, int(round(fps_in / fps_out)))

    mean = dataset.mean
    std = dataset.std

    model.eval()
    with torch.no_grad():
        for clip_id in clip_ids:
            indices = clip_to_indices.get(clip_id, [])
            if not indices:
                continue

            clip_dir = os.path.join(output_dir, clip_id)
            images_dir = os.path.join(clip_dir, "images")
            gt_dir = os.path.join(clip_dir, "GT")
            pred_dir = os.path.join(clip_dir, experiment_name)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            prev_query_embed = None
            saved_frames = 0

            for frame_idx, sample_idx in enumerate(indices):
                item = dataset[sample_idx]
                image = item["image"].to(device)
                gt_mask = item["mask"].to(device)

                if is_atlas:
                    outputs = model(
                        image.unsqueeze(0),
                        prev_query_embed=prev_query_embed,
                        return_query_embedding=True,
                    )
                    mask_logits_per_block, class_logits_per_block, _, query_embed = outputs
                    prev_query_embed = query_embed

                    mask_logits = mask_logits_per_block[-1]
                    class_logits = class_logits_per_block[-1]
                    per_pixel_logits = torch.einsum(
                        "bqhw, bqc -> bchw",
                        mask_logits.sigmoid(),
                        class_logits.softmax(dim=-1)[..., :-1],
                    )
                    if per_pixel_logits.shape[-2:] != gt_mask.shape[-2:]:
                        per_pixel_logits = torch.nn.functional.interpolate(
                            per_pixel_logits,
                            size=gt_mask.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    preds = torch.argmax(per_pixel_logits, dim=1)
                else:
                    outputs = model(image.unsqueeze(0))
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                if apply_class_mapping:
                    preds = map_atlas_to_cholecseg8k(preds)

                if frame_idx % stride != 0:
                    continue

                img_np = denormalize(image.cpu(), mean, std)
                gt_np = gt_mask.squeeze().cpu().numpy().astype(np.int32)
                pred_np = preds.squeeze().cpu().numpy().astype(np.int32)

                if pred_np.shape != gt_np.shape:
                    pred_np = cv2.resize(
                        pred_np.astype(np.uint8),
                        (gt_np.shape[1], gt_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)

                if mask_background:
                    pred_np[gt_np == 0] = 0

                gt_overlay = apply_mask_overlay(img_np, gt_np, color_palette)
                pred_overlay = apply_mask_overlay(img_np, pred_np, color_palette)

                if img_np.shape[0] != img_size:
                    img_np = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    gt_overlay = cv2.resize(gt_overlay, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    pred_overlay = cv2.resize(pred_overlay, (img_size, img_size), interpolation=cv2.INTER_AREA)

                filename = item.get("filename", f"frame_{frame_idx:06d}.png")
                base_name = os.path.splitext(filename)[0] + ".png"

                cv2.imwrite(os.path.join(images_dir, base_name), img_np[:, :, ::-1])
                cv2.imwrite(os.path.join(gt_dir, base_name), gt_overlay[:, :, ::-1])
                cv2.imwrite(os.path.join(pred_dir, base_name), pred_overlay[:, :, ::-1])
                saved_frames += 1
                if max_frames_per_clip and saved_frames >= max_frames_per_clip:
                    break


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = args.img_size if args.img_size != 256 else get_image_size(args.model)

    test_transform = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
    ])

    if args.dataset == "atlas":
        dataset = AtlasDataset(
            zip_path=args.data_path,
            split="test",
            transform=test_transform,
            frame_percentage=args.frame_percentage,
            seed=args.seed,
            normalization_type="surgical",
        )
    else:
        dataset = CholecSeg8kDataset(
            zip_path=args.data_path,
            transform=test_transform,
            frame_percentage=args.frame_percentage,
            seed=args.seed,
            normalization_type="surgical",
        )

    clip_to_indices = build_clip_index(dataset, args.dataset)
    clip_ids = sorted(clip_to_indices.keys())

    if args.shuffle_clips:
        rng = random.Random(args.seed)
        rng.shuffle(clip_ids)

    clip_ids = clip_ids[: args.num_clips]

    model = load_model(args.model, args.checkpoint, args.num_classes, device)

    apply_class_mapping = args.apply_class_mapping and args.dataset == "cholecseg8k"
    is_atlas = args.model.startswith("atlas")

    save_clip_visualizations(
        model=model,
        dataset=dataset,
        clip_to_indices=clip_to_indices,
        clip_ids=clip_ids,
        device=device,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        fps_in=args.clip_fps,
        fps_out=args.save_fps,
        max_frames_per_clip=args.max_frames_per_clip,
        img_size=img_size,
        is_atlas=is_atlas,
        apply_class_mapping=apply_class_mapping,
        mask_background=args.mask_background,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize clip predictions for experiments")
    parser.add_argument("--dataset", type=str, required=True, choices=["atlas", "cholecseg8k"])
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset zip")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations_clips")
    parser.add_argument("--num_classes", type=int, default=30)
    parser.add_argument("--num_clips", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_percentage", type=int, default=100)
    parser.add_argument("--clip_fps", type=int, default=15)
    parser.add_argument("--save_fps", type=int, default=1)
    parser.add_argument("--max_frames_per_clip", type=int, default=10)
    parser.add_argument("--shuffle_clips", action="store_true")
    parser.add_argument("--no_shuffle_clips", action="store_false", dest="shuffle_clips")
    parser.add_argument("--apply_class_mapping", action="store_true")
    parser.add_argument("--no_class_mapping", action="store_false", dest="apply_class_mapping")
    parser.add_argument("--mask_background", action="store_true")
    parser.add_argument("--no_mask_background", action="store_false", dest="mask_background")

    parser.set_defaults(
        shuffle_clips=True,
        apply_class_mapping=True,
        mask_background=True,
    )

    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = args.model

    main(args)

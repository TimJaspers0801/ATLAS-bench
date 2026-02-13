"""
Test script for benchmarking all models on the ATLAS dataset.

Supports:
- Image-based models (DINOv2, DINOv3, SurgeNet variants, EOMT)
- Video-based models (VideoMT with online processing)
- Frame-by-frame evaluation on test set

Usage:
    python test_atlas.py --model lh-dinov3-vitl-256-surgenet2m \\
                         --checkpoint weights/DINOv3-vitl-256-surgenet2m.pth \\
                         --data_path atlas.zip
    
    python test_atlas.py --model videomt \\
                         --checkpoint checkpoints/videomt.pth \\
                         --data_path atlas.zip
"""

import argparse
import random
import torch
import os
import json
from tqdm import tqdm
from collections import defaultdict
import cv2

from torch.utils.data import DataLoader
from datasets.atlas import AtlasDataset
import torchvision.transforms.v2 as T
from torch import nn
import torch.nn.functional as F

from utils import load_checkpoint, color_palette
from models.load_models import (
    load_lh_vit_s_dinov1, load_lh_vit_b_dinov1, 
    load_lh_vit_s_dinov2, load_lh_vit_b_dinov2, load_lh_vit_l_dinov2,
    load_lh_vit_s_dinov3, load_lh_vit_b_dinov3, load_lh_vit_l_dinov3,
    load_surgenet_caformer_s18, load_surgenet_convnextv2_tiny, load_surgenet_pvtv2_b2,
    load_endofm, load_endovit, load_lh_gastronet5m,
    load_lh_dinov1_vitb_224_surgenet2m, load_lh_dinov2_vitb_336_surgenet2m,
    load_lh_dinov3_vitb_256_surgenet2m, load_lh_dinov3_vitl_256_surgenet2m
)
from evaluation.dataset_evaluation import evaluate_model
from evaluation.visual_logging import apply_mask_overlay, denormalize
import numpy as np


# Model registry
MODEL_REGISTRY = {
    # DINOv1 variants
    "lh-vit-s-dinov1": load_lh_vit_s_dinov1,
    "lh-vit-b-dinov1": load_lh_vit_b_dinov1,

    # DINOv2 variants
    "lh-vit-s-dinov2": load_lh_vit_s_dinov2,
    "lh-vit-b-dinov2": load_lh_vit_b_dinov2,
    "lh-vit-l-dinov2": load_lh_vit_l_dinov2,
    
    # DINOv3 variants
    "lh-vit-s-dinov3": load_lh_vit_s_dinov3,
    "lh-vit-b-dinov3": load_lh_vit_b_dinov3,
    "lh-vit-l-dinov3": load_lh_vit_l_dinov3,
    
    # DINOv1 SurgeNet2M
    "lh-dinov1-vitb-224-surgenet2m": load_lh_dinov1_vitb_224_surgenet2m,
    
    # DINOv2 SurgeNet2M
    "lh-dinov2-vitb-336-surgenet2m": load_lh_dinov2_vitb_336_surgenet2m,
    
    # DINOv3 SurgeNet2M
    "lh-dinov3-vitb-256-surgenet2m": load_lh_dinov3_vitb_256_surgenet2m,
    "lh-dinov3-vitl-256-surgenet2m": load_lh_dinov3_vitl_256_surgenet2m,
    
    # SurgeNet baselines
    "surgenet-caformer-s18": load_surgenet_caformer_s18,
    "surgenet-convnextv2-tiny": load_surgenet_convnextv2_tiny,
    "surgenet-pvtv2-b2": load_surgenet_pvtv2_b2,
    
    # Other vision models
    "endofm": load_endofm,
    "endovit": load_endovit,
    "gastronet5m": load_lh_gastronet5m,
}

# Image size mapping
IMAGE_SIZE_MAP = {
    "224": 224,
    "256": 256,
    "336": 336,
    "dinov2": 224,
    "dinov3": 256,
    "gastronet": 336,
    'videomt': 1280,
    'eomt': 336,
}


def get_image_size(model_name: str) -> int:
    """Infer image size from model name."""
    for key, size in IMAGE_SIZE_MAP.items():
        if key in model_name:
            return size
    return 256  # default


def load_model(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    """Load a model from the registry with optional checkpoint."""
    if model_name == "videomt":
        return load_videomt(checkpoint_path, num_classes, device)
    elif model_name.startswith("eomt"):
        return load_eomt(model_name, checkpoint_path, num_classes, device)
    elif model_name in MODEL_REGISTRY:
        # Load from registry
        loader = MODEL_REGISTRY[model_name]
        model = loader(num_classes)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path)
        
        return model.to(device)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available models:\n"
            f"  Image-based: {', '.join(MODEL_REGISTRY.keys())}\n"
            f"  Video-based: videomt\n"
            f"  EOMT variants: eomt_vits_dinov2, eomt_vitb_dinov2, eomt_vitl_dinov2, "
            f"eomt_vits_dinov3, eomt_vitb_dinov3, eomt_vitl_dinov3"
        )


def load_videomt(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load VideoMT model for online video processing."""
    from models.videomt.videomt_standalone import VideoMT
    
    # Initialize with training configuration from ATLAS config
    model = VideoMT(
        img_size=1280,
        num_classes=124,
        num_queries=200,
        task='vss',
        model_name='vit_large_patch14_dinov2.lvd142m',
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading VideoMT checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Handle key name mismatches between Detectron2 version and standalone
        key_mappings = {
            'q.weight': 'query_embedding.weight',  # Detectron2 uses 'q', standalone uses 'query_embedding'
        }
        
        for old_key, new_key in key_mappings.items():
            if old_key in state_dict and new_key not in state_dict:
                print(f"  Remapping key: {old_key} → {new_key}")
                state_dict[new_key] = state_dict.pop(old_key)
        
        # NOTE: Model is now created with class_token=False to match checkpoint structure
        # Both checkpoint and model now have: [spatial_tokens only], no class token
        # No pos_embed expansion needed
        
        # Remove keys that are not part of the model
        keys_to_remove = ['criterion.empty_weight', 'attn_mask_probs', 'encoder.backbone.reg_token']
        for key in keys_to_remove:

            if key in state_dict:
                print(f"  Removing non-model key: {key}")
                del state_dict[key]
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if not missing_keys and not unexpected_keys:
            print("✓ All keys loaded successfully")
        else:
            if missing_keys:
                print(f"⚠ Missing keys ({len(missing_keys)}):")
                # pixel_mean/pixel_std are buffers that will be initialized by register_buffer
                important_missing = [k for k in missing_keys if 'pixel_' not in k]
                if important_missing:
                    print(f"  Important missing keys:")
                    for key in important_missing[:5]:
                        print(f"    - {key}")
                    if len(important_missing) > 5:
                        print(f"    ... and {len(important_missing) - 5} more")
                else:
                    print(f"  (Only pixel normalization missing, which is acceptable)")
            if unexpected_keys:
                print(f"⚠ Unexpected keys ({len(unexpected_keys)}):")
                for key in unexpected_keys:
                    print(f"  - {key}")
    
    return model.to(device)


def load_eomt(model_name: str, checkpoint_path: str, num_classes: int, device: torch.device):
    """Load EOMT model variants."""
    from models.eomt.eomt import (
        eomt_vits_dinov2, eomt_vitb_dinov2, eomt_vitl_dinov2,
        eomt_vits_dinov3, eomt_vitb_dinov3, eomt_vitl_dinov3
    )
    
    # Map model names to loader functions
    eomt_loaders = {
        "eomt_vits_dinov2": eomt_vits_dinov2,
        "eomt_vitb_dinov2": eomt_vitb_dinov2,
        "eomt_vitl_dinov2": eomt_vitl_dinov2,
        "eomt_vits_dinov3": eomt_vits_dinov3,
        "eomt_vitb_dinov3": eomt_vitb_dinov3,
        "eomt_vitl_dinov3": eomt_vitl_dinov3,
    }
    
    if model_name not in eomt_loaders:
        raise ValueError(
            f"Unknown EOMT variant: {model_name}. Available: {', '.join(eomt_loaders.keys())}"
        )
    
    print(f"Loading EOMT model: {model_name}")
    model = eomt_loaders[model_name](num_classes=num_classes)
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading EOMT checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # EOMT checkpoints are already the state dict (not wrapped in a dict with 'model' key)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if not missing_keys and not unexpected_keys:
            print("✓ All keys loaded successfully")
        else:
            if missing_keys:
                print(f"⚠ Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
    
    return model.to(device)


def evaluate_videomt(model, test_loader, device, num_classes, window_size=16):
    """
    Evaluate VideoMT in online fashion, processing frames sequentially
    within each clip while maintaining temporal state.
    """
    model.eval()
    
    # Track scores per class
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)
    
    # AP tracking
    from evaluation.metrics import SegmentationAPEvaluator
    clip_ap = {}
    current_clip = None
    ap_evaluator = None
    
    mean = torch.tensor(test_loader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(test_loader.dataset.std).view(3, 1, 1)

    def process_clip(clip_id, frames, gts):
        nonlocal current_clip, ap_evaluator

        if not frames:
            return

        if ap_evaluator is not None:
            clip_ap[current_clip] = ap_evaluator.evaluate()
        current_clip = clip_id
        ap_evaluator = SegmentationAPEvaluator()

        classes_to_eval = range(1, num_classes + 1)

        for start in range(0, len(frames), window_size):
            end = min(start + window_size, len(frames))
            window_frames = frames[start:end]
            window_gts = gts[start:end]

            frames_tensor = torch.stack(window_frames, dim=0).to(device)
            outputs = model.process_video(frames_tensor, normalize=True, match_queries=True)

            pred_logits = outputs['pred_logits'][0]  # (Q, C+1)
            pred_masks = outputs['pred_masks'][0]    # (Q, T, H, W)

            target_h, target_w = window_gts[0].shape[-2:]
            if pred_masks.shape[-2:] != (target_h, target_w):
                pred_masks = F.interpolate(
                    pred_masks,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                )

            mask_cls = F.softmax(pred_logits, dim=-1)[:, :-1]  # (Q, C)
            mask_pred = pred_masks.sigmoid()  # (Q, T, H, W)
            semseg = torch.einsum("qc,qthw->cthw", mask_cls, mask_pred)
            sem_score, sem_mask = semseg.max(0)  # (T, H, W)

            for t in range(sem_mask.shape[0]):
                pred_np = sem_mask[t].cpu().numpy()
                gt_np = window_gts[t].squeeze().cpu().numpy()

                gt_binary = (gt_np > 0).astype(np.uint8)
                pred_binary = (pred_np > 0).astype(np.uint8)
                max_score = sem_score[t].max().item()
                ap_evaluator.add_frame(gt_binary, pred_binary, max_score)

                from evaluation.metrics import compute_class_metrics
                for c in classes_to_eval:
                    iou_c, dice_c = compute_class_metrics(pred_np, gt_np, c)
                    if iou_c is not None:
                        class_ious[c].append(iou_c)
                        class_dices[c].append(dice_c)

    with torch.no_grad():
        current_clip = None
        clip_frames = []
        clip_gts = []

        for batch in tqdm(test_loader, desc="Evaluating VideoMT"):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            for i in range(images.shape[0]):
                clip_id = f"{batch['procedure'][i]}/{batch['video'][i]}/{batch['clip'][i]}"

                if current_clip is None:
                    current_clip = clip_id

                if clip_id != current_clip:
                    process_clip(current_clip, clip_frames, clip_gts)
                    clip_frames = []
                    clip_gts = []
                    current_clip = clip_id

                frame = images[i].cpu() * std + mean
                clip_frames.append(frame.clamp(0.0, 1.0))
                clip_gts.append(gt_masks[i].cpu())

        process_clip(current_clip, clip_frames, clip_gts)
    
    if ap_evaluator is not None:
        clip_ap[current_clip] = ap_evaluator.evaluate()

    # Aggregate metrics
    from evaluation.metrics import compute_class_metrics
    classes_to_eval = range(1, num_classes+1)
    
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
    
    # Print report
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


def save_random_visualizations(
    model,
    dataloader,
    device,
    output_dir,
    num_samples,
    img_size,
    is_videomt,
    seed,
):
    if num_samples <= 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    samples = []
    seen = 0

    mean = torch.tensor(dataloader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(dataloader.dataset.std).view(3, 1, 1)

    with torch.no_grad():
        current_video = None

        for batch in tqdm(dataloader, desc="Collecting visualizations"):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            if is_videomt:
                batch_video = f"{batch['procedure'][0]}/{batch['video'][0]}"
                if batch_video != current_video:
                    model.reset_memory()
                    current_video = batch_video

                images_denorm = images.cpu() * std + mean
                images_denorm = images_denorm.clamp(0.0, 1.0)
                images_denorm = images_denorm.to(device)

                outputs_list = []
                for i in range(images.shape[0]):
                    frame = images_denorm[i:i+1]
                    outputs_list.append(model.forward_frame(frame))

                pred_logits = torch.cat([o['pred_logits'] for o in outputs_list], dim=0)
                pred_masks = torch.cat([o['pred_masks'] for o in outputs_list], dim=0)

                mask_cls = F.softmax(pred_logits, dim=-1)[:, :, :-1]
                pred_masks_prob = torch.sigmoid(pred_masks)
                semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, pred_masks_prob)
                preds = semseg.argmax(dim=1)
            else:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            for i in range(images.shape[0]):
                seen += 1
                if len(samples) < num_samples:
                    samples.append((images[i].cpu(), gt_masks[i].cpu(), preds[i].cpu()))
                else:
                    j = rng.randint(0, seen - 1)
                    if j < num_samples:
                        samples[j] = (images[i].cpu(), gt_masks[i].cpu(), preds[i].cpu())

    for idx, (img, gt_mask, pred_mask) in enumerate(samples):
        img_np = denormalize(img, mean, std)
        gt_np = gt_mask.squeeze().numpy().astype(np.int32)
        pred_np = pred_mask.squeeze().numpy().astype(np.int32)

        if pred_np.shape != gt_np.shape:
            pred_np = cv2.resize(
                pred_np.astype(np.uint8),
                (gt_np.shape[1], gt_np.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        gt_overlay = apply_mask_overlay(img_np, gt_np, color_palette)
        pred_overlay = apply_mask_overlay(img_np, pred_np, color_palette)

        row = np.concatenate([img_np, gt_overlay, pred_overlay], axis=1)
        if row.shape[0] != img_size:
            row = cv2.resize(row, (img_size * 3, img_size), interpolation=cv2.INTER_AREA)

        out_path = os.path.join(output_dir, f"sample_{idx:02d}.png")
        cv2.imwrite(out_path, row[:, :, ::-1])


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Infer image size
    img_size = args.img_size if args.img_size != 256 else get_image_size(args.model)
    print(f"Image size: {img_size}")
    
    # Create test dataset
    test_transform = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
    ])
    
    test_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="test",
        transform=test_transform,
        frame_percentage=args.test_percentage,
        seed=args.seed,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: maintain order for clips
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_model(
        args.model,
        args.checkpoint,
        args.num_classes,
        device
    )
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    if args.model == "videomt":
        metrics = evaluate_videomt(
            model,
            test_loader,
            device,
            args.num_classes,
            window_size=args.videomt_window,
        )
    else:
        metrics = evaluate_model(model, test_loader, device, args.num_classes)

    # Save visualizations
    if args.visualize_samples > 0:
        output_dir = os.path.join(args.visualize_dir, args.model)
        save_random_visualizations(
            model=model,
            dataloader=test_loader,
            device=device,
            output_dir=output_dir,
            num_samples=args.visualize_samples,
            img_size=img_size,
            is_videomt=(args.model == "videomt"),
            seed=args.seed,
        )
        print(f"\nSaved {args.visualize_samples} visualizations to: {output_dir}")
    
    # Save results if requested
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        results = {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "metrics": {
                "mIoU": float(metrics["mIoU"]),
                "Dice": float(metrics["Dice"]),
                "AP": float(metrics["AP"]),
                "AP50": float(metrics["AP50"]),
                "AP75": float(metrics["AP75"]),
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ATLAS models")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (see MODEL_REGISTRY or use 'videomt')"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file"
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
        help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size (automatically inferred from model if not specified)"
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
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--test_percentage",
        type=int,
        default=10,
        help="Percentage of test frames to evaluate (1-100)"
    )
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=25,
        help="Number of random samples to visualize (0 to disable)"
    )
    parser.add_argument(
        "--visualize_dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save visualization images"
    )
    parser.add_argument(
        "--videomt_window",
        type=int,
        default=4,
        help="Window size for VideoMT clip processing to limit memory usage"
    )
    
    args = parser.parse_args()
    metrics = main(args)

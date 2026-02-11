"""
Example: Using VideoMT Standalone Model for Inference

This script demonstrates how to use the standalone VideoMT model
for video segmentation without Detectron2.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from videomt_standalone import build_videomt_model, VideoMT


def load_video_from_images(image_folder: str, max_frames: int = None):
    """
    Load video from a folder of images.
    
    Args:
        image_folder: Path to folder containing images
        max_frames: Maximum number of frames to load
        
    Returns:
        Video tensor (T, 3, H, W) with values in [0, 1]
    """
    image_paths = sorted(Path(image_folder).glob('*.png')) + \
                  sorted(Path(image_folder).glob('*.jpg'))
    
    if max_frames:
        image_paths = image_paths[:max_frames]
    
    frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        frames.append(img_tensor)
    
    return torch.stack(frames, dim=0)


def example_batch_inference():
    """Example: Process entire video at once."""
    print("=" * 60)
    print("Example 1: Batch Video Processing")
    print("=" * 60)
    
    # Initialize model
    model = build_videomt_model(
        checkpoint_path="checkpoints/segmentator_vitl.pth",
        img_size=256,
        num_classes=47,
        task='vss',  # Video Semantic Segmentation
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create dummy video or load real video
    # video = load_video_from_images('path/to/video/frames')
    video = torch.rand(20, 3, 256, 256)  # Dummy: 20 frames, 256x256
    
    # Run inference
    with torch.no_grad():
        results = model.inference(video, normalize=True)
    
    # Results
    print(f"Task: {results['task']}")
    print(f"Predicted masks shape: {results['pred_masks'].shape}")  # (T, H, W)
    print(f"Unique classes in video: {torch.unique(results['pred_masks'])}")
    
    # Save first frame visualization (optional)
    first_frame_seg = results['pred_masks'][0].numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(video[0].permute(1, 2, 0))
    plt.title('Input Frame')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(first_frame_seg)
    plt.title('Segmentation')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('example_batch_output.png')
    print("Saved visualization to example_batch_output.png")


def example_online_inference():
    """Example: Process video frame-by-frame (online mode)."""
    print("\n" + "=" * 60)
    print("Example 2: Online Frame-by-Frame Processing")
    print("=" * 60)
    
    # Initialize model
    model = build_videomt_model(
        checkpoint_path="checkpoints/segmentator_vitl.pth",
        img_size=256,
        num_classes=47,
        task='vss',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create dummy video
    video = torch.rand(10, 3, 256, 256)
    
    # Reset memory for new video
    model.reset_memory()
    
    # Process frame by frame
    all_outputs = []
    for frame_idx in range(video.shape[0]):
        # Normalize frame
        frame = model.normalize_image(video[frame_idx]).to(model.pixel_mean.device)
        
        # Forward pass
        with torch.no_grad():
            output = model.forward_frame(frame)
        
        all_outputs.append(output)
        print(f"Processed frame {frame_idx + 1}/{video.shape[0]}")
    
    # Post-process to get final segmentation
    print("\nPost-processing results...")
    pred_logits = torch.stack([o['pred_logits'][0] for o in all_outputs], dim=0)  # (T, Q, C)
    pred_masks = torch.stack([o['pred_masks'][0] for o in all_outputs], dim=0)  # (T, Q, H, W)
    
    # Average logits over time and apply softmax
    avg_logits = pred_logits.mean(dim=0)  # (Q, C)
    mask_cls = torch.softmax(avg_logits, dim=-1)[:, :-1]  # Remove no-object class
    
    # Compute semantic segmentation per frame
    semantic_masks = []
    for t in range(pred_masks.shape[0]):
        frame_masks = pred_masks[t].sigmoid()  # (Q, H, W)
        semseg = torch.einsum("qc,qhw->chw", mask_cls, frame_masks)
        sem_mask = semseg.argmax(0)
        semantic_masks.append(sem_mask)
    
    semantic_video = torch.stack(semantic_masks, dim=0)  # (T, H, W)
    print(f"Final semantic segmentation shape: {semantic_video.shape}")
    print(f"Unique classes: {torch.unique(semantic_video)}")


def example_instance_segmentation():
    """Example: Video Instance Segmentation (VIS)."""
    print("\n" + "=" * 60)
    print("Example 3: Video Instance Segmentation")
    print("=" * 60)
    
    # Initialize model for instance segmentation
    model = build_videomt_model(
        checkpoint_path="checkpoints/segmentator_vitl.pth",
        img_size=256,
        num_classes=47,
        task='vis',  # Video Instance Segmentation
        max_num_instances=20,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create dummy video
    video = torch.rand(15, 3, 256, 256)
    
    # Run inference
    with torch.no_grad():
        results = model.inference(video, normalize=True)
    
    # Results
    print(f"Task: {results['task']}")
    print(f"Number of detected instances: {len(results['pred_masks'])}")
    print(f"Instance labels: {results['pred_labels']}")
    print(f"Instance scores: {results['pred_scores']}")
    
    # Each mask is a binary tensor of shape (T, H, W)
    if len(results['pred_masks']) > 0:
        print(f"First instance mask shape: {results['pred_masks'][0].shape}")


def example_custom_model():
    """Example: Create custom VideoMT model with specific parameters."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Model Configuration")
    print("=" * 60)
    
    # Create model with custom parameters
    model = VideoMT(
        img_size=512,  # Different resolution
        num_classes=80,  # Different number of classes
        num_queries=200,  # More queries
        model_name='vit_large_patch14_dinov2.lvd142m',
        segmenter_blocks=[21, 21, 22, 23],
        task='vss',
        object_mask_threshold=0.3,
        overlap_threshold=0.8,
        max_num_instances=50,
        pixel_mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        pixel_std=[0.229, 0.224, 0.225],
    )
    
    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"Created custom model with:")
    print(f"  - Resolution: {model.img_size}x{model.img_size}")
    print(f"  - Classes: {model.num_classes}")
    print(f"  - Queries: {model.num_queries}")
    print(f"  - Task: {model.task}")
    
    # You can load pretrained weights if available
    # model.load_weights('path/to/checkpoint.pth', strict=False)


def example_with_resize():
    """Example: Process video with automatic resizing."""
    print("\n" + "=" * 60)
    print("Example 5: Processing with Different Input/Output Sizes")
    print("=" * 60)
    
    # Initialize model
    model = build_videomt_model(
        checkpoint_path="checkpoints/segmentator_vitl.pth",
        img_size=256,
        num_classes=47,
        task='vss',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create video with different size (model will handle resizing internally)
    video = torch.rand(10, 3, 480, 640)  # Different aspect ratio
    
    # Resize to model input size
    resized_video = torch.nn.functional.interpolate(
        video,
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )
    
    # Run inference
    with torch.no_grad():
        results = model.inference(
            resized_video,
            output_size=(480, 640),  # Specify original output size
            normalize=True
        )
    
    print(f"Input video shape: {video.shape}")
    print(f"Model input shape: {resized_video.shape}")
    print(f"Output masks shape: {results['pred_masks'].shape}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("VideoMT Standalone Inference Examples")
    print("=" * 60 + "\n")
    
    try:
        example_batch_inference()
    except Exception as e:
        print(f"Batch inference example failed: {e}")
    
    try:
        example_online_inference()
    except Exception as e:
        print(f"Online inference example failed: {e}")
    
    try:
        example_instance_segmentation()
    except Exception as e:
        print(f"Instance segmentation example failed: {e}")
    
    try:
        example_custom_model()
    except Exception as e:
        print(f"Custom model example failed: {e}")
    
    try:
        example_with_resize()
    except Exception as e:
        print(f"Resize example failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

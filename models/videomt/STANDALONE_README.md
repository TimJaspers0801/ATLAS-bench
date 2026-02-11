# VideoMT Standalone Model

A pure PyTorch implementation of VideoMT for video segmentation inference without Detectron2 dependencies.

## Features

- **Pure PyTorch**: No Detectron2, Lightning, or other framework dependencies
- **Online Inference**: Process videos frame-by-frame with temporal memory
- **Multiple Tasks**: Supports Video Semantic Segmentation (VSS), Video Instance Segmentation (VIS), and Video Panoptic Segmentation (VPS)
- **Easy to Use**: Simple API for both batch and online processing
- **Checkpoint Loading**: Load weights from your trained Detectron2 models

## Installation

### Requirements

```bash
pip install torch torchvision timm einops scipy numpy pillow matplotlib
```

### Minimum versions:
- Python >= 3.8
- PyTorch >= 2.0
- timm >= 0.9.0

## Quick Start

### 1. Batch Inference (Process entire video)

```python
import torch
from videomt_standalone import build_videomt_model

# Build model
model = build_videomt_model(
    checkpoint_path="checkpoints/segmentator_vitl.pth",
    img_size=256,
    num_classes=47,
    task='vss',  # 'vss', 'vis', or 'vps'
    device='cuda'
)

# Prepare video (T, 3, H, W) with values in [0, 1]
video = torch.rand(20, 3, 256, 256)

# Run inference
with torch.no_grad():
    results = model.inference(video, normalize=True)

# Results
print(results['pred_masks'].shape)  # (T, H, W) for VSS
```

### 2. Online Inference (Frame-by-frame)

```python
import torch
from videomt_standalone import build_videomt_model

# Build model
model = build_videomt_model(
    checkpoint_path="checkpoints/segmentator_vitl.pth",
    img_size=256,
    num_classes=47,
    task='vss',
    device='cuda'
)

# Reset memory for new video
model.reset_memory()

# Process frame by frame
for frame in video_frames:
    # Normalize and move to device
    frame = model.normalize_image(frame).to('cuda')
    
    # Process frame
    with torch.no_grad():
        output = model.forward_frame(frame)
    
    # output contains:
    # - 'pred_logits': Class predictions (1, Q, C+1)
    # - 'pred_masks': Mask predictions (1, Q, H, W)
    # - 'query_embeddings': Query features
```

## API Reference

### `VideoMT` Class

Main model class for video segmentation.

#### Constructor Parameters

- **img_size** (int): Input image size (e.g., 256, 512)
- **num_classes** (int): Number of semantic classes
- **num_queries** (int): Number of object queries (default: 100)
- **model_name** (str): Timm model name (default: 'vit_large_patch14_dinov2.lvd142m')
- **segmenter_blocks** (List[int]): Transformer blocks for segmentation (default: [21,21,22,23])
- **task** (str): Task type - 'vis', 'vss', or 'vps'
- **object_mask_threshold** (float): Threshold for filtering masks (default: 0.0)
- **overlap_threshold** (float): Overlap threshold for panoptic (default: 0.8)
- **max_num_instances** (int): Max instances for VIS (default: 20)
- **pixel_mean** (List[float]): Normalization mean [B,G,R] (default: DINOv2 mean)
- **pixel_std** (List[float]): Normalization std [B,G,R] (default: DINOv2 std)

#### Methods

##### `forward_frame(frame, resume=None)`
Process a single frame in online mode.

**Args:**
- `frame` (torch.Tensor): Input frame (B, 3, H, W) or (3, H, W), already normalized
- `resume` (bool): Whether to use memory from previous frame

**Returns:**
- Dict with 'pred_logits', 'pred_masks', 'query_embeddings'

##### `process_video(frames, normalize=True, match_queries=True)`
Process a complete video sequence.

**Args:**
- `frames` (torch.Tensor or List): Video frames (T, 3, H, W) or list of frames
- `normalize` (bool): Whether to normalize input
- `match_queries` (bool): Whether to match queries across frames

**Returns:**
- Dict with 'pred_logits', 'pred_masks', 'frame_logits'

##### `inference(frames, output_size=None, normalize=True)`
Run inference and get task-specific outputs.

**Args:**
- `frames` (torch.Tensor or List): Video frames
- `output_size` (Tuple[int,int]): Output size (H, W)
- `normalize` (bool): Whether to normalize input

**Returns:**
- Task-specific output dict (see below)

##### `reset_memory()`
Reset temporal memory. Call at the start of a new video.

##### `normalize_image(image)`
Normalize an image using model's pixel_mean and pixel_std.

**Args:**
- `image` (torch.Tensor): Image (3, H, W) or (B, 3, H, W) with values in [0, 1]

**Returns:**
- Normalized image tensor

##### `load_weights(checkpoint_path, strict=True)`
Load model weights from checkpoint.

**Args:**
- `checkpoint_path` (str): Path to checkpoint
- `strict` (bool): Strict weight matching

### `build_videomt_model()`

Convenience function to build a VideoMT model.

**Args:**
- `checkpoint_path` (str): Path to checkpoint (optional)
- `img_size` (int): Input image size
- `num_classes` (int): Number of classes
- `task` (str): Task type
- `device` (str): Device to load model on
- `**kwargs`: Additional VideoMT arguments

**Returns:**
- VideoMT model ready for inference

## Output Formats

### Video Semantic Segmentation (VSS)

```python
{
    'pred_masks': torch.Tensor,  # Shape: (T, H, W), class IDs
    'task': 'vss',
    'image_size': (H, W)
}
```

### Video Instance Segmentation (VIS)

```python
{
    'pred_masks': List[torch.Tensor],  # List of (T, H, W) binary masks
    'pred_labels': List[int],          # Class ID per instance
    'pred_scores': List[float],        # Confidence per instance
    'task': 'vis',
    'image_size': (H, W)
}
```

### Video Panoptic Segmentation (VPS)

```python
{
    'pred_masks': torch.Tensor,        # Shape: (T, H, W), instance IDs
    'segments_info': List[Dict],       # Info per instance
    'task': 'vps',
    'image_size': (H, W)
}
```

## Advanced Usage

### Custom Model Configuration

```python
from videomt_standalone import VideoMT

model = VideoMT(
    img_size=512,
    num_classes=80,
    num_queries=200,
    model_name='vit_large_patch14_dinov2.lvd142m',
    task='vis',
    max_num_instances=50,
    pixel_mean=[0.485, 0.456, 0.406],  # Custom normalization
    pixel_std=[0.229, 0.224, 0.225],
)

model.load_weights('path/to/checkpoint.pth', strict=False)
model = model.to('cuda')
model.eval()
```

### Loading from Different Checkpoint Formats

The model automatically handles:
- Detectron2 checkpoints (with 'model' key and 'backbone.' prefix)
- Standard PyTorch checkpoints
- Direct state_dict files

```python
# All these work:
model.load_weights('detectron2_checkpoint.pth', strict=False)
model.load_weights('pytorch_checkpoint.pth', strict=True)
model.load_weights('state_dict.pt', strict=False)
```

### Processing Videos with Different Resolutions

```python
# Original video at 720p
video = torch.rand(50, 3, 720, 1280)

# Resize to model input size
resized = torch.nn.functional.interpolate(
    video,
    size=(256, 256),
    mode='bilinear',
    align_corners=False
)

# Run inference and get output at original size
results = model.inference(
    resized,
    output_size=(720, 1280),
    normalize=True
)

# results['pred_masks'] will be at 720x1280
```

### Working with Real Video Files

```python
import cv2
import torch
from videomt_standalone import build_videomt_model

def load_video(video_path, max_frames=None):
    """Load video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return torch.stack(frames, dim=0)

# Load and process
video = load_video('path/to/video.mp4')
model = build_videomt_model(checkpoint_path='checkpoint.pth', task='vss')

with torch.no_grad():
    results = model.inference(video)
```

## Performance Tips

1. **Use appropriate batch size**: For online inference, process one frame at a time. For batch inference, you can load the entire video.

2. **Enable mixed precision**: Use torch.cuda.amp for faster inference
   ```python
   with torch.cuda.amp.autocast():
       results = model.inference(video)
   ```

3. **Clear memory between videos**: Always call `model.reset_memory()` when starting a new video

4. **Use appropriate resolution**: Lower resolutions (256x256) are faster but less accurate

## Differences from Detectron2 Version

- **No training support**: This is inference-only
- **Simplified API**: Direct PyTorch interface without Detectron2 abstractions
- **No auxiliary losses**: Only final layer predictions
- **Memory management**: Explicit reset_memory() call required

## Troubleshooting

### "PytorchStreamReader failed" error
Your checkpoint file may be corrupted. Try:
1. Re-downloading the checkpoint
2. Training a new model
3. Using `strict=False` when loading: `model.load_weights(path, strict=False)`

### Out of memory errors
- Reduce img_size (e.g., 256 instead of 512)
- Process fewer frames at once
- Use online inference (frame-by-frame) instead of batch

### Wrong predictions
- Check that input images are in [0, 1] range
- Verify you're using the correct normalization (pixel_mean/pixel_std)
- Ensure num_classes matches your checkpoint

## License

Same as the original VideoMT project.

## Citation

If you use this code, please cite the original VideoMT paper.

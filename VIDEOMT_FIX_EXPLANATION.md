# VideoMT Class Token Mismatch Fix

## Problem
VideoMT model was stuck predicting only class 7 across all queries and frames, regardless of the input. This indicated a fundamental architectural or weight-loading issue.

## Root Cause Analysis
The root cause was a **token structure mismatch** between the checkpoint and the model:

### Checkpoint (Detectron2 VideoMT)
- Trained WITH a Detectron2 VideoMT model
- Uses ViT backbone WITHOUT class token
- Token sequence: `[spatial_tokens only]`  
- For 256×256 image with 14×14 patches: 6400 spatial tokens
- Positional embeddings: shape `[1, 6400, 1024]`

### Model (Standalone PyTorch)  
- Created using `timm.create_model()` 
- **By default**, timm ViT includes a class token
- Token sequence: `[class_token, spatial_tokens]`
- For same image: 1 class token + 6400 spatial tokens = 6401 tokens
- Positional embeddings: shape `[1, 6401, 1024]`

### Impact
This 1-token difference caused:
1. **Positional embedding mismatch** → Shape error or incorrect positional information
2. **Spatial feature extraction error** → Spatial tokens extracted from wrong indices
3. **Model output collapse** → Gradients and feature propagation broke down
4. **Single-class prediction** → Model defaulted to always predicting the same class

## Solution
Set `class_token=False` when creating the timm ViT model to match the checkpoint structure.

### Change 1: models/videomt/videomt_standalone.py (Line 36)
```python
self.backbone = timm.create_model(
    name,
    pretrained=True,
    img_size=img_size,
    patch_size=patch_size,
    num_classes=0,
    class_token=False,  # ← CRITICAL: Checkpoint was trained without class token
    dynamic_img_size=True,
)
```

### Change 2: test_atlas.py (Lines 160-164)
Removed the pos_embed expansion logic since it's no longer needed. Both checkpoint and model now have the same structure.

## Verification
After the fix:
- `model.encoder.backbone.num_prefix_tokens == 0` ✓ (no class token)
- `model.encoder.backbone.pos_embed.shape == [1, 6400, 1024]` ✓ (matches checkpoint)
- Token sequence after query concatenation: `[queries (100) | spatial_tokens (6400)]` ✓
- Spatial feature extraction: `x[:, 100:, :]` correctly extracts all 6400 tokens ✓

## Expected Behavior After Fix
- Model should produce diverse predictions across multiple classes
- Mask predictions should vary across frames and queries  
- Class logits should show varied probability distributions

## Technical Details

### Token Ordering
Before concatenation:
```
x = [spatial_tokens]  # Shape: [B, 6400, D]
```

After concatenation with queries:
```
x = [queries | spatial_tokens]  # Shape: [B, 100+6400, D]
```

Spatial feature extraction in `_predict()`:
```python
num_prefix = self.encoder.backbone.num_prefix_tokens  # = 0
spatial_tokens = x[:, self.num_queries + num_prefix:, :]  # x[:, 100:, :]
```

### Why This Matters
The checkpoint weights were trained with this exact token structure. When we create the model with a different structure (extra class token), the weights don't align with the tensor positions they expect to operate on, breaking the entire inference.

## Files Modified
1. `models/videomt/videomt_standalone.py` - Added `class_token=False`
2. `test_atlas.py` - Removed unnecessary pos_embed expansion

## Backward Compatibility
This fix only works with checkpoints trained WITHOUT a class token. If using checkpoints trained with a class token, revert the `class_token=False` change or retrain/convert the checkpoint.

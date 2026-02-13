"""Comprehensive test of VideoMT class_token fix."""

import torch
import os
import sys
from pathlib import Path

# Get the checkpoint path
checkpoint_path = Path("checkpoints/segmentator_vitl.pth")

print("=" * 80)
print("VideoMT Class Token Fix Validation Test")
print("=" * 80)

# Check if checkpoint exists
if checkpoint_path.exists():
    print(f"\n✓ Found checkpoint: {checkpoint_path}")
    print(f"  Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Try loading and checking checkpoint structure
    print("\nAnalyzing checkpoint structure...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'encoder.backbone.pos_embed' in ckpt:
        pos_shape = ckpt['encoder.backbone.pos_embed'].shape
        print(f"  pos_embed shape: {pos_shape}")
        print(f"  Token count (without batch): {pos_shape[1]}")
        print(f"  Embed dim: {pos_shape[2]}")
        
        # 6400 = 256/14 * 256/14 (ViT-L with 14x14 patches)
        if pos_shape[1] == 6400:
            print(f"  ✓ Checkpoint trained without class token (6400 = 256/14 * 256/14)")
        elif pos_shape[1] == 6401:
            print(f"  ✗ Checkpoint trained WITH class token (6401 = 1 + 6400)")
    
    # Also check if there's a cls_token in checkpoint
    has_cls_token = 'encoder.backbone.cls_token' in ckpt
    print(f"\nCheckpoint has cls_token key: {has_cls_token}")
    
    if has_cls_token:
        print(f"  ⚠️  ERROR: Checkpoint contains cls_token but model is configured without it!")
        print(f"           This will cause loading errors.")
    else:
        print(f"  ✓ Checkpoint does not have cls_token (matches model configuration)")
    
else:
    print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
    print(f"   Expected location: {checkpoint_path}")

print("\n" + "=" * 80)
print("Model Configuration Test")
print("=" * 80)

try:
    from models.videomt.videomt_standalone import VideoMT
    
    # Create model
    print("\nCreating VideoMT model with class_token=False...")
    model = VideoMT(
        img_size=256,
        num_classes=30,
        num_queries=100,
        model_name='vit_large_patch14_dinov2.lvd142m',
        segmenter_blocks=[22, 23],
        task='vss',
        device='cpu'
    )
    
    # Check if cls_token exists in model
    has_cls_token = hasattr(model.encoder.backbone, 'cls_token') and model.encoder.backbone.cls_token is not None
    print(f"Model has cls_token: {has_cls_token}")
    
    if has_cls_token and model.encoder.backbone.cls_token is not None:
        print(f"  ⚠️  ERROR: Model still has cls_token despite class_token=False!")
    else:
        print(f"  ✓ Model correctly initialized without cls_token")
    
    # Check num_prefix_tokens
    print(f"\nModel num_prefix_tokens: {model.encoder.backbone.num_prefix_tokens}")
    if model.encoder.backbone.num_prefix_tokens == 0:
        print(f"  ✓ Correct: 0 prefix tokens (no class token)")
    else:
        print(f"  ✗ ERROR: Expected 0 prefix tokens, got {model.encoder.backbone.num_prefix_tokens}")
    
    # Check pos_embed shape
    print(f"\nModel pos_embed shape: {model.encoder.backbone.pos_embed.shape}")
    if model.encoder.backbone.pos_embed.shape[1] == 6400:
        print(f"  ✓ Correct: 6400 spatial tokens (no class token)")
    else:
        print(f"  ✗ ERROR: Expected 6400 tokens, got {model.encoder.backbone.pos_embed.shape[1]}")
    
    # Try loading checkpoint
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint into model...")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Key remapping
            if 'q.weight' in state_dict and 'query_embedding.weight' not in state_dict:
                state_dict['query_embedding.weight'] = state_dict.pop('q.weight')
            
            # Remove non-model keys
            for key in ['criterion.empty_weight', 'attn_mask_probs', 'encoder.backbone.reg_token']:
                if key in state_dict:
                    del state_dict[key]
            
            # Load
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Checkpoint loaded")
            
            if missing:
                print(f"  Missing keys ({len(missing)}):")
                for k in list(missing)[:3]:
                    print(f"    - {k}")
                if len(missing) > 3:
                    print(f"    ... and {len(missing) - 3} more")
            
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}):")
                for k in list(unexpected)[:3]:
                    print(f"    - {k}")
                if len(unexpected) > 3:
                    print(f"    ... and {len(unexpected) - 3} more")
            
        except Exception as e:
            print(f"  ✗ Error loading checkpoint: {e}")
    
    # Test inference
    print(f"\nTesting inference...")
    model.eval()
    with torch.no_grad():
        frame = torch.randn(1, 3, 256, 256)
        output = model.forward_frame(frame)
    
    print(f"  ✓ Inference successful")
    print(f"    pred_logits shape: {output['pred_logits'].shape}")
    print(f"    pred_masks shape: {output['pred_masks'].shape}")
    
    # Check for diverse predictions
    logits = output['pred_logits']
    classes = logits.argmax(dim=-1)
    unique = torch.unique(classes).tolist()
    
    print(f"\n  Class predictions:")
    print(f"    Unique classes: {unique}")
    print(f"    Count: {len(unique)}")
    
    if len(unique) == 1:
        print(f"    ⚠️  WARNING: Still only predicting class {unique[0]}")
        print(f"               This suggests the spatial features are still not contributing")
    else:
        print(f"    ✓ Good: Diverse predictions across {len(unique)} classes")
    
    # Histogram of class predictions
    for i in range(0, 31, 5):
        count = (classes == i).sum().item()
        if count > 0:
            print(f"      Class {i:2d}: {count:3d} queries")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

"""
Test script to diagnose DINOv2 vs DINOv3 differences.
This will help identify the issue with DINOv2 model performance.
"""

import torch
import torch.nn as nn

print("=" * 80)
print("Testing DINOv2 vs DINOv3 Model Loading")
print("=" * 80)

# Test 1: Raw DINOv2 model (current implementation)
print("\n[TEST 1] Loading raw DINOv2 model (current broken implementation)...")
try:
    from models.load_models import load_dinov2_l
    model_dinov2_raw = load_dinov2_l(img_size=256)
    print(f"✓ Loaded raw DINOv2 model")
    print(f"  Type: {type(model_dinov2_raw)}")
    print(f"  Has get_intermediate_layers: {hasattr(model_dinov2_raw, 'get_intermediate_layers')}")
    print(f"  get_intermediate_layers signature: {model_dinov2_raw.get_intermediate_layers.__doc__ if hasattr(model_dinov2_raw, 'get_intermediate_layers') else 'N/A'}")
    
    # Check backbone attributes
    print(f"  Has patch_embed: {hasattr(model_dinov2_raw, 'patch_embed')}")
    print(f"  Has embed_dim: {hasattr(model_dinov2_raw, 'embed_dim')}")
    print(f"  embed_dim value: {model_dinov2_raw.embed_dim if hasattr(model_dinov2_raw, 'embed_dim') else 'N/A'}")
    print(f"  patch_embed.patch_size: {model_dinov2_raw.patch_embed.patch_size if hasattr(model_dinov2_raw, 'patch_embed') else 'N/A'}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: DINOv3 model (current working implementation)
print("\n[TEST 2] Loading DINOv3 model (current working implementation)...")
try:
    from models.load_models import load_dinov3_l
    model_dinov3 = load_dinov3_l()
    print(f"✓ Loaded DINOv3 model")
    print(f"  Type: {type(model_dinov3)}")
    print(f"  Has backbone: {hasattr(model_dinov3, 'backbone')}")
    print(f"  backbone type: {type(model_dinov3.backbone) if hasattr(model_dinov3, 'backbone') else 'N/A'}")
    print(f"  Has get_intermediate_layers: {hasattr(model_dinov3.backbone, 'get_intermediate_layers') if hasattr(model_dinov3, 'backbone') else 'N/A'}")
    
    # Check backbone attributes
    if hasattr(model_dinov3, 'backbone'):
        print(f"  Has patch_embed: {hasattr(model_dinov3.backbone, 'patch_embed')}")
        print(f"  Has embed_dim: {hasattr(model_dinov3.backbone, 'embed_dim')}")
        print(f"  embed_dim value: {model_dinov3.backbone.embed_dim if hasattr(model_dinov3.backbone, 'embed_dim') else 'N/A'}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Forward pass with raw DINOv2
print("\n[TEST 3] Forward pass with raw DINOv2 model...")
try:
    from models.load_models import load_lh_vit_l_dinov2
    model = load_lh_vit_l_dinov2(n_classes=30)
    model.eval()
    model.to("cpu")
    
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: (1, 30, 256, 256)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Forward pass with DINOv3  
print("\n[TEST 4] Forward pass with DINOv3 model...")
try:
    from models.load_models import load_lh_vit_l_dinov3
    model = load_lh_vit_l_dinov3(n_classes=30)
    model.eval()
    model.to("cpu")
    
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: (1, 30, 256, 256)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)

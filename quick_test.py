"""Quick test of VideoMT with fixed class_token setting."""

import torch
import sys
sys.path.insert(0, '/workspace')

# Test if the model can now be imported and initialized
try:
    from models.videomt.videomt_standalone import VideoMT, build_videomt_model
    print("✓ Imports successful")
    
    # Try building model with class_token=False
    print("\nBuilding VideoMT model...")
    model = VideoMT(
        img_size=256,
        num_classes=30,
        num_queries=100,
        model_name='vit_large_patch14_dinov2.lvd142m',
        segmenter_blocks=[22, 23],
        task='vss',
        device='cpu'
    )
    print("✓ Model built successfully")
    
    # Check backbone config
    print(f"\nBackbone configuration:")
    print(f"  num_prefix_tokens: {model.encoder.backbone.num_prefix_tokens}")
    print(f"  Has cls_token: {hasattr(model.encoder.backbone, 'cls_token')}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_frame = torch.randn(1, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        output = model.forward_frame(test_frame)
    
    print(f"✓ Forward pass successful")
    print(f"  pred_logits shape: {output['pred_logits'].shape}")
    print(f"  pred_masks shape: {output['pred_masks'].shape}")
    
    # Check class predictions
    class_logits = output['pred_logits']
    pred_classes = class_logits.argmax(dim=-1)
    unique_classes = torch.unique(pred_classes).tolist()
    print(f"\nClass prediction analysis:")
    print(f"  Predicted classes: {unique_classes}")
    print(f"  Number of different classes: {len(unique_classes)}")
    
    if len(unique_classes) == 1:
        print(f"  ⚠️  WARNING: Still only predicting single class {unique_classes[0]}")
    else:
        print(f"  ✓ Model is producing diverse predictions!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

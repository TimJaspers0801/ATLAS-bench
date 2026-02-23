#!/usr/bin/env python3
"""
Quick test to verify DINOv3 model loading and forward pass work correctly.
"""

import torch
import torch.nn as nn
from models.load_models import load_lh_vit_l_dinov3

def test_dinov3_model():
    """Test that DINOv3 model loads and can perform forward pass."""
    print("Testing DINOv3 model loading and forward pass...")
    
    # Load model
    print("Loading lh-vit-l-dinov3 model...")
    try:
        model = load_lh_vit_l_dinov3(n_classes=30)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test forward pass
    print("Testing forward pass with dummy input...")
    try:
        # Create dummy input (batch_size=2, channels=3, height=256, width=256)
        dummy_input = torch.randn(2, 3, 256, 256)
        
        # Move model to CPU (or GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        # Check output shape
        expected_shape = (2, 30, 256, 256)  # (batch, classes, height, width)
        if output.shape == expected_shape:
            print(f"✓ Output shape is correct: {output.shape} == {expected_shape}")
            return True
        else:
            print(f"✗ Output shape mismatch: {output.shape} != {expected_shape}")
            return False
            
    except AttributeError as e:
        print(f"✗ AttributeError during forward pass: {e}")
        return False
    except Exception as e:
        print(f"✗ Exception during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dinov3_model()
    exit(0 if success else 1)

"""
Simple test to verify the standalone VideoMT model works correctly.
Run this to check if the model loads and runs without errors.
"""

import torch
import sys

def test_model_creation():
    """Test creating the model without checkpoint."""
    print("Test 1: Model Creation")
    print("-" * 40)
    
    try:
        from videomt_standalone import VideoMT
        
        model = VideoMT(
            img_size=256,
            num_classes=47,
            num_queries=100,
            task='vss',
            model_name='vit_large_patch14_dinov2.lvd142m',
        )
        
        print(f"✓ Model created successfully")
        print(f"  - Embed dim: {model.embed_dim}")
        print(f"  - Num queries: {model.num_queries}")
        print(f"  - Num classes: {model.num_classes}")
        print(f"  - Task: {model.task}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTest 2: Forward Pass (Single Frame)")
    print("-" * 40)
    
    try:
        from videomt_standalone import VideoMT
        
        model = VideoMT(
            img_size=256,
            num_classes=47,
            num_queries=100,
            task='vss',
        )
        model.eval()
        
        # Dummy input
        frame = torch.rand(1, 3, 256, 256)
        
        # Forward pass
        with torch.no_grad():
            output = model.forward_frame(frame)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {frame.shape}")
        print(f"  - Logits shape: {output['pred_logits'].shape}")
        print(f"  - Masks shape: {output['pred_masks'].shape}")
        print(f"  - Query embeddings shape: {output['query_embeddings'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_online_inference():
    """Test online multi-frame inference."""
    print("\nTest 3: Online Inference (Multiple Frames)")
    print("-" * 40)
    
    try:
        from videomt_standalone import VideoMT
        
        model = VideoMT(
            img_size=256,
            num_classes=47,
            num_queries=100,
            task='vss',
        )
        model.eval()
        
        # Reset memory
        model.reset_memory()
        
        # Process 5 frames
        num_frames = 5
        for i in range(num_frames):
            frame = torch.rand(1, 3, 256, 256)
            
            with torch.no_grad():
                output = model.forward_frame(frame)
        
        print(f"✓ Online inference successful")
        print(f"  - Processed {num_frames} frames")
        print(f"  - Frame count: {model.frame_count}")
        print(f"  - Memory state: {'Active' if model.last_query_embed is not None else 'Empty'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Online inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference():
    """Test batch video processing."""
    print("\nTest 4: Batch Video Processing")
    print("-" * 40)
    
    try:
        from videomt_standalone import VideoMT
        
        model = VideoMT(
            img_size=256,
            num_classes=47,
            num_queries=100,
            task='vss',
        )
        model.eval()
        
        # Dummy video
        video = torch.rand(10, 3, 256, 256)
        
        # Process video
        with torch.no_grad():
            outputs = model.process_video(video, normalize=False, match_queries=True)
        
        print(f"✓ Batch processing successful")
        print(f"  - Input video shape: {video.shape}")
        print(f"  - Average logits shape: {outputs['pred_logits'].shape}")
        print(f"  - Masks shape: {outputs['pred_masks'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_api():
    """Test high-level inference API."""
    print("\nTest 5: High-Level Inference API")
    print("-" * 40)
    
    try:
        from videomt_standalone import VideoMT
        
        # Test all three tasks
        tasks = ['vss', 'vis', 'vps']
        
        for task in tasks:
            model = VideoMT(
                img_size=256,
                num_classes=47,
                num_queries=100,
                task=task,
            )
            model.eval()
            
            video = torch.rand(5, 3, 256, 256)
            
            with torch.no_grad():
                results = model.inference(video, normalize=False)
            
            print(f"✓ Task '{task}' successful")
            print(f"  - Output task: {results['task']}")
            print(f"  - Pred masks shape: {results['pred_masks'].shape if isinstance(results['pred_masks'], torch.Tensor) else f'{len(results['pred_masks'])} instances'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference API failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normalization():
    """Test image normalization."""
    print("\nTest 6: Image Normalization")
    print("-" * 40)
    
    try:
        from videomt_standalone import VideoMT
        
        model = VideoMT(
            img_size=256,
            num_classes=47,
            task='vss',
        )
        
        # Test with different input shapes
        img_3d = torch.rand(3, 256, 256)
        img_4d = torch.rand(2, 3, 256, 256)
        
        normalized_3d = model.normalize_image(img_3d)
        normalized_4d = model.normalize_image(img_4d)
        
        print(f"✓ Normalization successful")
        print(f"  - Input 3D shape: {img_3d.shape} -> Output: {normalized_3d.shape}")
        print(f"  - Input 4D shape: {img_4d.shape} -> Output: {normalized_4d.shape}")
        print(f"  - Mean: {normalized_3d.mean().item():.4f}")
        print(f"  - Std: {normalized_3d.std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_function():
    """Test the build_videomt_model convenience function."""
    print("\nTest 7: Build Function")
    print("-" * 40)
    
    try:
        from videomt_standalone import build_videomt_model
        
        model = build_videomt_model(
            checkpoint_path=None,  # No checkpoint
            img_size=256,
            num_classes=47,
            task='vss',
            device='cpu',
        )
        
        print(f"✓ Build function successful")
        print(f"  - Model device: {next(model.parameters()).device}")
        print(f"  - Model eval mode: {not model.training}")
        
        return True
        
    except Exception as e:
        print(f"✗ Build function failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_reset():
    """Test memory reset functionality."""
    print("\nTest 8: Memory Reset")
    print("-" * 40)
    
    try:
        from videomt_standalone import VideoMT
        
        model = VideoMT(img_size=256, num_classes=47, task='vss')
        model.eval()
        
        # Process a frame to set memory
        frame = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            _ = model.forward_frame(frame)
        
        assert model.frame_count > 0, "Frame count should be > 0"
        assert model.last_query_embed is not None, "Memory should be set"
        
        # Reset
        model.reset_memory()
        
        assert model.frame_count == 0, "Frame count should be reset to 0"
        assert model.last_query_embed is None, "Memory should be cleared"
        
        print(f"✓ Memory reset successful")
        print(f"  - Frame count reset to: {model.frame_count}")
        print(f"  - Memory cleared: {model.last_query_embed is None}")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VideoMT Standalone Model Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_online_inference,
        test_batch_inference,
        test_inference_api,
        test_normalization,
        test_build_function,
        test_memory_reset,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

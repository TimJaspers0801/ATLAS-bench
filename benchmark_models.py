"""
Benchmark script for measuring model performance metrics:
- FPS (Frames Per Second) with batch size 1 (online inference)
- Number of Parameters
- GFLOPs (Giga Floating Point Operations)

Usage:
    python benchmark_models.py --model lh-dinov3-vitl-256-surgenet2m \\
                               --checkpoint outputs/lh_dinov3_vitl_256_surgenet2m_atlas/best_model.pth \\
                               --img_size 256 \\
                               --num_classes 30
    
    # Benchmark all models from test_atlas.sh
    python benchmark_models.py --benchmark_all
"""

import argparse
import torch
import time
import numpy as np
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import model loading functions from test_atlas.py
from test_atlas import (
    load_model,
    get_image_size,
    MODEL_REGISTRY,
)


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_fps(model, device, img_size, batch_size=1, warmup_iters=200, test_iters=10000):
    """
    Measure FPS (Frames Per Second) for online inference.
    
    Args:
        model: PyTorch model
        device: torch.device
        img_size: Input image size
        batch_size: Batch size (default 1 for online inference)
        warmup_iters: Number of warmup iterations
        test_iters: Number of test iterations for measurement
        
    Returns:
        fps: Frames per second
        latency_ms: Average latency in milliseconds per frame
    """
    model.eval()
    torch.backends.cudnn.benchmark = True
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Check if model is ATLAS (requires special handling for temporal models)
    is_atlas = hasattr(model, 'forward') and 'prev_query_embed' in model.forward.__code__.co_varnames
    
    # Warmup
    print(f"  Warming up ({warmup_iters} iterations)...")
    with torch.inference_mode():
        for _ in range(warmup_iters):
            if is_atlas:
                _ = model(dummy_input, prev_query_embed=None, return_query_embedding=False)
            else:
                _ = model(dummy_input)
    
    _synchronize(device)
    
    # Measure inference time
    print(f"  Benchmarking ({test_iters} iterations)...")
    times = []
    with torch.inference_mode():
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(test_iters):
                start_event.record()
                if is_atlas:
                    _ = model(dummy_input, prev_query_embed=None, return_query_embedding=False)
                else:
                    _ = model(dummy_input)
                end_event.record()
                _synchronize(device)
                times.append(start_event.elapsed_time(end_event) / 1000.0)
        else:
            for _ in range(test_iters):
                start_time = time.perf_counter()
                if is_atlas:
                    _ = model(dummy_input, prev_query_embed=None, return_query_embedding=False)
                else:
                    _ = model(dummy_input)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    fps = batch_size / avg_time
    latency_ms = avg_time * 1000 / batch_size
    
    print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency: {latency_ms:.2f} ms/frame")
    
    return fps, latency_ms


def measure_flops(model, img_size, batch_size=1, device='cuda'):
    """
    Measure FLOPs using fvcore library.
    Falls back to estimation if fvcore is not available.
    
    Args:
        model: PyTorch model
        img_size: Input image size
        batch_size: Batch size
        device: Device to use
        
    Returns:
        gflops: Giga FLOPs
        gparams: Giga parameters
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        # Check if model is ATLAS
        is_atlas = hasattr(model, 'forward') and 'prev_query_embed' in model.forward.__code__.co_varnames
        
        if is_atlas:
            # For ATLAS models, we need to handle the special forward signature
            print("  Note: ATLAS model detected, using forward without query propagation for FLOP count")
            flop_counter = FlopCountAnalysis(
                model, 
                inputs=(dummy_input,),
            )
            # Override forward call for ATLAS
            flop_counter._kwargs = {'prev_query_embed': None, 'return_query_embedding': False}
        else:
            flop_counter = FlopCountAnalysis(model, dummy_input)
        
        total_flops = flop_counter.total()
        gflops = total_flops / 1e9
        
        return gflops
        
    except ImportError:
        print("  Warning: fvcore not installed. Install with: pip install fvcore")
        print("  Attempting to estimate FLOPs...")
        return estimate_flops(model, img_size, batch_size)
    except Exception as e:
        print(f"  Warning: FLOPs measurement failed: {e}")
        print("  Attempting to estimate FLOPs...")
        return estimate_flops(model, img_size, batch_size)


def estimate_flops(model, img_size, batch_size=1):
    """
    Rough estimation of FLOPs based on model architecture.
    This is a fallback when fvcore measurement fails.
    """
    total_params, _ = count_parameters(model)
    
    # Rough estimation: 
    # For ViT models: ~2 * params * num_tokens (forward + backward pass, but we only count forward)
    # num_tokens = (img_size / patch_size)^2
    
    # Assume patch_size = 14 or 16 based on common architectures
    patch_size = 14 if img_size in [518, 224] else 16
    num_tokens = (img_size // patch_size) ** 2
    
    # Rough FLOP estimate: 2 operations per parameter per token
    estimated_flops = 2 * total_params * num_tokens * batch_size
    gflops = estimated_flops / 1e9
    
    print(f"  Estimated GFLOPs (rough): {gflops:.2f}")
    return gflops


def benchmark_model(
    model_name,
    checkpoint_path,
    num_classes,
    img_size,
    device,
    output_file=None,
    warmup_iters=50,
    test_iters=200,
):
    """
    Benchmark a single model.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to checkpoint (None for pretrained)
        num_classes: Number of output classes
        img_size: Input image size
        device: torch.device
        output_file: Optional path to save results JSON
        
    Returns:
        dict: Benchmark results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path or 'Pretrained'}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Num classes: {num_classes}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    try:
        model = load_model(model_name, checkpoint_path, num_classes, device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Count parameters
    print("\nCounting parameters...")
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Measure FLOPs
    print("\nMeasuring GFLOPs...")
    try:
        gflops = measure_flops(model, img_size, batch_size=1, device=device)
        print(f"  GFLOPs: {gflops:.2f}")
    except Exception as e:
        print(f"  Error measuring GFLOPs: {e}")
        gflops = None
    
    # Measure FPS
    print("\nMeasuring FPS (batch_size=1, online inference)...")
    try:
        fps, latency_ms = measure_fps(
            model,
            device,
            img_size,
            batch_size=1,
            warmup_iters=warmup_iters,
            test_iters=test_iters,
        )
    except Exception as e:
        print(f"  Error measuring FPS: {e}")
        fps, latency_ms = None, None
    
    # Compile results
    results = {
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "img_size": img_size,
        "num_classes": num_classes,
        "total_params": total_params,
        "total_params_M": round(total_params / 1e6, 2),
        "trainable_params": trainable_params,
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "gflops": round(gflops, 2) if gflops is not None else None,
        "fps": round(fps, 2) if fps is not None else None,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
    }
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Parameters: {results['total_params_M']:.2f}M")
    print(f"GFLOPs: {results['gflops']}")
    print(f"FPS: {results['fps']}")
    print(f"Latency: {results['latency_ms']} ms")
    print(f"{'='*80}\n")
    
    return results


def parse_test_atlas_models():
    """
    Parse model configurations from test_atlas.sh.
    
    Returns:
        list: List of model configurations
    """
    # Define all models from test_atlas.sh (same as in the file)
    models_config = [
        # DINOv2 Pretrained
        ("lh-vit-s-dinov2", None, "lh_vits_dinov2_atlas", 0, 32),
        ("lh-vit-b-dinov2", None, "lh_vitb_dinov2_atlas", 0, 32),
        ("lh-vit-l-dinov2", None, "lh_vitl_dinov2_atlas", 0, 32),
        
        # DINOv3 Pretrained
        ("lh-vit-b-dinov3", None, "lh_vitb_dinov3_atlas", 0, 32),
        ("lh-vit-l-dinov3", None, "lh_vitl_dinov3_atlas", 0, 32),
        
        # # DINOv1 SurgeNet2M
        # ("lh-dinov1-vitb-224-surgenet2m", "best_model.pth", "lh_dinov1_vitb_224_surgenet2m_atlas", 0, 32),
        
        # # DINOv2 SurgeNet2M
        # ("lh-dinov2-vitb-336-surgenet2m", "best_model.pth", "lh_dinov2_vitb_336_surgenet2m_atlas", 0, 32),
        
        # # DINOv3 SurgeNet2M
        # ("lh-dinov3-vitb-256-surgenet2m", "best_model.pth", "lh_dinov3_vitb_256_surgenet2m_atlas", 0, 32),
        # ("lh-dinov3-vitl-256-surgenet2m", "best_model.pth", "lh_dinov3_vitl_256_surgenet2m_atlas", 0, 32),
        
        # ATLAS models (temporal)
        ("atlas_vitl_dinov3", "best_model.pth", "atlas_vitl_dinov3_surgenet", 0, 32),
        ("atlas_vitb_dinov3", "best_model.pth", "atlas_vitb_dinov3_surgenet", 0, 32),
        ("atlas_vits_dinov3", "best_model.pth", "atlas_vits_dinov3_surgenet", 0, 32),
        ("atlas_vitl_dinov3_tracking", "best_model.pth", "atlas_vitl_dinov3_tracking_surgenet", 0, 32),

        # EOMT SurgeNet models
        ("eomt_vitl_dinov3", "best_model.pth", "eomt_dinov3_vitl_surgenet_256", 0, 32),

        # EOMT ImageNet models 
        ("eomt_vitl_dinov3", "best_model.pth", "eomt_dinov3_vitl_256", 0, 32),
        ("eomt_vitb_dinov2", "best_model.pth", "eomt_dinov2_vitb_518", 0, 32),
        ("eomt_vitb_dinov3", "best_model.pth", "eomt_dinov3_vitb_256", 0, 32),

        # SurgeNet Baselines
        ("surgenet-pvtv2-b2", "best_model.pth", "pvtv2_atlas", 0, 32),
        ("surgenet-convnextv2-tiny", "best_model.pth", "convnextv2_atlas", 0, 32),
        ("surgenet-caformer-s18", "best_model.pth", "caformer_atlas", 0, 32),
           
        # Other models 
        ("endofm", "best_model.pth", "endofm_atlas", 0, 32),
        ("endovit", "best_model.pth", "endovit_atlas", 0, 32),
        ("gastronet5m", "best_model.pth", "lh_gastronet5m_atlas", 0, 32),
    ]
    
    return models_config


def benchmark_all_models(output_root, num_classes=30):
    """
    Benchmark all models from test_atlas.sh.
    
    Args:
        output_root: Root directory for outputs
        num_classes: Number of classes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models_config = parse_test_atlas_models()
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    failed_models = []
    
    for model_name, checkpoint_pattern, experiment_pattern, seed, batch_size in models_config:
        experiment_name = f"{experiment_pattern}_seed{seed}"
        
        # Determine checkpoint path
        if checkpoint_pattern and checkpoint_pattern != "None":
            checkpoint_base = Path(output_root) / experiment_name / checkpoint_pattern
            
            # Try exact match first
            if checkpoint_base.exists():
                checkpoint_path = str(checkpoint_base)
            else:
                # Try glob pattern for models with metadata in filename
                base_name = checkpoint_base.stem
                parent_dir = checkpoint_base.parent
                
                # Try both .pt and .pth extensions
                checkpoint_path = None
                for ext in ['pt', 'pth']:
                    matches = list(parent_dir.glob(f"{base_name}_*.{ext}"))
                    if matches:
                        checkpoint_path = str(matches[0])
                        break
                
                if not checkpoint_path:
                    print(f"\n⚠️  Checkpoint not found for {experiment_name}")
                    print(f"   Tried: {checkpoint_base}")
                    print(f"   Skipping...")
                    failed_models.append(experiment_name)
                    continue
        else:
            checkpoint_path = None
        
        # Determine image size
        img_size = get_image_size(model_name)
        
        # Output file for this model
        output_file = results_dir / f"{experiment_name}_benchmark.json"
        
        # Benchmark model
        try:
            result = benchmark_model(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                num_classes=num_classes,
                img_size=img_size,
                device=device,
                output_file=str(output_file)
            )
            
            if result:
                result["experiment_name"] = experiment_name
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error benchmarking {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append(experiment_name)
            continue
    
    # Generate summary report
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY - ALL MODELS")
    print("="*80)
    print(f"{'Model':<50} {'Params(M)':>10} {'GFLOPs':>10} {'FPS':>10} {'Latency(ms)':>12}")
    print("-"*80)
    
    for result in all_results:
        name = result['experiment_name']
        params = result['total_params_M']
        gflops = result['gflops'] if result['gflops'] else 'N/A'
        fps = result['fps'] if result['fps'] else 'N/A'
        latency = result['latency_ms'] if result['latency_ms'] else 'N/A'
        
        print(f"{name:<50} {params:>10.2f} {str(gflops):>10} {str(fps):>10} {str(latency):>12}")
    
    print("="*80)
    
    if failed_models:
        print("\n⚠️  Failed to benchmark:")
        for name in failed_models:
            print(f"  - {name}")
    
    # Save summary CSV
    csv_path = results_dir / "benchmark_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("experiment_name,model_name,img_size,total_params_M,gflops,fps,latency_ms\n")
        for result in all_results:
            f.write(
                f"{result['experiment_name']},"
                f"{result['model_name']},"
                f"{result['img_size']},"
                f"{result['total_params_M']:.2f},"
                f"{result['gflops'] if result['gflops'] else 'N/A'},"
                f"{result['fps'] if result['fps'] else 'N/A'},"
                f"{result['latency_ms'] if result['latency_ms'] else 'N/A'}\n"
            )
    
    print(f"\nSummary saved to: {csv_path}")
    
    # Save full results JSON
    json_path = results_dir / "benchmark_summary.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Full results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance metrics")
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (required unless --benchmark_all is used)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (None for pretrained)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=30,
        help="Number of output classes"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Input image size (auto-inferred from model if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=50,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--test_iters",
        type=int,
        default=200,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--benchmark_all",
        action="store_true",
        help="Benchmark all models from test_atlas.sh"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for model outputs (used with --benchmark_all)"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.benchmark_all:
        # Benchmark all models from test_atlas.sh
        benchmark_all_models(args.output_root, args.num_classes)
    else:
        # Benchmark single model
        if not args.model:
            parser.error("--model is required unless --benchmark_all is used")
        
        # Infer image size if not provided
        img_size = args.img_size if args.img_size else get_image_size(args.model)
        
        benchmark_model(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            num_classes=args.num_classes,
            img_size=img_size,
            device=device,
            output_file=args.output,
            warmup_iters=args.warmup_iters,
            test_iters=args.test_iters,
        )


if __name__ == "__main__":
    main()

"""
Compare predictions from multiple experiments on a single clip.

Creates side-by-side visualizations for easy comparison of different models.

Output structure:
  <output_dir>/<frame_name>_comparison.png

Where each comparison image has columns:
  [Image | GT | Experiment1 | Experiment2 | ... | ExperimentN]
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def get_sorted_frames(folder: str) -> list:
    """Get sorted list of PNG files from a folder."""
    if not os.path.exists(folder):
        return []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    return files


def create_comparison_row(image_path, gt_path, pred_paths, frame_name: str):
    """
    Create a horizontal comparison row.
    
    Args:
        image_path: Path to original image
        gt_path: Path to GT overlay
        pred_paths: Dict of {experiment_name: path_to_pred}
        frame_name: Name of frame for logging
    
    Returns:
        Combined image array (H, W*num_cols, 3) or None if any image missing
    """
    images = []
    
    # Load original image
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
    
    # Load GT
    if os.path.exists(gt_path):
        gt = cv2.imread(gt_path)
        if gt is not None:
            images.append(gt)
    
    # Load predictions in order
    for exp_name in sorted(pred_paths.keys()):
        pred_path = pred_paths[exp_name]
        if os.path.exists(pred_path):
            pred = cv2.imread(pred_path)
            if pred is not None:
                images.append(pred)
    
    if not images:
        print(f"  ⚠️  No valid images found for frame: {frame_name}")
        return None
    
    # Ensure all images have same height
    heights = [img.shape[0] for img in images]
    if len(set(heights)) > 1:
        target_height = max(heights)
        images = [
            cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
            if img.shape[0] != target_height else img
            for img in images
        ]
    
    # Stack horizontally
    comparison = np.hstack(images)
    return comparison


def compare_clip(
    clip_dir: str,
    experiments: list,
    output_dir: str,
    verbose: bool = True,
):
    """
    Compare predictions for a single clip across experiments.
    
    Args:
        clip_dir: Path to clip folder (contains images/, GT/, experiment folders)
        experiments: List of experiment names to compare
        output_dir: Where to save comparison images
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images_dir = os.path.join(clip_dir, "images")
    gt_dir = os.path.join(clip_dir, "GT")
    
    if not os.path.exists(images_dir) or not os.path.exists(gt_dir):
        print(f"  ⚠️  Missing images/ or GT/ in {clip_dir}")
        return 0
    
    # Get frames from images folder
    frame_files = get_sorted_frames(images_dir)
    if not frame_files:
        print(f"  ⚠️  No frames found in {images_dir}")
        return 0
    
    saved_count = 0
    
    for frame_file in frame_files:
        frame_name = os.path.splitext(frame_file)[0]
        
        image_path = os.path.join(images_dir, frame_file)
        gt_path = os.path.join(gt_dir, frame_file)
        
        # Build prediction paths
        pred_paths = {}
        for exp_name in experiments:
            exp_dir = os.path.join(clip_dir, exp_name)
            pred_path = os.path.join(exp_dir, frame_file)
            if os.path.exists(pred_path):
                pred_paths[exp_name] = pred_path
        
        if not pred_paths:
            if verbose:
                print(f"  ⚠️  No experiment predictions found for {frame_file}")
            continue
        
        # Create comparison
        comparison = create_comparison_row(image_path, gt_path, pred_paths, frame_file)
        if comparison is None:
            continue
        
        # Save
        output_path = os.path.join(output_dir, f"{frame_name}_comparison.png")
        cv2.imwrite(output_path, comparison)
        saved_count += 1
    
    if verbose:
        print(f"  ✓ Saved {saved_count} comparisons for {len(frame_files)} frames")
    
    return saved_count


def main(args):
    if not os.path.exists(args.clip_dir):
        print(f"Error: clip_dir not found: {args.clip_dir}")
        return
    
    clip_name = os.path.basename(args.clip_dir)
    print(f"\nComparing clip: {clip_name}")
    print(f"Experiments: {', '.join(args.experiments)}")
    
    saved = compare_clip(
        clip_dir=args.clip_dir,
        experiments=args.experiments,
        output_dir=args.output_dir,
        verbose=True,
    )
    
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare experiment predictions on a clip")
    parser.add_argument("--clip_dir", type=str, required=True, help="Path to clip folder")
    parser.add_argument(
        "--experiments",
        type=str,
        required=True,
        nargs="+",
        help="Experiment names to compare (space-separated)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for comparison images"
    )
    
    args = parser.parse_args()
    main(args)

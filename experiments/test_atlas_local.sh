#!/bin/bash

# Local test script for ATLAS models (without SLURM/Apptainer)
# Usage: ./test_atlas_local.sh [model_name] [checkpoint_path]
# Example: ./test_atlas_local.sh lh-dinov3-vitl-256-surgenet2m outputs/lh_dinov3_vitl_256_surgenet2m_atlas_seed0/best_model.pth

# ===========================
# Configuration
# ===========================

DATA_ZIP="atlas.zip"  # Update this to your local path
NUM_CLASSES=47
NUM_WORKERS=4
BATCH_SIZE=32
SEED=0

# Create results directory
RESULTS_DIR="test_results"
mkdir -p ${RESULTS_DIR}

# ===========================
# Test single model or all models
# ===========================

if [ $# -eq 2 ]; then
    # Single model test (provided as arguments)
    MODEL_NAME="$1"
    CHECKPOINT_PATH="$2"
    
    echo "========================================"
    echo "Testing: ${MODEL_NAME}"
    echo "Checkpoint: ${CHECKPOINT_PATH}"
    echo "========================================"
    
    RESULT_FILE="${RESULTS_DIR}/$(basename ${CHECKPOINT_PATH%.*})_test_results.json"
    
    python test_atlas120k.py \
        --model ${MODEL_NAME} \
        --checkpoint ${CHECKPOINT_PATH} \
        --data_path ${DATA_ZIP} \
        --num_classes ${NUM_CLASSES} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS} \
        --seed ${SEED} \
        --output ${RESULT_FILE}
    
    echo "Results saved to: ${RESULT_FILE}"
    
else
    # Test all models
    echo "========================================"
    echo "Testing all ATLAS models"
    echo "========================================"
    
    # Define all models to test
    # Format: "model_name|checkpoint_path|experiment_name|batch_size"
    
    MODELS=(
        # DINOv2 Pretrained
        "lh-vit-s-dinov2||lh_vits_dinov2_atlas_seed0|32"
        "lh-vit-b-dinov2||lh_vitb_dinov2_atlas_seed0|32"
        "lh-vit-l-dinov2||lh_vitl_dinov2_atlas_seed0|32"
        
        # DINOv3 Pretrained
        "lh-vit-b-dinov3||lh_vitb_dinov3_atlas_seed0|32"
        "lh-vit-l-dinov3||lh_vitl_dinov3_atlas_seed0|32"
        
        # DINOv1 SurgeNet2M
        "lh-dinov1-vitb-224-surgenet2m|outputs/lh_dinov1_vitb_224_surgenet2m_atlas_seed0/best_model.pth|lh_dinov1_vitb_224_surgenet2m_atlas_seed0|32"
        
        # DINOv2 SurgeNet2M
        "lh-dinov2-vitb-336-surgenet2m|outputs/lh_dinov2_vitb_336_surgenet2m_atlas_seed0/best_model.pth|lh_dinov2_vitb_336_surgenet2m_atlas_seed0|32"
        
        # DINOv3 SurgeNet2M
        "lh-dinov3-vitb-256-surgenet2m|outputs/lh_dinov3_vitb_256_surgenet2m_atlas_seed0/best_model.pth|lh_dinov3_vitb_256_surgenet2m_atlas_seed0|32"
        "lh-dinov3-vitl-256-surgenet2m|outputs/lh_dinov3_vitl_256_surgenet2m_atlas_seed0/best_model.pth|lh_dinov3_vitl_256_surgenet2m_atlas_seed0|32"
        
        # SurgeNet Baselines
        "surgenet-pvtv2-b2|outputs/pvtv2_atlas_seed0/best_model.pth|pvtv2_atlas_seed0|32"
        "surgenet-convnextv2-tiny|outputs/convnextv2_atlas_seed0/best_model.pth|convnextv2_atlas_seed0|32"
        "surgenet-caformer-s18|outputs/caformer_atlas_seed0/best_model.pth|caformer_atlas_seed0|32"

        # Other models (commented out by default)
        # "endofm|outputs/endofm_atlas_seed0/best_model.pth|endofm_atlas_seed0|32"
        # "endovit|outputs/endovit_atlas_seed0/best_model.pth|endovit_atlas_seed0|32"
        # "gastronet5m|outputs/gastronet5m_atlas_seed0/best_model.pth|gastronet5m_atlas_seed0|32"
    )
    
    for model_config in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_NAME CHECKPOINT_PATH EXPERIMENT_NAME MODEL_BATCH_SIZE <<< "$model_config"
        
        # Use model-specific batch size if provided, otherwise default
        ACTUAL_BATCH_SIZE=${MODEL_BATCH_SIZE:-$BATCH_SIZE}
        
        echo ""
        echo "========================================"
        echo "Testing: ${EXPERIMENT_NAME}"
        echo "Model: ${MODEL_NAME}"
        echo "Batch Size: ${ACTUAL_BATCH_SIZE}"
        echo "======================================="
        
        # Build checkpoint argument
        if [ -n "${CHECKPOINT_PATH}" ]; then
            # Check if checkpoint exists
            if [ ! -f "${CHECKPOINT_PATH}" ]; then
                echo "⚠️  Checkpoint not found: ${CHECKPOINT_PATH}"
                echo "Skipping..."
                continue
            fi
            CHECKPOINT_ARG="--checkpoint ${CHECKPOINT_PATH}"
        else
            CHECKPOINT_ARG=""
        fi
        
        RESULT_FILE="${RESULTS_DIR}/${EXPERIMENT_NAME}_test_results.json"
        
        python test_atlas120k.py \
            --model ${MODEL_NAME} \
            ${CHECKPOINT_ARG} \
            --data_path ${DATA_ZIP} \
            --num_classes ${NUM_CLASSES} \
            --batch_size ${ACTUAL_BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --seed ${SEED} \
            --output ${RESULT_FILE}
        
        if [ $? -eq 0 ]; then
            echo "✅ Test completed successfully"
            echo "Results saved to: ${RESULT_FILE}"
        else
            echo "❌ Test failed"
        fi
    done
    
    echo ""
    echo "========================================"
    echo "All tests completed!"
    echo "Results saved in: ${RESULTS_DIR}"
    echo "========================================"
    
    # Generate summary
    python << 'EOF'
import json
import os
from pathlib import Path

results_dir = Path("test_results")

if not results_dir.exists():
    print("No results directory found")
    exit()

results = []
for result_file in sorted(results_dir.glob("*_test_results.json")):
    try:
        with open(result_file) as f:
            data = json.load(f)
            results.append({
                "experiment": result_file.stem.replace("_test_results", ""),
                "model": data.get("model", "unknown"),
                "mIoU": data["metrics"]["mIoU"],
                "Dice": data["metrics"]["Dice"],
                "AP": data["metrics"]["AP"],
                "AP50": data["metrics"]["AP50"],
                "AP75": data["metrics"]["AP75"],
            })
    except Exception as e:
        print(f"Error reading {result_file}: {e}")

if results:
    print("\n" + "="*80)
    print("ATLAS Test Results Summary")
    print("="*80)
    print(f"{'Experiment':<50} {'mIoU':>8} {'Dice':>8} {'AP':>8} {'AP50':>8} {'AP75':>8}")
    print("-"*80)
    for r in results:
        print(f"{r['experiment']:<50} {r['mIoU']:>8.4f} {r['Dice']:>8.4f} {r['AP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f}")
    print("="*80)
    
    # Save as CSV
    csv_path = results_dir / "summary.csv"
    with open(csv_path, 'w') as f:
        f.write("experiment,model,mIoU,Dice,AP,AP50,AP75\n")
        for r in results:
            f.write(f"{r['experiment']},{r['model']},{r['mIoU']:.4f},{r['Dice']:.4f},{r['AP']:.4f},{r['AP50']:.4f},{r['AP75']:.4f}\n")
    print(f"\nSummary saved to: {csv_path}")
else:
    print("No results found")
EOF
fi

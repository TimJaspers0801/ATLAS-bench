#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=16                      # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=4:00:00                          # Specify the maximum time the job can run

# ===========================
# Environment info
# ===========================

echo "========================================"
echo "Testing ATLAS models on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

export HF_TOKEN=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

# ===========================
# Paths
# ===========================

PROJECT_ROOT=/gpfs/work5/0/tesr0602/Tim/atlas-bench/
OUTPUT_ROOT_HOST=${PROJECT_ROOT}/outputs
RESULTS_DIR=${PROJECT_ROOT}/test_results
CONTAINER=${PROJECT_ROOT}/atlas.sif

mkdir -p ${RESULTS_DIR}

cd ${PROJECT_ROOT} || exit 1

# ===========================
# Dataset config
# ===========================

DATA_ZIP=/gpfs/work5/0/tesr0602/Tim/videomt/datasets/atlas/atlas.zip
NUM_CLASSES=30
NUM_WORKERS=16
BATCH_SIZE=32

# ===========================
# Model-to-checkpoint mapping
# ===========================

# Define all models to test
# Format: "model_name|checkpoint_pattern|experiment_pattern|seed|batch_size"
#
# NOTE: Checkpoint naming conventions:
#   - VideoMT/EOMT models: save as "best_model.pth" (exact name)
#   - Other models: save as "best_model_epoch_N_dice_X.XXXX.pt" (with metadata)
#   
# The script automatically handles both patterns:
#   - Tries exact match first (for best_model.pth)
#   - Falls back to glob pattern matching for best_model_*.pt variants
MODELS=(
    # # DINOv2 Pretrained
    # "lh-vit-s-dinov2|None|lh_vits_dinov2_atlas|0|32"
    # "lh-vit-b-dinov2|None|lh_vitb_dinov2_atlas|0|32"
    # "lh-vit-l-dinov2|None|lh_vitl_dinov2_atlas|0|32"
    
    # # DINOv3 Pretrained
    # "lh-vit-b-dinov3|None|lh_vitb_dinov3_atlas|0|32"
    # "lh-vit-l-dinov3|None|lh_vitl_dinov3_atlas|0|32"
    
    # # DINOv1 SurgeNet2M
    # "lh-dinov1-vitb-224-surgenet2m|best_model.pth|lh_dinov1_vitb_224_surgenet2m_atlas|0|32"
    
    # # DINOv2 SurgeNet2M
    # "lh-dinov2-vitb-336-surgenet2m|best_model.pth|lh_dinov2_vitb_336_surgenet2m_atlas|0|32"
    
    # # DINOv3 SurgeNet2M
    # "lh-dinov3-vitb-256-surgenet2m|best_model.pth|lh_dinov3_vitb_256_surgenet2m_atlas|0|32"
    # "lh-dinov3-vitl-256-surgenet2m|best_model.pth|lh_dinov3_vitl_256_surgenet2m_atlas|0|32"
    
    # ATLAS models (temporal)
    # "atlas_vitl_dinov3|best_model.pth|atlas_vitl_dinov3_surgenet|0|32"
    # "atlas_vitb_dinov3|best_model.pth|atlas_vitb_dinov3_surgenet|0|32"
    # "atlas_vits_dinov3|best_model.pth|atlas_vits_dinov3_surgenet|0|32"
    # "atlas_vitl_dinov3_tracking|best_model.pth|atlas_vitl_dinov3_tracking_surgenet|0|32"
    "atlas_vitb_dinov2|best_model.pth|atlas_vitb_dinov2_surgenet_336|0|32"
    "atlas_vitb_dinov1|best_model.pth|atlas_vitb_dinov1_surgenet_224|0|32"

    # # EOMT SurgeNet models
    # "eomt_vitl_dinov3|best_model.pth|eomt_dinov3_vitl_surgenet_256|0|32"

    # EOMT ImageNet models 
    # "eomt_vitl_dinov3|best_model.pth|eomt_dinov3_vitl_256|0|32"
    # "eomt_vitb_dinov2|best_model.pth|eomt_dinov2_vitb_518|0|32"
    # "eomt_vitb_dinov3|best_model.pth|eomt_dinov3_vitb_256|0|32"

    # # # # SurgeNet Baselines
    # "surgenet-pvtv2-b2|best_model.pth|pvtv2_atlas|0|32"
    # "surgenet-convnextv2-tiny|best_model.pth|convnextv2_atlas|0|32"
    # "surgenet-caformer-s18|best_model.pth|caformer_atlas|0|32"
       
    # # Other models 
    # "endofm|best_model.pth|endofm_atlas|0|32"
    # "endovit|best_model.pth|endovit_atlas|0|32"
    # "gastronet5m|best_model.pth|lh_gastronet5m_atlas|0|32"
)

# ===========================
# Test each model
# ===========================

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME CHECKPOINT_PATTERN EXPERIMENT_PATTERN SEED BATCH_SIZE <<< "$model_config"
    
    EXPERIMENT_NAME="${EXPERIMENT_PATTERN}_seed${SEED}"
    OUTPUT_PATH="${OUTPUT_ROOT_HOST}/${EXPERIMENT_NAME}"
    RESULT_FILE="${RESULTS_DIR}/${EXPERIMENT_NAME}_test_results.json"
    
    echo ""
    echo "========================================"
    echo "Testing: ${EXPERIMENT_NAME}"
    echo "Model: ${MODEL_NAME}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "========================================"
    
    # Check if experiment directory exists
    if [ ! -d "${OUTPUT_PATH}" ]; then
        echo "⚠️  Experiment directory not found: ${OUTPUT_PATH}"
        echo "Skipping..."
        continue
    fi
    
    # Build checkpoint argument
    if [ "${CHECKPOINT_PATTERN}" != "None" ]; then
        CHECKPOINT_PATH=""
        
        # Try exact match first (for videomt/eomt with simple best_model.pth)
        if [ -f "${OUTPUT_PATH}/${CHECKPOINT_PATTERN}" ]; then
            CHECKPOINT_PATH="${OUTPUT_PATH}/${CHECKPOINT_PATTERN}"
        else
            # Try glob pattern for models with metadata in filename (e.g., best_model_epoch_50_dice_0.8234.pt)
            # Get base name (without extension)
            base_name="${CHECKPOINT_PATTERN%.*}"
            
            # Try both .pt and .pth extensions
            for ext in pt pth; do
                for f in "${OUTPUT_PATH}/${base_name}"_*.${ext}; do
                    if [ -f "$f" ]; then
                        CHECKPOINT_PATH="$f"
                        break 2  # Break out of both loops
                    fi
                done
            done
        fi
        
        if [ -z "${CHECKPOINT_PATH}" ] || [ ! -f "${CHECKPOINT_PATH}" ]; then
            echo "⚠️  Checkpoint not found matching pattern: ${OUTPUT_PATH}/${CHECKPOINT_PATTERN}"
            echo "Tried: ${OUTPUT_PATH}/${base_name}_*.{pt,pth}"
            echo "Skipping..."
            continue
        fi
        
        CHECKPOINT_ARG="--checkpoint ${CHECKPOINT_PATH}"
    else
        CHECKPOINT_ARG=""
    fi
    
    echo "Running test with checkpoint: ${CHECKPOINT_PATH:-'None (pretrained)'}"
    
    # Run test in container
    apptainer exec --nv \
        --bind ${PROJECT_ROOT}:/workspace \
        --bind ${DATA_ZIP}:/data/atlas.zip \
        --pwd /workspace \
        ${CONTAINER} \
        python3 test_atlas.py \
            --model ${MODEL_NAME} \
            ${CHECKPOINT_ARG} \
            --data_path /data/atlas.zip \
            --num_classes ${NUM_CLASSES} \
            --batch_size ${BATCH_SIZE} \
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

# ===========================
# Generate summary report
# ===========================

echo ""
echo "Generating summary report..."

python3 << 'EOF'
import json
import os
from pathlib import Path

results_dir = Path("/gpfs/work5/0/tesr0602/Tim/atlas-bench/test_results")

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

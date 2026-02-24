#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=16                      # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=6:00:00                          # Specify the maximum time the job can run

# ===========================
# Environment info
# ===========================

export HF_TOKEN=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


echo "========================================"
echo "Evaluating SAM models on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# ===========================
# Paths
# ===========================

PROJECT_ROOT=/gpfs/work5/0/tesr0602/Tim/atlas-bench/
OUTPUT_ROOT_HOST=${PROJECT_ROOT}/outputs
RESULTS_DIR=${PROJECT_ROOT}/test_results/sam
CONTAINER=${PROJECT_ROOT}/atlas.sif

mkdir -p ${RESULTS_DIR}

cd ${PROJECT_ROOT} || exit 1

# ===========================
# Dataset config
# ===========================

DATA_ZIP=/gpfs/work5/0/tesr0602/Tim/videomt/datasets/atlas/atlas.zip
NUM_CLASSES=30
NUM_WORKERS=16
BATCH_SIZE=1  # SAM evaluation requires batch_size=1
IMG_SIZE=1024  # SAM models use 1024x1024 input
VIZ_SAMPLES=25  # Number of visualization samples to save
SEED=0

# ===========================
# SAM Model configurations
# ===========================

# Define SAM models and click counts to test
# Format: "model_name|num_clicks"
SAM_CONFIGS=(
    "sam2-hiera-large|1"
    "sam2-hiera-large|10"
)

# ===========================
# Debugging mode (set to true to debug instead of full eval)
# ===========================

DEBUG_MODE=true

if [ "$DEBUG_MODE" = true ]; then
    echo ""
    echo "========================================"
    echo "RUNNING IN DEBUG MODE"
    echo "========================================"
    echo "Debugging first SAM configuration..."
    
    MODEL_NAME="sam2-hiera-large"
    NUM_CLICKS=3
    
    echo ""
    echo "Running debug_sam.py on single clip..."
    echo "This will test with only 3 frames for fast diagnostics"
    echo ""
    
    apptainer exec --nv \
        --bind ${PROJECT_ROOT}:/workspace \
        --bind ${DATA_ZIP}:/data/atlas.zip \
        --pwd /workspace \
        ${CONTAINER} \
        python3 debug_sam.py \
            --model ${MODEL_NAME} \
            --data_path /data/atlas.zip \
            --num_classes ${NUM_CLASSES} \
            --num_clicks ${NUM_CLICKS} \
            --img_size ${IMG_SIZE} \
            --seed ${SEED}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Debug completed successfully"
        echo ""
        echo "If you see:"
        echo "  - Frame 0 with predictions but Frames 1+ empty → SAM2 temporal tracking issue"
        echo "  - All frames empty → No prompts or SAM not working"
        echo "  - Low Dice scores → Model not aligning with GT"
        echo ""
    else
        echo "❌ Debug failed"
    fi
    
    echo "To run full evaluation after debugging, set DEBUG_MODE=false"
    exit 0
fi

# ===========================
# Test each configuration
# ===========================

for sam_config in "${SAM_CONFIGS[@]}"; do
    IFS='|' read -r MODEL_NAME NUM_CLICKS <<< "$sam_config"
    
    EXPERIMENT_NAME="${MODEL_NAME}_clicks${NUM_CLICKS}_seed${SEED}"
    VIZ_DIR="${OUTPUT_ROOT_HOST}/sam_visualizations/${EXPERIMENT_NAME}"
    RESULT_FILE="${RESULTS_DIR}/${EXPERIMENT_NAME}_results.json"
    
    echo ""
    echo "========================================"
    echo "Evaluating: ${EXPERIMENT_NAME}"
    echo "Model: ${MODEL_NAME}"
    echo "Clicks per class: ${NUM_CLICKS}"
    echo "========================================"
    
    # Option to run with debug output (verbose diagnostics)
    # To enable: add --debug flag below
    
    # Run evaluation in container
    apptainer exec --nv \
        --bind ${PROJECT_ROOT}:/workspace \
        --bind ${DATA_ZIP}:/data/atlas.zip \
        --pwd /workspace \
        ${CONTAINER} \
        python3 eval_sam.py \
            --model ${MODEL_NAME} \
            --data_path /data/atlas.zip \
            --num_classes ${NUM_CLASSES} \
            --num_clicks ${NUM_CLICKS} \
            --img_size ${IMG_SIZE} \
            --batch_size ${BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --seed ${SEED} \
            --output ${RESULT_FILE} \
            --visualize_samples ${VIZ_SAMPLES} \
            --visualize_dir /workspace/outputs/sam_visualizations
    
    if [ $? -eq 0 ]; then
        echo "✅ Evaluation completed successfully"
        echo "Results saved to: ${RESULT_FILE}"
        echo "Visualizations saved to: ${VIZ_DIR}"
    else
        echo "❌ Evaluation failed"
    fi
done

echo ""
echo "========================================"
echo "All SAM evaluations completed!"
echo "Results saved in: ${RESULTS_DIR}"
echo "========================================"

# ===========================
# DEBUGGING & TROUBLESHOOTING
# ===========================

echo ""
echo "========================================"
echo "Debugging & Troubleshooting Guide"
echo "========================================"
echo ""
echo "If you're getting Dice scores of 0, check the following:"
echo ""
echo "Option 1: Run debug_sam.py (fast, single clip)"
echo "  - Set DEBUG_MODE=true at the top of this script"
echo "  - Tests only first 3 frames for quick diagnosis"
echo "  - Shows frame-by-frame predictions vs GT"
echo ""
echo "Option 2: Run eval_sam.py with --debug flag (full evaluation)"
echo "  - Edit the eval_sam.py call to add: --debug"
echo "  - Prints detailed SAM model inputs/outputs"
echo "  - Shows mask shapes, value ranges, thresholding results"
echo ""
echo "Option 3: Check SAM_DEBUGGING_GUIDE.md"
echo "  - Detailed explanation of common issues"
echo "  - How to interpret debug output"
echo "  - Suggested fixes for each type of failure"
echo ""
echo "Common issues:"
echo "  • Frame 0 has predictions, Frames 1+ are empty"
echo "    → SAM2 not using temporal tracking (frame-by-frame processing)"
echo "  • All frames empty with valid clicks generated"
echo "    → SAM model not returning valid masks"
echo "  • Dice scores vary wildly"
echo "    → Click generation or SAM configuration issue"
echo ""
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

results_dir = Path("/gpfs/work5/0/tesr0602/Tim/atlas-bench/test_results/sam")

if not results_dir.exists():
    print("No results directory found")
    exit()

results = []
for result_file in sorted(results_dir.glob("*_results.json")):
    try:
        with open(result_file) as f:
            data = json.load(f)
            results.append({
                "experiment": result_file.stem.replace("_results", ""),
                "model": data.get("model", "unknown"),
                "clicks": data.get("num_clicks_per_class", "unknown"),
                "mIoU": data["metrics"]["mIoU"],
                "Dice": data["metrics"]["Dice"],
                "AP": data["metrics"]["AP"],
                "AP50": data["metrics"]["AP50"],
                "AP75": data["metrics"]["AP75"],
            })
    except Exception as e:
        print(f"Error reading {result_file}: {e}")

if results:
    print("\n" + "="*90)
    print("SAM Models - ATLAS Evaluation Results Summary")
    print("="*90)
    print(f"{'Experiment':<40} {'Clicks':>6} {'mIoU':>8} {'Dice':>8} {'AP':>8} {'AP50':>8} {'AP75':>8}")
    print("-"*90)
    for r in results:
        print(f"{r['experiment']:<40} {r['clicks']:>6} {r['mIoU']:>8.4f} {r['Dice']:>8.4f} {r['AP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f}")
    print("="*90)
    
    # Save as CSV
    csv_path = results_dir / "sam_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("experiment,model,clicks_per_class,mIoU,Dice,AP,AP50,AP75\n")
        for r in results:
            f.write(f"{r['experiment']},{r['model']},{r['clicks']},{r['mIoU']:.4f},{r['Dice']:.4f},{r['AP']:.4f},{r['AP50']:.4f},{r['AP75']:.4f}\n")
    print(f"\nSummary saved to: {csv_path}")
else:
    print("No results found")
EOF

echo ""
echo "Done!"

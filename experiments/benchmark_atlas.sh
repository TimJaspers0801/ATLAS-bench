#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=8                       # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=2:00:00                          # Specify the maximum time the job can run

# ===========================
# Environment info
# ===========================

echo "========================================"
echo "Benchmarking ATLAS models on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

export HF_TOKEN=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

# ===========================
# Paths
# ===========================

PROJECT_ROOT=/gpfs/work5/0/tesr0602/Tim/atlas-bench/
OUTPUT_ROOT_HOST=${PROJECT_ROOT}/outputs
RESULTS_DIR=${PROJECT_ROOT}/benchmark_results
CONTAINER=${PROJECT_ROOT}/atlas.sif

mkdir -p ${RESULTS_DIR}

cd ${PROJECT_ROOT} || exit 1

# ===========================
# Benchmark config
# ===========================

NUM_CLASSES=30

# ===========================
# Model-to-checkpoint mapping
# ===========================

# Define all models to benchmark
# Format: "model_name|checkpoint_pattern|experiment_pattern|seed"
#
# NOTE: Checkpoint naming conventions:
#   - VideoMT/EOMT/ATLAS models: save as "best_model.pth" (exact name)
#   - Other models: save as "best_model_epoch_N_dice_X.XXXX.pt" (with metadata)
#   
# The script automatically handles both patterns:
#   - Tries exact match first (for best_model.pth)
#   - Falls back to glob pattern matching for best_model_*.pt variants
MODELS=(
    # DINOv2 Pretrained
    "lh-vit-s-dinov2|None|lh_vits_dinov2_atlas|0"
    "lh-vit-b-dinov2|None|lh_vitb_dinov2_atlas|0"
    "lh-vit-l-dinov2|None|lh_vitl_dinov2_atlas|0"
    
    # DINOv3 Pretrained
    "lh-vit-b-dinov3|None|lh_vitb_dinov3_atlas|0"
    "lh-vit-l-dinov3|None|lh_vitl_dinov3_atlas|0"
    
    # DINOv1 SurgeNet2M
    "lh-dinov1-vitb-224-surgenet2m|best_model.pth|lh_dinov1_vitb_224_surgenet2m_atlas|0"
    
    # DINOv2 SurgeNet2M
    "lh-dinov2-vitb-336-surgenet2m|best_model.pth|lh_dinov2_vitb_336_surgenet2m_atlas|0"
    
    # DINOv3 SurgeNet2M
    "lh-dinov3-vitb-256-surgenet2m|best_model.pth|lh_dinov3_vitb_256_surgenet2m_atlas|0"
    "lh-dinov3-vitl-256-surgenet2m|best_model.pth|lh_dinov3_vitl_256_surgenet2m_atlas|0"
    
    # ATLAS models (temporal)
    "atlas_vitl_dinov3|best_model.pth|atlas_vitl_dinov3_surgenet|0"
    "atlas_vitb_dinov3|best_model.pth|atlas_vitb_dinov3_surgenet|0"
    "atlas_vits_dinov3|best_model.pth|atlas_vits_dinov3_surgenet|0"
    "atlas_vitl_dinov3_tracking|best_model.pth|atlas_vitl_dinov3_tracking_surgenet|0"

    # EOMT SurgeNet models
    "eomt_vitl_dinov3|best_model.pth|eomt_dinov3_vitl_surgenet_256|0"

    # EOMT ImageNet models 
    "eomt_vitl_dinov3|best_model.pth|eomt_dinov3_vitl_256|0"
    "eomt_vitb_dinov2|best_model.pth|eomt_dinov2_vitb_518|0"
    "eomt_vitb_dinov3|best_model.pth|eomt_dinov3_vitb_256|0"

    # SurgeNet Baselines
    "surgenet-pvtv2-b2|best_model.pth|pvtv2_atlas|0"
    "surgenet-convnextv2-tiny|best_model.pth|convnextv2_atlas|0"
    "surgenet-caformer-s18|best_model.pth|caformer_atlas|0"
       
    # Other models 
    "endofm|best_model.pth|endofm_atlas|0"
    "endovit|best_model.pth|endovit_atlas|0"
    "gastronet5m|best_model.pth|lh_gastronet5m_atlas|0"
)

# ===========================
# Benchmark each model
# ===========================

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME CHECKPOINT_PATTERN EXPERIMENT_PATTERN SEED <<< "$model_config"
    
    EXPERIMENT_NAME="${EXPERIMENT_PATTERN}_seed${SEED}"
    OUTPUT_PATH="${OUTPUT_ROOT_HOST}/${EXPERIMENT_NAME}"
    RESULT_FILE="${RESULTS_DIR}/${EXPERIMENT_NAME}_benchmark.json"
    
    echo ""
    echo "========================================"
    echo "Benchmarking: ${EXPERIMENT_NAME}"
    echo "Model: ${MODEL_NAME}"
    echo "========================================"
    
    # Check if experiment directory exists (for checkpointed models)
    if [ "${CHECKPOINT_PATTERN}" != "None" ] && [ ! -d "${OUTPUT_PATH}" ]; then
        echo "⚠️  Experiment directory not found: ${OUTPUT_PATH}"
        echo "Skipping..."
        continue
    fi
    
    # Build checkpoint argument
    if [ "${CHECKPOINT_PATTERN}" != "None" ]; then
        CHECKPOINT_PATH=""
        
        # Try exact match first (for videomt/eomt/atlas with simple best_model.pth)
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
    
    echo "Running benchmark with checkpoint: ${CHECKPOINT_PATH:-'None (pretrained)'}"
    
    # Run benchmark in container
    apptainer exec --nv \
        --bind ${PROJECT_ROOT}:/workspace \
        --pwd /workspace \
        ${CONTAINER} \
        python3 benchmark_models.py \
            --model ${MODEL_NAME} \
            ${CHECKPOINT_ARG} \
            --num_classes ${NUM_CLASSES} \
            --warmup_iters 200 \
            --test_iters 10000 \
            --output ${RESULT_FILE}
    
    if [ $? -eq 0 ]; then
        echo "✅ Benchmark completed successfully"
        echo "Results saved to: ${RESULT_FILE}"
    else
        echo "❌ Benchmark failed"
    fi
done

echo ""
echo "========================================"
echo "All benchmarks completed!"
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

results_dir = Path("/gpfs/work5/0/tesr0602/Tim/atlas-bench/benchmark_results")

if not results_dir.exists():
    print("No results directory found")
    exit()

results = []
for result_file in sorted(results_dir.glob("*_benchmark.json")):
    try:
        with open(result_file) as f:
            data = json.load(f)
            results.append({
                "experiment": result_file.stem.replace("_benchmark", ""),
                "model": data.get("model_name", "unknown"),
                "img_size": data.get("img_size", "N/A"),
                "params_M": data.get("total_params_M", "N/A"),
                "gflops": data.get("gflops", "N/A"),
                "fps": data.get("fps", "N/A"),
                "latency_ms": data.get("latency_ms", "N/A"),
            })
    except Exception as e:
        print(f"Error reading {result_file}: {e}")

if results:
    print("\n" + "="*100)
    print("ATLAS Benchmark Results Summary")
    print("="*100)
    print(f"{'Experiment':<50} {'Img Size':>10} {'Params(M)':>12} {'GFLOPs':>10} {'FPS':>10} {'Latency(ms)':>12}")
    print("-"*100)
    for r in results:
        img_size = str(r['img_size'])
        params = f"{r['params_M']:.2f}" if isinstance(r['params_M'], (int, float)) else str(r['params_M'])
        gflops = f"{r['gflops']:.2f}" if isinstance(r['gflops'], (int, float)) else str(r['gflops'])
        fps = f"{r['fps']:.2f}" if isinstance(r['fps'], (int, float)) else str(r['fps'])
        latency = f"{r['latency_ms']:.2f}" if isinstance(r['latency_ms'], (int, float)) else str(r['latency_ms'])
        
        print(f"{r['experiment']:<50} {img_size:>10} {params:>12} {gflops:>10} {fps:>10} {latency:>12}")
    print("="*100)
    
    # Save as CSV
    csv_path = results_dir / "benchmark_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("experiment,model,img_size,params_M,gflops,fps,latency_ms\n")
        for r in results:
            f.write(f"{r['experiment']},{r['model']},{r['img_size']},{r['params_M']},{r['gflops']},{r['fps']},{r['latency_ms']}\n")
    print(f"\nSummary saved to: {csv_path}")
else:
    print("No results found")
EOF

echo ""
echo "========================================"
echo "Benchmark summary generated!"
echo "========================================"

#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=8                       # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=2:00:00                          # Specify the maximum time the job can run

# ===========================
# Compare experiment predictions
# ===========================

echo "========================================"
echo "Comparing experiments on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================"

# ===========================
# Configuration
# ===========================

PROJECT_ROOT=/gpfs/work5/0/tesr0602/Tim/atlas-bench/
CONTAINER=${PROJECT_ROOT}/atlasv2.sif

# Default datasets to process
DATASETS=(atlas cholecseg8k)

# Allow override via command line
if [ $# -gt 0 ]; then
    DATASETS=("$@")
fi

# Default experiments to compare
EXPERIMENTS=(
    "atlas_vitl_dinov3_surgenet_seed0"
    #"eomt_dinov3_vitl_surgenet_256_seed0"
    "eomt_dinov3_vitl_256_seed0"
)

cd ${PROJECT_ROOT} || exit 1

# ===========================
# Process each dataset
# ===========================

for DATASET in "${DATASETS[@]}"; do

echo ""
echo "========================================"
echo "Processing dataset: ${DATASET}"
echo "========================================"

VISUALIZATIONS_ROOT=${PROJECT_ROOT}/outputs/visualizations_clips/${DATASET}
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/comparisons3/${DATASET}

mkdir -p ${OUTPUT_ROOT}

# ===========================
# Process all clips
# ===========================

echo ""
echo "Experiments: ${EXPERIMENTS[@]}"
echo "Visualizations root: ${VISUALIZATIONS_ROOT}"
echo ""

if [ ! -d "${VISUALIZATIONS_ROOT}" ]; then
    echo "⚠️  Visualizations root not found: ${VISUALIZATIONS_ROOT}"
    continue
fi

clip_count=0
processed_count=0

# Find all clip folders (atlas: procedure/video/clip, cholecseg8k: video/clip)
if [ "${DATASET}" = "atlas" ]; then
    CLIP_GLOB="${VISUALIZATIONS_ROOT}/*/*/*"
else
    CLIP_GLOB="${VISUALIZATIONS_ROOT}/*/*"
fi

for clip_folder in ${CLIP_GLOB}; do
    if [ ! -d "${clip_folder}" ]; then
        continue
    fi
    
    # Extract clip ID for organizing output
    # clip_folder is like: /path/visualizations_clips/atlas/procedure/video/clip
    clip_name=$(echo "${clip_folder}" | sed "s|${VISUALIZATIONS_ROOT}/||")
    clip_count=$((clip_count + 1))
    
    # Create output folder mirroring clip structure
    output_dir="${OUTPUT_ROOT}/${clip_name}"
    mkdir -p "${output_dir}"
    
    echo "Processing clip ($clip_count): ${clip_name}"
    
    # Check if clip folder has required subfolders
    if [ ! -d "${clip_folder}/images" ] || [ ! -d "${clip_folder}/GT" ]; then
        echo "  ⚠️  Missing images/ or GT/ folder, skipping"
        continue
    fi
    
    # Check if at least one experiment exists
    has_experiments=false
    for exp in "${EXPERIMENTS[@]}"; do
        if [ -d "${clip_folder}/${exp}" ]; then
            has_experiments=true
            break
        fi
    done
    
    if [ "$has_experiments" = false ]; then
        echo "  ⚠️  No experiment directories found, skipping"
        continue
    fi
    
    # Run comparison
    apptainer exec --nv \
        --bind ${PROJECT_ROOT}:/workspace \
        --pwd /workspace \
        ${CONTAINER} \
        python3 compare_experiments.py \
            --clip_dir "${clip_folder}" \
            --experiments "${EXPERIMENTS[@]}" \
            --output_dir "${output_dir}"
    
    if [ $? -eq 0 ]; then
        processed_count=$((processed_count + 1))
    fi
done

echo ""
echo "Dataset ${DATASET} complete!"
echo "Processed: ${processed_count}/${clip_count} clips"
echo "Output root: ${OUTPUT_ROOT}"
echo ""

done  # End dataset loop

echo ""
echo "========================================"
echo "All comparisons complete!"
echo "======================================"

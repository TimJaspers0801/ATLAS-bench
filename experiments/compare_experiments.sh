#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

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

DATASET=${1:-atlas}  # atlas | cholecseg8k
shift
EXPERIMENTS=("$@")  # Remaining args are experiment names

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "Usage: bash visualize_experiments.sh <dataset> <experiment1> [experiment2] ..."
    echo "Example: bash compare_experiments.sh atlas atlas_vitl_dinov3_surgenet eomt_dinov3_vitl_surgenet_256"
    exit 1
fi

VISUALIZATIONS_ROOT=${PROJECT_ROOT}/outputs/visualizations_clips/${DATASET}
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/comparisons/${DATASET}

mkdir -p ${OUTPUT_ROOT}

cd ${PROJECT_ROOT} || exit 1

# ===========================
# Process all clips
# ===========================

echo ""
echo "Dataset: ${DATASET}"
echo "Experiments: ${EXPERIMENTS[@]}"
echo "Visualizations root: ${VISUALIZATIONS_ROOT}"
echo ""

if [ ! -d "${VISUALIZATIONS_ROOT}" ]; then
    echo "⚠️  Visualizations root not found: ${VISUALIZATIONS_ROOT}"
    exit 1
fi

clip_count=0
processed_count=0

# Find all clip folders (one level deep in dataset folder)
for clip_folder in ${VISUALIZATIONS_ROOT}/*/*; do
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
    python3 compare_experiments.py \
        --clip_dir "${clip_folder}" \
        --experiments "${EXPERIMENTS[@]}" \
        --output_dir "${output_dir}"
    
    if [ $? -eq 0 ]; then
        processed_count=$((processed_count + 1))
    fi
done

echo ""
echo "========================================"
echo "Comparison complete!"
echo "Processed: ${processed_count}/${clip_count} clips"
echo "Output root: ${OUTPUT_ROOT}"
echo "========================================"

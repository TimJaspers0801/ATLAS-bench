#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=4:00:00

# ===========================
# Environment info
# ===========================

echo "========================================"
echo "Visualizing experiments on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

export HF_TOKEN=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

# ===========================
# Paths
# ===========================

PROJECT_ROOT=/gpfs/work5/0/tesr0602/Tim/atlas-bench/
OUTPUT_ROOT_HOST=${PROJECT_ROOT}/outputs/visualizations_clips
CONTAINER=${PROJECT_ROOT}/atlas.sif

mkdir -p ${OUTPUT_ROOT_HOST}

cd ${PROJECT_ROOT} || exit 1

# ===========================
# Dataset config
# ===========================

DATASET=atlas  # atlas | cholecseg8k
NUM_CLIPS=5
CLIP_FPS=15
SAVE_FPS=1
MAX_FRAMES_PER_CLIP=10
NUM_WORKERS=16
BATCH_SIZE=1
SEED=0

if [ "${DATASET}" = "atlas" ]; then
    DATA_ZIP=/gpfs/work5/0/tesr0602/Tim/videomt/datasets/atlas/atlas.zip
    NUM_CLASSES=30
else
    DATA_ZIP=/gpfs/work5/0/tesr0602/Tim/videomt/datasets/cholecseg8k/cholecseg8k.zip
    NUM_CLASSES=30
fi

# ===========================
# Model-to-checkpoint mapping
# ===========================

# Format: "model_name|checkpoint_pattern|experiment_pattern|seed"
MODELS=(
    "atlas_vitl_dinov3|best_model.pth|atlas_vitl_dinov3_surgenet|0"
    "eomt_vitl_dinov3|best_model.pth|eomt_dinov3_vitl_surgenet_256|0"
    "eomt_vitl_dinov3|best_model.pth|eomt_dinov3_vitl_256|0"
)

# ===========================
# Visualize each experiment
# ===========================

for model_config in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME CHECKPOINT_PATTERN EXPERIMENT_PATTERN SEED <<< "$model_config"

    EXPERIMENT_NAME="${EXPERIMENT_PATTERN}_seed${SEED}"
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"
    OUTPUT_DIR="${OUTPUT_ROOT_HOST}/${DATASET}"

    echo ""
    echo "========================================"
    echo "Visualizing: ${EXPERIMENT_NAME}"
    echo "Model: ${MODEL_NAME}"
    echo "Dataset: ${DATASET}"
    echo "Clips: ${NUM_CLIPS}"
    echo "========================================"

    if [ ! -d "${OUTPUT_PATH}" ]; then
        echo "⚠️  Experiment directory not found: ${OUTPUT_PATH}"
        echo "Skipping..."
        continue
    fi

    if [ "${CHECKPOINT_PATTERN}" != "None" ]; then
        CHECKPOINT_PATH=""

        if [ -f "${OUTPUT_PATH}/${CHECKPOINT_PATTERN}" ]; then
            CHECKPOINT_PATH="${OUTPUT_PATH}/${CHECKPOINT_PATTERN}"
        else
            base_name="${CHECKPOINT_PATTERN%.*}"
            for ext in pt pth; do
                for f in "${OUTPUT_PATH}/${base_name}"_*.${ext}; do
                    if [ -f "$f" ]; then
                        CHECKPOINT_PATH="$f"
                        break 2
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

    apptainer exec --nv \
        --bind ${PROJECT_ROOT}:/workspace \
        --bind ${DATA_ZIP}:/data/dataset.zip \
        --pwd /workspace \
        ${CONTAINER} \
        python3 visualize_experiments.py \
            --dataset ${DATASET} \
            --data_path /data/dataset.zip \
            --model ${MODEL_NAME} \
            ${CHECKPOINT_ARG} \
            --num_classes ${NUM_CLASSES} \
            --num_clips ${NUM_CLIPS} \
            --clip_fps ${CLIP_FPS} \
            --save_fps ${SAVE_FPS} \
            --max_frames_per_clip ${MAX_FRAMES_PER_CLIP} \
            --seed ${SEED} \
            --output_dir ${OUTPUT_DIR} \
            --experiment_name ${EXPERIMENT_NAME}

done

echo ""
echo "========================================"
echo "All visualizations completed!"
echo "Output root: ${OUTPUT_ROOT_HOST}"
echo "========================================"

#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=16                      # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs to use
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=12:00:00                         # Specify the maximum time the job can run

export WANDB_API_KEY=1cf878a1b1aafcd37a1f6e6ba8fdd18ba1c4affb
export WANDB_DIR=/gpfs/work5/0/tesr0602/Tim/SSL_Pretraining/dino/experiments/$OUTPUT_FOLDER/wandb
export WANDB_CONFIG_DIR=/gpfs/work5/0/tesr0602/Tim/SSL_Pretraining/dino/experiments/$OUTPUT_FOLDER/wandb
export WANDB_CACHE_DIR=/gpfs/work5/0/tesr0602/Tim/SSL_Pretraining/dino/experiments/$OUTPUT_FOLDER/wandb
export WANDB_START_METHOD="thread"
wandb login



# ===========================
# Environment info
# ===========================

echo "========================================"
echo "Starting job on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# ===========================
# Paths
# ===========================

PROJECT_ROOT=/gpfs/work5/0/tesr0602/Tim/atlas-bench/
OUTPUT_ROOT_HOST=${PROJECT_ROOT}/outputs
CONTAINER=${PROJECT_ROOT}/atlas.sif

mkdir -p ${OUTPUT_ROOT_HOST}
mkdir -p ${PROJECT_ROOT}/logs

cd ${PROJECT_ROOT} || exit 1

# ===========================
# Experiment config
# ===========================

DATA_ZIP=/gpfs/work5/0/tesr0602/Tim/videomt/datasets/atlas/atlas.zip
OUTPUT_PATH=/outputs
IMG_SIZE=256
EPOCHS=1
BATCH_SIZE=128
NUM_CLASSES=47
NUM_WORKERS=16
FRAMES_PERCENTAGE=100
SEEDS=(0 1 2)

# ===========================
# Experiment — pvtv2 surgenet
# ===========================

WANDB_GROUP=pvtv2_atlas

for SEED in "${SEEDS[@]}"; do
  EXPERIMENT_NAME=pvtv2_atlas_seed${SEED}

  echo "========================================"
  echo "Running ${EXPERIMENT_NAME}"
  echo "========================================"

  srun apptainer exec --nv \
    --bind ${PROJECT_ROOT}:/workspace \
    --bind ${DATA_ROOT_HOST}:/data \
    --bind ${OUTPUT_ROOT_HOST}:/outputs \
    ${CONTAINER} \
    python3 /workspace/train_frame_level.py \
      --data_path ${DATA_ZIP} \
      --experiment_name ${EXPERIMENT_NAME} \
      --model pvtv2 \
      --num_classes ${NUM_CLASSES} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --img_size ${IMG_SIZE} \
      --output_dir ${OUTPUT_PATH} \
      --num_workers ${NUM_WORKERS} \
      --seed ${SEED} \
      --wandb_group ${WANDB_GROUP} \
      --visualize
done


# ===========================
# Experiment — convnext surgenet
# ===========================

WANDB_GROUP=convnextv2_atlas

for SEED in "${SEEDS[@]}"; do
  EXPERIMENT_NAME=convnextv2_atlas_seed${SEED}

  echo "========================================"
  echo "Running ${EXPERIMENT_NAME}"
  echo "========================================"

  srun apptainer exec --nv \
    --bind ${PROJECT_ROOT}:/workspace \
    --bind ${DATA_ROOT_HOST}:/data \
    --bind ${OUTPUT_ROOT_HOST}:/outputs \
    ${CONTAINER} \
    python3 /workspace/train_frame_level.py \
      --data_path ${DATA_ZIP} \
      --experiment_name ${EXPERIMENT_NAME} \
      --model convnextv2 \
      --num_classes ${NUM_CLASSES} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --img_size ${IMG_SIZE} \
      --output_dir ${OUTPUT_PATH} \
      --num_workers ${NUM_WORKERS} \
      --seed ${SEED} \
      --wandb_group ${WANDB_GROUP} \
      --visualize
done

# ===========================
# Experiment — caformer surgenet
# ===========================

WANDB_GROUP=caformer_atlas

for SEED in "${SEEDS[@]}"; do
  EXPERIMENT_NAME=caformer_atlas_seed${SEED}

  echo "========================================"
  echo "Running ${EXPERIMENT_NAME}"
  echo "========================================"

  srun apptainer exec --nv \
    --bind ${PROJECT_ROOT}:/workspace \
    --bind ${DATA_ROOT_HOST}:/data \
    --bind ${OUTPUT_ROOT_HOST}:/outputs \
    ${CONTAINER} \
    python3 /workspace/train_frame_level.py \
      --data_path ${DATA_ZIP} \
      --experiment_name ${EXPERIMENT_NAME} \
      --model caformer \
      --num_classes ${NUM_CLASSES} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --img_size ${IMG_SIZE} \
      --output_dir ${OUTPUT_PATH} \
      --num_workers ${NUM_WORKERS} \
      --seed ${SEED} \
      --wandb_group ${WANDB_GROUP} \
      --visualize
done


echo "========================================"
echo "Job finished"
echo "========================================"

#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of Nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=16                      # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs to use
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=12:00:00                         # Specify the maximum time the job can run


export APPTAINER_TMPDIR=/gpfs/home1/tjaspers2/temp/


apptainer pull atlas.sif docker://tjmjaspers/atlas:v2
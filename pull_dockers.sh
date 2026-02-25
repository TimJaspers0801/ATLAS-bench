#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of Nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=12                      # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs to use
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=12:00:00                         # Specify the maximum time the job can run


export APPTAINER_TMPDIR=/gpfs/work5/0/tesr0602/Tim/temp/


apptainer pull atlasv2.sif docker://tjmjaspers/atlas:v4
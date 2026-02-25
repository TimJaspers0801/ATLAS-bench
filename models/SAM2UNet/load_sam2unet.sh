#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bme.gpu2.q                        #  elec.gpu.q / bme.gpuresearch.q / bme.gpustudent.q  / bme.gpu2.q
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --time=0-48:00:00					                    # Time in D-HH:MM:SS
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE                  		        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=r.l.p.d.d.jong@student.tue.nl
#SBATCH --mem=40G

#module load cuda11.2/toolkit/11.2.2
#module load cudnn/8.1.1 

#module load cuda11.7/toolkit/11.7.1
#module load cudnn/8.1.1

#module load cuda11.8/toolkit/11.8.0 
#module load cuDNN/8.7.0.84-CUDA-11.8.0

python SAM2UNet.py
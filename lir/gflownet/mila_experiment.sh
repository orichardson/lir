#!/bin/bash

#SBATCH -o /home/mila/v/vivianoj/scratch/lir/logs/gflownet/slurm-%j.out
#SBATCH -e /home/mila/v/vivianoj/scratch/lir/logs/gflownet/slurm-%j.err
#SBATCH -J tb_normalize
#SBATCH --get-user-env
#SBATCH --partition=long
#SBATCH --gres=gpu
#SBATCH --mem=32gb
#SBATCH --time=23:59:59

# Initalize the conda environment on the target node, which should automatically set all
# oneapi variables for the user.
source /home/mila/v/vivianoj/miniconda3/bin/activate
conda activate lir
cd /home/mila/v/vivianoj/code/lir

python -u -m lir.gflownet.tb_normalize --ndim 4 --height 32 --n_iterations 10000

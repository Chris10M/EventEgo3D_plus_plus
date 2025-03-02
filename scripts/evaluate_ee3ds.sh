#!/bin/bash
#SBATCH --signal=B:SIGTERM@120
#SBATCH -p gpu20
#SBATCH --mem=192G
#SBATCH --gres gpu:1
#SBATCH -o ./logs/slurm_evaluations/evalutate-ee3d-s-%j.out
#SBATCH -t 24:00:00
#SBATCH --mail-type=fail
#SBATCH --mail-user=mrkristen@gmail.com

eval "$(conda shell.bash hook)"
  
conda activate EE3D

export BATCH_SIZE=27
export TEMPORAL_STEPS="20"

# export MODEL_PATH='' 
python evaluate_ee3d_s.py

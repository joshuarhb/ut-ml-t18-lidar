#!/bin/bash
#SBATCH --job-name=ground_removal
#SBATCH --output=ground_removal_%j.out
#SBATCH --error=ground_removal_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

source ~/miniconda3/bin/activate 
conda activate lidar_project

mpirun -np 8 python ground_removal.py

conda deactivate

#!/bin/bash
#SBATCH -J ground_removal
#SBATCH --partition=testing
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

source ~/miniconda3/bin/activate 
conda activate lidar_project

python ground_removal.py

conda deactivate
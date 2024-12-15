#!/bin/bash
#SBATCH -J initial_image_generation_from_pcd_files
#SBATCH --partition=testing
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

source ~/miniconda3/bin/activate 
conda activate lidar_project

python test_job_generate_image.py

conda deactivate
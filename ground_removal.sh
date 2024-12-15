#!/bin/bash
#SBATCH --job-name=ground_removal
#SBATCH --output=ground_removal_%j.out
#SBATCH --error=ground_removal_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

# Load necessary modules and activate conda environment
source ~/miniconda3/bin/activate
conda activate lidar_project

# Set number of threads for Python to match SLURM allocation
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run Python script with allocated CPUs
python ground_removal.py

# Deactivate conda environment
conda deactivate
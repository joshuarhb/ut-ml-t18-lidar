#!/bin/bash
#SBATCH --job-name=verify_kiss
#SBATCH --output=verify_kiss_%j.out
#SBATCH --error=verify_kiss_%j.err
#SBATCH --partition=main 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

source ~/miniconda3/bin/activate
conda create -n o3d python=3.8
conda activate o3d
pip install open3d==0.15.2 numpy matplotlib

python verify_kiss.py
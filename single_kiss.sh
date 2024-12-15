#!/bin/bash
#SBATCH --job-name=kiss_icp
#SBATCH --output=kiss_icp_%j.out
#SBATCH --error=kiss_icp_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

source ~/miniconda3/bin/activate
conda activate lidar_project

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found"
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV}" != "lidar_project" ]]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

pip install polyscope

pip install "kiss-icp[all]"

kiss_icp_pipeline --help

kiss_icp_pipeline --visualize /gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes/2024-03-25-15-42-22_mapping_tartu/non_ground_points

conda deactivate
#!/bin/bash
#SBATCH --job-name=generate_video
#SBATCH --output=generate_video_%j.out
#SBATCH --error=generate_video_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=04:00:00  # Adjust based on expected runtime

# Enable debug mode (optional)
set -x

# Load necessary modules (adjust based on your cluster's module system)

# Activate your Conda environment (ensure Open3D and other dependencies are installed)
source ~/miniconda3/bin/activate
conda activate lidar_project  # Replace with your environment name

mkdir /gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes/2024-03-25-15-40-16_mapping_tartu/non_ground_video
# Run the Python script
python generate_video.py \
    --input_dir /gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes/2024-03-25-15-40-16_mapping_tartu/non_ground_points \
    --output_dir /gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes/2024-03-25-15-40-16_mapping_tartu/non_ground_video \
    --video_name output_video.mp4 \
    --fps 30 \
    --voxel_size 0.5 \
    --max_correspondence_distance 2.0 \
    --color_by z

# Deactivate the Conda environment (optional)
conda deactivate

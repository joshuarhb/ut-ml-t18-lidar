#!/bin/bash
#SBATCH --job-name=open3d_icp
#SBATCH --output=open3d_icp_%j.out
#SBATCH --error=open3d_icp_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=16GB

# Enable debug mode to log each command
set -x

# Exit immediately if a command exits with a non-zero status
set -e

# Treat unset variables as an error and exit immediately
set -u

# Prevent errors in a pipeline from being masked
set -o pipefail

# Load necessary modules (adjust based on your cluster's module system)
module load gcc/9.3.0
module load python/3.10.4
module load parallel/20220522  # GNU Parallel

# Activate Conda environment
source ~/miniconda3/bin/activate
conda activate lidar_project  # Replace with your environment name

# Verify Conda environment activation
if [[ "${CONDA_DEFAULT_ENV}" != "lidar_project" ]]; then
    echo "Error: Failed to activate conda environment 'lidar_project'"
    exit 1
fi

# Check if Open3D is installed
python -c "import open3d" || { echo "Error: Open3D is not installed in the 'lidar_project' environment"; exit 1; }

# Define base directories
BASE_INPUT_DIR="/gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes"
BASE_OUTPUT_DIR="/gpfs/space/projects/mlcourse/2024/t18-lidar"
START_DIRECTORY="2024-04-12-14-44-20_mapping_tartu_streets"

# Define the function to process each scene
process_scene() {
    local scene_dir="$1"
    local scene_name
    scene_name=$(basename "$scene_dir")

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting processing for scene: $scene_name"

    # Skip scenes before the start_directory
    if [[ "$scene_name" < "$START_DIRECTORY" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Skipping scene: $scene_name (before start directory)"
        return
    fi

    INPUT_DIR="$scene_dir/non_ground_points"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/ground_removed_scenes/$scene_name/ego_estimation"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Input directory: $INPUT_DIR"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Output directory: $OUTPUT_DIR"

    # Check if input directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Warning: Directory $INPUT_DIR does not exist, skipping..."
        return
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Define paths for transformations and metrics
    TRANSFORMATION_PATH="$OUTPUT_DIR/results/latest/non_ground_points_poses.npy"
    METRICS_PATH="$OUTPUT_DIR/results/latest/chamfer_distances.npy"

    # Ensure the transformations directory exists
    mkdir -p "$(dirname "$TRANSFORMATION_PATH")"
    mkdir -p "$(dirname "$METRICS_PATH")"

    # Run the ICP registration Python script
    python icp_registration.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --transformation_output "$TRANSFORMATION_PATH" \
        --metrics_output "$METRICS_PATH"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed processing for scene: $scene_name"
}

export -f process_scene
export BASE_INPUT_DIR
export BASE_OUTPUT_DIR
export START_DIRECTORY

# Find all scene directories to process
echo "$(date '+%Y-%m-%d %H:%M:%S') - Discovering scene directories..."
processing_dirs=$(find "$BASE_INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# Log the directories to be processed
echo "$(date '+%Y-%m-%d %H:%M:%S') - Directories to process:"
echo "$processing_dirs"

# Run the processing function in parallel using GNU Parallel
echo "$(date '+%Y-%m-%d %H:%M:%S') - Initiating parallel processing..."
echo "$processing_dirs" | parallel --jobs 8 process_scene {}

echo "$(date '+%Y-%m-%d %H:%M:%S') - All scenes have been processed."

# Deactivate Conda environment
conda deactivate

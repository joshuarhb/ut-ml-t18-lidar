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

# Enable debug mode
set -x

# Exit immediately if a command exits with a non-zero status
set -e
# Treat unset variables as an error
set -u
# Prevent errors in a pipeline from being masked
set -o pipefail

# Activate Conda environment
source ~/miniconda3/bin/activate
conda activate lidar_project

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found"
    exit 1
fi



# Verify conda environment activation
if [[ "${CONDA_DEFAULT_ENV}" != "lidar_project" ]]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

# Check if kiss_icp_pipeline is available
if ! command -v kiss_icp_pipeline &> /dev/null; then
    echo "Error: kiss_icp_pipeline not found"
    exit 1
fi

# Load GNU Parallel module
module load parallel/20220522

# Verify GNU Parallel is available
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU Parallel not found after loading module"
    exit 1
fi

# Define base directories
base_input_dir="/gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes"
base_output_dir="/gpfs/space/projects/mlcourse/2024/t18-lidar"

# Check if directories exist
if [ ! -d "$base_input_dir" ]; then
    echo "Error: Base input directory does not exist: $base_input_dir"
    exit 1
fi

# Define the start directory BEFORE exporting variables
start_directory="2024-04-12-14-44-20_mapping_tartu_streets"

# Define the function to process each scene
process_scene() {
    local scene_dir="$1"
    echo "DEBUG: Starting process_scene with argument: $scene_dir"
    
    local scene_name
    scene_name=$(basename "$scene_dir")
    echo "DEBUG: Scene name: $scene_name"

    # Skip scenes before the start_directory
    if [[ "$scene_name" < "$start_directory" ]]; then
        echo "DEBUG: Skipping scene: $scene_name (before start directory)"
        return
    fi

    echo "DEBUG: Processing scene: $scene_name"

    # Define input and output paths for this scene
    local input_dir="$scene_dir/non_ground_points"
    local output_dir="$base_output_dir/ground_removed_scenes/$scene_name/ego_estimation"
    
    echo "DEBUG: Input directory: $input_dir"
    echo "DEBUG: Output directory: $output_dir"

    # Check if input directory exists
    if [ ! -d "$input_dir" ]; then
        echo "Warning: Directory $input_dir does not exist, skipping..."
        return
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Navigate to the output directory
    cd "$output_dir" || { 
        echo "Error: Failed to cd to $output_dir"
        return
    }

    echo "DEBUG: Current directory: $(pwd)"
    echo "Running KISS-ICP on: $input_dir"

    # Run the KISS-ICP pipeline with more verbose output
    if kiss_icp_pipeline "$input_dir" > "kiss_icp_${scene_name}.log" 2> "kiss_icp_${scene_name}.err"; then
        echo "KISS-ICP completed successfully for scene: $scene_name"
        echo "DEBUG: Check logs at: $output_dir/kiss_icp_${scene_name}.log"
    else
        echo "Error: KISS-ICP failed for scene: $scene_name"
        echo "DEBUG: Error log at: $output_dir/kiss_icp_${scene_name}.err"
        cat "kiss_icp_${scene_name}.err"
    fi
}

# Export the function and variables for GNU Parallel
export -f process_scene
export start_directory
export base_input_dir
export base_output_dir

echo "DEBUG: Starting parallel processing"
# Run with verbose parallel output
find "$base_input_dir" -mindepth 1 -maxdepth 1 -type d | sort | \
    parallel --verbose -j 8 process_scene {}

echo "DEBUG: Parallel processing complete"

# Deactivate Conda environment
conda deactivate
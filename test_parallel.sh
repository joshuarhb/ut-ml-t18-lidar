#!/bin/bash
#SBATCH --job-name=parallel_test
#SBATCH --output=parallel_test_%j.out
#SBATCH --error=parallel_test_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem=2GB

# Exit on errors and undefined variables
set -euo pipefail

# Function to log messages with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Define a simple test function
test_parallel() {
    local id="$1"
    echo "Test parallel job $id started."
    sleep 5
    echo "Test parallel job $id completed."
}

export -f test_parallel

# Load GNU Parallel module
module load parallel/20220522

# Run a test parallel job
echo -e "1\n2\n3\n4\n5\n6\n7\n8" | parallel -j 8 test_parallel {}

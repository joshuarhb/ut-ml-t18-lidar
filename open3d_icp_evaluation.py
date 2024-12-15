#!/usr/bin/env python3
"""
ICP Registration and Chamfer Distance Evaluation Script

This script processes a folder of PCD files, aligns each consecutive pair using ICP,
computes the Chamfer Distance between them, and logs the results.

Usage:
    python icp_evaluation.py --input_dir /path/to/pcd_folder --output_dir /path/to/output

Author: Your Name
Date: YYYY-MM-DD
"""

import open3d as o3d
import numpy as np
import argparse
import os
import sys
import logging
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="ICP Registration and Chamfer Distance Evaluation")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory containing PCD files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory where results will be saved.')
    parser.add_argument('--initial_pcd', type=str, default=None,
                        help='Filename of the initial PCD to serve as the reference (optional).')
    parser.add_argument('--voxel_size', type=float, default=0.5,
                        help='Voxel size for downsampling point clouds (default: 0.5 meters).')
    parser.add_argument('--max_correspondence_distance', type=float, default=2.0,
                        help='Maximum correspondence distance for ICP (default: 2.0 meters).')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of alignments and residuals.')
    args = parser.parse_args()
    return args

def setup_logging(output_dir):
    """
    Sets up logging to file and console.

    Args:
        output_dir (str): Directory where the log file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'icp_evaluation.log')
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.info("Logging is set up.")

def chamfer_distance(pcd1, pcd2):
    """
    Computes the Chamfer Distance between two point clouds.

    Args:
        pcd1 (open3d.geometry.PointCloud): Source point cloud.
        pcd2 (open3d.geometry.PointCloud): Target point cloud.

    Returns:
        float: Chamfer Distance.
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    # Build KD-Trees
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    
    # Compute distances from pcd1 to pcd2
    distances1 = []
    for point in points1:
        _, idx, d = pcd2_tree.search_knn_vector_3d(point, 1)
        if d:
            distances1.append(np.sqrt(d[0]))
        else:
            distances1.append(np.inf)  # Handle no neighbor found
    
    # Compute distances from pcd2 to pcd1
    distances2 = []
    for point in points2:
        _, idx, d = pcd1_tree.search_knn_vector_3d(point, 1)
        if d:
            distances2.append(np.sqrt(d[0]))
        else:
            distances2.append(np.inf)
    
    # Chamfer Distance
    chamfer_dist = (np.mean(np.square(distances1)) + np.mean(np.square(distances2))) / 2
    return chamfer_dist

def load_and_preprocess_pcd(pcd_path, voxel_size):
    """
    Loads a PCD file and preprocesses it (downsampling and normal estimation).

    Args:
        pcd_path (str): Path to the PCD file.
        voxel_size (float): Voxel size for downsampling.

    Returns:
        open3d.geometry.PointCloud: Preprocessed point cloud.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        logging.warning(f"Point cloud {pcd_path} has no points.")
        return None
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    return pcd_down

def visualize_alignment(transformed_pcd, target_pcd, frame_idx):
    """
    Visualizes the alignment between two point clouds.

    Args:
        transformed_pcd (open3d.geometry.PointCloud): Transformed source point cloud.
        target_pcd (open3d.geometry.PointCloud): Target point cloud.
        frame_idx (int): Index of the current frame.
    """
    transformed_pcd.paint_uniform_color([1, 0, 0])  # Red
    target_pcd.paint_uniform_color([0, 1, 0])        # Green
    o3d.visualization.draw_geometries([transformed_pcd, target_pcd],
                                      window_name=f"Frame {frame_idx} Alignment",
                                      width=800, height=600)

def visualize_residuals(residuals, frame_idx, output_dir):
    """
    Plots a histogram of residuals and saves the figure.

    Args:
        residuals (list of float): List of residual distances.
        frame_idx (int): Index of the current frame.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, alpha=0.75, color='blue')
    plt.title(f"Residuals Histogram After Transformation {frame_idx}")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"residuals_histogram_frame_{frame_idx}.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved residuals histogram to {plot_path}")

def main():
    args = parse_arguments()
    setup_logging(args.output_dir)
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    initial_pcd_filename = args.initial_pcd
    voxel_size = args.voxel_size
    max_correspondence_distance = args.max_correspondence_distance
    visualize = args.visualize
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # List and sort PCD files
    pcd_files = sorted([
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".pcd")
    ])
    
    if not pcd_files:
        logging.error(f"No PCD files found in the input directory: {input_dir}")
        sys.exit(1)
    
    logging.info(f"Found {len(pcd_files)} PCD files in {input_dir}")
    
    # If initial_pcd is specified, set it as the first PCD; otherwise, use the first file
    if initial_pcd_filename:
        initial_pcd_path = os.path.join(input_dir, initial_pcd_filename)
        if not os.path.isfile(initial_pcd_path):
            logging.error(f"Specified initial PCD file does not exist: {initial_pcd_path}")
            sys.exit(1)
        # Reorder pcd_files to start with initial_pcd
        pcd_files.remove(initial_pcd_path)
        pcd_files = [initial_pcd_path] + pcd_files
        logging.info(f"Using specified initial PCD: {initial_pcd_filename}")
    
    # Initialize lists to store results
    transformations = []
    chamfer_distances = []
    
    # Process each consecutive pair
    for i in range(len(pcd_files) - 1):
        source_pcd_path = pcd_files[i]
        target_pcd_path = pcd_files[i + 1]
        
        frame_idx = i + 1  # For logging purposes
        
        logging.info(f"\nProcessing Frame {frame_idx}:")
        logging.info(f"Source PCD: {source_pcd_path}")
        logging.info(f"Target PCD: {target_pcd_path}")
        
        # Load and preprocess point clouds
        source_pcd = load_and_preprocess_pcd(source_pcd_path, voxel_size)
        target_pcd = load_and_preprocess_pcd(target_pcd_path, voxel_size)
        
        if source_pcd is None or target_pcd is None:
            logging.warning(f"Skipping Frame {frame_idx} due to empty point cloud.")
            transformations.append(np.identity(4))
            chamfer_distances.append(np.inf)
            continue
        
        # Perform ICP Registration
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, max_correspondence_distance,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        transformation = icp_result.transformation
        transformations.append(transformation)
        logging.info(f"Transformation Matrix for Frame {frame_idx}:\n{transformation}")
        
        # Apply transformation to the original (non-downsampled) source point cloud
        transformed_source = o3d.io.read_point_cloud(source_pcd_path)  # Reload original
        transformed_source.transform(transformation)
        
        # Compute Chamfer Distance
        cd = chamfer_distance(transformed_source, o3d.io.read_point_cloud(target_pcd_path))
        chamfer_distances.append(cd)
        logging.info(f"Chamfer Distance for Frame {frame_idx}: {cd:.6f}")
        
        # Visualization (Optional)
        if visualize:
            visualize_alignment(transformed_source, o3d.io.read_point_cloud(target_pcd_path), frame_idx)
        
        # Residuals Visualization (Optional)
        # Here, residuals are distances from transformed source to target
        if visualize:
            points1 = np.asarray(transformed_source.points)
            points2 = np.asarray(o3d.io.read_point_cloud(target_pcd_path).points)
            pcd2_tree = o3d.geometry.KDTreeFlann(o3d.io.read_point_cloud(target_pcd_path))
            residuals = []
            for point in points1:
                _, idx, d = pcd2_tree.search_knn_vector_3d(point, 1)
                if d:
                    residuals.append(np.sqrt(d[0]))
                else:
                    residuals.append(np.inf)
            visualize_residuals(residuals, frame_idx, output_dir)
    
    # Save transformation matrices and Chamfer Distances
    os.makedirs(output_dir, exist_ok=True)
    transformation_output_path = os.path.join(output_dir, 'transformations.npy')
    metrics_output_path = os.path.join(output_dir, 'chamfer_distances.npy')
    
    np.save(transformation_output_path, np.array(transformations))
    np.save(metrics_output_path, np.array(chamfer_distances))
    
    logging.info(f"\nSaved transformation matrices to {transformation_output_path}")
    logging.info(f"Saved Chamfer Distances to {metrics_output_path}")
    
    # Plot Chamfer Distances Over Frames (Optional)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(chamfer_distances) + 1), chamfer_distances, marker='o', linestyle='-')
    plt.title("Chamfer Distance Over Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Chamfer Distance")
    plt.grid(True)
    chamfer_plot_path = os.path.join(output_dir, 'chamfer_distance_plot.png')
    plt.savefig(chamfer_plot_path)
    plt.close()
    logging.info(f"Saved Chamfer Distance plot to {chamfer_plot_path}")
    
    logging.info("ICP Registration and Chamfer Distance Evaluation Completed Successfully.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_video.py

A script to process a folder of PCD files, perform ICP registration, generate images,
and compile them into a video. Designed to be run within a SLURM job.

Usage:
    python generate_video.py --input_dir /path/to/pcd_folder --output_dir /path/to/output_video --video_name output.mp4 --fps 30

Author: Your Name
Date: YYYY-MM-DD
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import argparse
import logging
from sklearn.linear_model import RANSACRegressor

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process PCD files and generate a video.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory containing PCD files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory where results will be saved.')
    parser.add_argument('--video_name', type=str, default='output_video.mp4',
                        help='Filename for the output video.')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video.')
    parser.add_argument('--voxel_size', type=float, default=0.5,
                        help='Voxel size for downsampling point clouds.')
    parser.add_argument('--max_correspondence_distance', type=float, default=2.0,
                        help='Maximum correspondence distance for ICP.')
    parser.add_argument('--color_by', type=str, choices=['z', 'intensity'], default='z',
                        help='Color mapping for image generation.')
    parser.add_argument('--generate_video_only', action='store_true',
                        help='Skip processing PCD files and only generate video from existing images.')
    parser.add_argument('--image_folder', type=str, default='frames',
                        help='Folder name within output_dir to save frames.')
    args = parser.parse_args()
    return args

def setup_logging(output_dir):
    """
    Sets up logging to file and console.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'generate_video.log')
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.info("Logging initialized.")
    logging.info(f"Log file: {log_file}")

def point_cloud_to_image(pcd, color_by='z', limits=None):
    """
    Converts a point cloud to a 2D image by projecting onto the XY-plane.
    
    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to convert.
        color_by (str): 'z' to color by Z-axis, 'intensity' to color by intensity.
        limits (dict): Optional. Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'.
    
    Returns:
        matplotlib.pyplot.Figure: The generated figure.
    """
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Optionally, handle empty point clouds
    if points.size == 0:
        logging.warning("Empty point cloud received for image generation.")
        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        return fig
    
    # Project the points onto the XY plane
    x = points[:, 0]
    y = points[:, 1]
    
    # Color mapping
    if color_by == 'z':
        colors = points[:, 2]  # Z-axis
    elif color_by == 'intensity':
        colors = np.linalg.norm(points[:, :3], axis=1)  # Intensity as distance from origin
    else:
        colors = np.zeros_like(x)
    
    # Define plot limits
    if limits:
        x_min, x_max = limits['x_min'], limits['x_max']
        y_min, y_max = limits['y_min'], limits['y_max']
    else:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(x, y, c=colors, s=1, cmap='viridis')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def save_frame_as_image(fig, filename):
    """
    Saves a matplotlib figure as an image.
    
    Args:
        fig (matplotlib.pyplot.Figure): The figure to save.
        filename (str): The path to save the image.
    """
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    logging.info(f"Saved image: {filename}")

def compute_chamfer_distance(pcd1, pcd2):
    """
    Computes the Chamfer Distance between two point clouds.
    
    Args:
        pcd1 (open3d.geometry.PointCloud): The first point cloud.
        pcd2 (open3d.geometry.PointCloud): The second point cloud.
    
    Returns:
        float: The Chamfer Distance.
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    if points1.size == 0 or points2.size == 0:
        logging.warning("One of the point clouds is empty. Chamfer Distance set to infinity.")
        return float('inf')
    
    # Build KD-Trees
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    
    # Compute distances from pcd1 to pcd2
    distances1 = []
    for point in points1:
        [k, idx, d] = pcd2_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            distances1.append(np.sqrt(d[0]))
        else:
            distances1.append(float('inf'))
    
    # Compute distances from pcd2 to pcd1
    distances2 = []
    for point in points2:
        [k, idx, d] = pcd1_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            distances2.append(np.sqrt(d[0]))
        else:
            distances2.append(float('inf'))
    
    # Chamfer Distance
    chamfer_dist = (np.mean(np.square(distances1)) + np.mean(np.square(distances2))) / 2
    return chamfer_dist

def perform_ground_separation(pcd, voxel_size=0.5, max_distance=2.0):
    """
    Separates ground and non-ground points using RANSAC.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): Voxel size for downsampling.
        max_distance (float): Maximum correspondence distance for RANSAC.

    Returns:
        tuple: (ground_pcd, non_ground_pcd)
    """
    points = np.asarray(pcd.points)
    if points.shape[0] < 3:
        logging.warning("Point cloud too small for RANSAC. Returning empty clouds.")
        return o3d.geometry.PointCloud(), pcd

    X = points[:, :2]  # x and y coordinates
    y = points[:, 2]   # z coordinate (height)

    # Define the RANSAC regressor
    ransac = RANSACRegressor()

    # Fit the model
    try:
        ransac.fit(X, y)
    except Exception as e:
        logging.error(f"RANSAC fitting failed: {e}")
        return o3d.geometry.PointCloud(), pcd

    # Predict inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Separate ground points (inliers) and non-ground points (outliers)
    ground_points = points[inlier_mask]
    non_ground_points = points[outlier_mask]

    # Create Open3D point clouds
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
    ground_pcd.paint_uniform_color([0, 1, 0])  # Green for ground points

    non_ground_pcd = o3d.geometry.PointCloud()
    non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
    non_ground_pcd.paint_uniform_color([1, 0, 0])  # Red for non-ground points

    return ground_pcd, non_ground_pcd

def create_video_from_images(image_folder, video_filename, fps=30):
    """
    Creates a video from a sequence of images.

    Args:
        image_folder (str): Path to the folder containing images.
        video_filename (str): Path to the output video file.
        fps (int): Frames per second for the video.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the images are in the correct order

    if not images:
        logging.error(f"No PNG images found in {image_folder}. Cannot create video.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        logging.error(f"Failed to read the first image: {first_image_path}")
        return
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to read image: {img_path}. Skipping.")
            continue
        video.write(img)

    video.release()
    logging.info(f"Video saved as {video_filename}")

def main():
    args = parse_arguments()
    setup_logging(args.output_dir)

    input_dir = args.input_dir
    output_dir = args.output_dir
    video_name = args.video_name
    fps = args.fps
    voxel_size = args.voxel_size
    max_correspondence_distance = args.max_correspondence_distance
    color_by = args.color_by
    generate_video_only = args.generate_video_only
    image_folder = os.path.join(output_dir, args.image_folder)

    # Create image folder
    os.makedirs(image_folder, exist_ok=True)

    if not generate_video_only:
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

        # Initialize lists to store transformations and Chamfer Distances
        transformations = []
        chamfer_distances = []

        # Iterate over consecutive PCD pairs
        for i in range(len(pcd_files) - 1):
            source_pcd_path = pcd_files[i]
            target_pcd_path = pcd_files[i + 1]

            frame_idx = i + 1  # For naming and logging

            logging.info(f"\nProcessing Frame {frame_idx}:")
            logging.info(f"Source PCD: {source_pcd_path}")
            logging.info(f"Target PCD: {target_pcd_path}")

            # Load point clouds
            source_pcd = o3d.io.read_point_cloud(source_pcd_path)
            target_pcd = o3d.io.read_point_cloud(target_pcd_path)

            if source_pcd.is_empty() or target_pcd.is_empty():
                logging.warning(f"One of the point clouds is empty. Skipping Frame {frame_idx}.")
                transformations.append(np.identity(4))
                chamfer_distances.append(float('inf'))
                continue

            # Downsample point clouds
            source_down = source_pcd.voxel_down_sample(voxel_size)
            target_down = target_pcd.voxel_down_sample(voxel_size)

            # Ground separation using RANSAC
            ground_pcd, non_ground_pcd = perform_ground_separation(source_down, voxel_size, max_correspondence_distance)

            # Perform ICP registration between non-ground points
            if non_ground_pcd.is_empty() or target_down.is_empty():
                logging.warning(f"Non-ground or target point cloud is empty after separation. Skipping ICP for Frame {frame_idx}.")
                transformations.append(np.identity(4))
                chamfer_distances.append(float('inf'))
                continue

            icp_result = o3d.pipelines.registration.registration_icp(
                non_ground_pcd, target_down, max_correspondence_distance,
                np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            transformation = icp_result.transformation
            transformations.append(transformation)
            logging.info(f"Transformation Matrix for Frame {frame_idx}:\n{transformation}")

            # Apply transformation to the original (non-downsampled) source point cloud
            transformed_source = source_pcd.transform(transformation)

            # Compute Chamfer Distance between transformed source and target
            chamfer_dist = compute_chamfer_distance(transformed_source, target_pcd)
            chamfer_distances.append(chamfer_dist)
            logging.info(f"Chamfer Distance for Frame {frame_idx}: {chamfer_dist:.6f}")

            # Generate and save image
            fig = point_cloud_to_image(transformed_source, color_by=color_by)
            image_filename = os.path.join(image_folder, f"frame_{frame_idx:06d}.png")
            save_frame_as_image(fig, image_filename)

        # Save transformations and Chamfer Distances
        transformations_path = os.path.join(output_dir, 'transformations.npy')
        chamfer_distances_path = os.path.join(output_dir, 'chamfer_distances.npy')
        np.save(transformations_path, np.array(transformations))
        np.save(chamfer_distances_path, np.array(chamfer_distances))
        logging.info(f"Saved transformation matrices to {transformations_path}")
        logging.info(f"Saved Chamfer Distances to {chamfer_distances_path}")

    # Create video from images
    video_filename = os.path.join(output_dir, video_name)
    create_video_from_images(image_folder, video_filename, fps)
    logging.info("Video creation completed.")

if __name__ == "__main__":
    main()

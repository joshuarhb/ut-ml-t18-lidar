import open3d as o3d
import jcp_liso
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

def process_directory(dir_index, sub_dir, main_dir):
    try:
        selected_dir = os.path.join(main_dir, sub_dir, "lidar_center")
        lidar_files = sorted(os.listdir(selected_dir))
        logging.info(f"Thread processing directory {sub_dir} with {len(lidar_files)} files")
        
        for i, file in enumerate(lidar_files):
            logging.info(f"Processing {sub_dir}: {i + 1}/{len(lidar_files)} files")
            pcd = o3d.io.read_point_cloud(os.path.join(selected_dir, file))
            points = np.asarray(pcd.points)

            per_point_is_ground = jcp_liso.JPCGroundRemove(
                pcl=points,
                range_img_width=400,
                range_img_height=200,
                sensor_height=1.73,
                delta_R=0.2,
            )

            ground_points = points[per_point_is_ground]
            non_ground_points = points[~per_point_is_ground]

            output_dir = os.path.join("/gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes", sub_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            ground_points_dir = os.path.join(output_dir, "ground_points")
            os.makedirs(ground_points_dir, exist_ok=True)
            ground_pcd = o3d.geometry.PointCloud()
            ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
            o3d.io.write_point_cloud(os.path.join(ground_points_dir, f"{file}"), ground_pcd)

            non_ground_points_dir = os.path.join(output_dir, "non_ground_points")
            os.makedirs(non_ground_points_dir, exist_ok=True)
            non_ground_pcd = o3d.geometry.PointCloud()
            non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
            o3d.io.write_point_cloud(os.path.join(non_ground_points_dir, f"{file}"), non_ground_pcd)

    except Exception as e:
        logging.error(f"Error processing directory {sub_dir}: {str(e)}")

def main():
    logging.basicConfig(level=logging.INFO)
    main_dir = "/gpfs/space/projects/ml2024"
    sub_dirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    # Create tasks for round-robin distribution
    tasks = [(i, sub_dir, main_dir) for i, sub_dir in enumerate(sub_dirs)]
    
    # Use 4 threads to process directories
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda x: process_directory(*x), tasks)


if __name__ == "__main__":
    main()
import open3d as o3d
import numpy as np
import jcp_liso
import os

main_dir = "/gpfs/space/projects/ml2024/2024-07-08-12-15-50_mapping_tartu_streets/lidar_center"

lidar_files = sorted(os.listdir(main_dir))

for file in lidar_files[:20]:
    pcd = o3d.io.read_point_cloud(os.path.join(main_dir, file))
    points = np.asarray(pcd.points)

    range_img_width = 400
    range_img_height = 200
    sensor_height = 1.73
    delta_R = 0.2

    per_point_is_ground = jcp_liso.JPCGroundRemove(
        pcl=points,
        range_img_width=range_img_width,
        range_img_height=range_img_height,
        sensor_height=sensor_height,
        delta_R=delta_R,
    )

    ground_points = points[per_point_is_ground]
    non_ground_points = points[~per_point_is_ground]

    output_dir = "processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ground_points_dir = os.path.join(output_dir, "ground_points")
    if not os.path.exists(ground_points_dir):
        os.makedirs(ground_points_dir)
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
    o3d.io.write_point_cloud(os.path.join(ground_points_dir, f"{file}"), ground_pcd)


    non_ground_points_dir = os.path.join(output_dir, "non_ground_points")
    if not os.path.exists(non_ground_points_dir):
        os.makedirs(non_ground_points_dir)
    non_ground_pcd = o3d.geometry.PointCloud()
    non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
    o3d.io.write_point_cloud(os.path.join(non_ground_points_dir, f"{file}"), non_ground_pcd)
import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt
import os

initial_pcd = o3d.io.read_point_cloud("/gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes/2024-03-25-15-40-16_mapping_tartu/non_ground_points/000000.pcd")
icp_transformation_path = "/gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes/2024-03-25-15-40-16_mapping_tartu/ego_estimation/results/latest/non_ground_points_poses.npy"

transformations = np.load(icp_transformation_path)

print(f"Loaded transformations shape: {transformations.shape}")

pcd_folder = "/gpfs/space/projects/mlcourse/2024/t18-lidar/ground_removed_scenes/2024-03-25-15-40-16_mapping_tartu/non_ground_points/"

print(f"Number of pcd files found: {len(os.listdir(pcd_folder))}")

num_transforms = transformations.shape[0]
print(f"Number of transformations: {num_transforms}")

pcd_files = sorted([
    os.path.join(pcd_folder, file)
    for file in os.listdir(pcd_folder)
    if file.endswith(".pcd")
])

pcd_list = [o3d.io.read_point_cloud(pcd_file) for pcd_file in pcd_files]

def chamfer_distance(pcd1, pcd2):
    """Compute Chamfer Distance between two point clouds."""
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    # Build KD-Trees
    pcd2_tree = o3d.geometry.KDTreeFlann()
    pcd2_tree.set_geometry(pcd2)
    pcd1_tree = o3d.geometry.KDTreeFlann()
    pcd1_tree.set_geometry(pcd1)
    
    # Compute distances from pcd1 to pcd2
    distances1 = []
    for point in points1:
        k, idx, d = pcd2_tree.search_knn_vector_3d(point, 1)
        distances1.append(np.sqrt(d[0]))
    
    # Compute distances from pcd2 to pcd1
    distances2 = []
    for point in points2:
        k, idx, d = pcd1_tree.search_knn_vector_3d(point, 1)
        distances2.append(np.sqrt(d[0]))
    
    # Chamfer Distance
    chamfer_dist = (np.mean(np.square(distances1)) + np.mean(np.square(distances2))) / 2
    return chamfer_dist

for i in range(num_transforms):
    print(f"\nProcessing transformation {i+1}/{num_transforms}")
    
    # Current and next PCD
    current_pcd = pcd_list[i]
    if i == len(pcd_list):
        break
    next_pcd = pcd_list[i + 1]
    
    # Load the transformation matrix
    transformation = transformations[i]
    
    # Apply the transformation to the current PCD
    transformed_pcd = current_pcd.clone()
    transformed_pcd.transform(transformation)
    
    # Visualization (Optional)
    # Assign colors for visualization
    transformed_pcd.paint_uniform_color([1, 0, 0])  # Red
    next_pcd.paint_uniform_color([0, 1, 0])         # Green
    
    # Visualize alignment
    # o3d.visualization.draw_geometries([transformed_pcd, next_pcd],
    #                                   window_name=f"Frame {i} Alignment",
    #                                   width=800, height=600)
    
    # Compute Chamfer Distance
    cd = chamfer_distance(transformed_pcd, next_pcd)
    print(f"Chamfer Distance between transformed PCD {i} and PCD {i+1}: {cd:.6f}")
    
    # Optional: Visualize Residuals
    # Sample points for residual visualization
    num_samples = min(1000, len(np.asarray(transformed_pcd.points)))
    sampled_indices = np.random.choice(len(transformed_pcd.points), size=num_samples, replace=False)
    sampled_points = np.asarray(transformed_pcd.points)[sampled_indices]
    
    # Find nearest neighbors in next_pcd
    residuals = []
    pcd2_tree = o3d.geometry.KDTreeFlann()
    pcd2_tree.set_geometry(next_pcd)

for point in sampled_points:
    k, idx, d = pcd2_tree.search_knn_vector_3d(point, 1)
    residuals.append(np.sqrt(d[0]))
    
    # Plot histogram of residuals
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, alpha=0.75, color='blue')
    plt.title(f"Residuals Histogram After Transformation {i+1}")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
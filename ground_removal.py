import open3d as o3d
import jcp_liso
import os
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

main_dir = "/gpfs/space/projects/ml2024/2024-07-08-12-15-50_mapping_tartu_streets/lidar_center"

if rank == 0:
    lidar_files = sorted(os.listdir(main_dir))
    chunks = np.array_split(lidar_files, size)
    # Ensure all chunks are the same size by padding with None
    max_chunk_size = max(len(chunk) for chunk in chunks)
    chunks = [np.pad(chunk, (0, max_chunk_size - len(chunk)), 'constant', constant_values=None) for chunk in chunks]
    chunks = np.array(chunks, dtype=object)
    print(f"Rank {rank}: Split {len(lidar_files)} files into {size} chunks")
else:
    chunks = None

# Create a receive buffer for all processes
chunk = np.empty(max_chunk_size, dtype=object)

print(f"Rank {rank}: Scattering chunks")
comm.Scatter(chunks, chunk, root=0)

# Remove None values from the received chunk
chunk = [file for file in chunk if file is not None]
print(f"Rank {rank}: Received chunk of {len(chunk)} files")

for i, file in enumerate(chunk):
    print(f"Rank {rank}: Processing {i + 1}/{len(chunk)} files")
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

    print(f"Rank {rank}: Processed {i + 1}/{len(chunk)} files")

    ground_points = points[per_point_is_ground]
    non_ground_points = points[~per_point_is_ground]

    output_dir = "processed"
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

comm.Barrier()

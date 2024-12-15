import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from io import BytesIO

# Set the full path to the PCD files directory and output video file
pcd_directory = "/gpfs/space/projects/ml2024/2024-07-08-12-15-50_mapping_tartu_streets/lidar_center"
output_video = 'minicloud.avi'

# Verify directory exists
if not os.path.exists(pcd_directory):
    raise ValueError(f"Directory not found: {pcd_directory}")

# Check if output directory is writable
output_dir = os.path.dirname(output_video) or '.'
if not os.access(output_dir, os.W_OK):
    raise ValueError(f"Cannot write to output directory: {output_dir}")

def point_cloud_to_image(pcd, color_by='z'):
    try:
        points = np.asarray(pcd.points)
        if len(points) == 0:
            print("Warning: Empty point cloud")
            return None
            
        x = points[:, 0]
        y = points[:, 1]
        
        if color_by == 'z':
            colors = points[:, 2]
        else:
            colors = np.linalg.norm(points[:, :3], axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(x, y, c=colors, s=1, cmap='viridis')
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        plt.close(fig)
        plt.close('all')  # Prevent memory leaks
        
        img_buf.seek(0)
        img_arr = np.frombuffer(img_buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, 1)
        return img
        
    except Exception as e:
        print(f"Error processing point cloud: {e}")
        return None

try:
    # Initialize video writer with first frame
    first_pcd_path = os.path.join(pcd_directory, '008692.pcd')
    if not os.path.isfile(first_pcd_path):
        raise ValueError(f"First frame not found: {first_pcd_path}")
        
    first_frame = point_cloud_to_image(o3d.io.read_point_cloud(first_pcd_path))
    if first_frame is None:
        raise ValueError("Failed to process first frame")
        
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_video, fourcc, 10, (width, height))
    
    # Add first frame to video
    video.write(first_frame)
    print("Processed first frame")

    # Process remaining frames
    total_frames = 9692 - 8692
    processed = 1  # We've already processed the first frame
    
    for i in range(8693, 9692):  # Start from next frame
        try:
            pcd_file = f'{i:06d}.pcd'
            file_path = os.path.join(pcd_directory, pcd_file)
            
            if os.path.isfile(file_path):
                pcd = o3d.io.read_point_cloud(file_path)
                frame = point_cloud_to_image(pcd)
                
                if frame is not None:
                    video.write(frame)
                    processed += 1
                    
                    # Progress update
                    progress = (processed / total_frames) * 100
                    print(f"Processed frame {i:06d} ({progress:.1f}% complete)")
            else:
                print(f"File {file_path} not found, skipping...")
                
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue
            
finally:
    # Ensure video writer is properly closed even if an error occurs
    try:
        if 'video' in locals():
            video.release()
            print(f"Video saved as {output_video}")
            print(f"Total frames processed: {processed} out of {total_frames}")
    except Exception as e:
        print(f"Error closing video writer: {e}")
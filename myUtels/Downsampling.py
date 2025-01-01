import open3d as o3d
import os

def downsample_folder(input_folder, output_folder, voxel_size=None, radius=None):
    """
    Downsample all point clouds in a folder using the specified method.
    Args:
        input_folder (str): Path to the folder containing input point clouds.
        output_folder (str): Path to save downsampled point clouds.
        voxel_size (float): Size for voxel downsampling (optional).
        radius (float): Radius for radius-based downsampling (optional).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ply"):
            print(f"Processing {file_name}...")
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            # Load point cloud
            point_cloud = o3d.io.read_point_cloud(input_path)
            
            # Apply voxel downsampling
            if voxel_size is not None:
                point_cloud = point_cloud.voxel_down_sample(voxel_size)
            
            # Apply radius-based sampling
            if radius is not None:
                points = np.asarray(point_cloud.points)
                selected_indices = []
                selected_points = []

                for i, point in enumerate(points):
                    if i == 0 or np.all(np.linalg.norm(points[selected_indices] - point, axis=1) > radius):
                        selected_indices.append(i)
                        selected_points.append(point)

                point_cloud.points = o3d.utility.Vector3dVector(selected_points)
            
            # Save the downsampled point cloud
            o3d.io.write_point_cloud(output_path, point_cloud)
            print(f"Saved downsampled file: {output_path}")

# Define folders
train_input_folder = "/data/train"
train_output_folder = "/data/train_downsampled"

val_input_folder = "/data/val"
val_output_folder = "/data/val_downsampled"

# Downsample train data
downsample_folder(train_input_folder, train_output_folder, voxel_size=0.1, radius=None)

# Downsample val data
downsample_folder(val_input_folder, val_output_folder, voxel_size=0.1, radius=None)

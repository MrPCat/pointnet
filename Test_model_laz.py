import laspy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG  # Adjust this import to your model's structure

def process_nrw_laz(file_path):
    # Open the LAZ file
    las = laspy.read(file_path)
    
    # Get unique classes
    unique_classes = np.unique(las.classification)

    # Create a mapping from class values to indices
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

    # Map classifications to their indices
    classification_indices = np.array([class_to_index[cls] for cls in las.classification])

    # One-hot encode classification
    one_hot_classification = np.eye(len(unique_classes))[classification_indices]

    # Combine features, substituting RGB with proxy features
    data = np.column_stack([
        las.x, las.y, las.z,  # XYZ coordinates
        las.intensity,  # Intensity as proxy
        las.num_returns,  # Number of returns
        las.return_num,  # Return number
        las.scan_angle_rank,  # Scan angle rank
        one_hot_classification  # One-hot encoded classification
    ])
    
    # Normalize numeric features
    data[:, 3] = data[:, 3] / np.max(data[:, 3]) * 255  # Normalize intensity
    data[:, 4] = data[:, 4] / np.max(data[:, 4])  # Normalize num_returns
    data[:, 5] = data[:, 5] / np.max(data[:, 5])  # Normalize return_num
    data[:, 6] = (data[:, 6] + 90) / 180 * 255  # Normalize scan_angle_rank

    return data

# Dataset Class
class PointCloudDatasetNRW(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        # Process NRW data
        self.data = process_nrw_laz(file_path)
        
        # Extract XYZ and features
        self.xyz = self.data[:, :3].dtype(np.float64)  # XYZ
        self.features = self.data[:, 3:].dtype(np.float64)  # Features
        
        # Normalize XYZ
        self.xyz_mean = np.mean(self.xyz, axis=0)
        self.xyz -= self.xyz_mean

        # Ensure data is divisible by points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
        self.features = self.features[:self.num_clouds * self.points_per_cloud]
        
        if debug:
            self.print_debug_info()

    def print_debug_info(self):
        print("\n--- Dataset Debugging Information ---")
        print(f"Total Points: {len(self.xyz)}")
        print(f"Points per Cloud: {self.points_per_cloud}")
        print(f"Number of Point Clouds: {self.num_clouds}")
        print(f"XYZ Shape: {self.xyz.shape}")
        print(f"Features Shape: {self.features.shape}")
    
    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud

        # Convert XYZ and features to float32 tensors
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)
        features = torch.tensor(self.features[start:end], dtype=torch.float32)

        xyz = xyz.transpose(0, 1)  # [3, points_per_cloud]
        features = features.transpose(0, 1)  # [feature_dim, points_per_cloud]

        return features, xyz

# Load Model
def load_model(model_path, input_dim, output_dim):
    model = PointNet2ClsSSG(in_dim=input_dim, out_dim=output_dim)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Initializing model from scratch.")
    return model

# Main Function
if __name__ == "__main__":
    # File paths
    test_file = r'/content/drive/MyDrive/t1/3dm_32_280_5652_1_nw.laz'  # Adjust path
    model_path = r'/content/drive/MyDrive/t1/checkpoints/pointnet_model.pth'  # Adjust path
    output_file = r'/content/drive/MyDrive/t1/3dm_32_280_5652_1_nw_predictions.txt'  # Adjust path

    # Load the test dataset
    test_dataset = PointCloudDatasetNRW(test_file, points_per_cloud=1024, debug=True)

    # Load the model
    input_dim = test_dataset.features.shape[1]
    output_dim = 11  # Adjust based on your number of classes
    model = load_model(model_path, input_dim=input_dim, output_dim=output_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # DataLoader
    print("CUDA Available:", torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Predictions
    all_predictions = []
    with torch.no_grad():
        for features, xyz in test_loader:
            # Move tensors to device
            features, xyz = features.to(device, dtype=torch.float32), xyz.to(device, dtype=torch.float32)
            
            # Pass through the model
            logits = model(features, xyz)  # Output logits
            predictions = torch.argmax(logits, dim=1)  # Class predictions
            all_predictions.extend(predictions.cpu().numpy())

    # Save predictions
    point_cloud_predictions = np.array(all_predictions).reshape(-1, 1).dtype(np.float64)
    denormalized_xyz = (test_dataset.xyz[:len(point_cloud_predictions) * test_dataset.points_per_cloud]
                        + test_dataset.xyz_mean).dtype(np.float64)

    augmented_data = np.hstack([denormalized_xyz,
                                test_dataset.features[:len(point_cloud_predictions) * test_dataset.points_per_cloud],
                                np.repeat(point_cloud_predictions, test_dataset.points_per_cloud, axis=0)])

    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%.15f',
               header='X\tY\tZ\tIntensity\tNumReturns\tReturnNum\tScanAngle\tClassification\tPrediction', comments='')
    print(f"Predictions saved to {output_file}")

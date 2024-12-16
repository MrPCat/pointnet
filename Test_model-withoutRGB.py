import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG
import laspy


class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        try:
            # Load LAS file
            las = laspy.read(file_path)

            # Extract XYZ
            self.xyz = np.column_stack([las.x, las.y, las.z])
            
            # Extract features based on available dimensions
            intensity = las.intensity / np.max(las.intensity)
            num_returns = las.num_returns / np.max(las.num_returns)
            return_num = las.return_num / np.max(las.return_num)
            
            # Combine features
            self.features = np.column_stack([intensity, num_returns, return_num])
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise

        # Normalize XYZ
        self.xyz_mean = np.mean(self.xyz, axis=0, dtype=np.float64)
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
        print(f"XYZ Mean: {self.xyz_mean}")

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud

        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)
        features = torch.tensor(self.features[start:end], dtype=torch.float32)

        xyz = xyz.transpose(0, 1)
        features = features.transpose(0, 1)

        return features, xyz


def load_model(model_path, input_dim, output_dim):
    model = PointNet2ClsSSG(in_dim=input_dim, out_dim=output_dim)
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Initializing model from scratch.")
    return model

def predict_point_cloud(test_file, model_path, output_file):
    # Load the test dataset
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)

    # Load the model
    input_dim = test_dataset.features.shape[1]  # Number of feature channels
    model = load_model(model_path, input_dim=input_dim, output_dim=11)  # Adjust output_dim based on your classes
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
    point_cloud_predictions = np.array(all_predictions).reshape(-1, 1).astype(np.float64)
    denormalized_xyz = (test_dataset.xyz[:len(point_cloud_predictions) * test_dataset.points_per_cloud]
                        + test_dataset.xyz_mean).astype(np.float64)

    # Determine feature names dynamically
    feature_columns = ['Reflectance', 'NumberOfReturns', 'ReturnNumber']
    
    augmented_data = np.hstack([
        denormalized_xyz,
        test_dataset.features[:len(point_cloud_predictions) * test_dataset.points_per_cloud],
        np.repeat(point_cloud_predictions, test_dataset.points_per_cloud, axis=0)
    ])

    # Save results
    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%.15f',
               header='\t'.join(['X', 'Y', 'Z'] + feature_columns + ['Classification']), 
               comments='')
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # File paths
    test_file = '/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las'
    model_path = '/content/drive/MyDrive/t1/checkpoints/pointnet_epoch_7.pth'
    output_file = '/content/drive/MyDrive/t1/3dm_32_280_5652_1_nw_predictions.txt'

    predict_point_cloud(test_file, model_path, output_file)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG  # Ensure this matches the model from training
import pandas as pd

class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        # Load the dataset
        data = pd.read_csv(file_path, delimiter='\t', dtype=np.float64).values  # Ensure float64 precision
        
        # Extract XYZ and Features (excluding XYZ columns explicitly)
        self.xyz = data[:, :3].astype(np.float64)  # XYZ as float64
        self.features = data[:, 3:].astype(np.float64)  # Features as float64
        
        # Save the mean for denormalization later
        self.xyz_mean = np.mean(self.xyz, axis=0).astype(np.float64)
        
        # Normalize XYZ
        self.xyz -= self.xyz_mean
        
        # Ensure data is divisible by points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        
        # Truncate or pad to ensure exact division
        if len(self.xyz) % self.points_per_cloud != 0:
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

        # Convert XYZ and features to float32 tensors
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)  # [points_per_cloud, 3]
        features = torch.tensor(self.features[start:end], dtype=torch.float32)  # [points_per_cloud, feature_dim]

        # Transpose to match PointNet2 expected input
        xyz = xyz.transpose(0, 1)  # [3, points_per_cloud]
        features = features.transpose(0, 1)  # [feature_dim, points_per_cloud]

        return features, xyz


def load_model(model_path, input_dim, output_dim):
    model = PointNet2ClsSSG(in_dim=input_dim, out_dim=output_dim)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Initializing model from scratch.")
    return model


if __name__ == "__main__":
    # File paths
    test_file = '/content/drive/MyDrive/output_with_rgb1 (1).txt'
    model_path = r"C:\Users\faars\Downloads\modelnet40ply2048-train-pointnet++.pth"
    output_file = '/content/drive/MyDrive/output_new1_predictions.txt'
    
    # Load the test dataset
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)

    # Load the model
    input_dim = test_dataset.features.shape[1]
    model = load_model(model_path, input_dim=input_dim, output_dim=11)  # Adjust `output_dim` based on your classes
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

# Save predictions (denormalization in float64)
point_cloud_predictions = np.array(all_predictions).reshape(-1, 1).astype(np.float64)
denormalized_xyz = (test_dataset.xyz[:len(point_cloud_predictions) * test_dataset.points_per_cloud]
                    + test_dataset.xyz_mean).astype(np.float64)

augmented_data = np.hstack([denormalized_xyz,
                            test_dataset.features[:len(point_cloud_predictions) * test_dataset.points_per_cloud],
                            np.repeat(point_cloud_predictions, test_dataset.points_per_cloud, axis=0)])

np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%.15f',
           header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
print(f"Predictions saved to {output_file}")



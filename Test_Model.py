import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pointnet_ import PointNet2ClsSSG

# Enable CUDA DSA for device-side assertions
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Dataset class
class TestDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

        self.xyz = self.data[:, :3]  # XYZ coordinates
        self.features = self.data[:, 3:]  # Features (no labels in test)

        self.xyz -= np.mean(self.xyz, axis=0)  # Normalize spatial coordinates

        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud

        if len(self.xyz) % self.points_per_cloud != 0:
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz

# Debugging wrapper for _ball_query
def _ball_query(src, query, radius, k):
    try:
        idx, dists = ball_query_pytorch(src, query, radius, k)
        # Debugging: Print tensor information
        print(f"src shape: {src.shape}, query shape: {query.shape}")
        print(f"radius: {radius}, k: {k}")
        print(f"idx shape: {idx.shape}, dists shape: {dists.shape}")
        return idx, dists
    except RuntimeError as e:
        print(f"RuntimeError in _ball_query: {e}")
        raise

# Load test dataset
test_file = "/content/drive/MyDrive/t1/Mar18_test.txt"
test_dataset = TestDataset(test_file, points_per_cloud=1024)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load trained model
model = PointNet2ClsSSG(in_dim=6, out_dim=11, downsample_points=(512, 128))
model.load_state_dict(torch.load("/content/drive/MyDrive/t1/pointnet_model.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Perform inference
predictions = []
with torch.no_grad():
    for features, xyz in test_loader:
        features, xyz = features.to(device), xyz.to(device)

        try:
            logits = model(features, xyz)  # Forward pass on GPU
            preds = torch.argmax(logits, dim=1)  # Get predicted classes
            predictions.extend(preds.cpu().numpy())
        except RuntimeError as e:
            print(f"Error during forward pass: {e}")
            torch.cuda.empty_cache()
            raise

# Save predictions
output_file = "/content/drive/MyDrive/t1/predictions.txt"
np.savetxt(output_file, predictions, fmt="%d")
print(f"Predictions saved to {output_file}")

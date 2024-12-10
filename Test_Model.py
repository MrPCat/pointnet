# Test the model
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pointnet_ import PointNet2ClsSSG

# Ensure deterministic behavior
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Test Dataset class
class TestDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

        self.xyz = self.data[:, :3]  # XYZ coordinates
        self.features = self.data[:, 3:]  # Features (no labels in test)

        # Normalize XYZ spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud

        # Truncate to make divisible
        total_points = self.num_clouds * self.points_per_cloud
        self.xyz = self.xyz[:total_points]
        self.features = self.features[:total_points]

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz


# Load the test dataset
test_file = "/content/drive/MyDrive/t1/Mar18_test.txt"
test_dataset = TestDataset(test_file, points_per_cloud=1024)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the trained model
model = PointNet2ClsSSG(in_dim=6, out_dim=11, downsample_points=(512, 128))  # Adjust in_dim if necessary
model.load_state_dict(torch.load("/content/drive/MyDrive/t1/pointnet_model.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Perform inference and save predictions
predictions = []
with torch.no_grad():
    for features, xyz in test_loader:
        # Debugging shape of input data
        print(f"Features shape: {features.shape}, XYZ shape: {xyz.shape}")
        features, xyz = features.to(device), xyz.to(device)
        logits = model(features, xyz)
        preds = torch.argmax(logits, dim=1)  # Predicted classes
        predictions.extend(preds.cpu().numpy())

# Save predictions to a file
output_file = "/content/drive/MyDrive/t1/predictions.txt"
np.savetxt(output_file, predictions, fmt="%d")
print(f"Predictions saved to {output_file}")

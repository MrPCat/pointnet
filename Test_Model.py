# Test the Model
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pointnet_ import PointNet2ClsSSG

# Enable CUDA Error Debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Load Test Dataset (without labels)
class TestDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

        # Split into spatial (xyz) and feature columns
        self.xyz = self.data[:, :3]  # XYZ coordinates
        self.features = self.data[:, 3:]  # Features (no labels in test)

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        self.points_per_cloud = points_per_cloud
        self.num_points = len(self.xyz)

        # Ensure the dataset points are divisible by points_per_cloud
        if self.num_points % self.points_per_cloud != 0:
            remainder = self.num_points % self.points_per_cloud
            self.xyz = self.xyz[:-remainder]
            self.features = self.features[:-remainder]
            self.num_points = len(self.xyz)

        self.num_clouds = self.num_points // self.points_per_cloud

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]

        # Ensure the feature dimensions match the model's input expectations
        num_features_needed = 6  # Example: Model expects 6 feature dimensions
        if features.shape[0] < num_features_needed:
            padding = torch.zeros((num_features_needed - features.shape[0], features.shape[1]))
            features = torch.cat([features, padding], dim=0)
        
        return features, xyz

# Load the test dataset
test_file = "/content/drive/MyDrive/t1/Mar18_test.txt"
test_dataset = TestDataset(test_file, points_per_cloud=1024)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the trained model
model = PointNet2ClsSSG(in_dim=6, out_dim=11, downsample_points=(512, 128))
model.load_state_dict(torch.load("/content/drive/MyDrive/t1/pointnet_model.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Perform inference
predictions = []
with torch.no_grad():
    for features, xyz in test_loader:
        print(f"Features shape: {features.shape}, XYZ shape: {xyz.shape}")  # Debugging output
        features, xyz = features.to(device), xyz.to(device)
        logits = model(features, xyz)
        preds = torch.argmax(logits, dim=1)  # Get the predicted classes
        predictions.extend(preds.cpu().numpy())

# Save predictions to a file
np.savetxt("/content/drive/MyDrive/t1/predictions.txt", predictions, fmt="%d")
print("Predictions saved to predictions.txt")

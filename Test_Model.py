import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG


class TestDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        # Load data from the test file
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=0)  # Adjust based on the file delimiter

        self.xyz = self.data[:, :3]  # Columns X, Y, Z
        self.features = self.data[:, 3:]  # Remaining columns (R, G, B, Reflectance, etc.)

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Ensure the number of points is divisible by `points_per_cloud`
        self.points_per_cloud = points_per_cloud
        total_points = len(self.xyz)
        excess_points = total_points % points_per_cloud
        if excess_points != 0:
            pad_size = points_per_cloud - excess_points
            self.xyz = np.vstack([self.xyz, self.xyz[:pad_size]])  # Pad with duplicates
            self.features = np.vstack([self.features, self.features[:pad_size]])

        self.num_clouds = len(self.xyz) // points_per_cloud

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Extract the points for the current cloud
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz
# File paths
test_file = "/path/to/your/test_file.txt"  # Replace with your actual test file path
model_path = "/path/to/your/model_checkpoint.pth"  # Replace with your model path
output_file = "/path/to/output/predictions.txt"  # Where predictions will be saved

# Dataset and DataLoader
test_dataset = TestDataset(test_file, points_per_cloud=1024)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNet2ClsSSG(in_dim=9, out_dim=11, downsample_points=(512, 128))  # in_dim=9 for X, Y, Z, and features
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Inference
predictions = []
with torch.no_grad():
    for features, xyz in test_loader:
        features, xyz = features.to(device), xyz.to(device)
        logits = model(features, xyz)
        preds = torch.argmax(logits, dim=1)  # Predicted class indices
        predictions.extend(preds.cpu().numpy())

# Save predictions
np.savetxt(output_file, predictions, fmt="%d")
print(f"Predictions saved to {output_file}")

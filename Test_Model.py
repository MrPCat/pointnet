import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG  # Assuming your model is imported here

# Define the Test Dataset
class TestDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        # Dynamically determine the number of features
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
        self.xyz = self.data[:, :3]  # XYZ coordinates
        self.features = self.data[:, 3:]  # All features beyond XYZ
        self.xyz -= np.mean(self.xyz, axis=0)  # Normalize spatial coordinates

        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud

        if len(self.xyz) % self.points_per_cloud != 0:
            print("Warning: Dataset points not divisible by points_per_cloud. Truncating extra points.")
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]

        self.in_dim = self.features.shape[1] + 3  # Dynamically set input dimensions

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz

# Initialize the Test Dataset
test_file = "/content/drive/MyDrive/t1/Mar18_test.txt"  # Update with the correct path
test_dataset = TestDataset(test_file, points_per_cloud=1024)

# Dynamically set in_dim based on dataset
in_dim = test_dataset.in_dim  # Includes XYZ + other features

# Initialize Model
model = PointNet2ClsSSG(in_dim=in_dim, out_dim=11, downsample_points=(512, 128))

# Load the Pretrained Model
model_path = "/content/drive/MyDrive/t1/pointnet_model.pth"  # Update with the correct model path
model.load_state_dict(torch.load(model_path))

# Move Model to Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Inference
predictions = []
with torch.no_grad():
    for features, xyz in test_loader:
        features, xyz = features.to(device), xyz.to(device)
        logits = model(features, xyz)
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Save Predictions
np.savetxt("/content/drive/MyDrive/t1/Mar18_test_predicted.txt", predictions, fmt="%d")
print("Predictions saved to predictions.txt")

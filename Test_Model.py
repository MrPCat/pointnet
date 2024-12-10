import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pointnet_ import PointNet2ClsSSG

# Define the test dataset class
class TestDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

        self.xyz = self.data[:, :3]  # XYZ coordinates
        self.features = self.data[:, 3:]  # Features (excluding XYZ)

        # If no additional features are present, add a dummy feature
        if self.features.shape[1] == 0:
            self.features = np.ones((len(self.xyz), 1))  # Dummy feature

        # Normalize XYZ
        self.xyz -= np.mean(self.xyz, axis=0)

        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud

        assert len(self.xyz) % self.points_per_cloud == 0, \
            "Dataset points not divisible by points_per_cloud."

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PointNet2ClsSSG(in_dim=6, out_dim=11, downsample_points=(512, 128))
model.load_state_dict(torch.load("/content/drive/MyDrive/t1/pointnet_model.pth"))
model.to(device)
model.eval()

# Perform inference
predictions = []

with torch.no_grad():
    for features, xyz in test_loader:
        features, xyz = features.to(device), xyz.to(device)
        logits = model(features, xyz)
        preds = torch.argmax(logits, dim=1)  # Predicted classes
        predictions.extend(preds.cpu().numpy())

# Save predictions to a file
np.savetxt("/content/drive/MyDrive/t1/predictions.txt", predictions, fmt="%d")
print("Predictions saved to predictions.txt")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG  # Assuming your model is imported here

import numpy as np
import torch
from torch.utils.data import Dataset

class DynamicFeatureDataset(Dataset):
    def __init__(self, file_path, features_to_match, points_per_cloud=1024):
        # Load the dataset
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
        
        # Define all possible features (assuming columns are known)
        all_feature_names = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Reflectance', 'NumberOfReturns', 'ReturnNumber']
        feature_indices = {name: idx for idx, name in enumerate(all_feature_names)}

        # Determine which features are available in the current file
        current_feature_names = features_to_match  # Pass in the training features

        # Match features
        self.matched_features = [name for name in all_feature_names if name in current_feature_names]
        self.feature_indices = [feature_indices[name] for name in self.matched_features]
        
        # Extract XYZ, matched features, and labels (if available)
        self.xyz = self.data[:, :3]  # Always include XYZ
        self.features = self.data[:, self.feature_indices]
        self.labels = self.data[:, -1] if self.data.shape[1] > max(self.feature_indices) + 1 else None

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Handle points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        if len(self.xyz) % self.points_per_cloud != 0:
            print("Warning: Dataset points not divisible by points_per_cloud. Truncating extra points.")
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]
            if self.labels is not None:
                self.labels = self.labels[:self.num_clouds * self.points_per_cloud]

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Get a subset of points for the current cloud
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        
        if self.labels is not None:
            labels = torch.tensor(self.labels[start:end], dtype=torch.long)  # Shape: [points_per_cloud]
            label = torch.mode(labels).values  # Use the most common label as the cloud's label
            return features, xyz, label
        else:
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

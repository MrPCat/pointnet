import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNetCls, STN
from pointnet_ import PointNet2ClsSSG 

# === Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        # Load the entire dataset
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

        # Split into xyz, features, and labels
        self.xyz = self.data[:, :3]  # First 3 columns are X, Y, Z
        self.features = self.data[:, 3:-1]  # Columns 4 to second last are features
        self.labels = self.data[:, -1]  # Last column is Classification

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Group points into point clouds
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Get a subset of points for the current cloud
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        labels = torch.tensor(self.labels[start:end], dtype=torch.long)  # Shape: [points_per_cloud]

        # Use the most common label as the cloud's label
        label = torch.mode(labels).values

        return features, xyz, label



# === Model Setup ===
def create_model(in_dim, num_classes):
    print(f"Creating model with input dimension: {in_dim}")
    stn_3d = STN(in_dim=in_dim, out_nd=3)
    model = PointNetCls(in_dim=in_dim, out_dim=num_classes, stn_3d=stn_3d)
    return model


# === Training Loop ===
def train_model(model, data_loader, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for features, xyz, labels in data_loader:
            print(f"Features shape: {features.shape}")
            print(f"XYZ shape: {xyz.shape}")
            print(f"Labels shape: {labels.shape}")

            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features, xyz)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")


# === Main Function ===
if __name__ == "__main__":
    # === Specify File Paths ===
    train_file = '/content/drive/MyDrive/t1/Mar18_train.txt'
    val_file = '/content/drive/MyDrive/t1/Mar18_val.txt'

    # === Dataset and DataLoader ===
    batch_size = 16

    # First, inspect the data
    sample_data = np.loadtxt(train_file, skiprows=1)
    print("Sample data shape:", sample_data.shape)
    print("Sample data first row:", sample_data[0])

    train_dataset = PointCloudDataset(train_file, points_per_cloud=1024)
    val_dataset = PointCloudDataset(val_file, points_per_cloud=1024)

    in_dim = train_dataset.features.shape[0]  # Number of feature channels
    num_classes = len(np.unique(train_dataset.labels))  # Number of classe

    print(f"Detected input dimension: {in_dim}")
    print(f"Detected number of classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Inspect a batch from DataLoader ===
    for features, xyz, label in train_loader:
        print(f"Features shape: {features.shape}")  # Expected: [Batch size, F, Points]
        print(f"XYZ shape: {xyz.shape}")            # Expected: [Batch size, 3, Points]
        print(f"Labels shape: {label.shape}")       # Expected: [Batch size]
        break  # Inspect only the first batch

    # === Check GPU Availability ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    downsample_points = (512, 128)  # Example: Use a maximum of 512 points per cloud
    model = PointNet2ClsSSG(
    in_dim=in_dim,
    out_dim=num_classes,
    downsample_points=downsample_points)

    # === Initialize Model ===
    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes)

    # Test model with a batch
    for batch_features, batch_xyz, batch_labels in train_loader:
        output = model(batch_features, batch_xyz)  # Pass features and xyz to model
        print(f"Model output shape: {output.shape}")  # Expected: [Batch size, num_classes]
        break

    # === Training ===
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion, epochs, device)

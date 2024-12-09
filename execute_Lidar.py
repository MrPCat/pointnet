import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNetCls, STN
from pointnet_ import PointNet2ClsSSG  # Replace with your actual file/module name

# === Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, file_path):
        # Load the entire dataset
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)

        # Split into xyz, features, and labels
        self.xyz = self.data[:, :3]  # First 3 columns are X, Y, Z
        self.features = self.data[:, 3:-1]  # Columns 4 to second last are features
        self.labels = self.data[:, -1]  # Last column is Classification

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Verify input dimensions
        print(f"XYZ shape: {self.xyz.shape}")  # e.g., (N, 3)
        print(f"Features shape: {self.features.shape}")  # e.g., (N, F)
        print(f"Labels shape: {self.labels.shape}")  # e.g., (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract xyz, features, and label for a given index
        xyz = torch.tensor(self.xyz[idx], dtype=torch.float32)  # Shape: [3]
        features = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: [F]
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Shape: scalar

        # Reshape xyz and features for PointNet++
        xyz = xyz.unsqueeze(0)  # Shape: [3, 1]
        features = features.unsqueeze(-1)  # Shape: [F, 1]

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

    # Dynamically determine input dimension
    train_dataset = PointCloudDataset(train_file)
    val_dataset = PointCloudDataset(val_file)

    in_dim = train_dataset.features.shape[1]  # Number of features
    num_classes = len(np.unique(train_dataset.labels))  # Number of classes

    print(f"Detected input dimension: {in_dim}")
    print(f"Detected number of classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Inspect a batch from DataLoader ===
    for batch_features, batch_xyz, batch_labels in train_loader:
        print(f"Batch Features shape: {batch_features.shape}")  # [Batch size, F, 1]
        print(f"Batch XYZ shape: {batch_xyz.shape}")            # [Batch size, 3, 1]
        print(f"Batch Labels shape: {batch_labels.shape}")      # [Batch size]
        break

    # === Check GPU Availability ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

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

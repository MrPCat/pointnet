import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNetCls, STN

# === Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, file_path):
        # Load the entire dataset
        self.data = np.loadtxt(file_path, skiprows=1)
        
        # Separate features and labels
        self.points = self.data[:, :-1]
        self.labels = self.data[:, -1]
        
        # Normalize points (assuming first 3 columns are XYZ)
        self.points[:, :3] -= np.mean(self.points[:, :3], axis=0)

        # Verify input dimensions
        print(f"Dataset shape: {self.points.shape}")
        print(f"Number of features: {self.points.shape[1]}")
        print(f"Number of samples: {len(self.points)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get point features and label
        point_features = self.points[idx]
        label = self.labels[idx]
        
        # Convert to tensor with shape [num_features, num_points]
        # Ensure 10 features
        features = torch.tensor(point_features, dtype=torch.float32)
        
        # Reshape to [num_channels, num_points]
        # If features is 1D, add a dimension
        if features.dim() == 1:
            features = features.unsqueeze(1)
        
        # Transpose to match PointNet expected input
        features = features.t()
        
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label

# === Model Setup ===
def create_model(in_dim, num_classes):
    # Verify input dimension
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
        for features, labels in data_loader:
            # Debug input shapes
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Ensure features are on the correct device
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
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
    in_dim = sample_data.shape[1] - 1  # Subtract 1 for the label column
    num_classes = len(np.unique(sample_data[:, -1]))
    
    print(f"Detected input dimension: {in_dim}")
    print(f"Detected number of classes: {num_classes}")

    train_dataset = PointCloudDataset(train_file)
    val_dataset = PointCloudDataset(val_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Check GPU Availability ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Create model with dynamically determined dimensions
    model = create_model(in_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # === Training ===
    epochs = 10
    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion, epochs, device)
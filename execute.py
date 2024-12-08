#An execution for https://github.com/kentechx/pointnet with torch

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet import PointNetCls, STN  # Ensure the PointNet repository is correctly set up

# === Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, file_path):
        """
        Initialize dataset with a single file path.
        :param file_path: Path to dataset file
        """
        self.data = np.loadtxt(file_path, skiprows=1)  # Load the dataset, skipping the header row

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data item.
        :param idx: Index of the point in the dataset
        :return: Tuple of features and label
        """
        point = self.data[idx]
        xyz = point[:3]  # Extract XYZ coordinates
        other_feats = point[3:-1]  # Extract additional features
        label = point[-1]  # Extract the classification label

        # Normalize XYZ
        xyz -= np.mean(self.data[:, :3], axis=0)

        # Combine XYZ and other features
        features = np.hstack([xyz, other_feats])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# === Model Setup ===
def create_model(in_dim, num_classes):
    """
    Create and return the PointNet model.
    :param in_dim: Input dimension (XYZ + additional features)
    :param num_classes: Number of output classes
    :return: PointNet model
    """
    stn_3d = STN(in_dim=in_dim, out_nd=3)
    model = PointNetCls(in_dim=in_dim, out_dim=num_classes, stn_3d=stn_3d)
    return model

# === Training Loop ===
def train_model(model, data_loader, optimizer, criterion, epochs):
    """
    Train the PointNet model.
    :param model: PointNet model
    :param data_loader: DataLoader for training
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param epochs: Number of epochs
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in data_loader:
            features = features.transpose(1, 2)  # Change shape to [batch, channels, points]
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# === Evaluation ===
def evaluate_model(model, data_loader):
    """
    Evaluate the PointNet model.
    :param model: Trained PointNet model
    :param data_loader: DataLoader for evaluation
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.transpose(1, 2)
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {correct / total:.2%}")

# === Main Function ===
if __name__ == "__main__":
    # === Specify File Paths ===
    train_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Epoch_March2018\LiDAR\Mar18_train.txt"  # Replace with the actual path to your train file
    val_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Epoch_March2018\LiDAR\Mar18_val.txt"      # Replace with the actual path to your validation file

    # === Dataset and DataLoader ===
    batch_size = 16
    train_dataset = PointCloudDataset(train_file)
    val_dataset = PointCloudDataset(val_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Model Parameters ===
    in_dim = 10  # 3 (XYZ) + 7 (additional features: R, G, B, Reflectance, etc.)
    num_classes = 5  # Adjust based on your dataset

    model = create_model(in_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # === Training ===
    epochs = 10
    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion, epochs)

    # === Evaluation ===
    print("Evaluating on validation set...")
    evaluate_model(model, val_loader)

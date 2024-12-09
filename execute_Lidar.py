import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet import PointNetCls, STN

# === Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.loadtxt(file_path, skiprows=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point = self.data[idx]
        xyz = point[:3]
        other_feats = point[3:-1]
        label = point[-1]

        xyz -= np.mean(self.data[:, :3], axis=0)
        features = np.hstack([xyz, other_feats])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# === Model Setup ===
def create_model(in_dim, num_classes):
    stn_3d = STN(in_dim=in_dim, out_nd=3)
    model = PointNetCls(in_dim=in_dim, out_dim=num_classes, stn_3d=stn_3d)
    return model

# === Training Loop ===
def train_model(model, data_loader, optimizer, criterion, epochs, device):
    model.to(device)  # Move the model to GPU
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in data_loader:
            features = features.transpose(1, 2).to(device)  # Move to GPU
            labels = labels.to(device)  # Move labels to GPU

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# === Evaluation ===
def evaluate_model(model, data_loader, device):
    model.to(device)  # Ensure the model is on the GPU
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.transpose(1, 2).to(device)  # Move to GPU
            labels = labels.to(device)  # Move labels to GPU

            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {correct / total:.2%}")

# === Main Function ===
if __name__ == "__main__":
    # === Specify File Paths ===
    train_file = '/content/drive/MyDrive/t1/Mar18_train.txt'
    val_file = '/content/drive/MyDrive/t1/Mar18_train.txt'

    # === Dataset and DataLoader ===
    batch_size = 16
    train_dataset = PointCloudDataset(train_file)
    val_dataset = PointCloudDataset(val_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Model Parameters ===
    in_dim = 10
    num_classes = 5

    # === Check GPU Availability ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = create_model(in_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # === Training ===
    epochs = 10
    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion, epochs, device)

    # === Evaluation ===
    print("Evaluating on validation set...")
    evaluate_model(model, val_loader, device)

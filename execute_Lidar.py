import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet import PointNetCls, STN

# ====== Dataset Class ======
class PointCloudDataset(Dataset):
    def __init__(self, file_path):
        # Load the entire dataset
        self.data = np.loadtxt(file_path, skiprows=1)
        
        # Separate features and labels
        self.points = self.data[:, :-1]
        self.labels = self.data[:, -1]
        
        # Normalize points (assuming first 3 columns are XYZ)
        self.points[:, :3] -= np.mean(self.points[:, :3], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get point features and label
        point_features = self.points[idx]
        label = self.labels[idx]
        
        # Convert to tensor with shape [num_features, num_points]
        # PointNet typically expects [num_features, num_points]
        features = torch.tensor(point_features, dtype=torch.float32).unsqueeze(1)
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label

# === Model Setup ===
def create_model(in_dim, num_classes):
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
            # Reshape features to [batch_size, num_features, num_points]
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# === Evaluation ===
def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {correct / total:.2%}")

# === Main Function ===
if __name__ == "__main__":
    # === Specify File Paths ===
    train_file = '/content/drive/MyDrive/t1/Mar18_train.txt'
    val_file = '/content/drive/MyDrive/t1/Mar18_val.txt'
    
    print("Checking train file:", train_file)
    print("File exists:", os.path.exists(train_file))

    # === Dataset and DataLoader ===
    batch_size = 16
    train_dataset = PointCloudDataset(train_file)
    val_dataset = PointCloudDataset(val_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Model Parameters ===
    in_dim = 10  # Make sure this matches your input feature dimension
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
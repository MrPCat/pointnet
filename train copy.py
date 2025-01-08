import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNetCls, STN
from pointnet_ import PointNet2ClsSSG
from torch.optim.lr_scheduler import StepLR
import logging

# === Configure Logging ===
log_file_path = r"C:\Users\faars\Downloads\drive-download-20250102T121839Z-001\rainingDown_logs.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)  # Print to console
    logging.info(message)  # Save to log file

# === Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        self.data = np.loadtxt(file_path, delimiter=' ', skiprows=1)
        self.xyz = self.data[:, :3]
        self.features = self.data[:, 3:-1]
        self.labels = self.data[:, -1]

        self.xyz -= np.mean(self.xyz, axis=0)
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T
        labels = torch.tensor(self.labels[start:end], dtype=torch.long)
        label = torch.mode(labels).values
        return features, xyz, label

# === Model Setup ===
def create_model(in_dim, num_classes):
    stn_3d = STN(in_dim=in_dim, out_nd=3)
    model = PointNetCls(in_dim=in_dim, out_dim=num_classes, stn_3d=stn_3d)
    return model

# === Training Loop ===
def train_model(model, train_file_paths, val_loader, optimizer, scheduler, criterion, epochs, device, save_dir):
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0

        for train_file in train_file_paths:
            train_dataset = PointCloudDataset(train_file, points_per_cloud=1024)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            model.train()

            for features, xyz, labels in train_loader:
                features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(features, xyz)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        train_loss = total_loss / len(train_file_paths)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        epoch_model_path = os.path.join(save_dir, f"pointnetDown_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        log_and_print(f"Model checkpoint saved to {epoch_model_path}")

        log_and_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# === Validate Model ===
def validate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, xyz, labels in data_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
            logits = model(features, xyz)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# "__main__"
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")

    # File paths
    data_dir = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\Augmentation"
    train_files = [os.path.join(data_dir, f"Vaihingen3D_AugmentTraininig_{i}.pts") for i in range(18)]
    val_file = os.path.join(data_dir, "Vaihingen3D_AugmentTraininig_19.pts")

    val_dataset = PointCloudDataset(val_file, points_per_cloud=1024)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Parameters
    batch_size = 16
    lr = 0.001
    epochs = 50

    # Model, Optimizer, and Scheduler
    train_dataset = PointCloudDataset(train_files[0], points_per_cloud=1024)
    in_dim = train_dataset.features.shape[1]
    num_classes = len(np.unique(train_dataset.labels))

    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes, downsample_points=(512, 128))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    save_dir = r"C:\Users\faars\Downloads\drive-download-20250102T121839Z-001"
    os.makedirs(save_dir, exist_ok=True)

    # Training
    log_and_print("Starting training...")
    train_model(model, train_files, val_loader, optimizer, scheduler, criterion, epochs, device, save_dir)

    model_path = os.path.join(save_dir, "pointnetDown_model.pth")
    torch.save(model.state_dict(), model_path)
    log_and_print(f"Model saved to {model_path}")
    print("Model saved to pointnet_model.pth")

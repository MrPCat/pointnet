import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNetCls, STN
from pointnet_ import PointNet2ClsSSG 
import logging

# === Configure Logging ===
log_file_path = "/content/drive/MyDrive/t1/training_logs.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)  # Print to console
    logging.info(message)  # Save to log file

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


# for validating the model and hyperparameters there
def validate_model(model, data_loader, criterion, device):

    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, xyz, labels in data_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
            logits = model(features, xyz)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# === Model Setup ===
def create_model(in_dim, num_classes):
    print(f"Creating model with input dimension: {in_dim}")
    stn_3d = STN(in_dim=in_dim, out_nd=3)
    model = PointNetCls(in_dim=in_dim, out_dim=num_classes, stn_3d=stn_3d)
    return model


# === Training Loop ===
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_dir):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for features, xyz, labels in train_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features, xyz)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Save model after each epoch
        epoch_model_path = os.path.join(save_dir, f"pointnet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        log_and_print(f"Model checkpoint saved to {epoch_model_path}")

        log_and_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


#Testin and evaluating the model 
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, xyz, labels in test_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
            logits = model(features, xyz)
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    log_and_print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Accuracy: {accuracy:.2f}%")            

#"__main__"
if __name__ == "__main__":

    # Check GPU Availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")
    print("Using device:", device)
    
    # Specify File Paths
    train_file = '/content/drive/MyDrive/t1/Mar18_train.txt'
    val_file = '/content/drive/MyDrive/t1/Mar18_val.txt'
    test_file = '/content/drive/MyDrive/t1/Mar18_test.txt'

    # Dataset and DataLoader
    batch_size = 16
    train_dataset = PointCloudDataset(train_file, points_per_cloud=1024)
    val_dataset = PointCloudDataset(val_file, points_per_cloud=1024)
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Optimizer, and Loss Function
    in_dim = train_dataset.features.shape[1]
    num_classes = len(np.unique(train_dataset.labels))

    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes, downsample_points=(512, 128))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    # Directory for saving checkpoints
    save_dir = "/content/drive/MyDrive/t1/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Training with Validation
    log_and_print("Starting training...")
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_dir)



    # Save the trained model
    model_path = "/content/drive/MyDrive/t1/pointnet_model.pth"
    torch.save(model.state_dict(), model_path)
    log_and_print(f"Model saved to {model_path}")
    print("Model saved to pointnet_model.pth")

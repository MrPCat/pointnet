import os
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNetCls, STN
from pointnet_ import PointNet2ClsSSG
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === Configure Logging ===
log_file_path = "/content/drive/MyDrive/Filtered/training_logs.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)  # Print to console
    logging.info(message)  # Save to log file

# === Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, file_list, points_per_cloud=1024):
        self.data = []
        
        # Load and validate data dimensions
        for file_path in file_list:
            data = np.loadtxt(file_path)
            if data.shape[1] != 7:  # Verify each file has 7 columns
                raise ValueError(f"Expected 7 columns but got {data.shape[1]} in {file_path}")
            self.data.append(data)
        
        self.data = np.concatenate(self.data, axis=0)
        
        # Split into xyz, features, and labels
        self.xyz = self.data[:, :3]  # First 3 columns are X, Y, Z
        self.features = self.data[:, 3:-1]  # Columns 4 to second last are features
        self.labels = self.data[:, -1].astype(np.int64)  # Last column is Classification
        
        # Verify label values are valid
        unique_labels = np.unique(self.labels)
        if np.min(unique_labels) < 0:
            raise ValueError(f"Found negative label values: {unique_labels}")
        
        # Normalize spatial coordinates
        self.xyz = (self.xyz - np.mean(self.xyz, axis=0)) / np.std(self.xyz, axis=0)
        
        # Group points into point clouds
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        
        # Truncate data to be evenly divisible by points_per_cloud
        valid_points = self.num_clouds * self.points_per_cloud
        self.xyz = self.xyz[:valid_points]
        self.features = self.features[:valid_points]
        self.labels = self.labels[:valid_points]
        
        log_and_print(f"Dataset initialized with {self.num_clouds} point clouds")
        log_and_print(f"XYZ shape: {self.xyz.shape}")
        log_and_print(f"Features shape: {self.features.shape}")
        log_and_print(f"Unique labels: {unique_labels}")

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        
        # Get the points for this cloud
        xyz = self.xyz[start:end]  # Shape: [points_per_cloud, 3]
        features = self.features[start:end]  # Shape: [points_per_cloud, feature_dim]
        labels = self.labels[start:end]  # Shape: [points_per_cloud]
        
        # Ensure we have enough points
        if len(xyz) < self.points_per_cloud:
            # If we don't have enough points, duplicate some existing points
            indices = np.random.choice(len(xyz), self.points_per_cloud - len(xyz), replace=True)
            xyz = np.concatenate([xyz, xyz[indices]], axis=0)
            features = np.concatenate([features, features[indices]], axis=0)
            labels = np.concatenate([labels, labels[indices]])
        
        # Convert to tensors with correct shapes
        xyz = torch.tensor(xyz, dtype=torch.float32)  # Shape: [points_per_cloud, 3]
        features = torch.tensor(features, dtype=torch.float32)  # Shape: [points_per_cloud, feature_dim]
        
        # Use majority vote for cloud label
        label = torch.tensor(np.bincount(labels).argmax(), dtype=torch.long)
        
        return features, xyz, label

# === Validation Function ===
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

# === Training Loop ===
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_dir):
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
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

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Save model after each epoch
        epoch_model_path = os.path.join(save_dir, f"pointnet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        log_and_print(f"Model checkpoint saved to {epoch_model_path}")

        log_and_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# === Main Script ===
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")

    # Setup directories
    dir_path = '/content/drive/MyDrive/Filtered'
    save_dir = os.path.join(dir_path, 'Checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Initialize datasets and dataloaders
        batch_size = 16
        points_per_cloud = 512  # Reduced number of points per cloud
        
        all_files = sorted(glob.glob(os.path.join(dir_path, 'Vaihingen3D_AugmentTraininig_*.pts')))
        if len(all_files) < 20:
            raise ValueError(f"Expected at least 20 files, but found {len(all_files)}")

        train_files = all_files[:19]
        val_files = [all_files[19]]

        train_dataset = PointCloudDataset(train_files, points_per_cloud=points_per_cloud)
        val_dataset = PointCloudDataset(val_files, points_per_cloud=points_per_cloud)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        in_dim = train_dataset.features.shape[1]
        num_classes = len(np.unique(train_dataset.labels))
        
        model = PointNet2ClsSSG(
            in_dim=in_dim,
            out_dim=num_classes,
            downsample_points=(256, 128),  # Adjusted for sparse density
            radii=(0.4, 0.8),  # Larger radii for sparse data
            ks=(32, 64),  # Adjusted neighbors
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 80
        log_and_print("Starting training...")
        train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_dir)
        
        final_model_path = os.path.join(save_dir, "final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_model_path)
        log_and_print(f"Final model saved to {final_model_path}")
        
    except Exception as e:
        log_and_print(f"Error during execution: {str(e)}")
        raise

import os
import numpy as np
import torch
import laspy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pointnet_ import PointNetCls, STN
from pointnet_ import PointNet2ClsSSG
from torch.optim.lr_scheduler import StepLR
import logging

# === Configure Logging ===
log_file_path = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\ALS\Training_Log.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)  # Print to console
    logging.info(message)  # Save to log file

# === Dataset Class for PTS files ===
class PtsPointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        log_and_print(f"Loading PTS file: {file_path}")
        
        # Read PTS file (assuming space-separated values with X Y Z intensity classification)
        try:
            data = np.loadtxt(file_path, delimiter=' ')
            if data.shape[1] >= 3:  # At minimum, we need XYZ coordinates
                self.xyz = data[:, 0:3]  # First 3 columns are XYZ
                
                # Extract intensity if available (4th column)
                if data.shape[1] >= 4:
                    self.features = data[:, 3:4]  # 4th column is intensity
                else:
                    self.features = np.zeros((len(self.xyz), 1))
                
                # Extract classification if available (5th column)
                if data.shape[1] >= 5:
                    self.labels = data[:, 4].astype(np.int64)
                else:
                    self.labels = np.zeros(len(self.xyz)).astype(np.int64)
                
                # Normalize XYZ coordinates
                self.xyz -= np.mean(self.xyz, axis=0)
                
                self.points_per_cloud = points_per_cloud
                self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)
                log_and_print(f"Loaded {len(self.xyz)} points, creating {self.num_clouds} clouds")
            else:
                raise ValueError("PTS file must have at least 3 columns for XYZ coordinates")
        except Exception as e:
            log_and_print(f"Error loading PTS file: {e}")
            # Create empty dataset to avoid crashing
            self.xyz = np.zeros((points_per_cloud, 3))
            self.features = np.zeros((points_per_cloud, 1))
            self.labels = np.zeros(points_per_cloud).astype(np.int64)
            self.points_per_cloud = points_per_cloud
            self.num_clouds = 1

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Handle the case where we might not have enough points
        if idx >= self.num_clouds:
            idx = idx % self.num_clouds
            
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        
        # If we don't have enough points, we'll use random sampling with replacement
        if end > len(self.xyz):
            indices = np.random.choice(len(self.xyz), self.points_per_cloud, replace=True)
            xyz = self.xyz[indices]
            features = self.features[indices]
            point_labels = self.labels[indices]
        else:
            xyz = self.xyz[start:end]
            features = self.features[start:end]
            point_labels = self.labels[start:end]
        
        # Convert to tensors
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).T  # Shape: (3, points_per_cloud)
        features_tensor = torch.tensor(features, dtype=torch.float32).T  # Shape: (1, points_per_cloud)
        
        # Get the most common label as the cloud label
        label = torch.tensor(np.bincount(point_labels).argmax(), dtype=torch.long)
        
        return features_tensor, xyz_tensor, label

# === LAS Dataset Class (for reference, if you need it) ===
class LasPointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        log_and_print(f"Loading LAS file: {file_path}")
        
        try:
            # Open LAS file
            las = laspy.read(file_path)

            # Extract XYZ coordinates
            self.xyz = np.vstack((las.x, las.y, las.z)).T  # Shape: (num_points, 3)

            # Extract intensity and other features (if available)
            try:
                self.features = np.vstack((las.intensity,)).T  # Shape: (num_points, 1)
            except:
                self.features = np.zeros((len(self.xyz), 1))  # Dummy feature if missing

            # Extract labels (if available)
            try:
                self.labels = las.classification.astype(np.int64)
            except:
                self.labels = np.zeros(len(self.xyz)).astype(np.int64)  # Dummy labels if missing

            # Normalize XYZ coordinates
            self.xyz -= np.mean(self.xyz, axis=0)

            self.points_per_cloud = points_per_cloud
            self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)
            log_and_print(f"Loaded {len(self.xyz)} points, creating {self.num_clouds} clouds")
        except Exception as e:
            log_and_print(f"Error loading LAS file: {e}")
            # Create empty dataset to avoid crashing
            self.xyz = np.zeros((points_per_cloud, 3))
            self.features = np.zeros((points_per_cloud, 1))
            self.labels = np.zeros(points_per_cloud).astype(np.int64)
            self.points_per_cloud = points_per_cloud
            self.num_clouds = 1

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Handle the case where we might not have enough points
        if idx >= self.num_clouds:
            idx = idx % self.num_clouds
            
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        
        # If we don't have enough points, we'll use random sampling with replacement
        if end > len(self.xyz):
            indices = np.random.choice(len(self.xyz), self.points_per_cloud, replace=True)
            xyz = self.xyz[indices]
            features = self.features[indices]
            point_labels = self.labels[indices]
        else:
            xyz = self.xyz[start:end]
            features = self.features[start:end]
            point_labels = self.labels[start:end]
        
        # Convert to tensors
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).T  # Shape: (3, points_per_cloud)
        features_tensor = torch.tensor(features, dtype=torch.float32).T  # Shape: (1, points_per_cloud)
        
        # Get the most common label as the cloud label
        label = torch.tensor(np.bincount(point_labels).argmax(), dtype=torch.long)
        
        return features_tensor, xyz_tensor, label

# === Helper function to determine file type and create appropriate dataset ===
def create_dataset(file_path, points_per_cloud=1024):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in ['.las', '.laz']:
        return LasPointCloudDataset(file_path, points_per_cloud)
    elif file_ext in ['.pts', '.txt', '.xyz']:
        return PtsPointCloudDataset(file_path, points_per_cloud)
    else:
        log_and_print(f"Unknown file type: {file_ext}, attempting to load as PTS")
        return PtsPointCloudDataset(file_path, points_per_cloud)

# === Model Setup ===
def create_model(in_dim, num_classes):
    stn_3d = STN(in_dim=in_dim, out_nd=3)
    model = PointNetCls(in_dim=in_dim, out_dim=num_classes, stn_3d=stn_3d)
    return model

# === Training Loop ===
def train_model(model, train_loaders, optimizer, scheduler, criterion, epochs, device, save_dir, val_loader=None):
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0

        model.train()
        for train_loader in train_loaders:
            for features, xyz, labels in train_loader:
                features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(features, xyz)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        train_loss = total_loss / max(1, total_batches)
        
        # Validation step
        val_metrics = ""
        if val_loader:
            val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
            val_metrics = f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        epoch_model_path = os.path.join(save_dir, f"pointnetDown_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        log_and_print(f"Model checkpoint saved to {epoch_model_path}")

        log_and_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}{val_metrics}")

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

    avg_loss = total_loss / max(1, len(data_loader))
    accuracy = 100 * correct / max(1, total)
    return avg_loss, accuracy

# __main__
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")

    # File paths
    data_dir = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\Augmentation"
    train_files = [os.path.join(data_dir, f"Vaihingen3D_AugmentTraininig_{i}.pts") for i in range(1,18)]
    val_file = os.path.join(data_dir, "Vaihingen3D_AugmentTraininig_19.pts")

    # Verify file paths exist
    existing_train_files = []
    for file_path in train_files:
        if os.path.exists(file_path):
            existing_train_files.append(file_path)
        else:
            log_and_print(f"Warning: Training file not found: {file_path}")
    
    if not existing_train_files:
        log_and_print("Error: No training files found!")
        exit(1)
    
    if not os.path.exists(val_file):
        log_and_print(f"Warning: Validation file not found: {val_file}")
        val_loader = None
    else:
        val_dataset = create_dataset(val_file, points_per_cloud=1024)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Parameters
    batch_size = 16
    lr = 0.001
    epochs = 50

    # Create train loaders
    train_loaders = []
    all_features = []
    all_labels = []
    
    for train_file in existing_train_files:
        train_dataset = create_dataset(train_file, points_per_cloud=1024)
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        
        # Collect feature dimensions and unique labels
        if hasattr(train_dataset, 'features'):
            all_features.append(train_dataset.features.shape[1])
        if hasattr(train_dataset, 'labels'):
            all_labels.extend(np.unique(train_dataset.labels))
    
    # Determine input dimension and number of classes from data
    in_dim = max(all_features) if all_features else 1
    num_classes = len(np.unique(all_labels)) if all_labels else 2
    
    log_and_print(f"Input dimension: {in_dim}, Number of classes: {num_classes}")

    # Model, Optimizer, and Scheduler
    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes, downsample_points=(512, 128))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    save_dir = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\ALS"
    os.makedirs(save_dir, exist_ok=True)

    # Training
    log_and_print("Starting training...")
    train_model(model, train_loaders, optimizer, scheduler, criterion, epochs, device, save_dir, val_loader)

    model_path = os.path.join(save_dir, "pointnetDown_final_model.pth")
    torch.save(model.state_dict(), model_path)
    log_and_print(f"Final model saved to {model_path}")
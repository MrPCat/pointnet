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

# Configure Logging
log_file_path = "/content/drive/MyDrive/Filtered/training_logs.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)
    logging.info(message)

class PointCloudDataset(Dataset):
    def __init__(self, file_list, points_per_cloud=256):
        self.points_per_cloud = points_per_cloud
        self.processed_clouds = []
        self.processed_labels = []
        
        total_points = 0
        cloud_sizes = []
        
        for file_path in file_list:
            data = np.loadtxt(file_path)
            if data.shape[1] != 7:
                raise ValueError(f"Expected 7 columns but got {data.shape[1]} in {file_path}")
                
            total_points += len(data)
            cloud_sizes.append(len(data))
            
            n_chunks = len(data) // points_per_cloud
            for i in range(n_chunks):
                chunk = data[i * points_per_cloud:(i + 1) * points_per_cloud]
                xyz = chunk[:, :3]
                features = chunk[:, 3:-1]
                labels = chunk[:, -1].astype(np.int64)
                
                # Normalize per chunk
                xyz = (xyz - np.mean(xyz, axis=0)) / (np.std(xyz, axis=0) + 1e-6)
                
                self.processed_clouds.append((xyz, features))
                self.processed_labels.append(np.bincount(labels).argmax())
        
        self.processed_labels = np.array(self.processed_labels)
        
        log_and_print(f"Total number of points: {total_points}")
        log_and_print(f"Cloud sizes: {cloud_sizes}")
        log_and_print(f"Minimum points in a cloud: {min(cloud_sizes)}")
        log_and_print(f"Maximum points in a cloud: {max(cloud_sizes)}")
        log_and_print(f"Dataset initialized with {len(self.processed_clouds)} point clouds")
        log_and_print(f"Unique labels: {np.unique(self.processed_labels)}")

    def __len__(self):
        return len(self.processed_clouds)

    def __getitem__(self, idx):
        xyz, features = self.processed_clouds[idx]
        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(xyz, dtype=torch.float32),
                torch.tensor(self.processed_labels[idx], dtype=torch.long))

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
    
    return total_loss / len(data_loader), 100 * correct / total

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_dir):
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, xyz, labels in train_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features, xyz)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, best_model_path)
            log_and_print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        train_accuracy = 100 * train_correct / train_total
        log_and_print(f"Epoch {epoch+1}/{epochs}")
        log_and_print(f"Train Loss: {total_train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        log_and_print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(val_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, checkpoint_path)

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration
    points_per_cloud = 512
    batch_size = 64
    dir_path = '/content/drive/MyDrive/Filtered'
    save_dir = os.path.join(dir_path, 'Checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")
    
    try:
        # Dataset setup
        all_files = sorted(glob.glob(os.path.join(dir_path, 'Vaihingen3D_AugmentTraininig_*.pts')))
        if len(all_files) < 20:
            raise ValueError(f"Expected at least 20 files, but found {len(all_files)}")
        
        train_files = all_files[:19]
        val_files = [all_files[19]]
        
        train_dataset = PointCloudDataset(train_files, points_per_cloud=points_per_cloud)
        val_dataset = PointCloudDataset(val_files, points_per_cloud=points_per_cloud)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Model setup
        in_dim = train_dataset.processed_clouds[0][1].shape[1]
        num_classes = len(np.unique(train_dataset.processed_labels))
        
        model = PointNet2ClsSSG(
        in_dim=in_dim,
        out_dim=num_classes,
        downsample_points=(256, 128),  # These values should be ≤ points_per_cloud
        radii=(0.4, 0.8),
        ks=(64, 128)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        log_and_print("Starting training...")
        train_model(model, train_loader, val_loader, optimizer, criterion, 80, device, save_dir)
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_model_path)
        log_and_print(f"Final model saved to {final_model_path}")
        
    except Exception as e:
        log_and_print(f"Error during execution: {str(e)}")
        raise
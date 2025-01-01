import os
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PointCloudDataset(Dataset):
    def __init__(self, file_list, points_per_cloud=256):
        self.points_per_cloud = points_per_cloud
        self.processed_clouds = []
        self.processed_labels = []
        
        for file_path in file_list:
            data = np.loadtxt(file_path)
            n_chunks = len(data) // points_per_cloud
            for i in range(n_chunks):
                chunk = data[i * points_per_cloud:(i + 1) * points_per_cloud]
                xyz = chunk[:, :3]
                features = chunk[:, 3:-1]
                labels = chunk[:, -1].astype(np.int64)
                
                # Normalize coordinates
                xyz = (xyz - np.mean(xyz, axis=0)) / (np.std(xyz, axis=0) + 1e-6)
                self.processed_clouds.append((xyz, features))
                self.processed_labels.append(np.bincount(labels).argmax())
        
        self.processed_labels = np.array(self.processed_labels)
        logging.info(f"Dataset initialized with {len(self.processed_clouds)} point clouds")
        logging.info(f"Unique labels: {np.unique(self.processed_labels)}")

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, os.path.join(save_dir, "best_model.pth"))
            logging.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        train_accuracy = 100 * train_correct / train_total
        logging.info(f"Epoch {epoch+1}/{epochs}")
        logging.info(f"Train Loss: {total_train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(val_loss)

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration
    points_per_cloud = 256
    batch_size = 32
    dir_path = '/content/drive/MyDrive/Filtered'
    save_dir = os.path.join(dir_path, 'Checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
        # Dataset setup
        all_files = sorted(glob.glob(os.path.join(dir_path, 'Vaihingen3D_AugmentTraininig_*.pts')))
        train_files = all_files[:19]
        val_files = [all_files[19]]
        
        train_dataset = PointCloudDataset(train_files, points_per_cloud=points_per_cloud)
        val_dataset = PointCloudDataset(val_files, points_per_cloud=points_per_cloud)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Model setup
        model = PointNet2ClsSSG(
            in_dim=train_dataset.processed_clouds[0][1].shape[1],
            out_dim=len(np.unique(train_dataset.processed_labels)),
            downsample_points=(128, 64),  # Must be smaller than input points
            radii=(0.2, 0.4),
            ks=(16, 16)
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        logging.info("Starting training...")
        train_model(model, train_loader, val_loader, optimizer, criterion, 80, device, save_dir)
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, "final_model.pth"))
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise
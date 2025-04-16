import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pointnet_ import PointNet2ClsSSG
from torch.optim.lr_scheduler import StepLR
import logging

# === Configure Logging ===
log_file_path = r"/content/drive/MyDrive/Dales/DALESObjects/DALESObjects/train_pts/Valid/Training_Log.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)
    logging.info(message)

# === Dataset Class ===
class PtsPointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        log_and_print(f"Loading PTS file: {file_path}")
        self.points_per_cloud = points_per_cloud
        
        # Define class mapping for DALES dataset
        self.class_mapping = {
            1: 0,  # Ground: impervious surfaces, grass, rough terrain
            2: 1,  # Vegetation: trees, shrubs, hedges, bushes
            3: 2,  # Cars: sedans, vans, SUVs
            4: 3,  # Trucks: semi-trucks, box-trucks, recreational vehicles
            5: 4,  # Power lines: transmission and distribution lines
            6: 5,  # Poles: power line poles, light poles and transmission towers
            7: 6,  # Fences: residential fences and highway barriers
            8: 7,  # Buildings: residential, high-rises and warehouses
        }
        
        # Store number of classes for later use
        self.num_classes = 8  # Explicitly set to 8 classes based on DALES dataset

        try:
            data = np.loadtxt(file_path, delimiter=' ')
            if data.shape[1] < 6:
                raise ValueError("PTS file must have at least 6 columns (X Y Z intensity sem_class ins_class)")

            self.xyz = data[:, 0:3]
            self.sem_class = data[:, 4:5].astype(int)
            self.xyz -= np.mean(self.xyz, axis=0)

            # Dummy feature (PointNet++ requires some feature input)
            self.features = np.zeros((self.xyz.shape[0], 1))
            self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)
            
            # Print unique semantic classes in this file
            unique_classes = np.unique(self.sem_class)
            log_and_print(f"Loaded {len(self.xyz)} points, creating {self.num_clouds} clouds.")
            log_and_print(f"Unique semantic classes in file: {unique_classes}")

        except Exception as e:
            log_and_print(f"Error loading PTS file: {e}")
            self.xyz = np.zeros((points_per_cloud, 3))
            self.features = np.zeros((points_per_cloud, 1))
            self.sem_class = np.zeros((points_per_cloud, 1), dtype=int)
            self.num_clouds = 1

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        if idx >= self.num_clouds:
            idx = idx % self.num_clouds
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud

        if end > len(self.xyz):
            indices = np.random.choice(len(self.xyz), self.points_per_cloud, replace=True)
        else:
            indices = np.arange(start, end)

        xyz = self.xyz[indices]
        features = self.features[indices]
        sem_class = self.sem_class[indices]

        # Most common semantic class is used as the label
        most_common_class = np.bincount(sem_class.flatten()).argmax()
        
        # Map the original class ID to a continuous index
        if most_common_class in self.class_mapping:
            mapped_class = self.class_mapping[most_common_class]
        else:
            log_and_print(f"Warning: Unknown class ID {most_common_class}, defaulting to class 0")
            mapped_class = 0

        label = torch.tensor(mapped_class, dtype=torch.long)

        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).T  # (3, N)
        features_tensor = torch.tensor(features, dtype=torch.float32).T  # (1, N)

        return features_tensor, xyz_tensor, label

# === Create dataset from file ===
def create_dataset(file_path, points_per_cloud=1024):
    return PtsPointCloudDataset(file_path, points_per_cloud)

# === Validation function ===
def validate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = np.zeros(8)  # For 8 classes in DALES
    class_total = np.zeros(8)

    with torch.no_grad():
        for features, xyz, labels in data_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
            logits = model(features, xyz)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    avg_loss = total_loss / max(1, len(data_loader))
    accuracy = 100 * correct / max(1, total)
    
    # Log per-class accuracy
    class_accuracy = np.zeros(8)
    for i in range(8):
        if class_total[i] > 0:
            class_accuracy[i] = 100 * class_correct[i] / class_total[i]
            log_and_print(f"Class {i} accuracy: {class_accuracy[i]:.2f}%")
    
    log_and_print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# === Training function ===
def train_model(model, train_loader, optimizer, scheduler, criterion, epochs, device, save_dir, val_loader=None):
    model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        total_loss = 0
        correct_train = 0
        total_train = 0
        total_batches = 0

        model.train()
        for features, xyz, labels in train_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
            
            # Check for invalid labels
            if torch.any(labels >= 8) or torch.any(labels < 0):
                invalid_labels = labels[torch.logical_or(labels >= 8, labels < 0)]
                log_and_print(f"Warning: Invalid labels detected: {invalid_labels}")
                # Skip this batch
                continue
            
            # Skip problematic batches that might cause BatchNorm issues
            if features.size(0) <= 1:  # Skip batches with only one sample
                log_and_print(f"Skipping batch with size {features.size(0)} to avoid BatchNorm errors")
                continue
                
            try:
                optimizer.zero_grad()
                logits = model(features, xyz)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)
                total_batches += 1
            except ValueError as e:
                if "Expected more than 1 value per channel" in str(e):
                    log_and_print(f"Skipping problematic batch with shape {features.shape}")
                    continue
                else:
                    raise e

        train_loss = total_loss / max(1, total_batches)
        train_accuracy = 100 * correct_train / max(1, total_train)
        val_metrics = ""
        val_accuracy = 0.0

        if val_loader:
            val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
            val_metrics = f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_path = os.path.join(save_dir, "pointnetDown_best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                log_and_print(f"New best model saved with accuracy {best_val_accuracy:.2f}%")

        scheduler.step()
        
        # Save checkpoint every 5 epochs to avoid cluttering
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(save_dir, f"pointnetDown_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            log_and_print(f"Model checkpoint saved to {model_path}")
            
        log_and_print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%{val_metrics}")

# === Main ===
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")

    # File paths
    train_dir = "/content/drive/MyDrive/Dales/DALESObjects/DALESObjects/train_pts/Train"
    val_dir = "/content/drive/MyDrive/Dales/DALESObjects/DALESObjects/train_pts/Valid"
    save_dir = val_dir

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pts')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pts')]

    # Create training datasets and combine them
    train_datasets = []
    for file in train_files:
        if not os.path.exists(file):
            log_and_print(f"Missing: {file}")
            continue
        dataset = create_dataset(file, 1024)
        train_datasets.append(dataset)

    if not train_datasets:
        log_and_print("No training data found. Exiting.")
        exit(1)
    
    # Combine all training datasets and create a single DataLoader
    combined_train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True)
    log_and_print(f"Combined {len(train_datasets)} training datasets with {len(combined_train_dataset)} total samples")

    # Combine validation datasets
    val_datasets = [create_dataset(f, 1024) for f in val_files if os.path.exists(f)]
    if val_datasets:
        combined_val_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(combined_val_dataset, batch_size=128, shuffle=False)
        log_and_print(f"Combined {len(val_datasets)} validation datasets with {len(combined_val_dataset)} total samples")
    else:
        val_loader = None
        log_and_print("No validation data found.")
    
    # Fixed number of classes for DALES dataset
    num_classes = 8
    in_dim = 1  # Single feature dimension (we're using dummy features)
    log_and_print(f"Input Dim: {in_dim}, Classes: {num_classes}")

    # Model
    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes, downsample_points=(512, 128))
    
    # Add weight initialization for better convergence
    def init_weights(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Loss function with class weights if needed
    # If your dataset is imbalanced, you might want to add class weights
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(save_dir, exist_ok=True)

    # Train
    log_and_print("Starting training...")
    train_model(model, train_loader, optimizer, scheduler, criterion, epochs=50, device=device, save_dir=save_dir, val_loader=val_loader)

    # Final save
    final_model_path = os.path.join(save_dir, "pointnetDown_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    log_and_print(f"Final model saved to {final_model_path}")
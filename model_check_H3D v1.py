%cd pointnet_


import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import logging
import time
from pathlib import Path
from pointnet_ import PointNet2ClsMSG


# === Configure Logging ===
def setup_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging.getLogger('')

def log_info(logger, message):
    logger.info(message)

# === Dataset Class with Augmentation ===
class PtsPointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=8192, augment=False):
        """
        Dataset for PTS point cloud files
        Args:
            file_path: Path to the PTS file
            points_per_cloud: Number of points per cloud sample
            augment: Whether to apply data augmentation
        """
        self.points_per_cloud = points_per_cloud
        self.augment = augment

        # Define class mapping for DALES dataset (1-based to 0-based)
        self.class_mapping = {
            1: 0,  # Ground
            2: 1,  # Vegetation
            3: 2,  # Cars
            4: 3,  # Trucks
            5: 4,  # Power lines
            6: 5,  # Poles
            7: 6,  # Fences
            8: 7,  # Buildings
        }
        self.num_classes = len(self.class_mapping)

        try:
            data = np.loadtxt(file_path, delimiter=' ')
            if data.shape[1] < 6:
                raise ValueError("PTS file must have at least 6 columns (X Y Z intensity sem_class ins_class)")

            self.xyz = data[:, 0:3].astype(np.float32)
            self.sem_class = data[:, 4].astype(np.int64)

            # Center the point cloud
            self.xyz -= np.mean(self.xyz, axis=0)

            # Normalize to unit sphere
            scale = np.max(np.sqrt(np.sum(self.xyz**2, axis=1)))
            self.xyz /= max(scale, 1e-6)

            # Create intensity feature (using actual intensity if available)
            if data.shape[1] > 3:
                self.features = data[:, 3:4].astype(np.float32)  # Intensity as feature
            else:
                self.features = np.ones((self.xyz.shape[0], 1), dtype=np.float32)  # Default feature

            self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)

            # Print dataset info
            unique_classes = np.unique(self.sem_class)
            mapped_classes = [self.class_mapping.get(c, -1) for c in unique_classes if c in self.class_mapping]
            print(f"Loaded {len(self.xyz)} points from {file_path}")
            print(f"Creating {self.num_clouds} clouds with {self.points_per_cloud} points each")
            print(f"Unique semantic classes (mapped): {mapped_classes}")

        except Exception as e:
            print(f"Error loading PTS file {file_path}: {e}")
            # Create minimal default data
            self.xyz = np.zeros((self.points_per_cloud, 3), dtype=np.float32)
            self.features = np.zeros((self.points_per_cloud, 1), dtype=np.float32)
            self.sem_class = np.zeros(self.points_per_cloud, dtype=np.int64)
            self.num_clouds = 1

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        if idx >= self.num_clouds:
            idx = idx % self.num_clouds

        start = idx * self.points_per_cloud
        end = min(start + self.points_per_cloud, len(self.xyz))

        if end - start < self.points_per_cloud:
            # If we don't have enough points, sample with replacement
            indices = np.random.choice(len(self.xyz), self.points_per_cloud, replace=True)
        else:
            indices = np.arange(start, end)

            # If we have more than enough points, randomly sample
            if len(indices) > self.points_per_cloud:
                indices = np.random.choice(indices, self.points_per_cloud, replace=False)

        xyz = self.xyz[indices].copy()
        features = self.features[indices].copy()
        sem_class = self.sem_class[indices].copy()

        # Data augmentation
        if self.augment:
            # Random rotation around z-axis
            if np.random.random() > 0.5:
                angle = np.random.uniform(0, 2*np.pi)
                cos, sin = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([
                    [cos, -sin, 0],
                    [sin, cos, 0],
                    [0, 0, 1]
                ])
                xyz = xyz @ rotation_matrix

            # Random jitter
            if np.random.random() > 0.5:
                xyz += np.random.normal(0, 0.01, size=xyz.shape)

            # Random scaling
            if np.random.random() > 0.5:
                xyz *= np.random.uniform(0.9, 1.1)

        # Most common semantic class is used as the label
        unique_classes, counts = np.unique(sem_class, return_counts=True)
        most_common_class = unique_classes[np.argmax(counts)]

        # Map the class ID to a continuous index
        if most_common_class in self.class_mapping:
            mapped_class = self.class_mapping[most_common_class]
        else:
            mapped_class = 0  # Default to class 0 if unknown

        label = torch.tensor(mapped_class, dtype=torch.long)

        # Convert to tensors with proper shape for PointNet++ (transpose for channels-first)
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).transpose(0, 1)  # (3, N)
        features_tensor = torch.tensor(features, dtype=torch.float32).transpose(0, 1)  # (1, N)

        return features_tensor, xyz_tensor, label

# === Create dataset and dataloader functions ===
def create_datasets(base_dirs, points_per_count=512, augment_train=True):
    """Create training and validation datasets from directories"""
    train_dir = os.path.join(base_dirs, 'Train')
    val_dir = os.path.join(base_dirs, 'Valid')

    # Find all .pts files
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pts')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pts')]

    # Create datasets
    train_datasets = [
        PtsPointCloudDataset(file, points_per_count, augment=augment_train)
        for file in train_files if os.path.exists(file)
    ]

    val_datasets = [
        PtsPointCloudDataset(file, points_per_count, augment=False)  # No augmentation for validation
        for file in val_files if os.path.exists(file)
    ]

    # Combine datasets
    if train_datasets:
        train_dataset = ConcatDataset(train_datasets)
    else:
        train_dataset = None

    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
    else:
        val_dataset = None

    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """Create DataLoaders from datasets"""
    train_loader = None
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Drop last batch to avoid BatchNorm issues
        )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader

# === Validation function ===
def validate_model(model, data_loader, criterion, device, logger):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Track per-class performance
    num_classes = 8  # DALES dataset has 8 classes
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for features, xyz, labels in data_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)

            # Forward pass
            with autocast():
                logits = model(features, xyz)
                loss = criterion(logits, labels)

            # Calculate metrics
            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update confusion matrix and per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                confusion_matrix[label, pred] += 1

                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    # Calculate overall metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / max(1, total)

    # Calculate per-class metrics
    class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = 100 * class_correct[i] / class_total[i]
            log_info(logger, f"Class {i} accuracy: {class_accuracy[i]:.2f}%")
        else:
            log_info(logger, f"Class {i}: No samples")

    # Overall metrics
    log_info(logger, f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

# === Training function ===
def train_model(model, train_loader, val_loader, args, device, save_dir, logger):
    """Train the PointNet2 model with mixed precision"""
    model.to(device)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['learning_rate'],
        weight_decay=args['weight_decay']
    )

    # Learning rate scheduler with warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args['epochs'],
        eta_min=args['learning_rate'] * 0.01
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Mixed precision training
    scaler = GradScaler()

    # Training metrics tracking
    best_val_accuracy = 0.0
    training_start_time = time.time()

    # Training loop
    for epoch in range(args['epochs']):
        epoch_start_time = time.time()
        model.train()

        # Metrics
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        # Training batches
        for features, xyz, labels in train_loader:
            features = features.to(device)
            xyz = xyz.to(device)
            labels = labels.to(device)

            # Skip problematic small batches
            if features.size(0) <= 1:
                continue

            try:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass with mixed precision
                with autocast():
                    logits = model(features, xyz)
                    loss = criterion(logits, labels)

                # Backward pass with scaling
                scaler.scale(loss).backward()

                # Gradient clipping to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update weights
                scaler.step(optimizer)
                scaler.update()

                # Track metrics
                total_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                batch_count += 1

            except Exception as e:
                log_info(logger, f"Error in batch: {str(e)}")
                continue

        # Calculate epoch metrics
        train_loss = total_loss / max(1, batch_count)
        train_accuracy = 100 * correct / max(1, total)

        # Validation
        if val_loader:
            val_loss, val_accuracy = validate_model(model, val_loader, criterion, device, logger)

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_path = os.path.join(save_dir, "pointnet2_best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': val_accuracy,
                    'args': args
                }, best_model_path)
                log_info(logger, f"✓ New best model saved with accuracy {best_val_accuracy:.2f}%")
        else:
            val_loss, val_accuracy = float('inf'), 0.0

        # Step the scheduler
        scheduler.step()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"pointnet2_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_accuracy if val_loader else train_accuracy,
                'args': args
            }, checkpoint_path)

        # Log progress
        epoch_time = time.time() - epoch_start_time
        log_info(logger, f"Epoch {epoch+1}/{args['epochs']} completed in {epoch_time:.2f}s")
        log_info(logger, f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        if val_loader:
            log_info(logger, f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        log_info(logger, f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Save final model
    final_model_path = os.path.join(save_dir, "pointnet2_final_model.pth")
    torch.save({
        'epoch': args['epochs'],
        'model_state_dict': model.state_dict(),
        'accuracy': best_val_accuracy,
        'args': args
    }, final_model_path)

    # Log training summary
    total_time = time.time() - training_start_time
    log_info(logger, f"Training completed in {total_time / 60:.2f} minutes")
    log_info(logger, f"Best validation accuracy: {best_val_accuracy:.2f}%")
    log_info(logger, f"Final model saved to {final_model_path}")

# === Main function ===
def main():
    # Training settings
    args = {
        'data_dir': "C:/Farshid/Uni/Semesters/Thesis/Data/DALESObjects/DALESObjects/train",
        'save_dir': "C:/Farshid/Uni/Semesters/Thesis/Data/DALESObjects/DALESObjects",
        'points_per_cloud': 8192,  # Number of points per cloud sample
        'batch_size': 64,         # Batch size for training
        'learning_rate': 0.001,   # Initial learning rate
        'weight_decay': 1e-4,     # Weight decay for regularization
        'epochs': 50,             # Number of training epochs
        'num_workers': 4,         # DataLoader workers
        'in_dim': 1,              # Input feature dimension
        'num_classes': 8,         # Output classes (DALES dataset)
    }

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args['save_dir'], exist_ok=True)
    logger = setup_logger(args['save_dir'])

    # Log system info
    log_info(logger, f"Using device: {device}")
    if device.type == 'cuda':
        log_info(logger, f"CUDA Device: {torch.cuda.get_device_name(0)}")
        log_info(logger, f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Log training parameters
    log_info(logger, f"Training with parameters: {args}")

    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(
        args['data_dir'],
        points_per_count=args['points_per_cloud']
    )

    if train_dataset:
        log_info(logger, f"Training dataset size: {len(train_dataset)} samples")
    else:
        log_info(logger, "No training data found. Exiting.")
        return

    if val_dataset:
        log_info(logger, f"Validation dataset size: {len(val_dataset)} samples")
    else:
        log_info(logger, "No validation data found.")

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers']
    )

    # Create model
    # Create model
    model = PointNet2ClsMSG(
        in_dim=args['in_dim'],
        out_dim=args['num_classes'],
        downsample_points=(8192, 4096, 2048, 1024),  # As specified
        ball_radii=(0.5, 1.0, 5.0, 15.0),  # As specified
        neighbor_counts=(16, 64, 64, 32),  # As specified
        head_norm=True,
        dropout=0.5
      )

    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Print model summary
    log_info(logger, f"Model: PointNet2ClsSSG")
    log_info(logger, f"Input dimension: {args['in_dim']}, Output classes: {args['num_classes']}")
    log_info(logger, f"Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    log_info(logger, "Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        save_dir=args['save_dir'],
        logger=logger
    )

if __name__ == "__main__":
    main()
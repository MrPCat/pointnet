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
import pickle
from tqdm import tqdm
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
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger('')

def log_info(logger, message):
    logger.info(message)


# === Dataset Class with Caching and Optimized Loading ===
class PtsPointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=2048, augment=False, cache_dir=None):
        """
        Dataset for PTS point cloud files with caching support
        Args:
            file_path: Path to the PTS file
            points_per_cloud: Number of points per cloud sample
            augment: Whether to apply data augmentation
            cache_dir: Directory to store cached data
        """
        self.file_path = file_path
        self.points_per_cloud = points_per_cloud
        self.augment = augment
        self.cache_dir = cache_dir

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

        # Try to load from cache first
        self.cache_loaded = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(
                cache_dir,
                f"{os.path.basename(file_path)}_{points_per_cloud}_{augment}.pkl"
            )

            if os.path.exists(cache_file):
                try:
                    print(f"Loading cached data from {cache_file}")
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.xyz = cached_data['xyz']
                        self.features = cached_data['features']
                        self.sem_class = cached_data['sem_class']
                        self.num_clouds = cached_data['num_clouds']
                        self.cache_loaded = True
                        print(f"Successfully loaded cached data with {self.num_clouds} clouds")
                except Exception as e:
                    print(f"Error loading cache: {e}, will load from original file")

        # Load data if not cached
        if not self.cache_loaded:
            self.load_and_preprocess_data()

            # Save to cache if cache_dir is provided
            if cache_dir:
                cache_file = os.path.join(
                    cache_dir,
                    f"{os.path.basename(file_path)}_{points_per_cloud}_{augment}.pkl"
                )
                try:
                    print(f"Saving data to cache: {cache_file}")
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'xyz': self.xyz,
                            'features': self.features,
                            'sem_class': self.sem_class,
                            'num_clouds': self.num_clouds
                        }, f)
                    print("Cache saved successfully")
                except Exception as e:
                    print(f"Error saving cache: {e}")

    def load_and_preprocess_data(self):
        """Load and preprocess the point cloud data"""
        try:
            print(f"Loading data from {self.file_path}")
            # Optimize data loading by using numpy's memmap for large files
            data = np.loadtxt(self.file_path, delimiter=' ')
            if data.shape[1] < 6:
                raise ValueError("PTS file must have at least 6 columns (X Y Z intensity sem_class ins_class)")

            # Only use a subset of points to speed up initial processing
            # For large files, randomly sample 5 million points maximum
            if len(data) > 5000000:
                indices = np.random.choice(len(data), 5000000, replace=False)
                data = data[indices]
                print(f"Sampled 5,000,000 points from original {len(data)} points")

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

            # Calculate number of potential cloud samples
            self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)

            # Print dataset info
            unique_classes = np.unique(self.sem_class)
            mapped_classes = [self.class_mapping.get(c, -1) for c in unique_classes if c in self.class_mapping]
            print(f"Loaded {len(self.xyz)} points from {self.file_path}")
            print(f"Creating {self.num_clouds} clouds with {self.points_per_cloud} points each")
            print(f"Unique semantic classes (mapped): {mapped_classes}")

        except Exception as e:
            print(f"Error loading PTS file {self.file_path}: {e}")
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

        # Simple data augmentation (faster than complex transformation)
        if self.augment:
            # Random rotation around z-axis (more efficient)
            if np.random.random() > 0.5:
                angle = np.random.uniform(0, 2*np.pi)
                cos, sin = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([
                    [cos, -sin, 0],
                    [sin, cos, 0],
                    [0, 0, 1]
                ])
                xyz = xyz @ rotation_matrix

            # Random scaling (only one augmentation to save time)
            elif np.random.random() > 0.5:
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
def create_datasets(base_dirs, points_per_count=1024, augment_train=True, subset_fraction=0.5):
    """Create training and validation datasets from directories with caching"""
    train_dir = os.path.join(base_dirs, 'Train')
    val_dir = os.path.join(base_dirs, 'Valid')
    cache_dir = os.path.join(base_dirs, 'cache')

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Find all .pts files
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pts')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pts')]

    # Use subset of training files to speed up training
    num_train_files = max(3, int(len(train_files) * subset_fraction))
    train_files = train_files[:num_train_files]

    # Use subset of validation files
    num_val_files = max(2, int(len(val_files) * subset_fraction))
    val_files = val_files[:num_val_files]

    print(f"Selected {len(train_files)}/{len(os.listdir(train_dir))} training files and {len(val_files)}/{len(os.listdir(val_dir))} validation files")

    # Create datasets with progress bar
    print("Creating training datasets...")
    train_datasets = []
    for file in tqdm(train_files):
        if os.path.exists(file):
            train_datasets.append(
                PtsPointCloudDataset(file, points_per_count, augment=augment_train, cache_dir=cache_dir)
            )

    print("Creating validation datasets...")
    val_datasets = []
    for file in tqdm(val_files):
        if os.path.exists(file):
            val_datasets.append(
                PtsPointCloudDataset(file, points_per_count, augment=False, cache_dir=cache_dir)
            )

    # Combine datasets
    if train_datasets:
        train_dataset = ConcatDataset(train_datasets)
        print(f"Combined training dataset with {len(train_dataset)} samples")
    else:
        train_dataset = None
        print("No training dataset created")

    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
        print(f"Combined validation dataset with {len(val_dataset)} samples")
    else:
        val_dataset = None
        print("No validation dataset created")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=16, num_workers=6):
    """Create DataLoaders from datasets"""
    train_loader = None
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Drop last batch to avoid BatchNorm issues
            prefetch_factor=3,  # Increased prefetch for better performance
            persistent_workers=True  # Keep workers alive between epochs
        )
        print(f"Created training dataloader with {len(train_loader)} batches")

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=3,  # Increased prefetch
            persistent_workers=True  # Keep workers alive
        )
        print(f"Created validation dataloader with {len(val_loader)} batches")

    return train_loader, val_loader


# === Early stopping class ===
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_accuracy):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


# === Improved validation function with error handling ===
def validate_model(model, data_loader, criterion, device, logger):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Track per-class metrics for better evaluation
    class_correct = {}
    class_total = {}

    val_bar = tqdm(data_loader, desc="Validating")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_bar):
            try:
                # Check if batch data is complete
                if len(batch_data) != 3:
                    log_info(logger, f"Skipping incomplete validation batch {batch_idx}: expected 3 items, got {len(batch_data)}")
                    continue

                features, xyz, labels = batch_data

                # Check for NaN or inf values
                if torch.isnan(features).any() or torch.isinf(features).any() or \
                torch.isnan(xyz).any() or torch.isinf(xyz).any():
                    log_info(logger, f"Skipping validation batch {batch_idx} with NaN or inf values")
                    continue

                features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)

                # Forward pass
                with autocast():
                    logits = model(features, xyz)
                    loss = criterion(logits, labels)

                # Calculate metrics if loss is valid
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    predicted = torch.argmax(logits, dim=1)
                    batch_correct = (predicted == labels).sum().item()
                    correct += batch_correct
                    total += labels.size(0)

                    # Update per-class metrics
                    for cls in torch.unique(labels):
                        cls_idx = cls.item()
                        if cls_idx not in class_correct:
                            class_correct[cls_idx] = 0
                            class_total[cls_idx] = 0

                        mask = (labels == cls_idx)
                        class_correct[cls_idx] += (predicted[mask] == labels[mask]).sum().item()
                        class_total[cls_idx] += mask.sum().item()

                    # Update progress bar
                    val_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{100 * batch_correct / max(1, labels.size(0)):.1f}%"
                    })
                else:
                    log_info(logger, f"Skipping validation batch {batch_idx} due to NaN or inf loss")

            except Exception as e:
                log_info(logger, f"Error in validation batch {batch_idx}: {str(e)}")
                import traceback
                log_info(logger, traceback.format_exc())
                continue

    # Calculate overall metrics safely
    if total > 0:
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
    else:
        avg_loss = float('inf')
        accuracy = 0.0

    # Log overall metrics
    log_info(logger, f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Log per-class accuracy for better model evaluation
    if class_total:
        class_metrics = "\nPer-class validation accuracy:\n" + "-" * 30 + "\n"
        for cls_idx in sorted(class_total.keys()):
            cls_acc = 100 * class_correct[cls_idx] / max(1, class_total[cls_idx])
            class_metrics += f"Class {cls_idx}: {cls_acc:.2f}% ({class_correct[cls_idx]}/{class_total[cls_idx]})\n"
        log_info(logger, class_metrics)

    return avg_loss, accuracy


# === Improved Training function with comprehensive error handling ===
def train_model(model, train_loader, val_loader, args, device, save_dir, logger):
    """Train the PointNet2 model with mixed precision and early stopping"""
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

    # Mixed precision training - try training without this first to debug
    scaler = GradScaler()

    # Early stopping
    early_stopping = EarlyStopping(patience=args['early_stopping_patience'])

    # Training metrics tracking
    best_val_accuracy = 0.0
    training_start_time = time.time()

    # Create lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_list = []

    # Add try-except block for the entire training loop
    try:
        # Training loop
        for epoch in range(args['epochs']):
            epoch_start_time = time.time()
            model.train()

            # Report CUDA memory usage
            if device.type == 'cuda':
                try:
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    log_info(logger, f"CUDA Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                except Exception as e:
                    log_info(logger, f"Error reporting CUDA memory: {str(e)}")

            # Metrics
            total_loss = 0
            correct = 0
            total = 0
            batch_count = 0

            # Training batches with progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}")

            # Wrap batch processing in try-except
            for batch_idx, batch_data in enumerate(progress_bar):
                try:
                    # Check if batch data is complete
                    if len(batch_data) != 3:
                        log_info(logger, f"Skipping incomplete batch {batch_idx}: expected 3 items, got {len(batch_data)}")
                        continue

                    features, xyz, labels = batch_data

                    # Check for NaN or inf values
                    if torch.isnan(features).any() or torch.isinf(features).any() or \
                       torch.isnan(xyz).any() or torch.isinf(xyz).any():
                        log_info(logger, f"Skipping batch {batch_idx} with NaN or inf values")
                        continue

                    features = features.to(device)
                    xyz = xyz.to(device)
                    labels = labels.to(device)

                    # Skip problematic small batches
                    if features.size(0) <= 1:
                        log_info(logger, f"Skipping batch {batch_idx} with size {features.size(0)}")
                        continue

                    # Zero gradients
                    optimizer.zero_grad()

                    # Gradual transition to mixed precision - first few epochs without it
                    if epoch < 51:  # Use normal precision for first 3 epochs
                        logits = model(features, xyz)
                        loss = criterion(logits, labels)
                        loss.backward()
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    else:
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

                    # Track metrics safely
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        predicted = torch.argmax(logits, dim=1)
                        batch_correct = (predicted == labels).sum().item()
                        correct += batch_correct
                        total += labels.size(0)
                        batch_count += 1

                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'acc': f"{100 * batch_correct / max(1, labels.size(0)):.1f}%",
                            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                        })

                        # Log batch info less frequently
                        if batch_idx % 50 == 0:
                            log_info(logger, f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                                    f"Loss: {loss.item():.4f}, "
                                    f"Acc: {100 * batch_correct / max(1, labels.size(0)):.1f}%")
                    else:
                        log_info(logger, f"Skipping batch {batch_idx} due to NaN or inf loss")

                except Exception as e:
                    log_info(logger, f"Error in batch {batch_idx}: {str(e)}")
                    # Print full stack trace for debugging
                    import traceback
                    log_info(logger, traceback.format_exc())
                    # Continue to next batch rather than breaking the entire training
                    continue

            # Calculate epoch metrics safely
            if batch_count > 0:
                train_loss = total_loss / batch_count
                train_accuracy = 100 * correct / max(1, total)
            else:
                log_info(logger, "Warning: No valid batches in this epoch")
                train_loss = float('inf')
                train_accuracy = 0.0

            # Store training metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            epochs_list.append(epoch + 1)

            # Step the scheduler
            scheduler.step()

            # Validation - wrap in try-except
            val_loss, val_accuracy = float('inf'), 0.0
            if val_loader:
                try:
                    log_info(logger, "Starting validation...")
                    val_loss, val_accuracy = validate_model(model, val_loader, criterion, device, logger)

                    # Store validation metrics
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)

                    # Save best model
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_model_path = os.path.join(save_dir, "pointnet2_best_model.pth")
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'accuracy': val_accuracy,
                                'args': args
                            }, best_model_path)
                            log_info(logger, f"✓ New best model saved with accuracy {best_val_accuracy:.2f}%")
                        except Exception as e:
                            log_info(logger, f"Error saving best model: {str(e)}")

                    # Check early stopping
                    if early_stopping(val_accuracy):
                        log_info(logger, f"Early stopping triggered after {epoch+1} epochs")
                        break
                except Exception as e:
                    log_info(logger, f"Error during validation: {str(e)}")
                    import traceback
                    log_info(logger, traceback.format_exc())

            # Save checkpoint less frequently and safely
            if (epoch + 1) % 10 == 0 or epoch == 0:
                checkpoint_path = os.path.join(save_dir, f"pointnet2_epoch_{epoch+1}.pth")
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'accuracy': val_accuracy if val_loader else train_accuracy,
                        'args': args,
                        'metrics': {
                            'train_losses': train_losses,
                            'train_accuracies': train_accuracies,
                            'val_losses': val_losses,
                            'val_accuracies': val_accuracies,
                            'epochs_list': epochs_list
                        }
                    }, checkpoint_path)
                    log_info(logger, f"✓ Checkpoint saved at epoch {epoch+1}")
                except Exception as e:
                    log_info(logger, f"Error saving checkpoint: {str(e)}")

            # Log progress with clear metrics display
            epoch_time = time.time() - epoch_start_time
            log_info(logger, f"Epoch {epoch+1}/{args['epochs']} completed in {epoch_time:.2f}s")
            log_info(logger, f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            if val_loader:
                log_info(logger, f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            log_info(logger, f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Create a formatted metrics table every epoch
            metrics_table = (
                f"\n{'=' * 50}\n"
                f"EPOCH {epoch+1} METRICS SUMMARY\n"
                f"{'-' * 50}\n"
                f"Training Loss:      {train_loss:.4f}\n"
                f"Training Accuracy:  {train_accuracy:.2f}%\n"
            )
            if val_loader:
                metrics_table += (
                    f"Validation Loss:    {val_loss:.4f}\n"
                    f"Validation Accuracy:{val_accuracy:.2f}%\n"
                )
            metrics_table += f"{'=' * 50}"
            log_info(logger, metrics_table)

            # Also print to console for visibility
            print(metrics_table)

        # Save final model with all metrics
        final_model_path = os.path.join(save_dir, "pointnet2_final_model.pth")
        try:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'accuracy': best_val_accuracy,
                'args': args,
                'metrics': {
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'epochs_list': epochs_list
                }
            }, final_model_path)
            log_info(logger, f"Final model saved to {final_model_path}")
        except Exception as e:
            log_info(logger, f"Error saving final model: {str(e)}")

    except Exception as e:
        log_info(logger, f"Critical error in training loop: {str(e)}")
        import traceback
        log_info(logger, traceback.format_exc())

    # Log training summary
    total_time = time.time() - training_start_time
    log_info(logger, f"Training completed in {total_time / 60:.2f} minutes")
    log_info(logger, f"Best validation accuracy: {best_val_accuracy:.2f}%")

    # Plot and save training curves safely
    try:
        import matplotlib.pyplot as plt
        # Set figure size and title
        plt.figure(figsize=(12, 10))

        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs_list, train_losses, 'b-', label='Training Loss')
        if val_loader and len(val_losses) == len(epochs_list):
            plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs_list, train_accuracies, 'b-', label='Training Accuracy')
        if val_loader and len(val_accuracies) == len(epochs_list):
            plt.plot(epochs_list, val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()

        # Save the figure
        plt.tight_layout()
        plots_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plots_path)
        plt.close()
        log_info(logger, f"Training curves saved to {plots_path}")
    except Exception as e:
        log_info(logger, f"Error creating training plots: {str(e)}")

    return model, best_val_accuracy

# === Main function ===
def main():
    # Optimized training settings
    args = {
        'data_dir': "/content/drive/MyDrive/Dales/DALESObjects/DALESObjects/train_pts",
        'save_dir': "/content/drive/MyDrive/Dales/DALESObjects/DALESObjects/models",
        'points_per_cloud': 1024,     # Number of points to sample per cloud
        'batch_size': 32,             # Increased for faster training
        'learning_rate': 0.001,       # Slightly higher learning rate
        'weight_decay': 1e-4,         # Weight decay for regularization
        'epochs': 66,                 # Reduced number of training epochs
        'num_workers': 6,             # Increased DataLoader workers
        'in_dim': 1,                  # Input feature dimension
        'num_classes': 8,             # Output classes (DALES dataset)
        'checkpoint_interval': 5,    # Save checkpoint less frequently
        'early_stopping_patience': 10, # Early stopping patience
        'subset_fraction': 0.5,       # Use only 50% of data for faster testing
    }

    # Setup device and directories
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args['save_dir'], exist_ok=True)
    logger = setup_logger(args['save_dir'])

    # Log system info
    log_info(logger, f"Using device: {device}")
    if device.type == 'cuda':
        log_info(logger, f"CUDA Device: {torch.cuda.get_device_name(0)}")
        log_info(logger, f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Set cudnn benchmark for possibly faster training
        torch.backends.cudnn.benchmark = True

    # Log training parameters
    log_info(logger, f"Training with parameters: {args}")

    # Try to catch all errors for better debugging
    try:
        # Create datasets and dataloaders
        log_info(logger, "Creating datasets...")
        train_dataset, val_dataset = create_datasets(
            args['data_dir'],
            points_per_count=args['points_per_cloud'],
            augment_train=True,  # Enable augmentation for training
            subset_fraction=args['subset_fraction']
        )

        if train_dataset is None or len(train_dataset) == 0:
            log_info(logger, "No training data found. Exiting.")
            return

        log_info(logger, f"Training dataset size: {len(train_dataset)} samples")

        if val_dataset is None or len(val_dataset) == 0:
            log_info(logger, "Warning: No validation data found. Training without validation.")
            val_dataset = None
        else:
            log_info(logger, f"Validation dataset size: {len(val_dataset)} samples")

        log_info(logger, "Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=args['batch_size'],
            num_workers=args['num_workers']
        )

        # Check if we have a valid dataloader
        if train_loader is None or len(train_loader) == 0:
            log_info(logger, "Error: Empty training dataloader. Exiting.")
            return

        # Create model with optimized architecture
        log_info(logger, "Creating model...")
        try:
            model = PointNet2ClsMSG(
                in_dim=args['in_dim'],
                out_dim=args['num_classes'],
                downsample_points=(2048, 1024, 512, 256),  # Reduced points for efficiency
                ball_radii=(0.5, 1.0, 5.0, 15.0),          # Multiple scales
                neighbor_counts=(16, 64, 64, 32),            # Reduced neighbors for speed
                head_norm=True,                            # Use normalization in head
                dropout=0.2                                # Reduced dropout for better stability
            )

            # Initialize weights for better convergence
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
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log_info(logger, f"Model: PointNet2ClsMSG")
            log_info(logger, f"Input dimension: {args['in_dim']}, Output classes: {args['num_classes']}")
            log_info(logger, f"Parameter count: {param_count:,}")

            # Check model size
            if device.type == 'cuda':
                try:
                    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                    log_info(logger, f"Approximate model size: {model_size_mb:.2f} MB")
                except Exception as e:
                    log_info(logger, f"Could not calculate model size: {str(e)}")

        except Exception as e:
            log_info(logger, f"Error creating model: {str(e)}")
            import traceback
            log_info(logger, traceback.format_exc())
            return

        # Train model
        log_info(logger, "=" * 50)
        log_info(logger, "Starting training...")
        log_info(logger, "=" * 50)

        try:
            # Wrap training in try-except for better error handling
            model, best_accuracy = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                args=args,
                device=device,
                save_dir=args['save_dir'],
                logger=logger
            )

            log_info(logger, f"Training completed successfully with best accuracy: {best_accuracy:.2f}%")

            # Save a lightweight deployment version (just weights)
            try:
                deployment_path = os.path.join(args['save_dir'], "pointnet2_deployment.pth")
                torch.save(model.state_dict(), deployment_path)
                log_info(logger, f"Deployment model saved to {deployment_path}")
            except Exception as e:
                log_info(logger, f"Error saving deployment model: {str(e)}")

        except Exception as e:
            log_info(logger, f"Error during training: {str(e)}")
            import traceback
            log_info(logger, traceback.format_exc())

    except Exception as e:
        log_info(logger, f"Critical error in training pipeline: {str(e)}")
        import traceback
        log_info(logger, traceback.format_exc())

    finally:
        # Clean up resources
        if 'train_loader' in locals() and train_loader is not None:
            try:
                # Close dataloader workers to prevent hanging
                train_loader._iterator._shutdown_workers()
            except:
                pass

        if 'val_loader' in locals() and val_loader is not None:
            try:
                val_loader._iterator._shutdown_workers()
            except:
                pass

        # Report final memory usage
        if device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                log_info(logger, f"Final CUDA Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                # Empty cache to free memory
                torch.cuda.empty_cache()
            except:
                pass

        log_info(logger, "=" * 50)
        log_info(logger, "Training script finished")
        log_info(logger, "=" * 50)


if __name__ == "__main__":
    main()

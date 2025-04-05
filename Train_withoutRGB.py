import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from pointnet_ import PointNet2ClsSSG

class PointCloudDataset(Dataset):
    def __init__(self, file_path, chunk_size=1024):
        self.chunk_size = chunk_size
        data = np.loadtxt(file_path)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Add shape verification
        print(f"Loaded data shape: {data.shape}")
        assert data.shape[1] >= 6, f"Data should have at least 6 columns, got {data.shape[1]}"
            
        xyz = data[:, :3].astype(np.float32)
        features = data[:, 3:6].astype(np.float32)
        self.label = data[0, -1].astype(np.int64)
        
        # Verify shapes before chunking
        print(f"XYZ shape: {xyz.shape}, Features shape: {features.shape}")
        
        # Calculate number of full chunks and remaining points
        n_points = len(xyz)
        n_full_chunks = n_points // chunk_size
        remaining_points = n_points % chunk_size
        
        # Split points into chunks
        self.chunks_xyz = []
        self.chunks_features = []
        
        # First, create all full chunks
        for i in range(n_full_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            self.chunks_xyz.append(xyz[start_idx:end_idx])
            self.chunks_features.append(features[start_idx:end_idx])
        
        # Handle the last incomplete chunk if it exists
        if remaining_points > 0:
            # Take all remaining points
            last_chunk_xyz = xyz[-remaining_points:]
            last_chunk_features = features[-remaining_points:]
            
            # Calculate how many points we need to add
            points_needed = chunk_size - remaining_points
            
            # Randomly select indices from the same file to fill the last chunk
            indices = np.random.choice(n_points - remaining_points, points_needed, replace=True)
            
            # Add the selected points to complete the last chunk
            last_chunk_xyz = np.vstack([last_chunk_xyz, xyz[indices]])
            last_chunk_features = np.vstack([last_chunk_features, features[indices]])
            
            self.chunks_xyz.append(last_chunk_xyz)
            self.chunks_features.append(last_chunk_features)
        
        self.n_chunks = len(self.chunks_xyz)
        print(f"Loaded {n_points} points from {file_path}")
        print(f"Created {self.n_chunks} chunks of {chunk_size} points each")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return (torch.from_numpy(self.chunks_xyz[idx]),
                torch.from_numpy(self.chunks_features[idx]),
                self.label)

def collate_fn(batch):
    xyz_list = []
    features_list = []
    labels_list = []
    
    for xyz, features, label in batch:
        xyz_list.append(xyz)
        features_list.append(features)
        labels_list.append(label)
    
    xyz_batch = torch.stack(xyz_list)
    features_batch = torch.stack(features_list)
    labels_batch = torch.tensor(labels_list, dtype=torch.int64)
    
    return xyz_batch, features_batch, labels_batch
# Load all dataset files
def load_datasets(file_dir):
    files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.pts')]
    datasets = []
    for f in files:
        try:
            dataset = PointCloudDataset(f)
            datasets.append(dataset)
        except Exception as e:
            print(f"Error loading {f}: {str(e)}")
    return datasets

# Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xyz, features, labels in dataloader:
        # Add dimension checks
        print(f"XYZ shape before transpose: {xyz.shape}")
        print(f"Features shape before transpose: {features.shape}")
        
        xyz = xyz.transpose(1, 2)
        features = features.transpose(1, 2)
        
        print(f"XYZ shape after transpose: {xyz.shape}")
        print(f"Features shape after transpose: {features.shape}")
        
        xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)
        optimizer.zero_grad()

        # Note: model expects features first, then xyz
        outputs = model(features, xyz)  # Changed order here
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# Evaluation loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for xyz, features, labels in dataloader:
            # Transpose the data to match model's expected format
            xyz = xyz.transpose(1, 2)  # from (b, n, 3) to (b, 3, n)
            features = features.transpose(1, 2)  # from (b, n, c) to (b, c, n)
            
            xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)

            # Note: model expects features first, then xyz
            outputs = model(features, xyz)  # Changed order here
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy


def main():
    # Configuration
    data_dir = r"C:\\Farshid\\Uni\\Semesters\\Thesis\\Data\\Vaihingen\\Vaihingen\\Augmentation"
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.001
    step_size = 5
    gamma = 0.5
    chunk_size = 1024  # Size of each point cloud chunk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Checking dataset directory: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.endswith('.pts')]
    print("Found files:", files)

    # Load datasets
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pts')]
    datasets = [PointCloudDataset(f, chunk_size=chunk_size) for f in files]
    
    # Get feature dimension from first chunk of first dataset
    feature_dim = datasets[0].chunks_features[0].shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    num_files = len(datasets)
    print(f"Number of datasets loaded: {num_files}")

    if num_files < 2:
        raise ValueError("Not enough .pts files to split into training and validation sets.")

    # Use 80% for training and 20% for validation
    split_idx = int(0.8 * num_files)
    train_datasets = datasets[:split_idx]
    val_dataset = datasets[split_idx:]

    print(f"Training on {len(train_datasets)} datasets, Validating on {len(val_dataset)} datasets.")

    if len(train_datasets) == 0 or len(val_dataset) == 0:
        raise ValueError("One of the datasets (train or validation) is empty. Check the dataset path!")

    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_datasets), 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # Model initialization
    in_dim = feature_dim
    out_dim = max([ds.label for ds in datasets]) + 1
    model = PointNet2ClsSSG(
        in_dim=in_dim,
        out_dim=out_dim,
        downsample_points=[512, 128],
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Use the existing train and evaluate functions
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(data_dir, 'checkpoints', 'best_model.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
        
        scheduler.step()
        print("-" * 50)

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()

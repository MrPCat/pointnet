import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG
import logging

# === Configure Logging ===
log_file_path = "/content/drive/MyDrive/t1/training_logs.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)  # Print to console
    logging.info(message)  # Save to log file

# === Dynamic Feature Dataset Class ===
class DynamicFeatureDataset(Dataset):
    def __init__(self, file_path, features_to_match, points_per_cloud=1024):
        # Load the dataset
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
        
        # Define all possible features (assuming columns are known)
        all_feature_names = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Reflectance', 'NumberOfReturns', 'ReturnNumber']
        feature_indices = {name: idx for idx, name in enumerate(all_feature_names)}

        # Determine which features are available in the current file
        self.matched_features = [name for name in all_feature_names if name in features_to_match]
        self.feature_indices = [feature_indices[name] for name in self.matched_features]
        
        # Extract XYZ, matched features, and labels (if available)
        self.xyz = self.data[:, :3]  # Always include XYZ
        self.features = self.data[:, self.feature_indices]
        self.labels = self.data[:, -1] if self.data.shape[1] > max(self.feature_indices) + 1 else None

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Handle points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        if len(self.xyz) % self.points_per_cloud != 0:
            print("Warning: Dataset points not divisible by points_per_cloud. Truncating extra points.")
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]
            if self.labels is not None:
                self.labels = self.labels[:self.num_clouds * self.points_per_cloud]

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Get a subset of points for the current cloud
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        
        if self.labels is not None:
            labels = torch.tensor(self.labels[start:end], dtype=torch.long)  # Shape: [points_per_cloud]
            label = torch.mode(labels).values  # Use the most common label as the cloud's label
            return features, xyz, label
        else:
            return features, xyz

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

# === Testing Function ===
def test_model(model, test_loader, device, output_path):
    model.eval()
    predictions = []

    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predicted = torch.argmax(logits, dim=1)
            predictions.append(predicted.cpu().numpy())

    predictions = np.concatenate(predictions)
    dataset_with_predictions = np.hstack((test_loader.dataset.data, predictions.reshape(-1, 1)))

    # Save the augmented dataset to a new file
    np.savetxt(output_path, dataset_with_predictions, delimiter='\t', fmt='%0.8f', 
               header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
    log_and_print(f"Augmented dataset saved to {output_path}")

# === Main ===
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")

    # Specify File Paths
    train_file = '/content/drive/MyDrive/t1/Mar18_train.txt'
    val_file = '/content/drive/MyDrive/t1/Mar18_val.txt'
    test_file = '/content/drive/MyDrive/t1/Mar18_test.txt'
    output_file = '/content/drive/MyDrive/t1/Mar18_test_with_predictions.txt'

    # Dataset and DataLoader
    train_features = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Reflectance', 'NumberOfReturns', 'ReturnNumber']
    batch_size = 16

    train_dataset = DynamicFeatureDataset(train_file, features_to_match=train_features, points_per_cloud=1024)
    val_dataset = DynamicFeatureDataset(val_file, features_to_match=train_features, points_per_cloud=1024)
    test_dataset = DynamicFeatureDataset(test_file, features_to_match=train_features, points_per_cloud=1024)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Optimizer, and Loss Function
    in_dim = len(train_dataset.matched_features)
    num_classes = len(np.unique(train_dataset.labels))

    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes, downsample_points=(512, 128))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    # Directory for saving checkpoints
    save_dir = "/content/drive/MyDrive/t1/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Training
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_dir)

    # Testing
    test_model(model, test_loader, device, output_file)

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG
from sklearn.metrics import confusion_matrix, accuracy_score


class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True, label_column_index=6):
        print(f"Reading data from {file_path}")
        
        # Try reading as space-delimited
        try:
            data = pd.read_csv(file_path, delimiter=' ', header=None)
            print(f"Read data with shape: {data.shape}")
            
            # Assign default column names
            num_cols = data.shape[1]
            if num_cols >= 3:
                # Rename columns for clarity
                col_names = ['X', 'Y', 'Z'] + [f'Feature_{i}' for i in range(num_cols-3)]
                data.columns = col_names
                print(f"Assigned column names: {col_names}")
            else:
                print(f"Warning: Expected at least 3 columns for XYZ, but found {num_cols}")
        
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}. Error: {e}")
        
        # Extract XYZ coordinates
        self.xyz = data.iloc[:, 0:3].values.astype(np.float64)
        
        # Extract labels from the specified column if available
        self.has_labels = False
        self.labels = None
        
        if label_column_index is not None and label_column_index < num_cols:
            try:
                print(f"Using column {label_column_index} as labels")
                self.labels = data.iloc[:, label_column_index].values.astype(np.int64)
                self.has_labels = True
                print(f"Loaded {len(self.labels)} labels")
            except Exception as e:
                print(f"Error loading labels: {e}")
        
        # Features are everything except XYZ and the label column
        feature_columns = list(range(3, num_cols))
        if label_column_index is not None and label_column_index >= 3:
            feature_columns.remove(label_column_index)
        
        if feature_columns:
            self.features = data.iloc[:, feature_columns].values.astype(np.float64)
        else:
            # If no features, create a dummy feature column of ones
            print("No feature columns found, creating dummy feature")
            self.features = np.ones((len(self.xyz), 1), dtype=np.float64)
        
        # Normalize XYZ and features
        self.xyz_mean = np.mean(self.xyz, axis=0).astype(np.float64)
        self.xyz -= self.xyz_mean
        
        # Apply feature normalization safely
        if self.features.shape[1] > 0:
            feature_std = np.std(self.features, axis=0)
            # Avoid division by zero
            feature_std[feature_std == 0] = 1.0
            self.features = (self.features - np.mean(self.features, axis=0)) / feature_std

        # Ensure divisibility by points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
        self.features = self.features[:self.num_clouds * self.points_per_cloud]
        
        if self.has_labels:
            # Make sure we have enough labels
            if len(self.labels) >= self.num_clouds * self.points_per_cloud:
                self.labels = self.labels[:self.num_clouds * self.points_per_cloud]
            else:
                print(f"Warning: Not enough labels ({len(self.labels)}) for all points ({self.num_clouds * self.points_per_cloud})")
                # Extend labels if needed
                extra_needed = self.num_clouds * self.points_per_cloud - len(self.labels)
                if extra_needed > 0:
                    self.labels = np.concatenate([self.labels, np.zeros(extra_needed, dtype=np.int64)])

        if debug:
            self.print_debug_info()
        

    def print_debug_info(self):
        print("\n--- Dataset Debugging Information ---")
        print(f"Total Points: {len(self.xyz)}")
        print(f"Points per Cloud: {self.points_per_cloud}")
        print(f"Number of Point Clouds: {self.num_clouds}")
        print(f"XYZ Shape: {self.xyz.shape}")
        print(f"Features Shape: {self.features.shape}")
        print(f"XYZ Mean: {self.xyz_mean}")
        
        if self.has_labels:
            unique_labels = np.unique(self.labels)
            print(f"Labels found: {unique_labels}")
            for label in unique_labels:
                count = np.sum(self.labels == label)
                percentage = (count / len(self.labels)) * 100
                print(f"  Class {label}: {count} points ({percentage:.2f}%)")
        else:
            print("No labels available")

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)
        features = torch.tensor(self.features[start:end], dtype=torch.float32)
        
        # Transpose for PointNet format
        xyz = xyz.transpose(0, 1)
        features = features.transpose(0, 1)
        
        if self.has_labels:
            # Determine the dominant label for this point cloud using voting
            labels = self.labels[start:end]
            majority_label = int(np.bincount(labels.astype(int)).argmax())
            label = torch.tensor(majority_label, dtype=torch.long)
            return features, xyz, label
        
        return features, xyz


def load_model(model_path, input_dim, output_dim):
    model = PointNet2ClsSSG(in_dim=input_dim, out_dim=output_dim)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Initializing model from scratch.")
    return model


def evaluate_model_with_checkpoint(test_file, model_path, checkpoint_with_labels, num_classes=11):
    """
    Evaluate the model using a checkpoint that contains ground truth labels
    
    Args:
        test_file: Path to the test data file
        model_path: Path to the model weights
        checkpoint_with_labels: Path to a file containing ground truth labels
        num_classes: Number of classes to predict
    """
    # Load dataset with labels from checkpoint
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True, label_file=checkpoint_with_labels)
    
    if not test_dataset.has_labels:
        print("No labels found in the checkpoint. Cannot evaluate accuracy.")
        return
    
    # Determine input dimension from the features
    input_dim = test_dataset.features.shape[1]
    model = load_model(model_path, input_dim=input_dim, output_dim=num_classes)
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for features, xyz, labels in test_loader:
            features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
            logits = model(features, xyz)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n===== ACCURACY EVALUATION =====")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Get all unique class labels (from both predictions and ground truth)
    unique_classes = np.unique(np.concatenate([all_labels, all_predictions]))
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    print("Class\tCount\tAccuracy")
    print("-" * 30)
    
    for cls in unique_classes:
        # Get samples for this class
        cls_indices = (all_labels == cls)
        cls_count = np.sum(cls_indices)
        
        if cls_count > 0:
            # Calculate accuracy for this class
            cls_correct = np.sum(all_predictions[cls_indices] == cls)
            cls_accuracy = cls_correct / cls_count
            print(f"{cls}\t{cls_count}\t{cls_accuracy:.4f}")
        else:
            print(f"{cls}\t0\tN/A")
    
    # Print confusion matrix if it's not too large
    if len(unique_classes) <= 20:  # Only print if not too large
        print("\nConfusion Matrix:")
        print("Rows: True labels, Columns: Predicted labels")
        print("\t", end="")
        for cls in unique_classes:
            print(f"{cls}\t", end="")
        print()
        
        for i, cls_true in enumerate(unique_classes):
            row_idx = np.where(unique_classes == cls_true)[0][0]
            print(f"{cls_true}\t", end="")
            for j, cls_pred in enumerate(unique_classes):
                col_idx = np.where(unique_classes == cls_pred)[0][0]
                if row_idx < cm.shape[0] and col_idx < cm.shape[1]:
                    print(f"{cm[row_idx, col_idx]}\t", end="")
                else:
                    print("0\t", end="")
            print()
    else:
        print("\nConfusion matrix too large to display. Skipping.")


def predict_with_model(test_file, model_path, num_classes=11):
    """
    Run predictions using the model without evaluation
    
    Args:
        test_file: Path to the test data file
        model_path: Path to the model weights
        num_classes: Number of classes to predict
    """
    # Load dataset
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)
    
    # Determine input dimension from the features
    input_dim = test_dataset.features.shape[1]
    model = load_model(model_path, input_dim=input_dim, output_dim=num_classes)
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_predictions = []
    
    print("\nRunning predictions...")
    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    
    # Count predictions per class
    unique_preds, counts = np.unique(all_predictions, return_counts=True)
    total = len(all_predictions)
    
    print("\n===== PREDICTION DISTRIBUTION =====")
    print("Class\tCount\tPercentage")
    print("-" * 30)
    
    for cls, count in zip(unique_preds, counts):
        percentage = (count / total) * 100
        print(f"{cls}\t{count}\t{percentage:.2f}%")


if __name__ == "__main__":
    # File paths
    test_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\3DLabeling\Vaihingen3D_EVAL_WITH_REF.pts\Vaihingen3D_EVAL_WITH_REF.pts"
    model_path = r"C:\Users\faars\Downloads\s3dis-train-pointnet++s3ids.pth"
    
    # Create dataset with labels from the 7th column (index 6)
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True, label_column_index=6)
    
    if test_dataset.has_labels:
        # Evaluate model with ground truth labels
        # Note: Update num_classes to 9 to match your dataset
        evaluate_model_with_checkpoint(test_file, model_path, None, num_classes=9)
    else:
        # Just run predictions and show distribution
        predict_with_model(test_file, model_path, num_classes=9)

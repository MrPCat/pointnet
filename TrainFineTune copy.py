import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pointnet_ import PointNet2ClsSSG
import numpy as np
from collections import Counter
import logging

# === Configure Logging ===
log_file_path = r"C:\Users\faars\Downloads\Prediction_Log.txt"
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

def log_and_print(message):
    print(message)  # Print to console
    logging.info(message)  # Save to log file

# === Dataset Class for PTS files (for reference) ===
class PtsPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        log_and_print(f"Loading PTS file: {file_path}")
        
        try:
            data = np.loadtxt(file_path, delimiter=' ')
            
            if data.shape[1] >= 7:
                # Extract XYZ, Intensity, return_number, number_of_returns, and label
                self.xyz = data[:, 0:3]  # X, Y, Z
                self.intensity = data[:, 3:4]  # Intensity
                self.return_number = data[:, 4:5]  # Return Number
                self.number_of_returns = data[:, 5:6]  # Number of Returns
                self.labels = data[:, 6].astype(np.int64)  # Label

                # Normalize XYZ coordinates
                self.xyz -= np.mean(self.xyz, axis=0)

                # Combine features into a single tensor (excluding XYZ)
                self.features = np.concatenate((self.intensity, self.return_number, self.number_of_returns), axis=1)  # Shape: (N, num_features)
                
                # Calculate the number of features (excluding XYZ)
                self.num_features = self.features.shape[1]

                self.points_per_cloud = points_per_cloud
                self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)

                log_and_print(f"Loaded {len(self.xyz)} points, creating {self.num_clouds} clouds with {self.num_features} features per point")
            else:
                raise ValueError("PTS file must have at least 7 columns (X Y Z Intensity return_number number_of_returns label)")

        except Exception as e:
            log_and_print(f"Error loading PTS file: {e}")
            self.xyz = np.zeros((points_per_cloud, 3))
            self.intensity = np.zeros((points_per_cloud, 1))
            self.return_number = np.zeros((points_per_cloud, 1))
            self.number_of_returns = np.zeros((points_per_cloud, 1))
            self.labels = np.zeros(points_per_cloud).astype(np.int64)
            self.features = np.zeros((points_per_cloud, 3))
            self.num_features = 3  # Default to 3 for XYZ
            self.points_per_cloud = points_per_cloud
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
            xyz = self.xyz[indices]
            intensity = self.intensity[indices]
            return_number = self.return_number[indices]
            number_of_returns = self.number_of_returns[indices]
            point_labels = self.labels[indices]
        else:
            xyz = self.xyz[start:end]
            intensity = self.intensity[start:end]
            return_number = self.return_number[start:end]
            number_of_returns = self.number_of_returns[start:end]
            point_labels = self.labels[start:end]
        
        features = np.concatenate((intensity, return_number, number_of_returns), axis=1)  # Shape: (N, num_features)
        
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32).T  # Shape: (3, points_per_cloud)
        features_tensor = torch.tensor(features, dtype=torch.float32).T  # Shape: (num_features, points_per_cloud)
        
        label = torch.tensor(np.bincount(point_labels).argmax(), dtype=torch.long)
        
        return features_tensor, xyz_tensor, label

# === Helper function to determine file type and create appropriate dataset ===
def create_dataset(file_path, points_per_cloud=1024):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in ['.pts', '.txt', '.xyz']:
        return PtsPointCloudDataset(file_path, points_per_cloud)
    else:
        log_and_print(f"Unknown file type: {file_ext}, attempting to load as PTS")
        return PtsPointCloudDataset(file_path, points_per_cloud)

# === Predict Labels for New File ===
def predict_labels(model, data_loader, device):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for features, xyz, _ in data_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predicted = torch.argmax(logits, dim=1)
            all_predictions.extend(predicted.cpu().numpy())  # Collect predictions as a list
    
    return np.array(all_predictions)

# === Compare Predicted Labels with True Labels ===
def compare_predictions(predicted_labels, true_labels):
    # Count the occurrences of each class in the predictions and true labels
    pred_counter = Counter(predicted_labels)
    true_counter = Counter(true_labels)

    log_and_print("Class-wise Comparison:")
    log_and_print(f"Predicted Class Distribution: {dict(pred_counter)}")
    log_and_print(f"True Class Distribution: {dict(true_counter)}")
    
    # Calculate accuracy per class
    correct_per_class = {}
    for class_label in pred_counter:
        correct_per_class[class_label] = (predicted_labels == class_label).sum()  # Correct predictions for each class
        accuracy_per_class = (correct_per_class[class_label] / true_counter.get(class_label, 1)) * 100
        log_and_print(f"Class {class_label} -> Correct Predictions: {correct_per_class[class_label]}, Accuracy: {accuracy_per_class:.2f}%")

    return correct_per_class

# === Main Prediction and Comparison ===
def predict_and_compare(model_path, test_file, reference_file, points_per_cloud=1024, batch_size=16):
    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")

    # Load the trained model
    model = PointNet2ClsSSG(in_dim=3, out_dim=4, downsample_points=(512, 128))  # Adjust input/output dimensions if necessary
    model.load_state_dict(torch.load(model_path))  # Load the trained model
    model.to(device)
    log_and_print(f"Loaded model from {model_path}")

    # Create the dataset and dataloader for the test file (unlabeled)
    test_dataset = create_dataset(test_file, points_per_cloud=points_per_cloud)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the dataset and dataloader for the reference file (labeled)
    reference_dataset = create_dataset(reference_file, points_per_cloud=points_per_cloud)
    reference_loader = DataLoader(reference_dataset, batch_size=batch_size, shuffle=False)
    
    # Predict the labels for the test file (unlabeled)
    log_and_print("Starting prediction for the test file...")
    predicted_labels = predict_labels(model, test_loader, device)

    # Extract the true labels from the reference dataset
    true_labels = []
    for features, xyz, labels in reference_loader:
        true_labels.extend(labels.numpy())  # Collect the true labels from the reference dataset
    true_labels = np.array(true_labels)
    
    # Compare predicted labels with true labels
    if len(predicted_labels) == len(true_labels):
        compare_predictions(predicted_labels, true_labels)
    else:
        log_and_print("Warning: The number of predicted labels does not match the number of true labels!")

# === Main Code Execution ===
if __name__ == "__main__":
    # Specify the paths for the trained model and the test files
    model_path = r"C:\Users\faars\Downloads\pointnetDown_epoch_26.pth" # Path to your trained model
    test_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\3DLabeling\Vaihingen3D_EVAL_WITHOUT_REF.pts\Vaihingen3D_EVAL_WITHOUT_REF.pts"  # Test file without labels
    reference_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\3DLabeling\Vaihingen3D_EVAL_WITH_REF.pts\Vaihingen3D_EVAL_WITH_REF.pts"  # Reference file with labels

    # Call the function to predict and compare labels
    predict_and_compare(model_path, test_file, reference_file)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pointnet_ import PointNet2ClsSSG  # Replace with your PointNet model import

class DynamicFeatureDataset(Dataset):
    def __init__(self, file_path, features_to_match, points_per_cloud=1024):
        # Load the dataset
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
        
        # Define all possible features (assuming columns are known)
        all_feature_names = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Reflectance', 'NumberOfReturns', 'ReturnNumber']
        feature_indices = {name: idx for idx, name in enumerate(all_feature_names)}

        # Match features
        self.matched_features = [name for name in all_feature_names if name in features_to_match]
        self.feature_indices = [feature_indices[name] for name in self.matched_features]
        
        # Extract XYZ, matched features
        self.xyz = self.data[:, :3]  # Always include XYZ
        self.features = self.data[:, self.feature_indices]

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Handle points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        if len(self.xyz) % self.points_per_cloud != 0:
            print("Warning: Dataset points not divisible by points_per_cloud. Truncating extra points.")
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Get a subset of points for the current cloud
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz

def predict_classes(model, test_file, features_to_match, points_per_cloud, output_path, batch_size=16):
    # Load the test dataset
    test_dataset = DynamicFeatureDataset(test_file, features_to_match, points_per_cloud)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predicted = torch.argmax(logits, dim=1)
            predictions.append(predicted.cpu().numpy())

    predictions = np.concatenate(predictions)
    dataset_with_predictions = np.hstack((test_dataset.data, predictions.reshape(-1, 1)))

    # Save the augmented dataset to a new file
    np.savetxt(output_path, dataset_with_predictions, delimiter='\t', fmt='%0.8f',
               header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
    print(f"Augmented dataset saved to {output_path}")

# Example Usage
if __name__ == "__main__":
    # Define paths
    test_file = '/path/to/test_file.txt'
    model_path = '/path/to/saved_model.pth'
    output_file = '/path/to/output_file.txt'

    # Define feature matching (must match features used during training)
    features_to_match = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Reflectance', 'NumberOfReturns', 'ReturnNumber']

    # Load the model
    in_dim = len(features_to_match)  # Dynamically determine input dimension
    out_dim = 5  # Set the number of classes based on your training setup
    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=out_dim, downsample_points=(512, 128))
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")

    # Predict classes
    predict_classes(model, test_file, features_to_match, points_per_cloud=1024, output_path=output_file)

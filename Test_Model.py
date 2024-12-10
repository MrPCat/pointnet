import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader
from pointnet_ import PointNet2ClsSSG

class MatchFeaturesDataset(Dataset):
    def __init__(self, train_file_path, test_file_path, points_per_cloud=1024):
        # Read the header from the training file
        with open(train_file_path, 'r') as f:
            header = f.readline().strip()
        train_feature_names = header.split('\t')[:-1]  # Exclude the label column

        # Load the test file
        with open(test_file_path, 'r') as f:
            test_header = f.readline().strip()
        test_feature_names = test_header.split('\t')

        # Dynamically match features
        self.matched_feature_indices = [
            test_feature_names.index(feature) for feature in train_feature_names if feature in test_feature_names
        ]
        print(f"Matched Features: {train_feature_names}")

        # Load the numerical data (skip header row)
        test_data = np.loadtxt(test_file_path, delimiter='\t', skiprows=1)

        # Extract XYZ and matched features from test data
        self.xyz = test_data[:, :3]  # Always include X, Y, Z
        self.features = test_data[:, self.matched_feature_indices]  # Only include matched features

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

# Example usage for prediction
if __name__ == "__main__":
    

    # File paths
    train_file = '/content/drive/MyDrive/t1/training.txt'
    test_file = '/path/to/test_data.txt'
    model_path = '/content/drive/MyDrive/t1/pointnet_model.pth'
    output_file = '/content/drive/MyDrive/t1/predictions.txt'

    # Create the dataset
    test_dataset = MatchFeaturesDataset(train_file, test_file, points_per_cloud=1024)

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2ClsSSG(in_dim=test_dataset.features.shape[1], out_dim=5)  # Adjust `out_dim` as per your classes
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Predict and save
    all_predictions = []
    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # Add predictions to dataset
    augmented_data = np.hstack([test_dataset.xyz, test_dataset.features, np.array(all_predictions).reshape(-1, 1)])
    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%0.8f',
               header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
    print(f"Predictions saved to {output_file}")
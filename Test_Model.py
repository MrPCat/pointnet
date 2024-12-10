import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MatchFeaturesDataset(Dataset):
    def __init__(self, train_file_path, test_file_path, points_per_cloud=1024):
        # Extract the header from the training file
        with open(train_file_path, 'r') as f:
            train_header = f.readline().strip()
        train_feature_names = train_header.split('\t')[:-1]  # Exclude the label column

        # Extract the header from the test file
        with open(test_file_path, 'r') as f:
            test_header = f.readline().strip()
        test_feature_names = test_header.split('\t')

        # Dynamically match features
        self.matched_feature_indices = [
            test_feature_names.index(feature) for feature in train_feature_names if feature in test_feature_names
        ]
        print(f"Matched Features: {train_feature_names}")

        # Load numerical data from the test file
        test_data = []
        with open(test_file_path, 'r') as f:
            for line in f:
                try:
                    # Attempt to parse the line as a tab-separated list of floats
                    test_data.append([float(value) for value in line.strip().split('\t')])
                except ValueError:
                    # Skip the line if it cannot be parsed as numerical data
                    continue

        test_data = np.array(test_data)

        # Extract XYZ and matched features
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


def predict_classes(model, test_loader, device, output_path):
    model.eval()
    predictions = []

    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(predicted_classes)

    # Save predictions to a new file
    np.savetxt(output_path, np.array(predictions), fmt='%d', header='Classification', comments='')
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # File paths
    train_file = '/content/drive/MyDrive/t1/training_logs.txt'
    test_file = '/path/to/test_data.txt'
    model_path = '/content/drive/MyDrive/t1/pointnet_model.pth'
    output_file = '/content/drive/MyDrive/t1/predictions.txt'

    # GPU/CPU device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    test_dataset = MatchFeaturesDataset(train_file, test_file, points_per_cloud=1024)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model
    in_dim = len(test_dataset.matched_feature_indices)  # Dynamically determine input feature dimension
    num_classes = 11 # Adjust to the number of classes in your training
    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes, downsample_points=(512, 128))
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Predict and save classes
    predict_classes(model, test_loader, device, output_file)

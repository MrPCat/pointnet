import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG


class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        # Load the text file
        try:
            # Adjust delimiter='\t' for tab-delimited file
            data = pd.read_csv(file_path, delimiter='\t')
            print("Dataset preview:\n", data.head())
            print("Columns in dataset:\n", list(data.columns))
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}. Error: {e}")

        # Select XYZ (columns 0-2) and features (columns 6-8)
        try:
            self.xyz = data.iloc[:, 0:3].values.astype(np.float64)  # X, Y, Z
            self.features = data.iloc[:, 6:9].values.astype(np.float64)  # Reflectance, NumberOfReturns, ReturnNumber
        except IndexError as e:
            raise ValueError(f"Error in selecting columns. Check file format: {e}")

        # Debug column extraction
        print("XYZ Shape:", self.xyz.shape)
        print("Feature Shape:", self.features.shape)

        # Normalize XYZ
        self.xyz_mean = np.mean(self.xyz, axis=0).astype(np.float64)
        self.xyz -= self.xyz_mean

        # Normalize features
        self.features = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0)

        # Ensure divisibility by points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
        self.features = self.features[:self.num_clouds * self.points_per_cloud]

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

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)
        features = torch.tensor(self.features[start:end], dtype=torch.float32)
        xyz = xyz.transpose(0, 1)
        features = features.transpose(0, 1)
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


def predict_point_cloud(test_file, model_path, output_file):
    # Load the dataset
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)

    # Get input dimensions
    input_dim = test_dataset.features.shape[1] + 3  # Add 3 for XYZ
    print(f"Input dimension for the model: {input_dim}")

    # Load the model
    model = load_model(model_path, input_dim=input_dim, output_dim=11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Predictions
    all_predictions = []
    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # Save predictions
    point_cloud_predictions = np.array(all_predictions).reshape(-1, 1).astype(np.float64)
    denormalized_xyz = (test_dataset.xyz[:len(point_cloud_predictions) * test_dataset.points_per_cloud]
                        + test_dataset.xyz_mean).astype(np.float64)
    feature_columns = ['Reflectance', 'NumberOfReturns', 'ReturnNumber']
    augmented_data = np.hstack([
        denormalized_xyz,
        test_dataset.features[:len(point_cloud_predictions) * test_dataset.points_per_cloud],
        np.repeat(point_cloud_predictions, test_dataset.points_per_cloud, axis=0)
    ])
    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%.15f',
               header='\t'.join(['X', 'Y', 'Z'] + feature_columns + ['Classification']),
               comments='')
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    test_file = '/content/drive/MyDrive/t1/Mar18_test.txt'
    model_path = '/content/drive/MyDrive/t1/checkpoints/pointnet_epoch_7.pth'
    output_file = '/content/drive/MyDrive/t1/Mar18_testWithoutRGB_predictions.txt'

    predict_point_cloud(test_file, model_path, output_file)

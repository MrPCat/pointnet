import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG


class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        try:
            # Inspect column names
            sample_data = pd.read_csv(file_path, delimiter='\t', nrows=5)  # Read first 5 rows
            print("Column names in the dataset:", list(sample_data.columns))
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path} or inspect columns. Error: {e}")

        required_columns = ['X', 'Y', 'Z', 'Reflectance', 'NumberOfReturns', 'ReturnNumber']

        # Map required columns to available ones
        column_mapping = {}
        for col in required_columns:
            for available_col in sample_data.columns:
                if col.lower() in available_col.lower():  # Case-insensitive matching
                    column_mapping[col] = available_col
                    break
            if col not in column_mapping:
                raise ValueError(f"Missing required column: {col}. Available columns: {list(sample_data.columns)}")

        print("Mapped column names:", column_mapping)  # Debug print for column mappings

        # Load full dataset with chunks
        print("Loading dataset in chunks...")
        chunks = pd.read_csv(file_path, delimiter='\t', chunksize=10000)
        data = pd.concat(chunks, ignore_index=True)
        print("Dataset loaded successfully. Shape:", data.shape)

        # Extract XYZ and features
        self.xyz = data[[column_mapping['X'], column_mapping['Y'], column_mapping['Z']]].values.astype(np.float64)
        self.features = data[
            [column_mapping['Reflectance'], column_mapping['NumberOfReturns'], column_mapping['ReturnNumber']]
        ].values.astype(np.float64)

        # Normalize XYZ and features
        self.xyz_mean = np.mean(self.xyz, axis=0).astype(np.float64)
        self.xyz -= self.xyz_mean
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
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)
    input_dim = test_dataset.features.shape[1] + 3
    model = load_model(model_path, input_dim=input_dim, output_dim=11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

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

import numpy as np
import torch
import laspy
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG


class LASPointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        self.feature_attributes = [
            'red', 'green', 'blue',  # R, G, B
            'intensity',  # Reflectance
            'number_of_returns',
            'return_number'
        ]

        # Load LAS file
        try:
            las = laspy.read(file_path)
            print("LAS file loaded successfully")

            # Print included point attributes
            print("\nUsing these attributes as features:")
            print("- X, Y, Z (coordinates)")
            for attr in self.feature_attributes:
                print(f"- {attr}")

        except Exception as e:
            raise ValueError(f"Failed to read LAS file {file_path}. Error: {e}")

        # Extract XYZ and features
        try:
            # Extract XYZ coordinates
            self.xyz = np.vstack((las.x, las.y, las.z)).transpose()

            # Initialize list to store all feature arrays
            feature_arrays = []

            # Extract RGB (normalize to 0-1 range if necessary)
            rgb_scale = 65535.0 if las.header.point_format.id >= 3 else 255.0
            feature_arrays.append(np.array(las.red).reshape(-1, 1) / rgb_scale)
            feature_arrays.append(np.array(las.green).reshape(-1, 1) / rgb_scale)
            feature_arrays.append(np.array(las.blue).reshape(-1, 1) / rgb_scale)

            # Extract other features
            for attr in self.feature_attributes[3:]:
                if hasattr(las, attr):
                    feature_arrays.append(np.array(getattr(las, attr)).reshape(-1, 1))
                else:
                    raise ValueError(f"Missing required LAS attribute: {attr}")

            # Combine all features
            self.features = np.hstack(feature_arrays).astype(np.float64)

        except Exception as e:
            raise ValueError(f"Error during feature extraction: {e}")

        # Debug feature dimensions
        if debug:
            print("\nFeature dimensions:")
            for i, attr in enumerate(self.feature_attributes):
                print(f"{attr.capitalize()}: {feature_arrays[i].shape[1]}")
            print(f"Total features: {self.features.shape[1]}")

        # Normalize XYZ
        self.xyz_mean = np.mean(self.xyz, axis=0).astype(np.float64)
        self.xyz = (self.xyz - self.xyz_mean).astype(np.float64)

        # Normalize features
        feature_means = np.mean(self.features, axis=0)
        feature_stds = np.std(self.features, axis=0)
        feature_stds[feature_stds == 0] = 1  # Prevent division by zero
        self.features = (self.features - feature_means) / feature_stds

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
    except FileNotFoundError:
        print(f"Model checkpoint not found at {model_path}. Initializing model from scratch.")
    return model


def predict_point_cloud(test_file, model_path, output_file):
    # Load the dataset
    test_dataset = LASPointCloudDataset(test_file, points_per_cloud=1024, debug=True)

    # Get input dimensions (features + XYZ, excluding classification)
    input_dim = test_dataset.features.shape[1]
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
    input_las = laspy.read(test_file)
    output_las = laspy.LasData(input_las.header)

    # Copy original attributes
    output_las.x = input_las.x
    output_las.y = input_las.y
    output_las.z = input_las.z
    output_las.red = input_las.red
    output_las.green = input_las.green
    output_las.blue = input_las.blue
    output_las.intensity = input_las.intensity
    output_las.number_of_returns = input_las.number_of_returns
    output_las.return_number = input_las.return_number

    # Adjust predictions to match the number of points
    original_point_count = len(input_las.points)
    point_cloud_predictions = np.array(all_predictions)

    if len(point_cloud_predictions) < original_point_count:
        # Pad with the most common prediction or zeros
        padding = np.full(original_point_count - len(point_cloud_predictions),
                          fill_value=np.bincount(point_cloud_predictions).argmax())
        point_cloud_predictions = np.concatenate((point_cloud_predictions, padding))
    else:
        point_cloud_predictions = point_cloud_predictions[:original_point_count]

    # Assign predictions as classification
    output_las.classification = point_cloud_predictions

    # Save the new LAS file
    output_las.write(output_file)
    print(f"Predictions saved to {output_file}")



if __name__ == "__main__":
    test_file = '/content/drive/MyDrive/Archive /output_with_rgb1.las'
    model_path = '/content/drive/MyDrive/Archive /1. first attempt with RGB and high Accuracy there /pointnet_model.pth'
    output_file = '/content/drive/MyDrive/output_with_rgb1_predictions.las'

    predict_point_cloud(test_file, model_path, output_file)
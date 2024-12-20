import numpy as np
import torch
import laspy
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG


class LASPointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        try:
            # Load the LAS file using laspy
            las = laspy.read(file_path)
            print("LAS file loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to read LAS file {file_path}. Error: {e}")

        # Extract XYZ coordinates
        try:
            self.xyz = np.vstack((las.x, las.y, las.z)).transpose()
            
            # Extract features (assuming these fields exist in your LAS file)
            # Adjust these based on your LAS file's available fields
            self.features = np.vstack((
                las.intensity,  # equivalent to reflectance
                las.number_of_returns,
                las.return_number
            )).transpose()
            
        except AttributeError as e:
            raise ValueError(f"Error accessing LAS attributes. Check file format: {e}")

        # Debug extracted data
        print("XYZ Shape:", self.xyz.shape)
        print("Feature shape:", self.features.shape)

        # Normalize XYZ
        self.xyz_mean = np.mean(self.xyz, axis=0).astype(np.float64)
        self.xyz = (self.xyz - self.xyz_mean).astype(np.float64)

        # Normalize features
        self.features = (self.features - np.mean(self.features, axis=0)) / (np.std(self.features, axis=0) + 1e-6)
        self.features = self.features.astype(np.float64)

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
    # Create a new LAS file with predictions
    input_las = laspy.read(test_file)
    output_las = laspy.LasData(input_las.header)
    
    # Copy all points and their attributes
    for dimension in input_las.point_format.dimension_names:
        setattr(output_las, dimension, getattr(input_las, dimension))
    
    # Add predictions as classification
    point_cloud_predictions = np.repeat(all_predictions, test_dataset.points_per_cloud)
    output_las.classification = point_cloud_predictions[:len(input_las.points)]
    
    # Save the new LAS file
    output_las.write(output_file)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    test_file = '/content/drive/MyDrive/Archive /output_with_rgb1.las'
    model_path = '/content/drive/MyDrive/Archive /1. first attempt with RGB and high Accuracy there /pointnet_model.pth'
    output_file = '/content/drive/MyDrive/t1/output_with_rgb1_predictions.las'

    predict_point_cloud(test_file, model_path, output_file)
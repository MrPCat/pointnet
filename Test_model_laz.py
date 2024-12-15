import laspy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG  # Adjust this import to your model's structure

def process_nrw_laz_with_priority_classes(file_path):
    las = laspy.read(file_path)

    # Define class mapping for your top 5 classes
    class_mapping = {
        2: 0,   # Ground points (Bodenpunkte)
        20: 1,  # Last return non-ground
    }

    # Map classifications to your prioritized indices
    mapped_classes = np.array([class_mapping.get(cls, -1) for cls in las.classification])
    
    # One-hot encode the 5 prioritized classes
    num_classes = len(class_mapping)  # 5 important classes
    one_hot_classes = np.zeros((len(mapped_classes), num_classes))
    for i in range(num_classes):
        one_hot_classes[:, i] = (mapped_classes == i).astype(float)
    
    # Filter out invalid classes (-1) if desired
    valid_mask = mapped_classes != -1
    filtered_xyz = np.column_stack([las.x[valid_mask], las.y[valid_mask], las.z[valid_mask]])
    filtered_intensity = las.intensity[valid_mask]
    filtered_num_returns = las.num_returns[valid_mask]
    filtered_return_num = las.return_num[valid_mask]
    one_hot_classes = one_hot_classes[valid_mask]

    # Add synthetic feature: normalized height
    normalized_height = (filtered_xyz[:, 2] - np.min(filtered_xyz[:, 2])) / (
        np.max(filtered_xyz[:, 2]) - np.min(filtered_xyz[:, 2])
    )

    # Combine features into a 9-dimensional feature set
    data = np.column_stack([
        filtered_xyz,                               # XYZ coordinates (3 features)
        filtered_intensity / np.max(filtered_intensity) * 255,  # Intensity (1 feature)
        filtered_num_returns / np.max(filtered_num_returns),    # Number of returns (1 feature)
        filtered_return_num / np.max(filtered_return_num),      # Return number (1 feature)
        one_hot_classes,                          # One-hot-encoded classes (5 features)
        normalized_height                         # Normalized height (1 feature)
    ])

    return data

class PointCloudDatasetNRW(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        self.data = process_nrw_laz_with_priority_classes(file_path)
        self.xyz = self.data[:, :3].astype(np.float64)  # XYZ coordinates
        self.features = self.data[:, 3:].astype(np.float32)  # Other features
        self.xyz_mean = np.mean(self.xyz, axis=0, dtype=np.float64)
        self.xyz -= self.xyz_mean
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


# Load Model
def load_model(model_path, input_dim, output_dim):
    model = PointNet2ClsSSG(in_dim=input_dim, out_dim=output_dim)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Initializing model from scratch.")
    return model

# Main Function
if __name__ == "__main__":
    # File paths
    test_file = r'/content/drive/MyDrive/t1/3dm_32_280_5652_1_nw.laz'  # Adjust path
    model_path = r'/content/drive/MyDrive/t1/checkpoints/pointnet_model.pth'  # Adjust path
    output_file = r'/content/drive/MyDrive/t1/3dm_32_280_5652_1_nw_predictions.txt'  # Adjust path

    # Load the test dataset
    test_dataset = PointCloudDatasetNRW(test_file, points_per_cloud=1024, debug=True)

    # Load the model
    input_dim = test_dataset.features.shape[1]
    output_dim = 11  # Adjust based on your number of classes
    model = load_model(model_path, input_dim=input_dim, output_dim=output_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # DataLoader
    print("CUDA Available:", torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Predictions
    all_predictions = []
    with torch.no_grad():
        for features, xyz in test_loader:
            # Move tensors to device
            features, xyz = features.to(device, dtype=torch.float32), xyz.to(device, dtype=torch.float32)
            
            # Pass through the model
            logits = model(features, xyz)  # Output logits
            predictions = torch.argmax(logits, dim=1)  # Class predictions
            all_predictions.extend(predictions.cpu().numpy())

    # Save predictions
    point_cloud_predictions = np.array(all_predictions).reshape(-1, 1).dtype(np.float64)
    denormalized_xyz = (test_dataset.xyz[:len(point_cloud_predictions) * test_dataset.points_per_cloud]
                        + test_dataset.xyz_mean).dtype(np.float64)

    augmented_data = np.hstack([denormalized_xyz,
                                test_dataset.features[:len(point_cloud_predictions) * test_dataset.points_per_cloud],
                                np.repeat(point_cloud_predictions, test_dataset.points_per_cloud, axis=0)])

    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%.15f',
               header='X\tY\tZ\tIntensity\tNumReturns\tReturnNum\tScanAngle\tClassification\tPrediction', comments='')
    print(f"Predictions saved to {output_file}")

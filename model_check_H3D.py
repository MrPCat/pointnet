import numpy as np
import torch
import laspy
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG

class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        print(f"Reading LAS data from {file_path}")

        # Open LAS file
        try:
            las = laspy.read(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read LAS file: {file_path}. Error: {e}")

        # Debug: Print available dimensions
        print("Available dimensions:", las.point_format.dimension_names)

        # Explicitly extract coordinates
        try:
            x = las.x
            y = las.y
            z = las.z
            self.xyz = np.vstack((x, y, z)).T.astype(np.float64)
        except Exception as e:
            raise ValueError(f"Error extracting XYZ coordinates: {e}")

        if len(self.xyz) == 0:
            raise ValueError(f"No points found in the LAS file: {file_path}")

        # Feature extraction with extensive error handling
        def safe_extract_feature(las, attr_name, default_val=0, normalize=False):
            try:
                if not hasattr(las, attr_name):
                    print(f"Warning: {attr_name} not found. Using default.")
                    return np.full(len(self.xyz), default_val, dtype=np.float64)
                
                feature = np.array(getattr(las, attr_name), dtype=np.float64)
                
                if normalize and feature.max() > 0:
                    return feature / feature.max()
                return feature
            except Exception as e:
                print(f"Error extracting {attr_name}: {e}")
                return np.full(len(self.xyz), default_val, dtype=np.float64)

        # Extract features
        red = safe_extract_feature(las, 'red', normalize=True)
        green = safe_extract_feature(las, 'green', normalize=True)
        blue = safe_extract_feature(las, 'blue', normalize=True)
        intensity = safe_extract_feature(las, 'intensity', normalize=True)
        num_returns = safe_extract_feature(las, 'num_returns')
        return_number = safe_extract_feature(las, 'return_number')

        # Stack features
        self.features = np.vstack((red, green, blue, intensity, num_returns, return_number)).T

        # Extract classification
        try:
            self.labels = np.array(las.classification, dtype=np.int64)
            self.has_labels = True
        except Exception as e:
            print(f"Warning: Could not extract classification. Error: {e}")
            self.labels = np.zeros(len(self.xyz), dtype=np.int64)
            self.has_labels = False

        # Store original LAS for reference
        self.original_las = las

        # Normalize features
        try:
            feature_mean = np.mean(self.features, axis=0)
            feature_std = np.std(self.features, axis=0)
            feature_std[feature_std == 0] = 1.0
            self.features = (self.features - feature_mean) / feature_std
        except Exception as e:
            print(f"Feature normalization error: {e}")

        # Prepare for batching
        self.points_per_cloud = points_per_cloud
        self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)

        # Truncate to be divisible by points_per_cloud
        max_points = self.num_clouds * self.points_per_cloud
        self.xyz = self.xyz[:max_points]
        self.features = self.features[:max_points]
        self.labels = self.labels[:max_points]

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud

        # Debugging prints
        print(f"Index {idx}: start = {start}, end = {end}")
        print(f"Shape of XYZ data: {self.xyz.shape}")
        print(f"Shape of Features data: {self.features.shape}")
        
        # Check that indices do not exceed bounds
        if end > len(self.xyz) or end > len(self.features):
            print(f"Error: Index out of bounds. start: {start}, end: {end}, total points: {len(self.xyz)}")
            return None  # Return None to avoid further error

        # Convert to tensors
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)
        features = torch.tensor(self.features[start:end], dtype=torch.float32)

        # Debugging prints for tensor shapes
        print(f"Shape of xyz tensor: {xyz.shape}")
        print(f"Shape of features tensor: {features.shape}")

        # Transpose for PointNet format
        xyz = xyz.transpose(0, 1)
        features = features.transpose(0, 1)

        return features, xyz

def predict_and_export(input_file, output_file, model_path, num_classes=11):
    """
    Predict classes for an entire LAS file and export with predictions
    """
    try:
        # Create dataset
        test_dataset = PointCloudDataset(input_file, points_per_cloud=1024)

        # Prepare model
        input_dim = test_dataset.features.shape[1]

        # Load model with error handling
        try:
            model = PointNet2ClsSSG(in_dim=input_dim, out_dim=num_classes)
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Model loading error: {e}")
            return

        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model.to(device)
        model.eval()

        # Create DataLoader
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Predict
        all_predictions = []
        with torch.no_grad():
            for features, xyz in test_loader:
                if features is None or xyz is None:
                    continue  # Skip this batch if there was an indexing issue

                features, xyz = features.to(device), xyz.to(device)

                # Debugging prints before passing data to the model
                print(f"Input Tensor Shape (features): {features.shape}")
                print(f"Index Tensor Shape (xyz): {xyz.shape}")

                logits = model(features, xyz)
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())

        # Create new LAS file and export predictions
        full_predictions = np.zeros(len(test_dataset.original_las.x), dtype=np.int64)
        full_predictions[:len(all_predictions)] = all_predictions

        # Create new LAS file
        new_las = laspy.create(point_format=test_dataset.original_las.header.point_format,
                                file_version=test_dataset.original_las.header.version)
        
        # Copy all existing fields
        for dimension in test_dataset.original_las.point_format.dimension_names:
            if dimension in test_dataset.original_las.point_names:
                new_las[dimension] = test_dataset.original_las[dimension]

        # Add new dimension for predictions
        new_las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="predicted_classification", 
                type=np.int64, 
                description="Model predictions"
            )
        )
        new_las.predicted_classification = full_predictions

        # Save the new LAS file
        new_las.write(output_file)

        print(f"Predictions exported to {output_file}")

        # Print prediction distribution
        unique_preds, counts = np.unique(full_predictions, return_counts=True)
        print("\nPrediction Distribution:")
        for cls, count in zip(unique_preds, counts):
            percentage = (count / len(full_predictions)) * 100
            print(f"Class {cls}: {count} points ({percentage:.2f}%)")

    except Exception as e:
        import traceback
        print(f"Unexpected error during prediction and export:")
        print(traceback.format_exc())


if __name__ == "__main__":
    # Paths
    input_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Epoch_March2018\LiDAR\Mar18_test.laz"
    output_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Epoch_March2018\LiDAR\Mar18_perdict.laz"
    model_path = r"C:\Users\faars\Downloads\modelnet40ply2048-train-pointnet++best.pth"
    
    # Predict and export
    predict_and_export(input_file, output_file, model_path, num_classes=11)

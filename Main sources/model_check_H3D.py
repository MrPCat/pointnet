import numpy as np
import torch
import laspy
from torch.utils.data import Dataset, DataLoader
from pointnet import PointNet2ClsSSG
from sklearn.metrics import confusion_matrix, accuracy_score


class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        print(f"Reading LAS data from {file_path}")
        
        # Open LAS file
        try:
            las = laspy.read(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read LAS file: {file_path}. Error: {e}")

        # Extract XYZ coordinates
        self.xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        
        if len(self.xyz) == 0:
            raise ValueError(f"No points found in the LAS file: {file_path}")

        # Extract additional features with proper error handling
        # Normalize RGB values (0-1)
        red = np.array(las.red, dtype=np.float64) / 65535.0 if hasattr(las, 'red') else np.zeros(len(self.xyz))
        green = np.array(las.green, dtype=np.float64) / 65535.0 if hasattr(las, 'green') else np.zeros(len(self.xyz))
        blue = np.array(las.blue, dtype=np.float64) / 65535.0 if hasattr(las, 'blue') else np.zeros(len(self.xyz))
        
        # Normalize intensity safely
        intensity_max = np.max(las.intensity) if hasattr(las, 'intensity') and np.max(las.intensity) > 0 else 1.0
        intensity = np.array(las.intensity, dtype=np.float64) / intensity_max if hasattr(las, 'intensity') else np.zeros(len(self.xyz))
        
        # Get return information
        num_returns = np.array(las.num_returns, dtype=np.float64) if hasattr(las, 'num_returns') else np.ones(len(self.xyz))
        return_number = np.array(las.return_number, dtype=np.float64) if hasattr(las, 'return_number') else np.ones(len(self.xyz))

        # Stack all features
        self.features = np.vstack((red, green, blue, intensity, num_returns, return_number)).T

        # Extract classification labels - convert SubFieldView to numpy array
        if hasattr(las, 'classification'):
            self.labels = np.array(las.classification, dtype=np.int64)
            self.has_labels = True
        else:
            print("Warning: No classification field found in LAS file. Using dummy labels.")
            self.labels = np.zeros(len(self.xyz), dtype=np.int64)
            self.has_labels = False
        
        # Normalize XYZ
        self.xyz_mean = np.mean(self.xyz, axis=0)
        self.xyz -= self.xyz_mean
        
        # Normalize Features safely
        feature_mean = np.mean(self.features, axis=0)
        feature_std = np.std(self.features, axis=0)
        feature_std[feature_std == 0] = 1.0  # Avoid division by zero
        self.features = (self.features - feature_mean) / feature_std

        # Ensure divisibility by points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)
        
        # Truncate data to be divisible by points_per_cloud
        max_points = self.num_clouds * self.points_per_cloud
        self.xyz = self.xyz[:max_points]
        self.features = self.features[:max_points]
        self.labels = self.labels[:max_points]

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

        unique_labels = np.unique(self.labels)
        print(f"Labels found: {unique_labels}")
        for label in unique_labels:
            count = np.sum(self.labels == label)
            percentage = (count / len(self.labels)) * 100
            print(f"  Class {label}: {count} points ({percentage:.2f}%)")

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud

        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)
        features = torch.tensor(self.features[start:end], dtype=torch.float32)

        # Transpose for PointNet format
        xyz = xyz.transpose(0, 1)
        features = features.transpose(0, 1)

        # Determine the dominant label for this point cloud
        labels = self.labels[start:end]
        majority_label = int(np.bincount(labels.astype(int)).argmax())
        label = torch.tensor(majority_label, dtype=torch.long)

        return features, xyz, label


def load_model(model_path, input_dim, output_dim):
    model = PointNet2ClsSSG(in_dim=input_dim, out_dim=output_dim)
    try:
        if not model_path:
            print("No model path provided. Initializing model from scratch.")
            return model
            
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model file not found: {model_path}. Initializing model from scratch.")
    except Exception as e:
        print(f"Error loading model: {e}. Initializing model from scratch.")
    return model


def evaluate_model(test_file, model_path, num_classes=11):
    """
    Evaluate the model using a `.las` dataset.
    """
    try:
        test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)
        
        if not test_dataset.has_labels:
            print("No labels found. Cannot evaluate accuracy.")
            return

        input_dim = test_dataset.features.shape[1]
        model = load_model(model_path, input_dim=input_dim, output_dim=num_classes)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model.to(device)
        model.eval()

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, xyz, labels in test_loader:
                features, xyz, labels = features.to(device), xyz.to(device), labels.to(device)
                logits = model(features, xyz)
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute accuracy and confusion matrix
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracy):
            if np.isnan(acc):
                continue
            print(f"Class {i} accuracy: {acc:.4f}")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")


def predict(test_file, model_path, num_classes=11):
    """
    Run predictions on `.las` data without evaluation.
    """
    try:
        test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)
        
        input_dim = test_dataset.features.shape[1]
        model = load_model(model_path, input_dim=input_dim, output_dim=num_classes)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model.to(device)
        model.eval()

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        all_predictions = []

        with torch.no_grad():
            for features, xyz, _ in test_loader:
                features, xyz = features.to(device), xyz.to(device)
                logits = model(features, xyz)
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())

        unique_preds, counts = np.unique(all_predictions, return_counts=True)
        print("\nPrediction Distribution:")
        for cls, count in zip(unique_preds, counts):
            percentage = (count / len(all_predictions)) * 100
            print(f"Class {cls}: {count} points ({percentage:.2f}%)")
            
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    # Paths
    test_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Epoch_March2018\LiDAR\Mar18_test_GroundTruth.las"
    model_path = r"C:\Users\faars\Downloads\modelnet40ply2048-train-pointnet++best.pth"

    # Evaluate model
    evaluate_model(test_file, model_path, num_classes=11)
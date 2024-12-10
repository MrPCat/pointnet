import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG
import pandas as pd

class MatchFeaturesDataset(Dataset):
    def __init__(self, train_file_path, test_file_path, points_per_cloud=1024, debug=True):
        # Read the headers and identify features
        train_cols = pd.read_csv(train_file_path, delimiter='\t', nrows=0).columns.tolist()
        test_cols = pd.read_csv(test_file_path, delimiter='\t', nrows=0).columns.tolist()

        # Exclude classification column if present in training data
        if 'Classification' in train_cols:
            train_cols = train_cols[:-1]

        # Match features
        matched_features = [col for col in train_cols if col in test_cols]
        unmatched_train = [col for col in train_cols if col not in test_cols]
        unmatched_test = [col for col in test_cols if col not in train_cols]

        if debug:
            print("\n--- Feature Matching ---")
            print(f"Matched Features: {matched_features}")
            print(f"Unmatched Train Features: {unmatched_train}")
            print(f"Unmatched Test Features: {unmatched_test}")

        # Load test data and extract matched features
        test_data = pd.read_csv(test_file_path, delimiter='\t').values
        self.xyz = test_data[:, :3]
        self.features = test_data[:, [test_cols.index(f) for f in matched_features]]
        self.matched_features = matched_features  # Store matched feature names for later use

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Handle points per cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        
        if len(self.xyz) % self.points_per_cloud != 0:
            print(f"Warning: {len(self.xyz)} points not divisible by {self.points_per_cloud}")
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]

        if debug:
            self.print_debug_info()

    def print_debug_info(self):
        print("\n--- Dataset Debugging Information ---")
        print(f"Total Points: {len(self.xyz)}")
        print(f"Points per Cloud: {self.points_per_cloud}")
        print(f"Number of Point Clouds: {self.num_clouds}")
        print("\nFeature Matching:")
        print(f"Matched Features: {self.matched_features}")
        print(f"XYZ Shape: {self.xyz.shape}")
        print(f"Features Shape: {self.features.shape}")

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz

def load_model_safely(model_path, input_dim, out_dim, matched_features):
    model = PointNet2ClsSSG(in_dim=input_dim, out_dim=out_dim)
    try:
        # Load checkpoint
        state_dict = torch.load(model_path, map_location='cpu')

        print("\n--- Checkpoint Debug Information ---")
        for name, param in state_dict.items():
            if 'sa1.conv_blocks.0.0.0.weight' in name:
                checkpoint_dim = param.shape[1]
                print(f"Checkpoint Feature Dimension: {checkpoint_dim}")
                print(f"Checkpoint Features: {matched_features[:checkpoint_dim]}")
        for name, param in model.state_dict().items():
            if 'sa1.conv_blocks.0.0.0.weight' in name:
                model_dim = param.shape[1]
                print(f"Current Model Feature Dimension: {model_dim}")
                print(f"Current Model Features: {matched_features[:model_dim]}")

        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully with adjusted dimensions")
    except Exception as e:
        print(f"Error loading model: {e}. Initializing model from scratch.")
    return model

if __name__ == "__main__":
    # File paths
    train_file = '/content/drive/MyDrive/t1/Mar18_train.txt'
    test_file = '/content/drive/MyDrive/t1/Mar18_test.txt'
    model_path = '/content/drive/MyDrive/t1/pointnet_model.pth'
    output_file = '/content/drive/MyDrive/t1/predictions.txt'

    # Load dataset
    test_dataset = MatchFeaturesDataset(train_file, test_file, points_per_cloud=1024, debug=True)

    # Load model
    input_dim = test_dataset.features.shape[1]
    model = load_model_safely(model_path, input_dim, out_dim=11, matched_features=test_dataset.matched_features)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Prediction
    all_predictions = []
    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to('cuda'), xyz.to('cuda')
            logits = model(features, xyz)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # Save predictions
    augmented_data = np.hstack([test_dataset.xyz, test_dataset.features, np.array(all_predictions).reshape(-1, 1)])
    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%0.8f',
               header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
    print(f"Predictions saved to {output_file}")

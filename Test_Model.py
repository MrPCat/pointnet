import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader
from pointnet_ import PointNet2ClsSSG
import pandas as pd

class MatchFeaturesDataset(Dataset):
    def __init__(self, train_file_path, test_file_path, points_per_cloud=1024, debug=True):
        # Read the headers
        train_df = pd.read_csv(train_file_path, delimiter='\t', nrows=1)
        test_df = pd.read_csv(test_file_path, delimiter='\t', nrows=1)
        
        # Comprehensive feature tracking
        self.debug_info = {
            'train_features': list(train_df.columns),
            'test_features': list(test_df.columns),
            'matched_features': [],
            'unmatched_train_features': [],
            'unmatched_test_features': []
        }

        # Read full datasets for comprehensive analysis
        train_full_df = pd.read_csv(train_file_path, delimiter='\t')
        test_full_df = pd.read_csv(test_file_path, delimiter='\t')

        # Identify matched and unmatched features
        train_cols = list(train_full_df.columns)
        test_cols = list(test_full_df.columns)
        
        # Exclude the last column in training data if it's a classification column
        if 'Classification' in train_cols:
            train_cols = train_cols[:-1]

        # Find matched features
        matched_features = [col for col in train_cols if col in test_cols]
        
        # Track unmatched features
        unmatched_train = [col for col in train_cols if col not in test_cols]
        unmatched_test = [col for col in test_cols if col not in train_cols]

        # Update debug info
        self.debug_info['matched_features'] = matched_features
        self.debug_info['unmatched_train_features'] = unmatched_train
        self.debug_info['unmatched_test_features'] = unmatched_test

        # Matched feature indices in test data
        self.matched_feature_indices = [
            list(test_full_df.columns).index(feature) for feature in matched_features
        ]

        # Load numerical data
        test_data = test_full_df.values

        # Extract XYZ and matched features
        self.xyz = test_data[:, :3]
        self.features = test_data[:, self.matched_feature_indices]

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Handle points per cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        
        if len(self.xyz) % self.points_per_cloud != 0:
            print(f"Warning: {len(self.xyz)} points not divisible by {self.points_per_cloud}")
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]

        # Debugging output
        if debug:
            self.print_debug_info()

    def print_debug_info(self):
        print("\n--- Dataset Debugging Information ---")
        print(f"Total Points: {len(self.xyz)}")
        print(f"Points per Cloud: {self.points_per_cloud}")
        print(f"Number of Point Clouds: {self.num_clouds}")
        print("\nFeature Matching:")
        print(f"Matched Features: {self.debug_info['matched_features']}")
        print(f"Unmatched Train Features: {self.debug_info['unmatched_train_features']}")
        print(f"Unmatched Test Features: {self.debug_info['unmatched_test_features']}")
        print("\nFeature Matrices:")
        print(f"XYZ Shape: {self.xyz.shape}")
        print(f"Features Shape: {self.features.shape}")
        print("Matched Feature Indices:", self.matched_feature_indices)

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz

# Example usage for prediction
if __name__ == "__main__":
    # File paths
    train_file = '/content/drive/MyDrive/t1/Mar18_train.txt'
    test_file = '/content/drive/MyDrive/t1/Mar18_test.txt'
    model_path = '/content/drive/MyDrive/t1/pointnet_model.pth'
    output_file = '/content/drive/MyDrive/t1/predictions.txt'

    # Create dataset with debugging enabled
    debug_dataset = MatchFeaturesDataset(train_file, test_file, debug=True)

    # Create the dataset for prediction
    test_dataset = MatchFeaturesDataset(train_file, test_file, points_per_cloud=1024, debug=False)

    # Print input dimension for model
    print(f"Input Feature Dimension: {test_dataset.features.shape[1]}")

    # Flexible model loading with error handling
    def load_model_safely(model_path, input_dim, out_dim):
        try:
            # Load model with weights only
            state_dict = torch.load(model_path, weights_only=True)
            
            # Dynamically adjust model based on input dimensions
            model = PointNet2ClsSSG(in_dim=input_dim, out_dim=out_dim)
            
            # Remove output layer weights if they don't match
            output_keys = [k for k in state_dict.keys() if 'head.8' in k]
            for k in output_keys:
                del state_dict[k]
            
            # Partially load the state dict
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully with adjusted dimensions")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing model from scratch")
            return PointNet2ClsSSG(in_dim=input_dim, out_dim=out_dim)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_safely(model_path, 
                               input_dim=test_dataset.features.shape[1], 
                               out_dim=11)  # Ensure this matches your original training classes
    model.to(device)
    model.eval()

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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
    
    # Save predictions with comprehensive header
    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%0.8f',
               header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
    print(f"Predictions saved to {output_file}")
    print(f"Total predictions: {len(all_predictions)}")
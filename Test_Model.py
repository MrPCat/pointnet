import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG  
import pandas as pd

class PointCloudDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024, debug=True):
        # Load the dataset
        data = pd.read_csv(file_path, delimiter='\t').values
        
        # Extract XYZ and Features (excluding XYZ columns explicitly)
        self.xyz = data[:, :3]  # Assuming the first three columns are always XYZ
        self.features = data[:, 3:]  # Starting from the fourth column onwards
        
        # Normalize XYZ
        self.xyz -= np.mean(self.xyz, axis=0)
        
        # Ensure data is divisible by points_per_cloud
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud
        
        # Truncate or pad to ensure exact division
        if len(self.xyz) % self.points_per_cloud != 0:
            # Option 1: Truncate
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]
            
            # Alternatively, Option 2: Pad with zeros or repeat last points
            # self.xyz = np.pad(self.xyz, ((0, self.points_per_cloud - (len(self.xyz) % self.points_per_cloud)), (0, 0)), mode='constant')
            # self.features = np.pad(self.features, ((0, self.points_per_cloud - (len(self.features) % self.points_per_cloud)), (0, 0)), mode='constant')
        
        if debug:
            self.print_debug_info()

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        
        # Ensure correct tensor shapes
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32)  # [points_per_cloud, 3]
        features = torch.tensor(self.features[start:end], dtype=torch.float32)  # [points_per_cloud, feature_dim]
        
        # Transpose to match PointNet2 expected input
        xyz = xyz.transpose(0, 1)  # [3, points_per_cloud]
        features = features.transpose(0, 1)  # [feature_dim, points_per_cloud]
        
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


if __name__ == "__main__":
    # File paths
    test_file = '/content/drive/MyDrive/t1/Mar18_test.txt'
    model_path = '/content/drive/MyDrive/t1/checkpoints/pointnet_model.pth'
    output_file = '/content/drive/MyDrive/t1/predictions.txt'

    # Load the test dataset
    test_dataset = PointCloudDataset(test_file, points_per_cloud=1024, debug=True)

    # Load the model
    input_dim = test_dataset.features.shape[1]
    model = load_model(model_path, input_dim=input_dim, output_dim=11)  # Adjust `output_dim` based on your classes
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # DataLoader
    print("CUDA Available:", torch.cuda.is_available())
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for features, xyz in test_loader:
            # Debug prints
            print("Features shape:", features.shape)
            print("XYZ shape:", xyz.shape)
            print("Features dtype:", features.dtype)
            print("XYZ dtype:", xyz.dtype)
            
            features, xyz = features.to('cuda'), xyz.to('cuda')
            
            try:
                logits = model(features, xyz)
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
            except RuntimeError as e:
                print(f"Error during inference: {e}")
                # Add more detailed error handling or debugging here
                raise

    # Save predictions
    augmented_data = np.hstack([test_dataset.xyz, test_dataset.features, np.array(all_predictions).reshape(-1, 1)])
    np.savetxt(output_file, augmented_data, delimiter='\t', fmt='%0.8f',
               header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
    print(f"Predictions saved to {output_file}")

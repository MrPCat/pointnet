import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pointnet_ import PointNet2ClsSSG

class TestDataset(Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        self.data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
        self.xyz = self.data[:, :3]  # XYZ coordinates
        self.features = self.data[:, 3:]  # Remaining columns are features

        # Normalize spatial coordinates
        self.xyz -= np.mean(self.xyz, axis=0)

        # Group points into point clouds
        self.points_per_cloud = points_per_cloud
        self.num_clouds = len(self.xyz) // self.points_per_cloud

        if len(self.xyz) % self.points_per_cloud != 0:
            print("Warning: Dataset points not divisible by points_per_cloud. Truncating extra points.")
            self.xyz = self.xyz[:self.num_clouds * self.points_per_cloud]
            self.features = self.features[:self.num_clouds * self.points_per_cloud]

        self.in_dim = self.features.shape[1] + 3  # Dynamically set input dimensions

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        start = idx * self.points_per_cloud
        end = start + self.points_per_cloud
        xyz = torch.tensor(self.xyz[start:end], dtype=torch.float32).T  # Shape: [3, points_per_cloud]
        features = torch.tensor(self.features[start:end], dtype=torch.float32).T  # Shape: [F, points_per_cloud]
        return features, xyz

def predict_and_save(model, test_loader, dataset, device, output_path):
    model.eval()  # Set model to evaluation mode
    predictions = []

    with torch.no_grad():
        for features, xyz in test_loader:
            features, xyz = features.to(device), xyz.to(device)
            logits = model(features, xyz)
            predicted_classes = torch.argmax(logits, dim=1)
            predictions.extend(predicted_classes.cpu().numpy())

    # Create a new dataset with predictions as the last column
    extended_dataset = []
    for i in range(len(predictions)):
        start = i * test_loader.dataset.points_per_cloud
        end = start + test_loader.dataset.points_per_cloud
        data_chunk = dataset[start:end]
        predicted_class = predictions[i]
        predicted_column = np.full((data_chunk.shape[0], 1), predicted_class)
        extended_chunk = np.hstack((data_chunk, predicted_column))
        extended_dataset.append(extended_chunk)

    extended_dataset = np.vstack(extended_dataset)

    # Save the augmented dataset to a new file
    np.savetxt(output_path, extended_dataset, delimiter='\t', fmt='%0.8f',
               header='X\tY\tZ\tR\tG\tB\tReflectance\tNumberOfReturns\tReturnNumber\tClassification', comments='')
    print(f"New dataset with predictions saved to {output_path}")

if __name__ == "__main__":
    # Check GPU Availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Specify File Paths
    test_file = '/content/drive/MyDrive/t1/Mar18_test.txt'
    output_path = '/content/drive/MyDrive/t1/Mar18_test_with_predictions.txt'

    # Test Dataset and DataLoader
    test_dataset = TestDataset(test_file, points_per_cloud=1024)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the Trained Model
    in_dim = test_dataset.in_dim
    num_classes = 3  # Update this based on the number of classes in your training data
    model = PointNet2ClsSSG(in_dim=in_dim, out_dim=num_classes, downsample_points=(512, 128))
    model_path = "/content/drive/MyDrive/t1/pointnet_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Predict and Save New File
    predict_and_save(model, test_loader, test_dataset.data, device, output_path)

import numpy as np
import laspy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet import PointNet2ClsSSG

# Label mapping for H3D dataset
H3D_LABELS = {
    0: 'C00 Low Vegetation',
    1: 'C01 Impervious Surface', 
    2: 'C02 Vehicle',
    3: 'C03 Urban Furniture',
    4: 'C04 Roof',
    5: 'C05 Facade',
    6: 'C06 Shrub', 
    7: 'C07 Tree',
    8: 'C08 Soil/Gravel',
    9: 'C09 Vertical Surface',
    10: 'C10 Chimney'
}

class PointCloudPredictor:
    def __init__(self, model_path, num_points=2048, num_classes=11):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, num_classes)
        self.model.to(self.device)
        self.num_points = num_points
    
    def load_model(self, model_path, num_classes):
        model = self.create_pointnet_model(num_classes)
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            new_state_dict = {k.replace("encoder.", "model."): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_pointnet_model(self, num_classes):
        class PointNetPlusPlus(nn.Module):
            def __init__(self, in_dim=3, out_dim=11, downsample_points=(512, 128), radii=(0.2, 0.4), ks=(32, 64), head_norm=True, dropout=0.5):
                super(PointNetPlusPlus, self).__init__()
                self.model = PointNet2ClsSSG(
                    in_dim=in_dim, out_dim=out_dim, downsample_points=downsample_points,
                    radii=radii, ks=ks, head_norm=head_norm, dropout=dropout
                )
            
            def forward(self, x):
                return self.model(x, x[:, :3, :])
        
        return PointNetPlusPlus(out_dim=num_classes)
    
    def preprocess_las_file(self, file_path):
        try:
            # Load only X, Y, Z (columns 0, 1, 2)
            points = np.loadtxt(file_path, delimiter='\t', skiprows=1, usecols=(0, 1, 2))
        except Exception as e:
            print(f"Error reading file: {e}")
            raise
        
        if points.shape[0] == 0:
            raise ValueError("No points found in the file")

        # Normalize spatial coordinates
        points = self.normalize_points(points)

        # Sample points
        sampled_points = self.sample_points(points)

        # Convert to tensor (3 features per point)
        points_tensor = torch.from_numpy(sampled_points).float().transpose(0, 1).unsqueeze(0).to(self.device)

        return points_tensor

    
    def normalize_points(self, points):
        centered_points = points - points.mean(axis=0)
        scale = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
        return centered_points / scale
    
    def sample_points(self, points):
        indices = np.random.choice(len(points), self.num_points, replace=(len(points) < self.num_points))
        return points[indices]
    
    def predict(self, las_file_path):
        input_tensor = self.preprocess_las_file(las_file_path)  # Shape: (1, 3, num_points)

        with torch.no_grad():
            outputs = self.model(input_tensor)  # Shape: (1, num_classes, num_points)
            predicted = torch.argmax(outputs, dim=1)  # Get the class index for each point

        # Debugging print statements
        print("Outputs shape:", outputs.shape)
        print("Predicted shape:", predicted.shape)
        print("Predicted tensor:", predicted)

        # Robust handling of different tensor shapes
        if predicted.ndim == 0:
            predicted_labels = [H3D_LABELS[predicted.item()]]
        elif predicted.ndim == 1:
            predicted_labels = [H3D_LABELS[pred.item()] for pred in predicted]
        else:
            predicted_labels = [H3D_LABELS[pred.item()] for pred in predicted.flatten()]

        return predicted_labels  # Return per-point classification


def main():
    model_path = r"C:\Users\faars\Downloads\s3dis-train-pointnet++_ckpt_best.pth"
    las_file_path = r"C:\Farshid\Uni\Semesters\Thesis\Data\Epoch_March2018\LiDAR\Mar18_test.txt"
    predictor = PointCloudPredictor(model_path)
    predictions = predictor.predict(las_file_path)
    print("Prediction Results:")
    for label in predictions:
        print(label)

if __name__ == '__main__':
    main()
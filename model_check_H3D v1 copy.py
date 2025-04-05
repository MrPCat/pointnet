import numpy as np
import torch
import torch.nn as nn
import laspy
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

class PointNetPlusPlus(nn.Module):
    def __init__(self, in_dim=3, out_dim=11, downsample_points=(512, 128), radii=(0.2, 0.4), ks=(32, 64), head_norm=True, dropout=0.5):
        super(PointNetPlusPlus, self).__init__()
        # Use the existing PointNet2ClsSSG as a base
        self.segmentation_model = PointNet2ClsSSG(
            in_dim=in_dim, out_dim=out_dim, downsample_points=downsample_points,
            radii=radii, ks=ks, head_norm=head_norm, dropout=dropout
        )
        
        # Add a segmentation head to support per-point predictions
        self.segmentation_head = nn.Sequential(
            nn.Conv1d(out_dim, out_dim//2, kernel_size=1),
            nn.BatchNorm1d(out_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim//2, out_dim, kernel_size=1)
        )
    
    def forward(self, x):
        # Ensure we pass both the input and xyz coordinates
        global_features = self.segmentation_model(x, x[:, :3, :])
        
        # Convert global features to per-point predictions
        point_wise_features = self.segmentation_head(global_features.unsqueeze(-1))
        return point_wise_features.squeeze(-1)

class PointCloudPredictor:
    def __init__(self, model_path, num_points=2048, num_classes=11):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, num_classes)
        self.model.to(self.device)
        self.num_points = num_points
    
    def load_model(self, model_path, num_classes):
        model = PointNetPlusPlus(out_dim=num_classes)
        try:
            # Try loading the state dict with flexible mapping
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Try different key mapping strategies
            if 'encoder.model.sa1.mlp_module.0.weight' in state_dict:
                # Rename keys if necessary
                new_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_las_file(self, file_path):
        try:
            # Load X, Y, Z coordinates
            points = np.loadtxt(file_path, delimiter='\t', skiprows=1, usecols=(0, 1, 2))
        except Exception as e:
            print(f"Error reading file: {e}")
            raise
        
        if points.shape[0] == 0:
            raise ValueError("No points found in the file")

        # Normalize spatial coordinates
        points = self.normalize_points(points)

        # Sample points to match model input
        sampled_points = self.sample_points(points)

        # Convert to tensor (3 features per point)
        points_tensor = torch.from_numpy(sampled_points).float().transpose(0, 1).unsqueeze(0).to(self.device)

        return points_tensor, points  # Return both tensor and original points
    
    def normalize_points(self, points):
        centered_points = points - points.mean(axis=0)
        scale = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
        return centered_points / scale
    
    def sample_points(self, points):
        # Ensure consistent number of points for model input
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
        return points[indices]
    
    def predict(self, las_file_path):
        # Preprocess input
        input_tensor, original_points = self.preprocess_las_file(las_file_path)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)  # Shape: (1, num_classes, num_points)
            predicted = torch.argmax(outputs, dim=1)  # Get the class index for each point

        # Convert indices to label names
        predicted_labels = [H3D_LABELS[pred.item()] for pred in predicted.squeeze()]

        return predicted_labels, original_points

def main():
    # Paths for model and test data
    model_path = r"C:\Users\faars\Downloads\s3dis-train-pointnet++_ckpt_best.pth"
    las_file_path = r"C:\Farshid\Uni\Semesters\Thesis\Data\Epoch_March2018\LiDAR\Mar18_test.txt"
    
    # Initialize predictor
    predictor = PointCloudPredictor(model_path)
    
    # Predict
    predictions, points = predictor.predict(las_file_path)
    
    # Print results
    print("Prediction Results:")
    print(f"Total points predicted: {len(predictions)}")
    
    # Summarize predictions
    from collections import Counter
    pred_summary = Counter(predictions)
    print("\nPrediction Summary:")
    for label, count in pred_summary.items():
        print(f"{label}: {count} points")
    
    # Optional: Save predictions to a file
    output_file = las_file_path.replace('.txt', '_predictions.txt')
    with open(output_file, 'w') as f:
        f.write("X\tY\tZ\tLabel\n")
        for point, label in zip(points, predictions):
            f.write(f"{point[0]}\t{point[1]}\t{point[2]}\t{label}\n")
    print(f"\nPredictions saved to {output_file}")

if __name__ == '__main__':
    main()
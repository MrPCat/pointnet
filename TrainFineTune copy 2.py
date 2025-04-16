import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pointnet_ import PointNet2ClsSSG

# === Dataset Class for PTS files ===
class PtsPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, points_per_cloud=1024):
        try:
            # Read the PTS file with custom handling for potential trailing whitespace
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse the lines manually to avoid numpy's strict parsing
            data_rows = []
            for line in lines:
                line = line.strip()  # Remove trailing/leading whitespace
                if line:  # Skip empty lines
                    values = line.split()
                    if len(values) >= 6:  # Only process lines with at least 6 values
                        row = [float(values[i]) for i in range(6)]
                        data_rows.append(row)
            
            # Convert to numpy array
            data = np.array(data_rows)
            print(f"Successfully loaded {len(data)} points from {file_path}")
            
            # Extract XYZ coordinates and features
            self.xyz = data[:, 0:3]  # X, Y, Z
            self.intensity = data[:, 3:4]  # Intensity
            self.return_number = data[:, 4:5]  # Return Number
            self.number_of_returns = data[:, 5:6]  # Number of Returns
            
            # Store original points before normalization for saving later
            self.original_points = data[:, 0:3].copy()
            
            # Normalize XYZ coordinates
            center = np.mean(self.xyz, axis=0)
            self.xyz = self.xyz - center
            
            # Combine features
            self.features = np.concatenate((self.intensity, self.return_number, self.number_of_returns), axis=1)
            
            self.points_per_cloud = points_per_cloud
            self.num_clouds = max(1, len(self.xyz) // self.points_per_cloud)
            print(f"Total points: {len(self.xyz)}, Creating {self.num_clouds} point clouds")
                
        except Exception as e:
            print(f"Error loading PTS file: {e}")
            raise

    def __len__(self):
        return self.num_clouds

    def __getitem__(self, idx):
        # Get a chunk of points for this cloud
        start_idx = idx * self.points_per_cloud
        end_idx = min((idx + 1) * self.points_per_cloud, len(self.xyz))
        
        # If we don't have enough points, just repeat the last points
        if end_idx - start_idx < self.points_per_cloud:
            # Calculate how many points we need to repeat
            points_needed = self.points_per_cloud - (end_idx - start_idx)
            
            # Get the available points
            cloud_xyz = self.xyz[start_idx:end_idx]
            cloud_features = self.features[start_idx:end_idx]
            
            # Repeat the last point to fill up to points_per_cloud
            repeat_xyz = np.tile(cloud_xyz[-1:], (points_needed, 1))
            repeat_features = np.tile(cloud_features[-1:], (points_needed, 1))
            
            # Concatenate the available points with the repeated points
            cloud_xyz = np.concatenate([cloud_xyz, repeat_xyz], axis=0)
            cloud_features = np.concatenate([cloud_features, repeat_features], axis=0)
        else:
            cloud_xyz = self.xyz[start_idx:start_idx + self.points_per_cloud]
            cloud_features = self.features[start_idx:start_idx + self.points_per_cloud]
        
        # Convert to tensors with proper shape for PointNet2 (Important: Remove the extra batch dimension)
        # Shape should be (3, N) for xyz and (num_features, N) for features
        xyz_tensor = torch.tensor(cloud_xyz, dtype=torch.float32).transpose(0, 1)  # Shape: (3, N)
        features_tensor = torch.tensor(cloud_features, dtype=torch.float32).transpose(0, 1)  # Shape: (num_features, N)
        
        # Store the indices for mapping predictions back to original points
        indices = np.arange(start_idx, min(start_idx + self.points_per_cloud, len(self.xyz)))
        if len(indices) < self.points_per_cloud:
            # Pad with the last index for any repeated points
            indices = np.pad(indices, (0, self.points_per_cloud - len(indices)), 'edge')
            
        return features_tensor, xyz_tensor, torch.tensor(indices, dtype=torch.long)

# === Helper function to create dataset ===
def create_dataset(file_path, points_per_cloud=1024):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in ['.pts', '.txt', '.xyz']:
        return PtsPointCloudDataset(file_path, points_per_cloud)
    else:
        print(f"Unknown file type: {file_ext}, attempting to load as PTS")
        return PtsPointCloudDataset(file_path, points_per_cloud)

# === Inspect the model structure ===
def inspect_model_forward(model):
    print("Model structure:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    print("Model expects:")
    print("- features input shape: (B, C, N) where B=batch, C=channels, N=points")
    print("- xyz input shape: (B, 3, N) where B=batch, 3=XYZ, N=points")

# === Predict Labels for All Points ===
def predict_labels_for_all_points(model, data_loader, device, num_total_points):
    model.eval()
    all_predictions = np.zeros(num_total_points, dtype=np.int64)
    point_processed = np.zeros(num_total_points, dtype=bool)

    with torch.no_grad():
        for batch_idx, (features, xyz, indices) in enumerate(data_loader):
            # Move to device
            features, xyz = features.to(device), xyz.to(device)
            
            # Debug shapes (for first batch only)
            if batch_idx == 0:
                print(f"Input features shape: {features.shape}")
                print(f"Input xyz shape: {xyz.shape}")
            
            # Forward pass through the model
            try:
                logits = model(features, xyz)
                predicted = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Map predictions back to original points using indices
                batch_indices = indices.numpy()
                
                # For each point cloud in the batch
                for i, cloud_indices in enumerate(batch_indices):
                    # For each point in the point cloud that's valid (not from padding)
                    for j, idx in enumerate(cloud_indices):
                        if idx < num_total_points and not point_processed[idx]:
                            all_predictions[idx] = predicted[i]
                            point_processed[idx] = True
                
                # Print progress every few batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                # Print shapes for debugging
                print(f"Features shape: {features.shape}")
                print(f"XYZ shape: {xyz.shape}")
                raise
    
    # Check if any points weren't processed and handle them
    if not np.all(point_processed):
        print(f"Warning: {np.sum(~point_processed)} points were not processed!")
        # Fill unprocessed points with nearest neighbor's prediction
        for idx in np.where(~point_processed)[0]:
            if idx > 0 and point_processed[idx-1]:
                all_predictions[idx] = all_predictions[idx-1]
            elif idx < num_total_points-1 and point_processed[idx+1]:
                all_predictions[idx] = all_predictions[idx+1]
    
    return all_predictions

# === Save Predicted Labels with Points ===
def save_predictions_with_points(predictions, original_points, intensity, return_number, number_of_returns, output_file):
    complete_data = np.concatenate((
        original_points,  # X, Y, Z
        intensity,       # Intensity
        return_number,   # Return Number
        number_of_returns,  # Number of Returns
        predictions.reshape(-1, 1)  # Predicted class
    ), axis=1)
    
    np.savetxt(output_file, complete_data, fmt='%.2f %.2f %.2f %.0f %.0f %.0f %.0f', delimiter=' ')
    print(f"Saved {len(predictions)} points with predictions to {output_file}")

# === New Function: Print Class Distribution ===
def print_class_distribution(predictions, class_names=None):
    """
    Print the number of points predicted for each class.
    
    Args:
        predictions: numpy array of predicted class indices
        class_names: optional list of class names corresponding to indices
    """
    unique_classes, counts = np.unique(predictions, return_counts=True)
    total_points = len(predictions)
    
    print("\n=== Class Distribution ===")
    print(f"Total points classified: {total_points}")
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(max(unique_classes) + 1)]
    
    # Print counts for each class
    for class_idx, count in sorted(zip(unique_classes, counts)):
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Unknown Class {class_idx}"
            
        percentage = (count / total_points) * 100
        print(f"{class_name}: {count} points ({percentage:.2f}%)")
    
    print("=========================")

# === Main Prediction and Saving ===
def predict_and_save(model_path, test_file, output_file, batch_size=4, points_per_cloud=128):
    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PointNet2ClsSSG(in_dim=3, out_dim=4, downsample_points=(64, 32))  # Adjust to smaller downsampling values
    # Use map_location and weights_only=True for safety
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    print(f"Model loaded successfully")
    
    # Display model info
    inspect_model_forward(model)

    # Create the dataset for the test file
    print(f"Loading test data from {test_file}...")
    test_dataset = create_dataset(test_file, points_per_cloud)
    
    # Store the original points before they get processed
    original_points = test_dataset.original_points
    num_total_points = len(original_points)
    
    # Also store intensity, return number, and number of returns for saving
    intensity = test_dataset.intensity
    return_number = test_dataset.return_number
    number_of_returns = test_dataset.number_of_returns
    
    # Create dataloader with appropriate batch size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Predict the labels for all points in the test file
    print(f"Starting prediction for {num_total_points} points in batches of {batch_size} point clouds...")
    predicted_labels = predict_labels_for_all_points(model, test_loader, device, num_total_points)

    # Save the predicted labels with the original points and other attributes
    save_predictions_with_points(
        predicted_labels, 
        original_points, 
        intensity, 
        return_number, 
        number_of_returns, 
        output_file
    )
    
    # Return the predicted labels for analysis
    return predicted_labels

# === Main Code Execution ===
if __name__ == "__main__":
    model_path = r"C:\Users\faars\Downloads\pointnetDown_epoch_26.pth"  # Path to your trained model
    test_file = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\3DLabeling\Vaihingen3D_EVAL_WITHOUT_REF\Vaihingen3D_EVAL_WITHOUT_REF.pts"  # Test file without labels
    output_file = r"C:\Users\faars\Downloads\predictions_with_points1.txt"  # Output file to save predictions

    # Call the function to predict and save
    # Use smaller values for batch_size and points_per_cloud to avoid memory issues
    predicted_labels = predict_and_save(model_path, test_file, output_file, batch_size=4, points_per_cloud=128)
    
    # Define class names for the 4 classes (modify based on your specific class names)
    class_names = ["Low Vegetation", "Buildings", "Trees", "Impervious Surfaces"]
    
    # Print distribution of points per class
    print_class_distribution(predicted_labels, class_names)
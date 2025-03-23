import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_scoreimport
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet_ import PointNet2ClsSSG  # Ensure this matches the model from training
import pandas as pd

# === 1. Load the Trained Model ===
def load_model(model_path):
    """
    Load a trained PointNet++ model.
    """
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))  # Load checkpoint
    model = PointNet2ClsSSG()  # Instantiate your model (change this to your model's class)
    
    # Load model weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])  # Ensure this key matches your saved checkpoint structure
    model.eval()  # Set to evaluation mode
    return model

# === 2. Preprocess Point Cloud Data ===
def preprocess_point_cloud(points):
    """
    Normalize point cloud (N, 3) format.
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid  # Centering
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = points / max_dist  # Normalization
    return points

# === 3. Load New Data (WITH Labels) ===
def load_new_points_with_labels(file_path):
    """
    Load a .pts file that contains both point coordinates and ground truth labels.
    """
    try:
        data = np.loadtxt(file_path)  # Load entire file
        if data.shape[1] >= 4:  # At least X, Y, Z, and Label
            points = data[:, :3]  # Extract XYZ coordinates
            labels = data[:, -1].astype(int)  # Extract the last column as integer labels
        else:
            raise ValueError("Invalid .pts file format. Expected at least 4 columns (X, Y, Z, Label).")
    except Exception as e:
        raise RuntimeError(f"Error loading .pts file: {e}")

    return preprocess_point_cloud(points), labels

# === 4. Make Predictions ===
def predict(model, points):
    """
    Predict using PointNet++ model.
    """
    points = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(points)  # Forward pass

    return output

# === 5. Define Label Mapping ===
label_mapping = {
    0: 0,  # Powerline
    1: 1,  # Low vegetation
    2: 2,  # Impervious surfaces
    3: 3,  # Car
    4: 4,  # Fence/Hedge
    5: 5,  # Roof
    6: 6,  # Facade
    7: 7,  # Shrub
    8: 8   # Tree
}

def remap_labels(predicted_labels, mapping_dict):
    """
    Map predicted labels to match the dataset's ground-truth labels.
    """
    return np.array([mapping_dict[label] for label in predicted_labels])

# === 6. Main Execution ===
if __name__ == "__main__":
    # Set correct file paths
    model_path = r"C:\Users\faars\Downloads\modelnet40ply2048-train-pointnet++.pth" # Change to your model path
    new_points_path = r"C:\Farshid\Uni\Semesters\Thesis\Data\Vaihingen\Vaihingen\3DLabeling\Vaihingen3D_EVAL_WITH_REF.pts"  # Change to your .pts file

    # Load model and data
    model = load_model(model_path)
    new_points, true_labels = load_new_points_with_labels(new_points_path)  # Load points & labels

    # Get predictions
    predictions = predict(model, new_points)

    # Convert predictions to class labels
    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()

    # 🔹 Apply Label Mapping if Needed 🔹
    predicted_labels = remap_labels(predicted_labels, label_mapping)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Print detailed classification report
    print("\nClassification Report:\n")
    print(classification_report(true_labels, predicted_labels))

import laspy
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from joblib import Parallel, delayed
import cupy as cp
from cuml.neighbors import NearestNeighbors  # GPU-accelerated nearest neighbors

# Step 1: Load Ground Truth Data from .las File
def load_ground_truth(las_file):
    las = laspy.read(las_file)
    ground_truth_data = pd.DataFrame({
        "x": las.x,
        "y": las.y,
        "z": las.z,
        "ground_truth_label": las.classification  # Replace with appropriate field
    })
    return ground_truth_data

# Step 2: Load Prediction Data from .txt File
def load_predictions(txt_file):
    predictions = pd.read_csv(txt_file, delimiter=",", names=["x", "y", "z", "predicted_label"])
    return predictions

# Step 3: Match Predictions to Ground Truth (Using GPU)
def match_predictions_to_ground_truth(predictions, ground_truth_data):
    # Convert ground truth and predictions to GPU arrays
    gt_coords = cp.array(ground_truth_data[['x', 'y', 'z']].values)
    pred_coords = cp.array(predictions[['x', 'y', 'z']].values)

    # Use GPU-accelerated nearest neighbors
    knn = NearestNeighbors(n_neighbors=1, algorithm="brute")
    knn.fit(gt_coords)
    distances, indices = knn.kneighbors(pred_coords)

    # Retrieve ground truth labels
    indices = cp.asnumpy(indices).flatten()
    predictions['ground_truth_label'] = ground_truth_data.iloc[indices].ground_truth_label.values
    return predictions

# Step 4: Evaluate Predictions
def evaluate_predictions(predictions):
    print("Classification Report:")
    print(classification_report(predictions['ground_truth_label'], predictions['predicted_label']))
    
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(predictions['ground_truth_label'], predictions['predicted_label'])
    print(conf_matrix)

# Main Function
if __name__ == "__main__":
    # File paths
    ground_truth_file = "ground_truth.las"  # Replace with your .las file path
    predictions_file = "predictions.txt"   # Replace with your .txt file path

    # Load data
    print("Loading ground truth data...")
    ground_truth_data = load_ground_truth(ground_truth_file)
    
    print("Loading predictions...")
    predictions = load_predictions(predictions_file)
    
    # Match and compare
    print("Matching predictions to ground truth...")
    predictions = match_predictions_to_ground_truth(predictions, ground_truth_data)
    
    print("Evaluating predictions...")
    evaluate_predictions(predictions)

import laspy
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Ground Truth Data from .las File
def load_ground_truth(las_file):
    las = laspy.read(las_file)
    ground_truth_data = pd.DataFrame({
        "x": las.x,
        "y": las.y,
        "z": las.z,
        "ground_truth_label": las.classification  # Replace with the appropriate field if different
    })
    return ground_truth_data

# Step 2: Load Prediction Data from .txt File
def load_predictions(txt_file):
    predictions = pd.read_csv(txt_file, delimiter=",", names=["x", "y", "z", "predicted_label"])
    return predictions

# Step 3: Match Predictions to Ground Truth
def match_predictions_to_ground_truth(predictions, ground_truth_data):
    # Build a KDTree for ground truth points
    gt_tree = cKDTree(ground_truth_data[['x', 'y', 'z']].values)
    
    # Find the nearest ground truth point for each prediction
    distances, indices = gt_tree.query(predictions[['x', 'y', 'z']].values)
    
    # Add ground truth labels to predictions DataFrame
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
    ground_truth_file = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"  # Replace with your .las file path
    predictions_file = "/content/drive/MyDrive/t1/predictions.txt"   # Replace with your .txt file path

    # Load data
    ground_truth_data = load_ground_truth(ground_truth_file)
    predictions = load_predictions(predictions_file)
    
    # Match and compare
    predictions = match_predictions_to_ground_truth(predictions, ground_truth_data)
    evaluate_predictions(predictions)

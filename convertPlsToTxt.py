import laspy
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import classification_report, confusion_matrix

def load_ground_truth(file_path):
    """
    Load ground truth data, whether it's a valid LAS file or a text file.
    """
    try:
        # Attempt to read as LAS file
        las = laspy.read(file_path)
        ground_truth_data = pd.DataFrame({
            "x": las.x,
            "y": las.y,
            "z": las.z,
            "ground_truth_label": las.classification  # Adjust field as necessary
        })
        print("Successfully loaded LAS file.")
        return ground_truth_data
    except laspy.errors.LaspyException:
        # Handle as text file
        print("File is not a valid LAS file. Attempting to load as text file.")
        # Read the text file as a DataFrame
        ground_truth_data = pd.read_csv("/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las", delimiter="\t")

        
        print("Successfully loaded text file.")
        return ground_truth_data

def load_predictions(txt_file):
    """
    Load prediction data from a text file.
    """
    predictions = pd.read_csv(txt_file, delimiter=",", names=["x", "y", "z", "predicted_label"])
    print("Successfully loaded predictions.")
    return predictions

def match_predictions_to_ground_truth(predictions, ground_truth_data):
    """
    Match predictions to the nearest ground truth points using cKDTree.
    """
    print("Matching predictions to ground truth...")
    gt_tree = cKDTree(ground_truth_data[['x', 'y', 'z']].values)
    distances, indices = gt_tree.query(predictions[['x', 'y', 'z']].values)
    predictions['ground_truth_label'] = ground_truth_data.iloc[indices].ground_truth_label.values
    print("Matching completed.")
    return predictions

def evaluate_predictions(predictions):
    """
    Evaluate predictions against ground truth labels.
    """
    print("Evaluating predictions...")
    print("Classification Report:")
    print(classification_report(predictions['ground_truth_label'], predictions['predicted_label']))
    
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(predictions['ground_truth_label'], predictions['predicted_label'])
    print(conf_matrix)

def main(ground_truth_file, predictions_file):
    """
    Main function to load data, match predictions, and evaluate performance.
    """
    # Load ground truth and predictions
    ground_truth_data = load_ground_truth(ground_truth_file)
    predictions = load_predictions(predictions_file)
    
    # Match predictions to ground truth
    predictions = match_predictions_to_ground_truth(predictions, ground_truth_data)
    
    # Evaluate predictions
    evaluate_predictions(predictions)

if __name__ == "__main__":
    # File paths
    ground_truth_file = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"  # Adjust path as needed
    predictions_file = "/content/drive/MyDrive/t1/predictions.txt"  # Adjust path as needed
    
    # Run the main function
    main(ground_truth_file, predictions_file)

import pandas as pd
import laspy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the .txt file
def load_txt_file(txt_file):
    # Load data as a pandas DataFrame
    df = pd.read_csv(txt_file, sep="\t")  # Adjust delimiter if needed
    print("Load text Done!")
    return df

# Step 2: Load the .las file
def load_las_file(las_file):
    las = laspy.read(las_file)
    # Extract classifications
    classifications = las.classification
    # Extract X, Y, Z coordinates for alignment
    coords = pd.DataFrame({'X': las.x, 'Y': las.y, 'Z': las.z, 'Classification': classifications})
    print("load las Done")
    return coords

# Step 3: Merge data based on coordinates
def merge_data(txt_df, las_df):
    # Perform a merge on X, Y, Z coordinates to align points
    merged = pd.merge(txt_df, las_df, on=["X", "Y", "Z"], suffixes=("_pred", "_ref"))
    print("merge Done")
    return merged

# Step 4: Evaluate accuracy and generate metrics
def evaluate_classes(merged_data):
    # Predicted and reference classes
    y_pred = merged_data["Classification_pred"]
    y_ref = merged_data["Classification_ref"]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_ref, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Generate confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_ref, y_pred))
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_ref, y_pred))

# Main function
def main(txt_file, las_file):
    # Load data
    txt_data = load_txt_file(txt_file)
    las_data = load_las_file(las_file)
    
    # Merge data
    merged_data = merge_data(txt_data, las_data)
    
    # Evaluate classifications
    evaluate_classes(merged_data)

# Replace with your actual file paths
txt_file = "/content/drive/MyDrive/t1/predictions.txt"
las_file = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"

# Run the comparison
main(txt_file, las_file)

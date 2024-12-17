import pandas as pd 
import laspy 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import numpy as np
import os

# Step 1: Load the .txt file 
def load_txt_file(txt_file): 
    # Verify file exists
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Text file not found: {txt_file}")
    
    # Load data as a pandas DataFrame 
    df = pd.read_csv(txt_file, sep="\t")  # Adjust delimiter if needed 
    
    # Print column info for diagnosis
    print("Text File Columns:")
    print(df.columns)
    print("\nText File Coordinate Columns Data Types:")
    print(df[['X', 'Y', 'Z']].dtypes)
    print("\nFirst few rows:")
    print(df.head())
    
    # Convert coordinate columns to float if they're not already
    for col in ['X', 'Y', 'Z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("\nAfter conversion:")
    print(df[['X', 'Y', 'Z']].dtypes)
    
    print("Load text Done!") 
    return df 
 
# Step 2: Load the .las file 
def load_las_file(las_file): 
    # Verify file exists
    if not os.path.exists(las_file):
        raise FileNotFoundError(f"LAS file not found: {las_file}")
    
    try:
        las = laspy.read(las_file) 
        
        # Convert ScaledArrayView to numpy array first
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)
        classifications = np.array(las.classification)
        
        # Create DataFrame with explicit index
        coords = pd.DataFrame({
            'X': x, 
            'Y': y, 
            'Z': z, 
            'Classification': classifications
        }, index=range(len(x)))
        
        print("load las Done") 
        print(f"Loaded {len(coords)} points")
        
        # Print coordinate info for diagnosis
        print("\nLAS File Coordinate Columns Data Types:")
        print(coords[['X', 'Y', 'Z']].dtypes)
        print("\nFirst few rows:")
        print(coords.head())
        
        return coords 
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        raise
    
# Step 3: Merge data based on coordinates 
def merge_data(txt_df, las_df): 
    # Use round to handle floating point imprecision
    def round_coords(df, decimals=3):
        df_rounded = df.copy()
        df_rounded[['X', 'Y', 'Z']] = df_rounded[['X', 'Y', 'Z']].round(decimals)
        return df_rounded
    
    txt_df_rounded = round_coords(txt_df)
    las_df_rounded = round_coords(las_df)
    
    # Perform a merge on X, Y, Z coordinates to align points 
    merged = pd.merge(txt_df_rounded, las_df_rounded, 
                      on=['X', 'Y', 'Z'], 
                      suffixes=("_pred", "_ref"))
    
    print(f"Merge Done. {len(merged)} points matched")
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
    # Print file paths for verification
    print(f"Text file path: {txt_file}")
    print(f"LAS file path: {las_file}")
    
    # Load data 
    txt_data = load_txt_file(txt_file) 
    las_data = load_las_file(las_file) 
     
    # Merge data 
    merged_data = merge_data(txt_data, las_data) 
     
    # Evaluate classifications 
    evaluate_classes(merged_data) 

# Replace with your actual file paths 
txt_file = r"/content/drive/MyDrive/t1/PredictTest_noRGB.txt"
las_file = r"/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las" 
 
# Run the comparison 
main(txt_file, las_file)
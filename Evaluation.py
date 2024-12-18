import pandas as pd
import laspy
import numpy as np
import os


# Step 1: Load the .las file
def load_las_file(las_file):
    # Verify file exists
    if not os.path.exists(las_file):
        raise FileNotFoundError(f"LAS file not found: {las_file}")
    
    try:
        # Read LAS file
        las = laspy.read(las_file)
        print("LAS object loaded successfully.")
        
        # Extract points into a DataFrame using numpy arrays
        coords = pd.DataFrame({
            'X': las.x.copy(),  # Use .copy() to create a standalone NumPy array
            'Y': las.y.copy(), 
            'Z': las.z.copy(),
            'Classification': las.classification.copy()
        }, index=np.arange(len(las.x)))  # Add an explicit index
        
        print("Loaded LAS file")
        print(f"Loaded {len(coords)} points")
        print(coords.head())
        return coords
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        raise

# Step 2: Load the .txt file
def load_txt_file(txt_file):
    # Verify file exists
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Text file not found: {txt_file}")
    
    try:
        # Load data as a pandas DataFrame
        df = pd.read_csv(txt_file, sep="\t", encoding='utf-8')  # Added encoding specification
        
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
    except Exception as e:
        print(f"Error reading text file: {e}")
        raise

# Rest of the code remains the same...
# [Previous merge_data and evaluate_classes functions remain unchanged]

# Main function
def main(txt_file, las_file):
    # Print file paths for verification
    print(f"Text file path: {txt_file}")
    print(f"LAS file path: {las_file}")
    
    try:
        # Verify file extensions
        if not txt_file.lower().endswith('.txt'):
            raise ValueError(f"Expected .txt file for predictions, got: {txt_file}")
        if not las_file.lower().endswith('.las'):
            raise ValueError(f"Expected .las file for ground truth, got: {las_file}")
        
        # Load data
        las_data = load_las_file(las_file)
        txt_data = load_txt_file(txt_file)
        
        # Merge data
        merged_data = merge_data(txt_data, las_data)
        
        # Evaluate classifications
        evaluate_classes(merged_data)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

# File paths - CORRECTED ORDER
txt_file = r"/content/drive/MyDrive/t1/Mar18_testWithoutRGB_predictions.txt"
las_file = r"/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"

# Run the comparison
if __name__ == "__main__":
    main(txt_file, las_file)
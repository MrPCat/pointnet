import pandas as pd
import laspy
import numpy as np
import os

def load_las_file(las_file):
    """Load and validate a LAS file"""
    if not os.path.exists(las_file):
        raise FileNotFoundError(f"LAS file not found: {las_file}")
    
    try:
        # Try to read first few bytes to validate file signature
        with open(las_file, 'rb') as f:
            signature = f.read(4)
            if signature != b'LASF':
                raise ValueError(f"Invalid LAS file: File does not have LASF signature. This might be a text file instead.")
        
        # If signature is valid, proceed with reading
        las = laspy.read(las_file)
        coords = pd.DataFrame({
            'X': las.x.copy(),
            'Y': las.y.copy(),
            'Z': las.z.copy(),
            'Classification': las.classification.copy()
        }, index=np.arange(len(las.x)))
        
        print(f"Successfully loaded LAS file with {len(coords)} points")
        return coords
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        print("Please verify that the file paths for LAS and TXT files are not swapped.")
        raise

def load_txt_file(txt_file):
    """Load and validate a text file"""
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Text file not found: {txt_file}")
    
    try:
        # Try to read the first line to check format
        with open(txt_file, 'r') as f:
            first_line = f.readline().strip()
            if 'LASF' in first_line:
                raise ValueError("This appears to be a LAS file, not a text file. File paths might be swapped.")
        
        # Load data as a pandas DataFrame
        df = pd.read_csv(txt_file, sep="\t", dtype={
            'X': float,
            'Y': float,
            'Z': float,
            'Classification': int
        })
        
        required_columns = ['X', 'Y', 'Z', 'Classification']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in text file: {missing_columns}")
        
        print(f"Successfully loaded text file with {len(df)} points")
        return df
    
    except pd.errors.EmptyDataError:
        print("Error: The text file is empty")
        raise
    except Exception as e:
        print(f"Error reading text file: {e}")
        raise

# Main function
def main(txt_file, las_file):
    print("Starting comparison process...")
    print(f"Text file path: {txt_file}")
    print(f"LAS file path: {las_file}")
    
    # Swap file paths if they appear to be incorrect
    try:
        with open(txt_file, 'rb') as f:
            if f.read(4) == b'LASF':
                print("Warning: File paths appear to be swapped. Auto-correcting...")
                txt_file, las_file = las_file, txt_file
    except:
        pass
    
    # Load data
    txt_data = load_txt_file(txt_file)
    las_data = load_las_file(las_file)
    
    # Continue with existing merge and evaluation functions...
    merged_data = merge_data(txt_data, las_data)
    evaluate_classes(merged_data)

# File paths (make sure these are in the correct order)
las_file = r"/content/drive/MyDrive/t1/Mar18_testWithoutRGB_predictions.txt"
txt_file = r"/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"

if __name__ == "__main__":
    main(txt_file, las_file)
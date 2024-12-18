import pandas as pd
import laspy
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Step 3: Merge data based on coordinates (Optimized)
def merge_data(txt_df, las_df, decimals=3):
    try:
        # Select only necessary columns for merging
        txt_df = txt_df[['X', 'Y', 'Z', 'Classification']]
        las_df = las_df[['X', 'Y', 'Z', 'Classification']]
        
        # Round coordinates to handle floating-point imprecision
        txt_df_rounded = txt_df.copy()
        las_df_rounded = las_df.copy()
        txt_df_rounded[['X', 'Y', 'Z']] = txt_df[['X', 'Y', 'Z']].round(decimals)
        las_df_rounded[['X', 'Y', 'Z']] = las_df[['X', 'Y', 'Z']].round(decimals)
        
        # Merge in chunks if memory issues persist
        chunk_size = 1_000_000  # Process 1 million rows at a time
        merged_chunks = []
        
        for i in range(0, len(txt_df_rounded), chunk_size):
            txt_chunk = txt_df_rounded.iloc[i:i + chunk_size]
            merged_chunk = pd.merge(txt_chunk, las_df_rounded, 
                                  on=['X', 'Y', 'Z'], 
                                  suffixes=("_pred", "_ref"))
            merged_chunks.append(merged_chunk)
        
        # Concatenate all merged chunks
        merged = pd.concat(merged_chunks, ignore_index=True)
        print(f"Merge Done. {len(merged)} points matched.")
        return merged
    except Exception as e:
        print(f"Error during merge operation: {e}")
        raise

# Step 4: Evaluate accuracy and generate metrics
def evaluate_classes(merged_data):
    try:
        # Predicted and reference classes
        y_pred = merged_data["Classification_pred"]
        y_ref = merged_data["Classification_ref"]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_ref, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(y_ref, y_pred)
        
        # Generate classification report
        class_report = classification_report(y_ref, y_pred)
        
        # Save results to file
        output_path = "accuracy_results.txt"  # You can modify this path as needed
        with open(output_path, "w") as file:
            file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            file.write("\nConfusion Matrix:\n")
            file.write(str(conf_matrix))
            file.write("\n\nClassification Report:\n")
            file.write(class_report)
        
        # Print results
        print(f"\nResults saved to: {output_path}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        
        return accuracy, conf_matrix, class_report
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

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
        print("\nLoading LAS file...")
        las_data = load_las_file(las_file)
        
        print("\nLoading TXT file...")
        txt_data = load_txt_file(txt_file)
        
        print("\nMerging data...")
        merged_data = merge_data(txt_data, las_data)
        
        print("\nEvaluating classifications...")
        accuracy, conf_matrix, class_report = evaluate_classes(merged_data)
        
        print("\nProcessing completed successfully!")
        return accuracy, conf_matrix, class_report
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # File paths - update these to your actual file paths
    txt_file = r"/path/to/your/predictions.txt"
    las_file = r"/path/to/your/ground_truth.las"
    
    try:
        # Run the comparison
        accuracy, conf_matrix, class_report = main(txt_file, las_file)
    except Exception as e:
        print(f"Program terminated with error: {e}")
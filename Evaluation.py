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

# Step 3: Merge data based on coordinates
# Step 3: Merge data based on coordinates (Optimized)
def merge_data(txt_df, las_df, decimals=3):
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

# Step 4: Evaluate accuracy and generate metrics
# Step 4: Evaluate accuracy and generate metrics
def evaluate_classes(merged_data):
    # Predicted and reference classes
    y_pred = merged_data["Classification_pred"]
    y_ref = merged_data["Classification_ref"]
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    accuracy = accuracy_score(y_ref, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Save accuracy to a file
    output_path = r"C:\Farshid\Uni\Semesters\Thesis\Data\Output\2. First attempt without RGB\accuracy_results.txt"  # Update this path as needed
    with open(output_path, "w") as file:
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        file.write("\nConfusion Matrix:\n")
        file.write(f"{confusion_matrix(y_ref, y_pred)}\n")
        file.write("\nClassification Report:\n")
        file.write(classification_report(y_ref, y_pred))
    
    print(f"Accuracy and metrics saved to {output_path}")
    
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
    las_data = load_las_file(las_file)
    txt_data = load_txt_file(txt_file)
    
    # Merge data
    merged_data = merge_data(txt_data, las_data)
    
    # Evaluate classifications
    evaluate_classes(merged_data)

# Replace with your actual file paths
txt_file = r"/content/drive/MyDrive/t1/Mar18_testWithoutRGB_predictions.txt"
las_file = r"/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"

# Run the comparison
main(txt_file, las_file)

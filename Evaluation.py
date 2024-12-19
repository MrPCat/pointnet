import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_txt_file(file_path, file_type=""):
    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Load data as a pandas DataFrame
        df = pd.read_csv(file_path, sep="\t", encoding='utf-8')
        
        # Print column info for diagnosis
        print(f"\nLoaded {file_type} File:")
        print("Columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        
        # Convert coordinate columns to float if they're not already
        for col in ['X', 'Y', 'Z']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Classification' not in df.columns:
            raise ValueError(f"Classification column missing in {file_type} file")
            
        return df
    except Exception as e:
        print(f"Error reading {file_type} file: {e}")
        raise

def merge_data(pred_df, ref_df, decimals=3):
    try:
        # Select only necessary columns for merging
        pred_df = pred_df[['X', 'Y', 'Z', 'Classification']]
        ref_df = ref_df[['X', 'Y', 'Z', 'Classification']]
        
        # Round coordinates to handle floating-point imprecision
        pred_df_rounded = pred_df.copy()
        ref_df_rounded = ref_df.copy()
        pred_df_rounded[['X', 'Y', 'Z']] = pred_df[['X', 'Y', 'Z']].round(decimals)
        ref_df_rounded[['X', 'Y', 'Z']] = ref_df[['X', 'Y', 'Z']].round(decimals)
        
        # Merge in chunks if memory issues persist
        chunk_size = 1_000_000  # Process 1 million rows at a time
        merged_chunks = []
        
        for i in range(0, len(pred_df_rounded), chunk_size):
            pred_chunk = pred_df_rounded.iloc[i:i + chunk_size]
            merged_chunk = pd.merge(pred_chunk, ref_df_rounded, 
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
        output_path = "accuracy_results.txt"
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

def main(predictions_file, reference_file):
    print(f"Predictions file: {predictions_file}")
    print(f"Reference file: {reference_file}")
    
    try:
        # Load both files as text files
        print("\nLoading predictions file...")
        pred_data = load_txt_file(predictions_file, "Predictions")
        
        print("\nLoading reference file...")
        ref_data = load_txt_file(reference_file, "Reference")
        
        print("\nMerging data...")
        merged_data = merge_data(pred_data, ref_data)
        
        print("\nEvaluating classifications...")
        accuracy, conf_matrix, class_report = evaluate_classes(merged_data)
        
        print("\nProcessing completed successfully!")
        return accuracy, conf_matrix, class_report
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        raise

if __name__ == "__main__":
    # Update these paths to your actual file paths
    predictions_file = "/content/drive/MyDrive/Archive /1. first attempt with RGB and high Accuracy there /predictions_First_Exp.txt"
    reference_file = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"
    
    try:
        accuracy, conf_matrix, class_report = main(predictions_file, reference_file)
    except Exception as e:
        print(f"Program terminated with error: {e}")
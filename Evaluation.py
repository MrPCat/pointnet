import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_txt_file(file_path, file_type=""):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, sep="\t", encoding='utf-8')
        print(f"\nLoaded {file_type} File:")
        print("Columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())

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
        pred_df = pred_df[['X', 'Y', 'Z', 'Classification']]
        ref_df = ref_df[['X', 'Y', 'Z', 'Classification']]

        pred_df_rounded = pred_df.copy()
        ref_df_rounded = ref_df.copy()
        pred_df_rounded[['X', 'Y', 'Z']] = pred_df[['X', 'Y', 'Z']].round(decimals)
        ref_df_rounded[['X', 'Y', 'Z']] = ref_df[['X', 'Y', 'Z']].round(decimals)

        chunk_size = 1_000_000
        merged_chunks = []

        for i in range(0, len(pred_df_rounded), chunk_size):
            pred_chunk = pred_df_rounded.iloc[i:i + chunk_size]
            merged_chunk = pd.merge(pred_chunk, ref_df_rounded, 
                                    on=['X', 'Y', 'Z'], 
                                    suffixes=("_pred", "_ref"))
            merged_chunks.append(merged_chunk)

        merged = pd.concat(merged_chunks, ignore_index=True)

        # Calculate unmatched points
        matched_coords = merged[['X', 'Y', 'Z']]
        unmatched_pred = pred_df_rounded[~pred_df_rounded[['X', 'Y', 'Z']].isin(matched_coords.to_numpy()).all(axis=1)]
        unmatched_ref = ref_df_rounded[~ref_df_rounded[['X', 'Y', 'Z']].isin(matched_coords.to_numpy()).all(axis=1)]
        
        print(f"Merge Done. {len(merged)} points matched.")
        print(f"Unmatched points in predictions: {len(unmatched_pred)}")
        print(f"Unmatched points in reference: {len(unmatched_ref)}")

        return merged
    except Exception as e:
        print(f"Error during merge operation: {e}")
        raise


def evaluate_classes(merged_data):
    try:
        y_pred = merged_data["Classification_pred"]
        y_ref = merged_data["Classification_ref"]

        accuracy = accuracy_score(y_ref, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        conf_matrix = confusion_matrix(y_ref, y_pred)
        class_report = classification_report(y_ref, y_pred)

        output_path = "accuracy_results.txt"
        with open(output_path, "w") as file:
            file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            file.write("\nConfusion Matrix:\n")
            file.write(str(conf_matrix))
            file.write("\n\nClassification Report:\n")
            file.write(class_report)

        print(f"\nResults saved to: {output_path}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

        # Visualize Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_ref), yticklabels=np.unique(y_ref))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        return accuracy, conf_matrix, class_report

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

def main(predictions_file, reference_file):
    print(f"Predictions file: {predictions_file}")
    print(f"Reference file: {reference_file}")

    try:
        print("\nLoading predictions file...")
        pred_data = load_txt_file(predictions_file, "Predictions")

        print("\nLoading reference file...")
        ref_data = load_txt_file(reference_file, "Reference")

        print("\nInspecting loaded data...")
        print("Predictions shape:", pred_data.shape)
        print("Reference shape:", ref_data.shape)
        print("Predictions classifications:", pred_data['Classification'].unique())
        print("Reference classifications:", ref_data['Classification'].unique())

        print("\nMerging data...")
        merged_data = merge_data(pred_data, ref_data)

        print("\nMerged data shape:", merged_data.shape)
        print("Sample of merged data:")
        print(merged_data.head())

        print("\nEvaluating classifications...")
        accuracy, conf_matrix, class_report = evaluate_classes(merged_data)

        print("\nProcessing completed successfully!")
        return accuracy, conf_matrix, class_report

    except Exception as e:
        print(f"\nError in main execution: {e}")
        raise

if __name__ == "__main__":
    predictions_file = "/content/drive/MyDrive/t1/PredictTest_noRGB.txt"
    reference_file = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"

    try:
        accuracy, conf_matrix, class_report = main(predictions_file, reference_file)
    except Exception as e:
        print(f"Program terminated with error: {e}")

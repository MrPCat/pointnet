import laspy
import pandas as pd
import numpy as np

def convert_las_to_filtered_csv(las_file_path, output_file_path, remove_zero_columns=True):
    """
    Converts a .las file to a tab-separated text file and removes zero-filled columns if specified.

    Parameters:
    las_file_path (str): Path to the input .las file.
    output_file_path (str): Path to the output file (.txt or .csv).
    remove_zero_columns (bool): Whether to remove columns filled with zeros.
    """
    # Open the LAS file
    print(f"Reading LAS file: {las_file_path}")
    las = laspy.read(las_file_path)
    
    # Collect point data
    print("Extracting point properties...")
    points_data = {}

    # Add core properties (X, Y, Z)
    points_data["X"] = las.x
    points_data["Y"] = las.y
    points_data["Z"] = las.z

    # Extract additional attributes dynamically
    for dimension in las.point_format.dimension_names:
        if dimension not in ["X", "Y", "Z"]:  # Avoid duplicating XYZ
            points_data[dimension] = las[dimension]

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.DataFrame(points_data)

    if remove_zero_columns:
        print("Removing zero-filled columns...")
        # Remove columns where all values are zero
        non_zero_columns = df.loc[:, (df != 0).any(axis=0)]
        print(f"Removed {len(df.columns) - len(non_zero_columns.columns)} zero-filled columns.")
        df = non_zero_columns

    # Save to tab-separated text file
    print(f"Saving to file: {output_file_path}")
    df.to_csv(output_file_path, sep="\t", index=False, float_format="%.8f")

    print("Conversion complete.")

# Example usage
if __name__ == "__main__":
    input_las_file = "/path/to/your/input_file.las"  # Replace with the actual path
    output_txt_file = "/path/to/your/output_file.txt"  # Replace with the desired output path
    convert_las_to_filtered_csv(input_las_file, output_txt_file, remove_zero_columns=True)

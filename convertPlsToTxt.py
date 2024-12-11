import laspy
import pandas as pd
import numpy as np

def convert_las_to_csv(las_file_path, output_file_path):
    """
    Converts a .las file to a tab-separated text file with all its properties.
    
    Parameters:
    las_file_path (str): Path to the input .las file.
    output_file_path (str): Path to the output file (.txt or .csv).
    """
    # Open the LAS file
    print(f"Reading LAS file: {las_file_path}")
    with laspy.read(las_file_path) as las:
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
        
        # Save to tab-separated text file
        print(f"Saving to file: {output_file_path}")
        df.to_csv(output_file_path, sep="\t", index=False, float_format="%.8f")

    print("Conversion complete.")

# Example usage
if __name__ == "__main__":
    input_las_file = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"
    output_txt_file = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.txt"
    convert_las_to_csv(input_las_file, output_txt_file)

import os
file_path = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"
if os.path.exists(file_path):
    print("File exists.")
else:
    print("File not found.")
import laspy
import pandas as pd

def convert_las_to_filtered_csv(las_file_path, output_file_path, remove_zero_columns=True):
    try:
        # Read the LAS file
        las = laspy.read(las_file_path)
        print(f"File version: {las.header.version}")
        print(f"Number of points: {len(las.points)}")
        
        # Collect point data
        points_data = {
            "X": las.x,
            "Y": las.y,
            "Z": las.z
        }
        
        # Include additional attributes
        for dim in las.point_format.dimension_names:
            if dim not in ["X", "Y", "Z"]:
                points_data[dim] = las[dim]
        
        # Convert to DataFrame
        df = pd.DataFrame(points_data)
        
        # Save as tab-separated text file
        df.to_csv(output_file_path, sep="\t", index=False)
        print(f"Conversion successful. Saved to {output_file_path}")
    
    except Exception as e:
        print(f"Error during conversion: {e}")

convert_las_to_filtered_csv(
    "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las",
    "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.txt"
)

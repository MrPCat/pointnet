import laspy
import pandas as pd

def convert_valid_las(las_file_path, output_file_path):
    try:
        las = laspy.read(las_file_path)
        print(f"File opened successfully. Version: {las.header.version}")

        # Extract and process point data
        points_data = {
            "X": las.x,
            "Y": las.y,
            "Z": las.z
        }
        for dim in las.point_format.dimension_names:
            if dim not in ["X", "Y", "Z"]:
                points_data[dim] = las[dim]
        df = pd.DataFrame(points_data)
        df.to_csv(output_file_path, sep="\t", index=False)
        print(f"File converted and saved to: {output_file_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")

convert_valid_las(
    "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las",
    "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.txt"
)

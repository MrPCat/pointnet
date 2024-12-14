import laspy
import numpy as np

def inspect_important_attributes(file_path):
    # Open the LAZ file
    las = laspy.read(file_path)
    
    print("\n--- Important Attribute Analysis ---")

    # Classification Analysis
    if 'classification' in las.header.point_format.dimension_names:
        unique_classes, counts = np.unique(las.classification, return_counts=True)
        print("\n--- Classification Details ---")
        print(f"Unique Classes: {unique_classes}")
        print(f"Counts per Class: {counts}")

    # Number of Returns Analysis
    if 'number_of_returns' in las.header.point_format.dimension_names:
        unique_num_returns, counts_num_returns = np.unique(las.num_returns, return_counts=True)
        print("\n--- Number of Returns Details ---")
        print(f"Unique Number of Returns: {unique_num_returns}")
        print(f"Counts per Return Number: {counts_num_returns}")

    # Return Number Analysis
    if 'return_number' in las.header.point_format.dimension_names:
        unique_return_numbers, counts_return_numbers = np.unique(las.return_num, return_counts=True)
        print("\n--- Return Number Details ---")
        print(f"Unique Return Numbers: {unique_return_numbers}")
        print(f"Counts per Return Number: {counts_return_numbers}")
    
    print("\n--- Summary Complete ---")
    return {
        "classification": (unique_classes, counts),
        "number_of_returns": (unique_num_returns, counts_num_returns),
        "return_number": (unique_return_numbers, counts_return_numbers),
    }

# Replace with your file path
file_path = r"C:\Users\faars\Downloads\3dm_32_280_5652_1_nw.laz"
result = inspect_important_attributes(file_path)

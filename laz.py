import laspy
import lazrs

def inspect_laz_file(file_path):
    # Open the LAZ file
    las = laspy.read(file_path)
    
    # Print basic metadata
    print("\n--- Basic File Information ---")
    print(f"Point Count: {len(las.points)}")
    print(f"Point Format ID: {las.header.point_format.id}")
    print(f"Dimensions Available: {las.header.point_format.dimension_names}")
    print("\n--- Point Attributes ---")

    # Iterate through all dimensions
    for dimension in las.header.point_format.dimension_names:
        print(f"Attribute: {dimension} - Example Value: {getattr(las, dimension)[0]}")
    
    # Optional: View scaling and offset details for spatial coordinates
    print("\n--- Scaling and Offsets ---")
    print(f"X Scale: {las.header.scales[0]}, Offset: {las.header.offsets[0]}")
    print(f"Y Scale: {las.header.scales[1]}, Offset: {las.header.offsets[1]}")
    print(f"Z Scale: {las.header.scales[2]}, Offset: {las.header.offsets[2]}")

    # Return the list of attributes
    return las.header.point_format.dimension_names

# Replace with your file path
file_path = r"C:\Users\faars\Downloads\3dm_32_280_5652_1_nw.laz"
attributes = inspect_laz_file(file_path)

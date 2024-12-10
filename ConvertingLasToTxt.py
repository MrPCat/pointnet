import laspy
import numpy as np

def las_to_txt(input_las_path, output_txt_path):
    las_file = laspy.read(input_las_path)
    
    # Combine XYZ coordinates and classification (or other attributes)
    data = np.vstack((las_file.x, las_file.y, las_file.z, las_file.classification)).T
    
    # Save as text file
    np.savetxt(output_txt_path, data, fmt="%.6f", delimiter="\t", header="X\tY\tZ\tClassification", comments='')
    print(f"Converted {input_las_path} to {output_txt_path}")

# Example usage
las_to_txt("/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las", "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las")

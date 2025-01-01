import laspy

def convert_las_to_txt(las_file_path, txt_file_path):
    try:
        # Read the LAS file
        las = laspy.read(las_file_path)
        
        # Open the TXT file for writing
        with open(txt_file_path, 'w') as txt_file:
            # Write the header
            header = ["X", "Y", "Z"]
            if hasattr(las, "intensity"):
                header.append("Intensity")
            if hasattr(las, "classification"):
                header.append("Classification")
            if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
                header.extend(["Red", "Green", "Blue"])
            
            txt_file.write(",".join(header) + "\n")
            
            # Write the data
            for i in range(len(las.points)):
                line = [las.x[i], las.y[i], las.z[i]]
                if hasattr(las, "intensity"):
                    line.append(las.intensity[i])
                if hasattr(las, "classification"):
                    line.append(las.classification[i])
                if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
                    line.extend([las.red[i], las.green[i], las.blue[i]])
                
                txt_file.write(",".join(map(str, line)) + "\n")
        
        print(f"LAS file successfully converted to TXT: {txt_file_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
las_file_path = r"C:\Users\faars\Downloads\output_with_rgb1_predictions.las"
txt_file_path = r"C:\Users\faars\Downloads\output_with_rgb1_predictions.txt"
convert_las_to_txt(las_file_path, txt_file_path)

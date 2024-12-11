import os

def load_ground_truth(file_path):
    """
    Load ground truth data from a valid LAS file or a misnamed text file.
    """
    try:
        # Check file signature for binary LAS file
        if file_path.lower().endswith(".las"):
            with open(file_path, "rb") as f:
                file_signature = f.read(4).decode(errors="ignore")
                if file_signature == "LASF":
                    print("Valid LAS binary file detected.")
                    las = laspy.read(file_path)
                    ground_truth_data = pd.DataFrame({
                        "x": las.x,
                        "y": las.y,
                        "z": las.z,
                        "ground_truth_label": las.Classification
                    })
                    print("Successfully loaded LAS file.")
                    return ground_truth_data
                else:
                    print("File has .las extension but is not a binary LAS file. Attempting as text file.")
        
        # Treat as text file if signature is not valid
        ground_truth_data = pd.read_csv(file_path, delimiter="\t")
        if {'X', 'Y', 'Z', 'ground_truth_label'}.issubset(ground_truth_data.columns):
            ground_truth_data.rename(columns={"X": "x", "Y": "y", "Z": "z"}, inplace=True)
        else:
            raise ValueError("Text file does not contain required columns.")
        print("Successfully loaded text file.")
        return ground_truth_data
    except laspy.errors.LaspyException as e:
        print(f"Error reading LAS file: {e}")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

import laspy
import os

def inspect_file(file_path):
    """
    Inspect the file to determine if it is a valid LAS file, has a text header, or is corrupted.
    """
    try:
        with open(file_path, "rb") as f:
            file_signature = f.read(4)
        
        if file_signature == b'LASF':
            print("[INFO] File is a valid LAS binary file. Signature detected: LASF.")
            return "las_binary"
        elif b'\t' in file_signature or file_signature.decode(errors="ignore").startswith("X"):
            print("[WARNING] File has a text-like header. Signature detected: Text header.")
            return "text_with_header"
        else:
            print("[ERROR] File signature does not match LAS or text header standards.")
            return "unknown_format"
    except Exception as e:
        print(f"[ERROR] Failed to inspect file: {e}")
        return "error"

def strip_text_header(file_path, temp_file_path):
    """
    Removes any text-based headers from a LAS file and saves the binary content to a new file.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()

        # Locate the start of the LAS binary header (b'LASF')
        binary_start = data.find(b'LASF')
        if binary_start == -1:
            raise ValueError("No valid LAS binary header (b'LASF') found in the file.")

        # Save the binary content to a new file
        with open(temp_file_path, "wb") as f:
            f.write(data[binary_start:])
        print(f"[INFO] Non-binary header stripped. Cleaned file saved to {temp_file_path}.")
        return temp_file_path
    except Exception as e:
        print(f"[ERROR] Failed to strip text header: {e}")
        raise

def load_las_file(file_path):
    """
    Attempts to load the LAS file after verifying and cleaning if necessary.
    """
    try:
        print("[INFO] Inspecting file format...")
        file_type = inspect_file(file_path)

        if file_type == "las_binary":
            print("[INFO] Attempting to load as LAS binary file.")
            las_file = laspy.read(file_path)
            print("[SUCCESS] LAS file loaded successfully.")
            return las_file
        elif file_type == "text_with_header":
            print("[INFO] Stripping text header and attempting to load the binary content.")
            temp_file_path = file_path.replace(".las", "_cleaned.las")
            cleaned_file_path = strip_text_header(file_path, temp_file_path)
            las_file = laspy.read(cleaned_file_path)
            print("[SUCCESS] LAS file loaded successfully after cleaning.")
            return las_file
        else:
            print("[ERROR] Unsupported or unknown file format.")
            return None
    except laspy.errors.LaspyException as e:
        print(f"[ERROR] Failed to read LAS file: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        raise

# Main driver function
def main():
    file_path = "/content/drive/MyDrive/t1/Mar18_test_GroundTruth.las"  # Replace with your LAS file path
    try:
        las_file = load_las_file(file_path)
        if las_file:
            print("[INFO] File processing complete. Here are the first few points:")
            print(f"X: {las_file.x[:5]}, Y: {las_file.y[:5]}, Z: {las_file.z[:5]}")
        else:
            print("[FAILURE] Could not process the file. Please check the logs above.")
    except Exception as e:
        print(f"[CRITICAL ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

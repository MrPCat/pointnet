import os

# Define the directory and output file
directory = r"C:\Users\faars\Downloads\drive-download-20250102T121839Z-001"
output_file = os.path.join(directory, "Vaihingen3D_AugmentTraininig_Merged.pts")

# Correct filenames based on directory listing
file_names = [
    os.path.join(directory, f"Vaihingen3D_AugmentTraininig_{i}.pts") for i in range(7)
]

# Open the output file
with open(output_file, "w") as outfile:
    files_merged = 0
    for file_name in file_names:
        if os.path.exists(file_name):
            print(f"Processing file: {file_name}")
            with open(file_name, "r") as infile:
                content = infile.read()
                if content.strip():
                    outfile.write(content)
                    outfile.write("\n")  # Newline between files
                    files_merged += 1
                else:
                    print(f"Warning: {file_name} is empty.")
        else:
            print(f"Warning: {file_name} not found and will be skipped.")

print(f"{files_merged} files have been merged into {output_file}")
if files_merged == 0:
    print("No files were merged. Check file paths and content.")

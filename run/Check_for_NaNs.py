import os
import pandas as pd

# Define the path to the folder containing the CSV files.
folder_path = "../SI_Toolkit_ASF/Experiments/Experiments_03_03_2025/Recordings/Train"

# Optionally define a destination folder to move corrupted CSVs.
# If set to None, no files will be moved.
MOVE_CORRUPTED_FILES_TO = "../SI_Toolkit_ASF/Experiments/Experiments_03_03_2025_random_mu/Recordings/Corrupted"

# List only CSV files from the given folder.
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Iterate through each CSV file to check for missing (NaN) values.
for file in csv_files:
    file_path = os.path.join(folder_path, file)  # Build the full file path.

    # Load the CSV into a DataFrame. The 'comment' parameter ignores commented lines.
    df = pd.read_csv(file_path, comment="#")

    # Use isnull().values.any() to check for any NaNs in the DataFrame.
    if df.isnull().values.any():
        print(f"{file} contains NaNs.")

        # If a destination folder is provided, move the corrupted file there.
        if MOVE_CORRUPTED_FILES_TO is not None:
            # Ensure the destination directory exists.
            # os.makedirs with exist_ok=True prevents errors if the folder already exists.
            os.makedirs(MOVE_CORRUPTED_FILES_TO, exist_ok=True)

            # Construct the destination file path.
            destination_file_path = os.path.join(MOVE_CORRUPTED_FILES_TO, file)

            # Move the file by renaming its path.
            # This operation effectively relocates the file without copying data.
            os.rename(file_path, destination_file_path)
            print(f"Moved {file} to {MOVE_CORRUPTED_FILES_TO}")
    else:
        # Optionally, you could log or handle files without NaNs here.
        pass

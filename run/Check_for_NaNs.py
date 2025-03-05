import os
import pandas as pd

# Define the path to the folder containing the CSV files.
# Replace 'xxx' with your actual folder path.
folder_path = "../SI_Toolkit_ASF/Experiments/Experiments_03_03_2025_random_mu/Recordings/Validate"

# List all files in the folder and filter out only those with a .csv extension.
# os.listdir returns all entries in the directory, so we use a list comprehension to ensure we're only checking CSV files.
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# Iterate through each CSV file found in the folder.
for file in csv_files:
    # Construct the full file path by joining the folder path and the file name.
    # os.path.join handles different operating systems' path separators.
    file_path = os.path.join(folder_path, file)

    # Load the CSV file into a pandas DataFrame.
    # Using pandas here simplifies data manipulation and checking for missing values.
    df = pd.read_csv(file_path, comment="#")

    # Check if any NaN values exist in the DataFrame.
    # The isnull() method creates a DataFrame of booleans (True for NaNs).
    # The values attribute converts it to a NumPy array, and any() checks if there's any True in the array.
    if df.isnull().values.any():
        print(f"{file} contains NaNs.")
    else:
        pass
        # print(f"{file} does not contain any NaNs.")

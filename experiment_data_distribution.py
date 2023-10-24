import os
import random
import shutil
import zipfile
from datetime import datetime

# Create the ZIP archive
def csv_comprimisation(folder_path):
    
    # Zip up old trainingsdata
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    zip_filename = os.path.join(folder_path, f"csv_files_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate through all CSV files in the input folder and add them to the ZIP archive
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

    # Delete all generated CSV files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                os.remove(file_path)

    print(f"CSV files have been compressed to {zip_filename}.")
    
    

# Initalizing distribution probability
train_distribution = 0.8
test_distribution = 0.1
validate_distribution = 0.1

# Input folder with CSV files
input_folder = "./ExperimentRecordings"

# Output folders for distribution
output_folder_train = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Train"
output_folder_test = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Test"
output_folder_validate = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Validate"

# Compremising older csvs
csv_comprimisation(output_folder_test)
csv_comprimisation(output_folder_train)
csv_comprimisation(output_folder_validate)

# List all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Calculate the number of files for each category
total_files = len(csv_files)
num_files_train = int(total_files * train_distribution)
num_files_test = int(total_files * test_distribution)
num_files_validate = int(total_files * validate_distribution)

# Shuffle the order of CSV files
random.shuffle(csv_files)

# Copy files to the output folders according to the distribution
for i, file in enumerate(csv_files):
    source_path = os.path.join(input_folder, file)
    if i < num_files_train:
        destination_folder = output_folder_train
    elif i < num_files_train + num_files_test:
        destination_folder = output_folder_test
    else:
        destination_folder = output_folder_validate

    destination_path = os.path.join(destination_folder, file)
    shutil.copy(source_path, destination_path)

print("Files have been distributed successfully.")

csv_comprimisation(input_folder)
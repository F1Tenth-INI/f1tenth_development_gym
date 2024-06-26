import os
import random
import shutil
import zipfile
from datetime import datetime

# Create the ZIP archive
def csv_comprimisation(folder_path, save_folder):
    
    # Zip up old trainingsdata
    ending = os.path.basename(folder_path)
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    zip_filename = os.path.join(save_folder, f"csv_files_{ending}_{timestamp}.zip")
    
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
    
def subfolder_data_compression(dir_path):
    
    # Name of the ZIP file to be created
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    zip_filename = os.path.join(dir_path, f"csv_files_data_{timestamp}.zip")

    # Create a ZIP object for the archive
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dir_path):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                arcname = os.path.relpath(folder_path, dir_path)
                zipf.write(folder_path, arcname)

    print(f'All subfolders of {dir_path} have been compressed to {zip_filename}.')
    
    # Delete subfolders and their contents
    for folder in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            # print(f'Deleted subfolder: {folder_path}')
            
    print(f'All subfolder of the {dir_path} were deleted.')
    
# Creating directory if it doesn't exists
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
# Initalizing distribution probability
train_distribution = 0.8
test_distribution = 0.1
validate_distribution = 0.1

# Input folder with CSV files
input_folder = "./ExperimentRecordings"
past_recordings = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Past_trainings"

# Output folders for distribution
output_folder_train = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Train"
create_directory(output_folder_train)
output_folder_test = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Test"
create_directory(output_folder_test)
output_folder_validate = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Validate"
create_directory(output_folder_validate)

# Compremising older csvs and metadata
csv_comprimisation(output_folder_train, past_recordings)
csv_comprimisation(output_folder_test, past_recordings)
csv_comprimisation(output_folder_validate, past_recordings)

subfolder_data_compression(input_folder)

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

csv_comprimisation(input_folder, input_folder)

print("Files have been distributed successfully.")


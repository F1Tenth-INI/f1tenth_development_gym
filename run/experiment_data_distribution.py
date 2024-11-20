import os
import random
import shutil
import zipfile
from datetime import datetime
import sys
import time


# This file automatically distributes the data into the SI_Toolkit_ASK experiment recordings folder
# The data is distributed into the Train, Test and Validate folders according to the distribution probabilities

# Fortschrittszeilenfunktion
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

# Create the ZIP archive
def csv_comprimisation(folder_path, save_folder):
    # Zip up old trainingsdata
    ending = os.path.basename(folder_path)
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    zip_filename = os.path.join(save_folder, f"csv_files_{ending}_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate through all CSV files in the input folder and add them to the ZIP archive
        files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith(".csv")]
        total_files = len(files)
        for i, file_path in enumerate(files):
            arcname = os.path.relpath(file_path, folder_path)
            zipf.write(file_path, arcname)
            print_progress_bar(i + 1, total_files, prefix='Compressing CSVs:', suffix='Complete', length=50)

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
    subfolders = [os.path.join(root, folder) for root, dirs, _ in os.walk(dir_path) for folder in dirs]
    total_folders = len(subfolders)
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, folder_path in enumerate(subfolders):
            arcname = os.path.relpath(folder_path, dir_path)
            zipf.write(folder_path, arcname)
            print_progress_bar(i + 1, total_folders, prefix='Compressing Folders:', suffix='Complete', length=50)

    print(f'All subfolders of {dir_path} have been compressed to {zip_filename}.')
    
    # Delete subfolders and their contents
    for i, folder in enumerate(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print_progress_bar(i + 1, total_folders, prefix='Deleting Folders:', suffix='Complete', length=50)
            
    print(f'All subfolders of the {dir_path} were deleted.')
    
# Creating directory if it doesn't exist
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
# Initializing distribution probability
train_distribution = 0.8
test_distribution = 0.1
validate_distribution = 0.1

# Change to desired directory in Experiment
root_dir = "./SI_Toolkit_ASF/Experiments"
experiment_dir = "/NigalsanSim1"

# Input folder with CSV files
input_folder = "./ExperimentRecordings"
past_recordings = root_dir + experiment_dir + "/Past_trainings"
create_directory(past_recordings)

# Output folders for distribution
output_folder_train = root_dir + experiment_dir + "/Recordings/Train"
create_directory(output_folder_train)
output_folder_test = root_dir + experiment_dir + "/Recordings/Test"
create_directory(output_folder_test)
output_folder_validate = root_dir + experiment_dir + "/Recordings/Validate"
create_directory(output_folder_validate)

# Compressing older csvs and metadata
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
print_progress_bar(0, total_files, prefix='Distributing Files:', suffix='Complete', length=50)
for i, file in enumerate(csv_files):
    source_path = os.path.join(input_folder, file)
    if i < num_files_train:
        destination_folder = output_folder_train
    elif i < num_files_train + num_files_test:
        destination_folder = output_folder_test
    elif i < total_files:
        destination_folder = output_folder_validate
    else:
        break

    destination_path = os.path.join(destination_folder, file)
    shutil.copy(source_path, destination_path)
    print_progress_bar(i + 1, total_files, prefix='Distributing Files:', suffix='Complete', length=50)

csv_comprimisation(input_folder, input_folder)

print("Files have been distributed successfully.")
import pandas as pd
from io import StringIO
import os
import random

# function to shift specified columns in a CSV file by a given value
def shift_column_with_comments(input_file, output_file, shift_value, column_name):
    
    # Step 1: Read the file and separate comments and data
    with open(input_file, 'r') as f:
        lines = f.readlines()

    comment_lines = []
    data_lines = []
    
    # Assume comments start with '#' (adjust this based on your file format)
    for line in lines:
        if line.startswith('#'):
            comment_lines.append(line)
        else:
            data_lines.append(line)

    # Step 2: Load the remaining data into a DataFrame
    data = ''.join(data_lines)
    df = pd.read_csv(StringIO(data))

    # Step 3: Shift the specified columns
    for column in column_names:
        df[column] = df[column].shift(shift_value)
    
    # Drop rows where all shifted columns have NaN values
    df = df.dropna(subset=column_names)

    # Step 4: Write the comments and updated table back to a new CSV
    with open(output_file, 'w') as f:
        # First write the comments
        f.writelines(comment_lines)
        # Then write the updated table
        df.to_csv(f, index=False)
        
    print(f"Die CSV-Datei '{input_file}' wurde erfolgreich kopiert und als '{output_file}' gespeichert.")

# function to trim CSV files in a folder to the length of the smallest CSV file (excluding comments)
def trim_csvs_with_comments_in_folder(folder_path, output_folder):
    # Step 1: Find all CSV files in the given folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    # Step 2: Read all CSVs, keeping comments separate and find the length of the smallest CSV (excluding comments)
    csv_data = {}
    min_length = float('inf')
    
    for file in csv_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Separate comments from the data (assuming comments start with '#')
        comment_lines = []
        data_lines = []
        for line in lines:
            if line.startswith('#'):
                comment_lines.append(line)
            else:
                data_lines.append(line)
        
        # Load the table data into a DataFrame
        data_str = ''.join(data_lines)
        df = pd.read_csv(pd.io.common.StringIO(data_str))
        
        # Store comments and DataFrame for later processing
        csv_data[file] = {'comments': comment_lines, 'data': df}
        
        # Update the minimum length based on the table data
        if len(df) < min_length:
            min_length = len(df)

    # Step 3: Trim all CSVs to the length of the smallest CSV and write them back
    for file, content in csv_data.items():
        # Trim the DataFrame to the smallest length
        df = content['data'].iloc[:min_length]

        # Combine comments and trimmed table
        output_file = os.path.join(output_folder, os.path.basename(file))
        with open(output_file, 'w') as f:
            # Write comments first
            f.writelines(content['comments'])
            # Write the trimmed table
            df.to_csv(f, index=False)
        
        print(f'Trimmed {file} to {min_length} rows (excluding comments) and saved to {output_file}')

# Create subfolders with shift values as title suffix
def distribute_files_to_subfolders(folder_path, probabilities, shift_values, column_names):
    # Ensure the probabilities sum to 1
    if sum(probabilities) != 1:
        raise ValueError("The probabilities must sum to 1.")
    
    # Create subfolders if they don't exist
    shifted_subfolders = [os.path.join(folder_path, f'columns_shifted_by_{abs(shift_value)}') for shift_value in shift_values]
    for subfolder in shifted_subfolders:
        os.makedirs(subfolder, exist_ok=True)
    
    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Distribute files based on the given probabilities
    for file in files:
        subfolder = random.choices(shifted_subfolders, probabilities)[0]
        src_path = os.path.join(folder_path, file)
        dest_path = os.path.join(subfolder, file)
        os.rename(src_path, dest_path)
        print(f"Moved {file} to {subfolder}")
        
    for i in range(len(shift_values)):
        for filename in os.listdir(shifted_subfolders[i]):
            if filename.endswith('.csv'):
                # Construct the full file paths
                input_file = os.path.join(shifted_subfolders[i], filename)
                output_file = os.path.join(shifted_subfolders[i], filename)
                shift_column_with_comments(input_file, output_file, shift_values[i], column_names)
                print(f"the file '{filename}' was successfully copied and saved as '{output_file}'.")

# parameter to modify
folder_path = 'ExperimentRecordings'
probabilities = [0.65, 0.35]
shift_values = [-3, -4]
column_names = ['angular_control_calculated', 'translational_control_calculated']

distribute_files_to_subfolders(folder_path, probabilities, shift_values, column_names)

# if distrbute_files_to_subfolders used, then you shouldn't change output
output = [folder_path + '/'+ f'columns_shifted_by_{abs(shift_value)}' for shift_value in shift_values]
for x in output : 
    trim_csvs_with_comments_in_folder(x, x)
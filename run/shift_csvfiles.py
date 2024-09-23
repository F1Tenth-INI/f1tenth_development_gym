import pandas as pd
from io import StringIO
import os

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

# parameter to modify
input_folder = 'ExperimentRecordings/test1'
output_folder = 'ExperimentRecordings/test2'
test_folder = 'ExperimentRecordings/test3'
shift_value = -2
column_names = ['angular_control_calculated', 'translational_control_calculated']

# Loop through all CSV files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # Construct the full file paths
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)

        shift_column_with_comments(input_file, output_file, shift_value, column_names)

        print(f"the file '{filename}' was successfully copied and saved as '{output_file}'.")

trim_csvs_with_comments_in_folder(output_folder, test_folder)
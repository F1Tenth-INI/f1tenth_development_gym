import os
import pandas as pd
"""Specify the path to folder where all csv files you want to merge are, and specify whether this should be saved
as a validation or training data. Code also automatically assigns index to each data set so you know
which values come from which data"""


# Path to the folder containing the CSV files
# folder_path = 'training_data'
folder_path = "validation_data"
folder_path = "F1tenth_data/hardware_data/sysid"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

dataframes = []

# Loop over each file and process
for idx, file in enumerate(sorted(csv_files), start=1):  # start from 1
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, comment='#')
    
    # Add dataset_id column to track source
    df['dataset_id'] = idx
    
    dataframes.append(df)

# Concatenate all data vertically
combined_df = pd.concat(dataframes, ignore_index=True)


combined_df.to_csv('combined_sysid_data.csv', index=False)
# combined_df.to_csv('combined_validation_data.csv', index=False)

print("All CSV files combined successfully.")

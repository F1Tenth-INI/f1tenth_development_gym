import pandas as pd
from io import StringIO
import os
from tqdm import tqdm

def modify_csv(filename: str, outfile, argument, value):
    comments = []
    data_lines = []
    
    # Einlesen der Datei und Aufteilen in Kommentare und Datenzeilen
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                comments.append(line.strip())  # Kommentarzeilen speichern
            else:
                data_lines.append(line)  # Datenzeilen speichern
        
        # Den DataFrame aus den Datenzeilen erstellen
        data = StringIO(''.join(data_lines))
        df = pd.read_csv(data)
        rows = list(df)
        
        if argument in rows:
            
            # Get the column values for the argument
            column_values = df[argument].values
            for i in range(len(column_values)):
                column_values[i] = value
            
            # Update the column values
            df[argument] = column_values
            
            if outfile is None:
                outfile = filename
            
            # Save the updated CSV file and save comment lines
            with open(outfile, 'w') as file:
                # Kommentare zuerst schreiben
                for comment in comments:
                    file.write(comment + "\n")
                
                # Daten schreiben
                df.to_csv(file, index=False)
            
        else:
            print(f'The argument {argument} is not in the CSV file')
            return

# change if other argument is necessary to modify the folder_path and argument and the value you want to set          
# Set the folder path
folder_path = 'SI_Toolkit_ASF/Experiments/MPPI-pacejka/Recordings/Validate/'
argument = 'mu'

# Get all the files in the folder
files = os.listdir(folder_path)

# Iterate over each file in a folder
for file in tqdm(files, desc="Processing files", ascii=True):
    # Check if the file is a CSV file
    if file.endswith('.csv'):
        # Create the full file path
        file_path = os.path.join(folder_path, file)
        
        # get value from the file name last part
        value = str(int(file.split('_')[-1].split('.')[1])/10 + int(file.split('_')[-1].split('.')[0])) 
        
        # Call the modify_csv function for each file
        modify_csv(file_path, file_path, argument, value)
        


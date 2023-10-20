import os
import random
import shutil
import zipfile
from datetime import datetime

train_distributuion = 0.8
test_distribution = 0.1
validate_distribution = 0.1

# Eingabeordner mit CSV-Dateien
input_folder = "./ExperimentRecordings"

# Ausgabeordner für die Verteilung
output_folder_train = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Train"
output_folder_test = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Test"
output_folder_validate = "./SI_Toolkit_ASF/Experiments/MPPI-Imitator/Recordings/Validate"


# Liste aller CSV-Dateien im Eingabeordner
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Berechne die Anzahl der Dateien für jede Kategorie
total_files = len(csv_files)
num_files_train = int(total_files * train_distributuion)
num_files_test = int(total_files * test_distribution)
num_files_validate = int(total_files * validate_distribution)

# Zufällige Reihenfolge der CSV-Dateien
random.shuffle(csv_files)

# Kopiere die Dateien in die Ausgabeordner entsprechend der Verteilung
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

print("Dateien wurden erfolgreich verteilt.")

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
zip_filename = os.path.join(input_folder, f"csv_files_{timestamp}.zip")

# Erstelle das ZIP-Archiv
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Gehe durch alle CSV-Dateien im Eingabeordner und füge sie dem ZIP-Archiv hinzu
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, input_folder)
                zipf.write(file_path, arcname)
                
# Delete all CSV generated
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            os.remove(file_path)


print(f"CSV-Dateien wurden zu {zip_filename} komprimiert.")

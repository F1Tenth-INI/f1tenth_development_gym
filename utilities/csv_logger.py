import os
import csv
import numpy as np

from pathlib import Path
from datetime import datetime

from utilities.Settings import Settings

try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except ModuleNotFoundError:
    pass


def create_csv_file(
        csv_name=None,
        keys=None,
        path_to_experiment_recordings=None,
        title=None,
        header=None
):

    # Make folder to save data (if not yet existing)
    Path(path_to_experiment_recordings).mkdir(parents=True, exist_ok=True)

    csv_filepath = os.path.join(path_to_experiment_recordings, csv_name)
    # If such file exists, append index to the end (do not overwrite)
    csv_filepath = csv_append_index_if_file_exists(csv_filepath)

    with open(csv_filepath, "a", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(header)

        writer.writerow(keys)

    return csv_filepath


def csv_append_index_if_file_exists(csv_filepath):
    """
    Checks if the file at the given file exists.
    If it does, appends an index to the file name and checks again.
    Repeats until a non-existing file name is found.
    Returns the new file name.
    Args:
        csv_filepath:

    Returns:
        csv_filepath:
    """
    index = 1
    logpath_new = csv_filepath
    while True:
        if os.path.isfile(logpath_new):
            logpath_new = csv_filepath[:-4]
        else:
            csv_filepath = logpath_new
            break
        logpath_new = logpath_new + '-' + str(index) + '.csv'
        index += 1
    return csv_filepath


def create_csv_file_name(Settings, csv_name=None):
    if csv_name is None or csv_name == '':

        if Settings.CONTROLLER is None:
            controller_name = 'None'
        else:
            controller_name = Settings.CONTROLLER
        
        reverse_string = 'reverse_' if Settings.REVERSE_DIRECTION else ''
            
        dataset_name = Settings.MAP_NAME + '_' + reverse_string + controller_name + '_' + str(
            int(1 / Settings.TIMESTEP_CONTROL)) + 'Hz' + '_vel_' + str(
            Settings.GLOBAL_WAYPOINT_VEL_FACTOR) + '_noise_c' + str(Settings.NOISE_LEVEL_CONTROL) + '_mu_' + str(
            Settings.SURFACE_FRICTION) + '_mu_c_' + str(Settings.FRICTION_FOR_CONTROLLER) + '_'
        timestamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        csv_file_name = timestamp + '_' + Settings.DATASET_NAME + '_' + str(Settings.RECORDING_INDEX) + '_' + dataset_name + '.csv'
    else:
        if csv_name[-4:] != '.csv':
            csv_name += '.csv'
        csv_file_name = csv_name

    return csv_file_name




def create_csv_header(Settings, controller_name, dt):

    # Initialize a list to store all rows to be written to the CSV header
    header = []

    # Append rows to the list
    header.append([
                        f'# This is F1TENTH simulation from {datetime.now().strftime("%d.%m.%Y")} at time {datetime.now().strftime("%H:%M:%S")}'])

    try:
        repo = Repo()
        git_revision = repo.head.object.hexsha
    except:
        git_revision = 'unknown'

    header.append([f'# Done with git-revision: {git_revision}'])

    header.append([f'# Starting position :', *np.array(Settings.STARTING_POSITION).tolist()])

    header.append([f'# Timestep: {dt} s'])

    header.append([f'# Speedfactor {Settings.GLOBAL_WAYPOINT_VEL_FACTOR}'])

    header.append([f'# Controller: {controller_name}'])

    header.append(['#'])
    header.append(['# Data:'])

    return header

def augment_csv_header(s: str, p: str, i: int = 0, after_header: bool = False):
    """
    Inserts the string `s` at the specified position in the CSV file at path `p`.
    If `i` is greater than the number of lines, the string will be added at the end.
    If `after_header` is True, the string will be added after the last header (lines starting with '#  ').
    Adds '#  ' at the beginning of the string before saving it.

    Args:
        s (str): The string to insert into the CSV.
        p (str): Path to the CSV file.
        i (int): The line number after which to insert the string (unless `after_header` is True).
        after_header (bool): If True, inserts the string after the last header line (i does not matter).
    """
    # Prepend "#  " to the string
    s = f"# {s}"

    # Read the contents of the file
    try:
        with open(p, mode='r', newline='') as file:
            lines = file.readlines()

        # Determine the correct insertion point
        if after_header:
            # Find the last header line (lines starting with "#  ")
            last_header_index = max(i for i, line in enumerate(lines) if line.startswith('# '))
            i = last_header_index

        # If i is larger than the available lines, append at the end
        if i >= len(lines):
            lines.append(s + '\n')
        else:
            # Insert the string after line `i`
            lines.insert(i + 1, s + '\n')

        # Write the updated contents back to the file
        with open(p, mode='w', newline='') as outfile:
            outfile.writelines(lines)
    except FileNotFoundError:
        # If the file does not exist, create it and write the string
        print(f"Cant save laptimes after crash. File {p} does not exist.")
        
def augment_csv_header_with_laptime(laptimes, csv_filepath):
    """
    Args:
        laptimes: Array of lap times
        csv_filepath: Path to the CSV file

    Returns: String to be used as part of csv header.
    """

    # Convert lap_times array to string in the format [1, 2, ...]
    lap_times_str = "[" + ", ".join([str(round(time, 3)) for time in laptimes]) + "]"
    
    lap_time_string = f"Lap times: {lap_times_str}"
    
    augment_csv_header(lap_time_string, csv_filepath, i=6)
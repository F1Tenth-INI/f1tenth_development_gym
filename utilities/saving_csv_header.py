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


def create_csv_header(path_to_recordings,
                      controller_name,
                      dt,
                      csv_name=None):

    # Make folder to save data (if not yet existing)
    Path(path_to_recordings).mkdir(parents=True, exist_ok=True)

    # Set path where to save the data
    if csv_name is None or csv_name == '':

        dataset_name = Settings.MAP_NAME + '_' + Settings.CONTROLLER + '_' + str(
            int(1 / Settings.TIMESTEP_CONTROL)) + 'Hz' + '_vel_' + str(
            Settings.GLOBAL_WAYPOINT_VEL_FACTOR) + '_noise_c' + str(Settings.NOISE_LEVEL_CONTROL) + '_mu_' + str(
            Settings.SURFACE_FRICITON)
        experiment_name = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + Settings.DATASET_NAME + '_' + dataset_name
        csv_filepath = os.path.join(path_to_recordings, experiment_name + '.csv')
    else:
        csv_filepath = csv_name
        if csv_name[-4:] != '.csv':
            csv_filepath += '.csv'

        # If such file exists, append index to the end (do not overwrite)
        net_index = 1
        logpath_new = csv_filepath
        while True:
            if os.path.isfile(logpath_new):
                logpath_new = csv_filepath[:-4]
            else:
                csv_filepath = logpath_new
                break
            logpath_new = logpath_new + '-' + str(net_index) + '.csv'
            net_index += 1

    # Write the header of .csv file
    with open(csv_filepath, "a", newline='') as outfile:
        writer = csv.writer(outfile)

        writer.writerow(['# ' + 'This is F1TENTH simulation from {} at time {}'
                        .format(datetime.now().strftime('%d.%m.%Y'), datetime.now().strftime('%H:%M:%S'))])
        try:
            repo = Repo()
            git_revision = repo.head.object.hexsha
        except:
            git_revision = 'unknown'
        writer.writerow(['# ' + 'Done with git-revision: {}'
                        .format(git_revision)])

        writer.writerow(['# Starting position :', *np.array(Settings.STARTING_POSITION).tolist()])

        writer.writerow(['# Timestep: {} s'.format(dt)])

        writer.writerow(['# Speedfactor {}'.format(Settings.GLOBAL_WAYPOINT_VEL_FACTOR)])

        writer.writerow(['# Controller: {}'.format(controller_name)])

        writer.writerow(['#'])
        writer.writerow(['# Data:'])

    return csv_filepath, experiment_name


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
    with open(p, mode='r', newline='') as file:
        reader = csv.reader(file)
        lines = list(reader)

    # Determine the correct insertion point
    if after_header:
        # Find the last header line (lines starting with "#  ")
        last_header_index = max(i for i, line in enumerate(lines) if line and line[0].startswith('#  '))
        i = last_header_index

    # If i is larger than the available lines, append at the end
    if i >= len(lines):
        lines.append([s])
    else:
        # Insert the string after line `i`
        lines.insert(i, [s])

    # Write the updated contents back to the file
    with open(p, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


def augment_csv_header_with_laptime(laptime, obs, settings, csv_filepath):
    """

    Args:
        laptime:
        obs:
        settings:

    Returns: String to be used as part of csv header.

    """
    # Adding Laptime to the recordings
    if laptime - 3 * settings.TIMESTEP_CONTROL <= obs['lap_times'][0]:  # FIXME: @Nigalsan, why is there 3 here?
        lap_time_string = str(round(laptime, 3)) + ' s (uncompleted)'
    else:
        lap_time_string = str(round(obs['lap_times'][0], 3)) + ' s'
    lap_time_string = str(settings.STOP_TIMER_AFTER_N_LAPS) + '-laptime: ' + lap_time_string

    augment_csv_header(lap_time_string, csv_filepath, i=6)
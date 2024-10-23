import os
import shutil

import pandas as pd

import matplotlib.pyplot as plt

from utilities.Settings import Settings
from utilities.ExperimentAnalyzer import ExperimentAnalyzer


def move_csv_to_crash_folder(csv_filepath, path_to_plots):
    import os
    import shutil

    path_to_experiment_recordings, _ = os.path.split(csv_filepath)

    # Check if the crash directory exists, if not create it
    dir_path = os.path.join(path_to_experiment_recordings, "crashes")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Move the file to the crash directory
    shutil.move(csv_filepath, dir_path)

    # Move the plot directory to the crash directory
    if path_to_plots is not None and os.path.isdir(path_to_plots):
        shutil.move(path_to_plots, dir_path)


def save_experiment_data(csv_filepath):
    """
    Copy relevant settings to the data folder and analyze experiment
    """

    path_to_experiment_recordings, experiment_name = os.path.split(csv_filepath)
    experiment_name = experiment_name[:-4]  # Remove .csv

    save_path = csv_filepath[:-4] + "_data"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df = pd.read_csv(csv_filepath, comment='#')

    # read out all states

    times = df['time'].to_numpy()[1:]

    # Plot time needed for calculation
    dts = []
    for i in range(15, len(times) - 1):
        last_row = times[i - 1]
        row = times[i]
        delta = float(row) - float(last_row)
        dts.append(delta)

    plt.plot(dts)
    plt.savefig(save_path + "/dt_control.png")

    # Copy Settings and configs
    config_sage_path = os.path.join(save_path, "configs")

    if not os.path.exists(config_sage_path):
        os.mkdir(config_sage_path)
    shutil.copy("Control_Toolkit_ASF/config_controllers.yml", config_sage_path)
    shutil.copy("Control_Toolkit_ASF/config_cost_function.yml", config_sage_path)
    shutil.copy("Control_Toolkit_ASF/config_optimizers.yml", config_sage_path)
    shutil.copy("utilities/Settings.py", config_sage_path)

    shutil.copy("utilities/Settings.py", config_sage_path)

    with open(config_sage_path + '/Settings_applied.json', 'w') as f:
        f.write(str(Settings.__dict__))

    shutil.copy(os.path.join(Settings.MAP_PATH, Settings.MAP_NAME + ".png"), config_sage_path)
    shutil.copy(os.path.join(Settings.MAP_PATH, Settings.MAP_NAME + "_wp.csv"), config_sage_path)

    try:
        shutil.copy(os.path.join(Settings.MAP_PATH, Settings.MAP_NAME + "_wp_reverse.csv"), config_sage_path)
    except:
        pass

    try:
        shutil.copy(os.path.join(Settings.MAP_PATH, "speed_scaling.yaml"), config_sage_path)
    except:
        pass

    try:
        experiment_analyzer = ExperimentAnalyzer(experiment_name, experiment_path=path_to_experiment_recordings)
        experiment_analyzer.plot_experiment()
    except Exception as e:
        print(f'Warning: experiment analysis did not work. Error: {e}')

    return save_path
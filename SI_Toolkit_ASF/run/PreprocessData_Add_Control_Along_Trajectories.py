"""
This script calculate the control signal along the prerecorded trajectories.
There need to be no relationship between the controller with which the trajectories were recorded
and the controller which is used here to calculate the control signal.

In Pycharm to get the progress bars display correctly you need to set
"Emulate terminal in output console" in the run configuration.

"""

import os
import glob
from SI_Toolkit.data_preprocessing import transform_dataset

from utilities.state_utilities import control_limits_low, control_limits_high
from SI_Toolkit_ASF.ToolkitCustomization.control_along_trajectories_car_helpers import controller_creator, df_modifier
import argparse

get_files_from_folder_root = './ExperimentRecordings/Experiments_03_03_2025/'

get_file_name = 'xxx.csv'  # Only used if no index is provided

save_files_to = './ExperimentRecordings/Experiments_03_03_2025_random_mu/'

controller_config = {
    "controller_name": "mpc",
    "state_components": 'state',
    "environment_attributes_dict": {  # keys are names used by controller, values the csv column names
        "lidar": "lidar",
        "next_waypoints": "next_waypoints",
        "mu": "mu_random_uniform_0.3_1.1",
    },
}

controller_output_variable_name = ['angular_control_random_mu', 'translational_control_random_mu']


def args_fun():
    parser = argparse.ArgumentParser(description='Generate Car data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--secondary_experiment_index', default=-1, type=int,
                        help='Additional index to the experiment folder (ML Pipeline mode) or file (otherwise) name. -1 to skip.')


    args = parser.parse_args()

    if args.secondary_experiment_index == -1:
        args.secondary_experiment_index = None

    digits = 3
    if args.secondary_experiment_index is not None:
        formatted_index_from_terminal = f"{args.secondary_experiment_index:0{digits}d}"
    else:
        formatted_index_from_terminal = None

    return formatted_index_from_terminal


formatted_index = args_fun()
if formatted_index is not None:
    pattern = os.path.join(get_files_from_folder_root, "*_" + formatted_index + ".csv")
    matching_files = glob.glob(pattern)
    get_files_from = matching_files[0]
else:
    get_files_from = os.path.join(get_files_from_folder_root, get_file_name)

control_limits = (control_limits_low, control_limits_high),

if __name__ == '__main__':
    transform_dataset(get_files_from, save_files_to,
                      transformation='add_control_along_trajectories',
                      controller_config=controller_config,
                      controller_creator=controller_creator,
                      df_modifier=df_modifier,
                      controller_output_variable_name=controller_output_variable_name,
                      integration_num_evals=4,
                      save_output_only=False,
                      )

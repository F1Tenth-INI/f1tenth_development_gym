"""
This script calculate the control signal along the prerecorded trajectories.
There need to be no relationship between the controller with which the trajectories were recorded
and the controller which is used here to calculate the control signal.

In Pycharm to get the progress bars display correctly you need to set
"Emulate terminal in output console" in the run configuration.

"""


from SI_Toolkit.data_preprocessing import transform_dataset

import numpy as np
from utilities.state_utilities import STATE_VARIABLES, control_limits_low, control_limits_high

import argparse

def args_fun():
    parser = argparse.ArgumentParser(description='Generate Car data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--secondary_experiment_index', default=-1, type=int,
                        help='Additional index to the experiment folder (ML Pipeline mode) or file (otherwise) name. -1 to skip.')


    args = parser.parse_args()

    if args.secondary_experiment_index == -1:
        args.secondary_experiment_index = None

    digits = 3
    if args.secondary_experiment_index is not None:
        formatted_index = f"{args.secondary_experiment_index:0{digits}d}"
    else:
        formatted_index = None

    return formatted_index

formatted_index = args_fun()
if formatted_index is not None:
    get_files_from = f'../Experiment_Recordings/Experiment_16_11_2024_pole_L_and_m_informed/Experiment-{formatted_index}.csv'
else:
    get_files_from = './SI_Toolkit_ASF/Experiments/Euler_RCA1_slip_little_noise/2025-02-05_17-13-40_Euler_RCA1_slip_little_noise_2_RCA1_mpc_50Hz_vel_0.5_noise_c[0.1, 0.1]_mu_0.45.csv'

save_files_to = './SI_Toolkit_ASF/Experiments'

controller_name = "mpc"
control_limits = (control_limits_low, control_limits_high),
controller = {
    "controller_name": "mpc",
    "environment_name": "Car",
    "control_limits": control_limits,
    "state_components": STATE_VARIABLES,
    "environment_attributes_dict": {  # keys are names used by controller, values the csv column names
        "lidar_points": "LIDAR",
        "next_waypoints": "WPT",
        "mu": "mu",
    },
}

controller_output_variable_name = 'Q_new'

if __name__ == '__main__':
    transform_dataset(get_files_from, save_files_to, transformation='add_control_along_trajectories_car',
                      controller_name=controller_name, controller_output_variable_name=controller_output_variable_name,
                      integration_num_evals=4,
                      save_output_only=True,
                      )

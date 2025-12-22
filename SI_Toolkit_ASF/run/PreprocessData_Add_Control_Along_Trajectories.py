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

# ======================================================================================
# INPUT / OUTPUT (hardcoded; no CLI flags except --secondary_experiment_index as below)
# --------------------------------------------------------------------------------------
# `get_files_from_folder_root` can be EITHER:
# - a folder path: processes all `*.csv` in that folder (and subfolders, per SI_Toolkit)
# - a single CSV file path: processes only that file
#
# When `get_files_from_folder_root` is a folder, you can additionally choose:
# - `get_file_name = None` (or '') to process ALL files in the folder
# - `get_file_name = 'some.csv'` to process ONLY that file (within the folder)
#
# When `--secondary_experiment_index` is provided (e.g. 7 -> "007"), the script will
# pick exactly one file matching `*_<idx>.csv` inside the folder.
# ======================================================================================
get_files_from_folder_root = './Mu_tests/2025-12-22_13-57-47_Experiments_22_12_2025_0_RCA2_mpc_100Hz_vel_1.0_noise_c[0.0, 0.0]_mu_0.3_mu_c_0.3_.csv'  # folder OR a single CSV path
get_file_name = None  # Only used when `get_files_from_folder_root` is a folder and no index is provided

# `save_files_to`:
# - for folder input: MUST be a folder (output root)
# - for single-file input: can be a folder OR a full output CSV path
save_files_to = './Mu_tests_mpc/2025-12-22_13-57-47_Experiments_22_12_2025_0_RCA2_mpc_100Hz_vel_1.0_noise_c[0.0, 0.0]_mu_0.3_mu_c_0.3_.csv'

controller_config = {
    "controller_name": "mpc",
    "state_components": 'state',
    "environment_attributes_dict": {  # keys are names used by controller, values the csv column names
        "lidar": "lidar",
        "next_waypoints": "next_waypoints",
        # Evaluate control for a whole vector of mu values at each row:
        # creates output columns like *_mu_0p3, *_mu_0p35, ... *_mu_1p1
        "mu": "mu_regular_grid_0.3_1.1_0.05",
    },
}

controller_output_variable_name = ['angular_control_mpc', 'translational_control_mpc']


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

def _is_csv_path(p: str) -> bool:
    return isinstance(p, str) and p.lower().endswith('.csv')


def _resolve_io_paths(input_root: str, output_root: str, formatted_index_: str | None, file_name_: str | None):
    """
    Returns:
        get_files_from (str): file OR directory path accepted by SI_Toolkit.transform_dataset
        save_dir (str|None): directory path accepted by SI_Toolkit.transform_dataset
        post_rename (tuple|None): (tmp_written_path, desired_final_path) if a rename is needed for single-file output
    """
    input_root = os.path.expanduser(input_root)
    output_root = os.path.expanduser(output_root) if output_root is not None else None

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    input_is_file = os.path.isfile(input_root)
    input_is_dir = os.path.isdir(input_root)
    if not (input_is_file or input_is_dir):
        raise ValueError(f"Input path exists but is neither file nor directory: {input_root}")

    # Decide which input(s) to pass to transform_dataset
    if formatted_index_ is not None:
        if input_is_file:
            raise ValueError("--secondary_experiment_index can only be used when get_files_from_folder_root is a folder.")
        pattern = os.path.join(input_root, f"*_{formatted_index_}.csv")
        matching_files = sorted(glob.glob(pattern))
        if not matching_files:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files matched pattern: {pattern}\nMatches:\n- " + "\n- ".join(matching_files))
        get_files_from_ = matching_files[0]
    else:
        if input_is_file:
            get_files_from_ = input_root
        else:
            # Folder mode: either process all files, or one explicit filename inside the folder.
            if file_name_ is None or str(file_name_).strip() == '':
                get_files_from_ = input_root  # let transform_dataset discover all csv files
            else:
                get_files_from_ = os.path.join(input_root, file_name_)

    # Decide output directory (transform_dataset expects a directory)
    post_rename = None
    if output_root is None:
        save_dir_ = None
    else:
        if input_is_dir and _is_csv_path(output_root):
            raise ValueError("For folder input, save_files_to must be a folder (not a .csv file path).")

        if _is_csv_path(output_root):
            # Single-file convenience: allow specifying full output file path
            if not os.path.isfile(get_files_from_):
                raise ValueError("save_files_to points to a .csv file, but the selected input is not a single file.")
            desired_final = output_root
            save_dir_ = os.path.dirname(desired_final) or '.'
            tmp_written = os.path.join(save_dir_, os.path.basename(get_files_from_))
            post_rename = (tmp_written, desired_final)
        else:
            save_dir_ = output_root

    return get_files_from_, save_dir_, post_rename


get_files_from, save_dir, rename_after = _resolve_io_paths(
    get_files_from_folder_root,
    save_files_to,
    formatted_index,
    get_file_name,
)

control_limits = (control_limits_low, control_limits_high),

if __name__ == '__main__':
    transform_dataset(get_files_from, save_dir,
                      transformation='add_control_along_trajectories',
                      controller_config=controller_config,
                      controller_creator=controller_creator,
                      df_modifier=df_modifier,
                      controller_output_variable_name=controller_output_variable_name,
                      integration_num_evals=4,
                      save_output_only=False,
                      )

    # Optional rename if user provided a full output CSV path for single-file processing.
    if rename_after is not None:
        tmp_written, desired_final = rename_after
        os.makedirs(os.path.dirname(desired_final) or '.', exist_ok=True)
        if not os.path.exists(tmp_written):
            raise FileNotFoundError(
                f"Expected output file was not created: {tmp_written}\n"
                f"(This is the default output name derived from the input file.)"
            )
        if os.path.abspath(tmp_written) != os.path.abspath(desired_final):
            os.replace(tmp_written, desired_final)

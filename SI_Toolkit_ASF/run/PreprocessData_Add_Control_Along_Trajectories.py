"""
This script calculates the control signal along prerecorded trajectories.
There need to be no relationship between the controller with which the trajectories were recorded
and the controller which is used here to calculate the control signal.

In Pycharm to get the progress bars display correctly you need to set
"Emulate terminal in output console" in the run configuration.

=== COUNTERFACTUAL TRAJECTORY MODE ===
Set COUNTERFACTUAL_ENABLED = True to enable counterfactual trajectory analysis. This mode:
1. Uses backward predictor to reconstruct alternative trajectories for various parameter values
2. Feeds counterfactual histories to stateful controllers to test behavior
3. Supports multi-rate: fine dt for dynamics, coarse DT for controller
4. Supports output stride: process every N-th timestep

Configure options in COUNTERFACTUAL_CONFIG dictionary.

"""

import os
import glob

from SI_Toolkit.data_preprocessing import transform_dataset

from utilities.state_utilities import control_limits_low, control_limits_high
from utilities.Settings import Settings
from SI_Toolkit_ASF.ToolkitCustomization.control_along_trajectories_car_helpers import controller_creator, df_modifier
import argparse
import pandas as pd

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
# Default paths for quick verification runs (can be edited as needed).
DEFAULT_INPUT_ROOT = '/Users/marcinpaluch/PycharmProjects/f1tenth_development_gym/ExperimentRecordings/2026-01-01_14-12-37_Recording1_0_RCA1_neural_50Hz_vel_1.0_noise_c[0.05, 0.1]_mu_None_mu_c_None_.csv'  # folder OR a single CSV path
DEFAULT_OUTPUT_ROOT = './ExperimentRecordings/_offline_verify/'  # folder for outputs
DEFAULT_FILE_NAME = None  # only used when input_root is a folder and -i is not provided

# If > 0 and the resolved input is a single CSV, create a truncated test CSV (first N rows) and process that.
DEFAULT_MAX_ROWS = 0  # 0 = process whole file (no truncation)

# Fake history via backward predictor (history forging). If True, offline replay will prime stateful controllers
# using a synthetic past based on the recorded trajectory.
DEFAULT_FORGE_HISTORY = True
# options: optimizer, network, hybrid, off
DEFAULT_FORGED_HISTORY_MODE = "optimizer"
# How to generate the "fake past" used to prime the RNN:
# - "backward": use BackwardPredictor output (but fed with RECORDED online controls)
# - "oracle": replay true past states/waypoints/lidar from the CSV (to isolate RNN convergence issues)
DEFAULT_FORGE_PAST_SOURCE = "backward"  # "backward" | "oracle"
# When to reset RNN hidden states before warmup:
# - "every_step": reset to zeros at every step (current behavior, sliding window)
# - "first_only": reset only at the first forged step (H), then let hidden state accumulate
DEFAULT_FORGE_RESET_MODE = "every_step"  # "every_step" | "first_only"
# Length of history window for warmup (number of past steps to replay before computing current control)
# Try 50, 100, 150, 200, etc. to see if longer warmup converges to online results
DEFAULT_FORGE_HISTORY_LENGTH = 100
# Output stride: only compute offline control for every Nth row (1 = all rows, 200 = every 200th row)
# Useful for quick tests on long trajectories
DEFAULT_OUTPUT_STRIDE = 50

# `save_files_to`:
# - for folder input: MUST be a folder (output root)
# - for single-file input: can be a folder OR a full output CSV path
#
# The default output is a folder; output filenames follow the input filenames.

DEFAULT_MU_MIN = 0.3
DEFAULT_MU_MAX = 1.1
DEFAULT_MU_STEP = 0.05
# If True, compute control for each mu in [MU_MIN, MU_MAX] with MU_STEP.
# Output columns will be suffixed with mu values, e.g. angular_control_offline_mu_0p30
DEFAULT_MU_SWEEP_ENABLED = True

controller_output_variable_name = ['angular_control_offline', 'translational_control_offline']

# ======================================================================================
# COUNTERFACTUAL TRAJECTORY MODE CONFIGURATION (hardcoded)
# ======================================================================================
COUNTERFACTUAL_ENABLED = False  # Set True to enable counterfactual trajectory analysis
COUNTERFACTUAL_CONFIG = {
    'parameter_name': 'mu',        # Name of the parameter to vary (e.g., 'mu', 'mass', 'inertia')
    'parameter_values': None,      # List of values to test, e.g. [0.3, 0.5, 0.7, 0.9]. None = auto
    'horizon_fine': 200,           # Horizon (timesteps) for backward predictor at fine dt
    'controller_dt': None,         # Controller timestep (seconds). None = use data dt (no subsampling)
    'output_stride': 1,            # Process every N-th timestep to reduce output size
    'traj_weight': 0.1,            # Weight for trajectory regularization
    'run_tests_only': False,       # Run alignment tests only (no processing)
    'verbose': False,              # Print verbose debug info
}


def args_fun():
    parser = argparse.ArgumentParser(
        description='Calculate control signal along prerecorded trajectories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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

get_files_from_folder_root = DEFAULT_INPUT_ROOT
save_files_to = DEFAULT_OUTPUT_ROOT
get_file_name = DEFAULT_FILE_NAME

mu_min = float(DEFAULT_MU_MIN)
mu_max = float(DEFAULT_MU_MAX)
mu_step = float(DEFAULT_MU_STEP)
if mu_step <= 0:
    raise ValueError(f"DEFAULT_MU_STEP must be > 0, got {mu_step}")
if mu_max < mu_min:
    raise ValueError(f"DEFAULT_MU_MAX must be >= DEFAULT_MU_MIN, got mu_min={mu_min}, mu_max={mu_max}")

# Build mu attribute: either single value from CSV column, or regular_grid sweep
if DEFAULT_MU_SWEEP_ENABLED:
    # regular_grid pattern: mu_regular_grid_{min}_{max}_{step}
    # This generates outputs for each mu value, e.g. angular_control_offline_mu_0p30
    _mu_attr = f"mu_regular_grid_{mu_min}_{mu_max}_{mu_step}"
    print(f"[MU SWEEP] Enabled: {_mu_attr}")
else:
    # Single value from CSV column 'mu'
    _mu_attr = "mu"

controller_config = {
    # Match online Settings.CONTROLLER == 'neural' (NeuralNetImitatorPlanner).
    "controller_name": "neural",
    "state_components": 'state',
    "environment_attributes_dict": {
        "lidar": "lidar",
        "next_waypoints": "next_waypoints",
        # mu: either from CSV column or regular_grid sweep (see DEFAULT_MU_SWEEP_ENABLED)
        "mu": _mu_attr,
        # For forged-history mode B: feed the BackwardPredictor with APPLIED controls (not raw network output).
        # angular_control/translational_control = actually applied to car (includes noise, clipping, etc.)
        # angular_control_calculated/translational_control_calculated = raw network output (for comparison only)
        "angular_control": "angular_control",
        "translational_control": "translational_control",
        "angular_control_calculated": "angular_control_calculated",
        "translational_control_calculated": "translational_control_calculated",
    },
}

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

    # Apply history-forging settings (mirrors online Settings knobs).
    Settings.FORGE_HISTORY = bool(DEFAULT_FORGE_HISTORY)
    Settings.FORGED_HISTORY_MODE = str(DEFAULT_FORGED_HISTORY_MODE)
    Settings.FORGE_PAST_SOURCE = str(DEFAULT_FORGE_PAST_SOURCE)
    Settings.FORGE_RESET_MODE = str(DEFAULT_FORGE_RESET_MODE)
    Settings.FORGE_HISTORY_LENGTH = int(DEFAULT_FORGE_HISTORY_LENGTH)
    Settings.OUTPUT_STRIDE = int(DEFAULT_OUTPUT_STRIDE)

    # Optional: create a truncated test CSV for faster iteration.
    if DEFAULT_MAX_ROWS and DEFAULT_MAX_ROWS > 0:
        if os.path.isdir(get_files_from):
            raise ValueError("--max_rows is only supported when the resolved input is a single CSV file (not a folder).")
        if save_dir is None:
            raise ValueError("--max_rows requires an output folder (DEFAULT_OUTPUT_ROOT must not be None).")
        df_in = pd.read_csv(get_files_from, comment='#')
        df_in = df_in.iloc[: DEFAULT_MAX_ROWS].copy()
        truncated_name = f"_truncated_{DEFAULT_MAX_ROWS}_" + os.path.basename(get_files_from)
        truncated_path = os.path.join(save_dir, truncated_name)
        os.makedirs(save_dir, exist_ok=True)
        df_in.to_csv(truncated_path, index=False)
        print(f"[max_rows] Wrote truncated test file: {truncated_path}")
        get_files_from = truncated_path
    
    # === COUNTERFACTUAL TRAJECTORY MODE ===
    if COUNTERFACTUAL_ENABLED or COUNTERFACTUAL_CONFIG.get('run_tests_only', False):
        from SI_Toolkit.General.counterfactual_trajectory_helpers import (
            add_control_with_counterfactual_trajectories,
            run_alignment_tests,
        )
        
        cfg = COUNTERFACTUAL_CONFIG
        param_name = cfg.get('parameter_name', 'mu')
        
        print("\n" + "=" * 60)
        print("COUNTERFACTUAL TRAJECTORY MODE")
        print("=" * 60)
        print(f"  Parameter: {param_name}")
        print(f"  Input: {get_files_from}")
        print(f"  Output: {save_dir}")
        print(f"  Horizon (fine): {cfg['horizon_fine']} timesteps")
        print(f"  Controller dt: {cfg['controller_dt']}")
        print(f"  Output stride: {cfg['output_stride']}")
        print(f"  Trajectory weight: {cfg['traj_weight']}")
        print(f"  Parameter values: {cfg['parameter_values'] or 'auto'}")
        
        if cfg.get('run_tests_only', False):
            # Test-only mode: load one file and run alignment tests
            import pandas as pd
            test_file = get_files_from
            if os.path.isdir(test_file):
                csv_files = glob.glob(os.path.join(test_file, '*.csv'))
                if csv_files:
                    test_file = csv_files[0]
                else:
                    raise FileNotFoundError(f"No CSV files found in {test_file}")
            
            print(f"\n=== Running alignment tests on: {test_file} ===")
            df = pd.read_csv(test_file)
            
            # Get dt from file
            from SI_Toolkit.load_and_normalize import get_sampling_interval_from_datafile
            dt = get_sampling_interval_from_datafile(df, test_file) or 0.01
            
            results = run_alignment_tests(
                df,
                horizon_fine=cfg['horizon_fine'],
                dt=dt,
                controller_dt=cfg['controller_dt'],
                parameter_name=param_name,
                verbose=cfg.get('verbose', False) or True,
            )
            
            if results['all_passed']:
                print("\n✓ ALL ALIGNMENT TESTS PASSED")
            else:
                print("\n✗ SOME TESTS FAILED")
            exit(0)
        
        # Full counterfactual trajectory processing
        transform_dataset(
            get_files_from, save_dir,
            transformation=add_control_with_counterfactual_trajectories,
            controller_config=controller_config,
            controller_creator=controller_creator,
            df_modifier=df_modifier,
            controller_output_variable_name=controller_output_variable_name,
            # Counterfactual trajectory options from config
            parameter_name=param_name,
            parameter_values=cfg['parameter_values'],
            horizon_fine=cfg['horizon_fine'],
            controller_dt=cfg['controller_dt'],
            output_stride=cfg['output_stride'],
            traj_weight=cfg['traj_weight'],
            continuation=True,
            verbose=cfg.get('verbose', False),
            run_verification=True,
        )
    
    # === STANDARD MODE ===
    else:
        transform_dataset(get_files_from, save_dir,
                          transformation='add_control_along_trajectories',
                          controller_config=controller_config,
                          controller_creator=controller_creator,
                          df_modifier=df_modifier,
                          controller_output_variable_name=controller_output_variable_name,
                          integration_num_evals=4,
                          # For mu_regular_grid_* evaluation: run full trajectory per mu to improve MPC convergence.
                          regular_grid_eval_order='by_grid',
                          regular_grid_reset_between_values=True,
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

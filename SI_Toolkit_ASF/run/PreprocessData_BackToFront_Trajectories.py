"""
This script calculates backward-then-forward trajectories from prerecorded test data.

It processes CSV files and generates trajectories by:
1. Predicting backward from seed states
2. Reconstructing forward from the furthest backward point

Output: CSV files with all trajectories, each row labeled with trajectory_index and phase.

In Pycharm to get the progress bars display correctly you need to set
"Emulate terminal in output console" in the run configuration.
"""

import os
import glob
from SI_Toolkit.data_preprocessing import transform_dataset
import argparse

# =============================================================================
# CONFIGURATION - Edit these variables for your use case
# =============================================================================

# Input files
get_files_from_folder_root = './SI_Toolkit_ASF/Experiments/Experiments_19_11_2025/Recordings/'
get_file_name = None  # Only used if no index is provided

# Output folder
save_files_to = './SI_Toolkit_ASF/Experiments/Experiments_19_11_2025_BackToFront/Recordings/'

# Predictor specifications
backward_predictor_specification = 'Dense-9IN-32H1-32H2-8OUT-1'
forward_predictor_specification = 'ODE'

# Test parameters
test_horizon = 30  # Prediction horizon
dataset_sampling_dt = 0.02  # Sampling time step (seconds)
verbose = False  # Set to True for detailed diagnostic output
max_batch_size = 512  # Fixed batch size for predictor reuse (increase if you have larger datasets)

# Control parameter randomization (optional)
randomize_param = 'mu'  # Name of control parameter to randomize (e.g., 'mu'). Set to None to disable
param_range = (0.3, 1.1)  # Range for random parameter sampling (min, max)
random_seed = 1  # Random seed for reproducibility (None = use file name hash)

# =============================================================================
# COMMAND LINE ARGUMENTS (optional)
# =============================================================================

def args_fun():
    parser = argparse.ArgumentParser(
        description='Generate backward-forward trajectories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--secondary_experiment_index',
        default=-1,
        type=int,
        help='Additional index to the file name. -1 to skip and use get_file_name.'
    )
    
    args = parser.parse_args()
    
    if args.secondary_experiment_index == -1:
        args.secondary_experiment_index = None
    
    digits = 3
    if args.secondary_experiment_index is not None:
        formatted_index_from_terminal = f"{args.secondary_experiment_index:0{digits}d}"
    else:
        formatted_index_from_terminal = None
    
    return formatted_index_from_terminal


# =============================================================================
# FILE SELECTION
# =============================================================================

formatted_index = args_fun()
if formatted_index is not None:
    # Search for file with specific index
    pattern = os.path.join(get_files_from_folder_root, "*_" + formatted_index + ".csv")
    matching_files = glob.glob(pattern)
    if matching_files:
        get_files_from = matching_files[0]
    else:
        raise FileNotFoundError(f"No file found matching pattern: {pattern}")
else:
    # Use specific file or entire folder
    # If get_file_name is None, will process all CSV files in folder
    if get_file_name:
        get_files_from = os.path.join(get_files_from_folder_root, get_file_name)
    else:
        get_files_from = get_files_from_folder_root

# =============================================================================
# RUN TRANSFORMATION
# =============================================================================

if __name__ == '__main__':
    from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
    
    print("="*80)
    print("BACKWARD-TO-FORWARD TRAJECTORY GENERATION")
    print("="*80)
    print(f"Input: {get_files_from}")
    print(f"Output: {save_files_to}")
    print(f"Backward predictor: {backward_predictor_specification}")
    print(f"Forward predictor: {forward_predictor_specification}")
    print(f"Horizon: {test_horizon}")
    print(f"Max batch size: {max_batch_size}")
    if randomize_param is not None:
        print(f"Parameter randomization: '{randomize_param}' (range: {param_range})")
    else:
        print(f"Parameter randomization: DISABLED")
    print("="*80)
    
    # Create predictors once for reuse across all files
    print("\nInitializing predictors (this will be reused for all files)...")
    backward_predictor = PredictorWrapper()
    backward_predictor.update_predictor_config_from_specification(
        predictor_specification=backward_predictor_specification
    )
    
    forward_predictor = PredictorWrapper()
    forward_predictor.update_predictor_config_from_specification(
        predictor_specification=forward_predictor_specification
    )
    print("Predictors initialized.\n")
    
    transform_dataset(
        get_files_from,
        save_files_to,
        transformation='back_to_front_trajectories',
        backward_predictor_specification=backward_predictor_specification,
        forward_predictor_specification=forward_predictor_specification,
        test_horizon=test_horizon,
        dataset_sampling_dt=dataset_sampling_dt,
        verbose=verbose,
        max_batch_size=max_batch_size,
        predictor=backward_predictor,
        forward_predictor=forward_predictor,
        randomize_param=randomize_param,
        param_range=param_range,
        random_seed=random_seed,
    )
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)


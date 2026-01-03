""" Run the race! """
from run.run_simulation import RacingSimulation
from utilities.Settings import Settings
import time
import datetime
import os
import zipfile
import subprocess
import numpy as np
from utilities.EncodeDecodeEulerFlag import euler_index, decode_flag
from itertools import product  # Enables Cartesian product iteration over multiple parameter lists

started = datetime.datetime.now()
print(f"Started at: {started}")

Settings.DATASET_NAME = "MPC_RCA1_Jan2026"
Settings.RECORDING_INDEX = euler_index


# Global Settings (for every recording)
Settings.MAP_NAME = 'RCA1'  # Same map as 04_08_RCA1_noise dataset

Settings.EXPERIMENT_LENGTH = 6000

# Noise level matching original 04_08_RCA1_noise dataset
Settings.NOISE_LEVEL_CONTROL = [0.1, 0.1]  # noise level [angular, translational]
Settings.CONTROL_NOISE_DURATION = 10  # Number of timesteps for which the control noise is applied


Settings.CONTROL_DELAY = 0.0
Settings.EXECUTE_NTH_STEP_OF_CONTROL_SEQUENCE = 0


Settings.START_FROM_RANDOM_POSITION = True
Settings.RECORDING_FOLDER = os.path.join(Settings.RECORDING_FOLDER, Settings.DATASET_NAME) + '/'

# Dont touch:
Settings.CONTROLLER = 'mpc'
Settings.RENDER_MODE = None
Settings.SAVE_RECORDINGS = True 
Settings.SAVE_PLOTS = True
Settings.APPLY_SPEED_SCALING_FROM_CSV = False
Settings.NUMBER_OF_EXPERIMENTS = 1  # How many experiment per simulation.run_experiments(), you probably want to set this to 1

runs_with_obstacles = 0
runs_without_obstacles = 1
runs_with_oponents = 0

# Friction sampling configuration - UNIFORM DISTRIBUTION
FRICTION_MIN = 0.3
FRICTION_MAX = 1.1
# 1190 total files = 7 velocities × 170 repetitions
NUM_REPETITIONS_PER_VELOCITY = 170

# Velocity factors - 7 discrete values
VELOCITY_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

reverse_direction_values = [False]
global_waypoint_velocity_factors = VELOCITY_FACTORS

# Settings for tuning before recording (uncomment to test locally)
# Settings.EXPERIMENT_LENGTH = 1000
# Settings.RENDER_MODE = "human_fast"
# NUM_REPETITIONS_PER_VELOCITY = 1

feature_A = global_waypoint_velocity_factors
feature_B_size = NUM_REPETITIONS_PER_VELOCITY

index_A, index_B, index_C = decode_flag(euler_index, len(feature_A), feature_B_size)

# Sample velocity from discrete list
global_waypoint_velocity_factors = [global_waypoint_velocity_factors[index_A]]

# Sample friction UNIFORMLY from [0.3, 1.1] using euler_index as seed
# Each job gets a different random mu value
friction_rng = np.random.RandomState(seed=euler_index)
sampled_friction = friction_rng.uniform(FRICTION_MIN, FRICTION_MAX)
global_surface_friction_values = [sampled_friction]
print(f"[Data Collection] euler_index={euler_index}, vel_factor={global_waypoint_velocity_factors[0]}, mu={sampled_friction:.4f}")

# Friction for controller is same as actual friction
global_surface_friction_for_controller_values = None  # If None, always same as global_surface_friction
    

# Save this file to the recordings for perfect reconstruction
def save_this_file():
    target_dir = os.path.join("ExperimentRecordings", f"{Settings.DATASET_NAME}/")
    os.makedirs(target_dir, exist_ok=True)
    
    # Compress the Python file
    source_file = "run/data_collection.py"
    target_zip = os.path.join(target_dir, f"{Settings.DATASET_NAME}_run.zip")
    with zipfile.ZipFile(target_zip, 'w') as zipf:
        zipf.write(source_file, os.path.basename(source_file))
    
    # Get Git status
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
    
    # Save Git status to a text file
    git_status_file = os.path.join(target_dir, "git_status.txt")
    with open(git_status_file, 'w') as f:
        f.write(f"Branch: {branch_name}\n")
        f.write(f"Commit: {commit_hash}\n")
    
    print(f"File {source_file} compressed and saved to {target_zip}")
    print(f"Git status saved to {git_status_file}")


save_this_file()


if global_surface_friction_for_controller_values is not None:
    friction_for_controller_values = global_surface_friction_for_controller_values
else:
    friction_for_controller_values = [None]

# Loop over all combinations of the settings.
# Each iteration gives one configuration of:
#   reverse_direction, global waypoint velocity factor, global surface friction, and friction for controller.
for reverse_direction, global_waypoint_velocity_factor, global_surface_friction, friction_for_controller in product(
        reverse_direction_values,
        global_waypoint_velocity_factors,
        global_surface_friction_values,
        friction_for_controller_values
):
    print("Start of new Experiment with the following settings:")

    # Set reverse direction parameter.
    Settings.REVERSE_DIRECTION = reverse_direction
    print("reverse_direction:", reverse_direction)

    # Set the global waypoint velocity factor.
    Settings.GLOBAL_WAYPOINT_VEL_FACTOR = global_waypoint_velocity_factor
    print("global_waypoint_velocity_factor:", global_waypoint_velocity_factor)

    # Set the global surface friction.
    Settings.SURFACE_FRICTION = global_surface_friction
    print("global_surface_friction:", global_surface_friction)

    # For friction for controller, use the provided value if available.
    # If it's None, default to the same value as global_surface_friction.
    if friction_for_controller is None:
        Settings.FRICTION_FOR_CONTROLLER = global_surface_friction
        print("FRICTION_FOR_CONTROLLER (default to global):", global_surface_friction)
    else:
        Settings.FRICTION_FOR_CONTROLLER = friction_for_controller
        print("FRICTION_FOR_CONTROLLER:", friction_for_controller)

    # Run experiments with obstacles enabled.
    for i in range(runs_with_obstacles):
        # Set the flag to enable random obstacles.
        Settings.PLACE_RANDOM_OBSTACLES = True
        print("runs_with_obstacles iteration:", i)
        time.sleep(1)
        try:
            simulation = RacingSimulation()
            simulation.run_experiments()
        except Exception as e:
            print(f"An error occurred while running the experiments: {e}")

    # Run experiments with obstacles disabled.
    for i in range(runs_without_obstacles):
        Settings.PLACE_RANDOM_OBSTACLES = False
        print("runs_without_obstacles iteration:", i)
        time.sleep(1)
        try:
            simulation = RacingSimulation()
            simulation.run_experiments()
        except Exception as e:
            print(f"An error occurred while running the experiments: {e}")

    # for i in range(runs_with_oponents):
    #     Settings.PLACE_RANDOM_OBSTACLES = False
    #     print("runs_with_oponents", i)
    #     time.sleep(1)
    #     simulation = RacingSimulation()
    #     simulation.run_experiments()

finished = datetime.datetime.now()
print(f"Finished at: {finished}")
print(f"Duration: {finished - started}")
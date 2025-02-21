""" Run the race! """
from run.run_simulation import RacingSimulation
from utilities.Settings import Settings
import time
import os
import zipfile
import subprocess 
import argparse

def args_fun():
    """
    This function is for use with Euler cluster to differentiate parallel runs with an index.
    Returns:
    """
    parser = argparse.ArgumentParser(description='Generate F1T data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--euler_experiment_index', default=-1, type=int,
                        help='Additional index. -1 to skip.')
    parser.add_argument('-s', '--speed_factor', type=float, default=None,
                        help='Float values for global speed factor')

    args = parser.parse_args()

    if args.euler_experiment_index == -1:
        args.euler_experiment_index = None

    return args

euler_index = args_fun().euler_experiment_index
speed_factor = args_fun().speed_factor

# Global Settings (for every recording)
Settings.MAP_NAME = 'RCA1'

Settings.EXPERIMENT_LENGTH = 20000
Settings.NUMBER_OF_EXPERIMENTS = 1 

# Settings.NOISE_LEVEL_TRANSLATIONAL_CONTROL = 1.0 # ftg: 0.5  # mppi: 2.0
# Settings.NOISE_LEVEL_ANGULAR_CONTROL = 0.3  # ftg: 0.05  # mppi: 3.0
# Settings.NOISE_LEVEL_CAR_STATE = [ 0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]


# Settings.NOISE_LEVEL_CAR_STATE = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0]
Settings.NOISE_LEVEL_CONTROL = [0.0, 0.0] # noise level [angular, translational]
Settings.CONTROL_DELAY = 0.0
Settings.EXECUTE_NTH_STEP_OF_CONTROL_SEQUENCE = 0

Settings.RECORDING_INDEX = euler_index


Settings.START_FROM_RANDOM_POSITION = False
Settings.DATASET_NAME = "MPC_mu_vs_mu_control"
Settings.RECORDING_FOLDER = os.path.join(Settings.RECORDING_FOLDER, Settings.DATASET_NAME) + '/'

Settings.CONTROLLER = 'mpc'
Settings.SURFACE_FRICITON = 0.8

Settings.RENDER_MODE = None

# Dont touch:
Settings.SAVE_RECORDINGS = True 
Settings.SAVE_PLOTS = True
Settings.APPLY_SPEED_SCALING_FROM_CSV = False 

runs_with_obstacles = 0
runs_without_obstacles = 1
runs_with_oponents = 0 
global_waypoint_velocity_factors = [0.4]
# global_waypoint_velocity_factors = [0.8,]
global_surface_friction_values = [0.5, 0.6, 0.7, 0.8, 0.9]
global_surface_friction_for_controller_values = [0.5, 0.6, 0.7, 0.8, 0.9]
reverse_direction_values = [False, True]

expected_number_of_experiments = len(global_waypoint_velocity_factors) * len(global_surface_friction_values) * len(reverse_direction_values) * (runs_with_obstacles + runs_without_obstacles)
print(f"Expected number of experiments: {expected_number_of_experiments}")

# Settings for tuning before recoriding
# Comment out during data collection
# Settings.EXPERIMENT_LENGTH = 1000  
# global_waypoint_velocity_factors = [1.0] 
# global_surface_friction_values = [ 0.7 ]
# Settings.RENDER_MODE = "human_fast"


if speed_factor is not None:
    global_waypoint_velocity_factors = [speed_factor]
    

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

for reverse_direction in reverse_direction_values:
    
    print("Start of new Experiment with the following settings:")
    
    Settings.REVERSE_DIRECTION = reverse_direction
    print("reverse_direction", reverse_direction)
    
    for global_waypoint_velocity_factor in global_waypoint_velocity_factors:
        Settings.GLOBAL_WAYPOINT_VEL_FACTOR = global_waypoint_velocity_factor
        print("global_waypoint_velocity_factor", global_waypoint_velocity_factor)
        
        for global_surface_friction in global_surface_friction_values:
            Settings.SURFACE_FRICITON = global_surface_friction
            print("global_surface_friction", global_surface_friction)

            for global_surface_friction_for_controller in global_surface_friction_for_controller_values:
                Settings.FRICTION_FOR_CONTROLLER = global_surface_friction_for_controller
                print("global_surface_friction_for_controller", global_surface_friction_for_controller)
            
                for i in range(runs_with_obstacles):
                    Settings.PLACE_RANDOM_OBSTACLES = True
                    print("runs_with_obstacles", i)
                    print("Speedfator: ", global_waypoint_velocity_factor)
                    time.sleep(1)
                    try:
                        simulation = RacingSimulation()
                        simulation.run_experiments()
                    except Exception as e:
                        print(f"An error occurred while running the experiments: {e}")

                for i in range(runs_without_obstacles):
                    Settings.PLACE_RANDOM_OBSTACLES = False
                    print("runs_without_obstacles", i)
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


""" Run the race! """
from run.run_simulation import run_experiments
from utilities.Settings import Settings
import time
import os
import zipfile
import subprocess



# Global Settings (for every recording)
Settings.MAP_NAME = 'RCA2'

Settings.EXPERIMENT_LENGTH = 2000  
Settings.NUMBER_OF_EXPERIMENTS = 1 

# Settings.NOISE_LEVEL_TRANSLATIONAL_CONTROL = 1.0 # ftg: 0.5  # mppi: 2.0
# Settings.NOISE_LEVEL_ANGULAR_CONTROL = 0.3  # ftg: 0.05  # mppi: 3.0
# Settings.NOISE_LEVEL_CAR_STATE = [ 0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]


# Settings.NOISE_LEVEL_CAR_STATE = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0]
Settings.NOISE_LEVEL_CONTROL = [0.1, 0.1] # noise level [angular, translational]
Settings.START_FROM_RANDOM_POSITION = True 


Settings.START_FROM_RANDOM_POSITION = True
Settings.DATASET_NAME = "RPGD_fixedspeed"
Settings.RECORDING_FOLDER = os.path.join(Settings.RECORDING_FOLDER, Settings.DATASET_NAME) + '/'

Settings.CONTROLLER = 'mpc'
Settings.SURFACE_FRICITON = 0.8

Settings.RENDER_MODE = None

# Dont touch:
Settings.SAVE_RECORDINGS = True 
Settings.SAVE_PLOTS = True
Settings.APPLY_SPEED_SCALING_FROM_YAML = False 

runs_with_obstacles = 0
runs_without_obstacles = 7
runs_with_oponents = 0 
global_waypoint_velocity_factors = [0.8]
global_surface_friction_values = [ 0.3, 0.5, 0.7, 0.9, 1.1]
reverse_direction_values = [False, True]


# Settings for tuning before recoriding
# Comment out during data collection
# Settings.EXPERIMENT_LENGTH = 1000  
# global_waypoint_velocity_factors = [0.8] 
# Settings.RENDER_MODE = "human_fast"


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
            
            for i in range(runs_with_obstacles):
                Settings.PLACE_RANDOM_OBSTACLES = True
                print("runs_with_obstacles", i)
                print("Speedfator: ", global_waypoint_velocity_factor)
                time.sleep(1)
                try:
                    run_experiments()
                except Exception as e:
                    print(f"An error occurred while running the experiments: {e}")        
                    
            for i in range(runs_without_obstacles):
                Settings.PLACE_RANDOM_OBSTACLES = False
                print("runs_without_obstacles", i)
                time.sleep(1)
                
                try:
                    run_experiments()
                except Exception as e:
                    print(f"An error occurred while running the experiments: {e}")    
            
        # for i in range(runs_with_oponents):
        #     Settings.PLACE_RANDOM_OBSTACLES = False
        #     print("runs_with_oponents", i)
        #     time.sleep(1)
        #     run_experiments()



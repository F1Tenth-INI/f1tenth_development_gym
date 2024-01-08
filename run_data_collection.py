""" Run the race! """
from utilities.run_simulation import run_experiments
from utilities.Settings import Settings
import time
import os

# Global Settings (for every recording)
Settings.DATASET_NAME = Settings.DATASET_NAME + "_DelayedControl"

Settings.MAP_NAME = 'RCA1'
Settings.EXPERIMENT_LENGTH = 3000  
Settings.NUMBER_OF_EXPERIMENTS = 1 
Settings.NOISE_LEVEL_CAR_STATE = [ 0., 0., 0., 0., 0., 0., 0., 0., 0.]


Settings.CONTROLLER = 'mpc'
Settings.CONTROL_AVERAGE_WINDOW = (1, 1)     # Window for avg filter [angular, translational]

# Dont touch:
Settings.START_FROM_RANDOM_POSITION = True
Settings.SAVE_RECORDINGS = True 
Settings.SAVE_PLOTS = True
Settings.APPLY_SPEED_SCALING_FROM_YAML = False 
Settings.RENDER_MODE = None

control_noise_settings = [
    [0,0], 
    # [0,0], 
    # [0.35, 0.5],
]

obstacle_settings = 7 * [False] + 0 * [True]
runs_with_oponents = 0 

global_waypoint_velocity_factors = [0.7]
reverse_direction_values = [False, True]


# Settings for tuning before recoriding
# Comment out during data collection
# Settings.EXPERIMENT_LENGTH = 1000  
# global_waypoint_velocity_factors = [0.8]
# Settings.RENDER_MODE = "human_fast"


run_index = 0

# Direction
for reverse_direction in reverse_direction_values:
    Settings.REVERSE_DIRECTION = reverse_direction
    print("reverse_direction", reverse_direction)
    
    # Velocity
    for global_waypoint_velocity_factor in global_waypoint_velocity_factors:
        Settings.GLOBAL_WAYPOINT_VEL_FACTOR = global_waypoint_velocity_factor
        print("global_waypoint_velocity_factor", global_waypoint_velocity_factor)

        # Control Noise
        for control_noise in control_noise_settings:
            Settings.NOISE_LEVEL_CONTROL = control_noise
            print("Noise on Control level", global_waypoint_velocity_factor)
            
            # Obstacles
            for place_obstacles in obstacle_settings:
                Settings.PLACE_RANDOM_OBSTACLES = place_obstacles
                                
                print("run_index", run_index)
                run_index += 1
                time.sleep(1)
                
                # Run experiment with adjusted settings
                try:
                    run_experiments()
                except Exception as e:
                    print(f"An error occurred while running the experiments: {e}")        
                
      
            
        # for i in range(runs_with_oponents):
        #     Settings.PLACE_RANDOM_OBSTACLES = False
        #     print("runs_with_oponents", i)
        #     time.sleep(1)
        #     run_experiments()



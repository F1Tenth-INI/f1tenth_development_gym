""" Run the race! """
from run.run_simulation import run_experiments
from utilities.Settings import Settings
import time
import os

# Global Settings (for every recording)
Settings.MAP_NAME = 'RCA2'

Settings.EXPERIMENT_LENGTH = 3000  
Settings.NUMBER_OF_EXPERIMENTS = 1 

# Settings.NOISE_LEVEL_TRANSLATIONAL_CONTROL = 1.0 # ftg: 0.5  # mppi: 2.0
# Settings.NOISE_LEVEL_ANGULAR_CONTROL = 0.3  # ftg: 0.05  # mppi: 3.0
# Settings.NOISE_LEVEL_CAR_STATE = [ 0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]


Settings.NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.0 # ftg: 0.5  # mppi: 2.0
Settings.NOISE_LEVEL_ANGULAR_CONTROL = 0.0  # ftg: 0.05  # mppi: 3.0
Settings.NOISE_LEVEL_CAR_STATE = [ 0., 0., 0., 0., 0., 0., 0., 0., 0.]


Settings.START_FROM_RANDOM_POSITION = True
Settings.DATASET_NAME = "_MPPI_with_delay_"

Settings.CONTROLLER = 'mpc'
Settings.CONTROL_AVERAGE_WINDOW = (1, 1)     # Window for avg filter [angular, translational]
Settings.SURFACE_FRICITON = 0.8

Settings.RENDER_MODE = None

# Dont touch:
Settings.SAVE_RECORDINGS = True 
Settings.SAVE_PLOTS = True
Settings.APPLY_SPEED_SCALING_FROM_YAML = False 

runs_with_obstacles = 0
runs_without_obstacles = 4
runs_with_oponents = 0 
global_waypoint_velocity_factors = [0.5, 0.6, 0.75, 1.0, 1.1]
global_surface_friction_values = [ 0.3, 0.5, 0.7, 0.8, 1.0]
reverse_direction_values = [False, True]


# Settings for tuning before recoriding
# Comment out during data collection
# Settings.EXPERIMENT_LENGTH = 1000  
# global_waypoint_velocity_factors = [0.8]
# Settings.RENDER_MODE = "human_fast"

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



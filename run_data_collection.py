""" Run the race! """
from utilities.run_simulation import run_experiments
from utilities.Settings import Settings
import time

# Speed scalling is globally set here... make sure its 1.0 everywhere in speed_scaling.yaml

# Global Settings (for every recording)
Settings.MAP_NAME = 'hangar12'
Settings.SAVE_RECORDINGS = True # Dont touch
Settings.SAVE_PLOTS = True
Settings.RENDER_MODE = None

Settings.EXPERIMENT_LENGTH = 3000  
Settings.NUMBER_OF_EXPERIMENTS = 1 

runs_with_obstacles = 5
runs_without_obstacles = 5
runs_with_oponents = 0 
global_waypoint_velocity_factors = [0.3, 0.4, 0.5, 0.6]
reverse_direction_values = (False, True)



for reverse_direction in reverse_direction_values:
    Settings.REVERSE_DIRECTION = reverse_direction
    print("reverse_direction", reverse_direction)
    for global_waypoint_velocity_factor in global_waypoint_velocity_factors:
        Settings.GLOBAL_WAYPOINT_VEL_FACTOR = global_waypoint_velocity_factor
        print("global_waypoint_velocity_factor", global_waypoint_velocity_factor)

        
        for i in range(runs_with_obstacles):
            Settings.PLACE_RANDOM_OBSTACLES = True
            print("runs_with_obstacles", i)
            time.sleep(1)
            run_experiments()        
                
        for i in range(runs_without_obstacles):
            Settings.PLACE_RANDOM_OBSTACLES = False
            print("runs_without_obstacles", i)
            time.sleep(1)
            run_experiments()        
            
        # for i in range(runs_with_oponents):
        #     Settings.PLACE_RANDOM_OBSTACLES = False
        #     print("runs_with_oponents", i)
        #     time.sleep(1)
        #     run_experiments()


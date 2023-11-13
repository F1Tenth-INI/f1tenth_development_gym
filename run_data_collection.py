""" Run the race! """
from utilities.run_simulation import run_experiments
from utilities.Settings import Settings
import time

# Speed scalling is globally set here... make sure its 1.0 everywhere in speed_scaling.yaml

# Global Settings (for every recording)
Settings.MAP_NAME = 'hangar14' # hangar3, hangar9, hangar12, hangar14, hangar16, london3_small, london3_large, ETF1, ini10, london3_large
Settings.SAVE_RECORDINGS = True # Dont touch
Settings.SAVE_PLOTS = True
Settings.RENDER_MODE = None

Settings.EXPERIMENT_LENGTH = 2000  
Settings.NUMBER_OF_EXPERIMENTS = 1 

runs_with_obstacles = 0
runs_without_obstacles = 20
runs_with_oponents = 0 
global_waypoint_velocity_factors = [0.3, 0.4, 0.5, 0.6, 0.7]
reverse_direction_values = (False, True)



for reverse_direction in reverse_direction_values:
    Settings.REVERSE_DIRECTION = reverse_direction
    print("reverse_direction", reverse_direction)
    for global_waypoint_velocity_factor in global_waypoint_velocity_factors:
        Settings.GLOBAL_WAYPOINT_VEL_FACTOR = global_waypoint_velocity_factor
        # print("global_waypoint_velocity_factor", global_waypoint_velocity_factor)

        
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
            print("Speedfator: ", global_waypoint_velocity_factor)
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



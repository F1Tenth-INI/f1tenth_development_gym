from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.path_helper_ros import *

import os.path
import numpy as np
import pandas as pd
import yaml

'''
HOW TO USE:

1. Generate/Get a Waypoints CSV File (https://github.com/TUMFTM/global_racetrajectory_optimization )
2. Set the waypoint_payth in the map config file (fe. config_Custom.yaml) to the generated file (without .csv)
3. Set Waypoint parameters in config.yml
4. Use inside planner class like this:

# Import 
from utilities.waypoint_utils import WaypointUtils

# Initialize
waypoint_utils = WaypointUtils()

# Update at every step for the next_waypoints with the current car's position
car_position = [0., 0.]
waypoint_utils.update_next_waypoints(car_position)

# Access next waypoint's positions list([x,y]):
waypoints_positions = waypoint_utils.next_waypoint_positions

# Or Access next full waypoints list([dist, x, y, abs_angle, rel_angle, v_x, acc_x]):
waypoints = waypoint_utils.next_waypoints

'''
     
class WaypointUtils:
    
    def __init__(self):
        
        gym_path = get_gym_path()
        self.look_ahead_steps = 0
        # config = yaml.load(open(os.path.join(gym_path, "config.yml"), "r"), Loader=yaml.FullLoader)
        self.waypoint_positions = np.array([])
        self.next_waypoints = np.zeros((self.look_ahead_steps, 7), dtype=np.float32)
        self.next_waypoint_positions = np.zeros((self.look_ahead_steps, 2), dtype=np.float32)
        
       
        
        return
        
        
        
    def update_next_waypoints(self, car_position):
        return
       
        
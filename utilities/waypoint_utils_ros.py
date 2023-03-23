from utilities.Settings import Settings
from utilities.state_utilities import *

from racecar.msg import WpntArray

import os.path
import numpy as np
import pandas as pd
import yaml
import rospkg
import rospy
import os

import time

from visualization_msgs.msg import Marker, MarkerArray


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
        
        self.interpolation_steps = 1
        self.decrease_resolution_factor = 1

        self.look_ahead_steps = self.interpolation_steps * 30
        self.ignore_steps = 0

        self.waypoint_positions = np.zeros((self.look_ahead_steps,2), dtype=np.float32)

        self.next_waypoints = np.zeros((self.look_ahead_steps, 7), dtype=np.float32)
        self.next_waypoint_positions = np.zeros((self.look_ahead_steps,2), dtype=np.float32)


        rospy.Subscriber('/local_waypoints', WpntArray, self.local_waypoints_cb)

    

    def local_waypoints_cb(self, data):
        next_waypoints = []
        next_waypoint_positions = []

        for wpnt in data.wpnts:
            waypoint = [
                wpnt.s_m,
                wpnt.x_m,
                wpnt.y_m,
                wpnt.psi_rad,
                wpnt.kappa_radpm,
                wpnt.vx_mps,
                wpnt.ax_mps2,
            ]
            next_waypoints.append(waypoint)
            next_waypoint_positions.append([ wpnt.x_m, wpnt.y_m,])
        self.next_waypoints = np.array(next_waypoints)
        self.next_waypoint_positions = np.array(next_waypoint_positions)


    # called by planner, needs to stay here
    def update_next_waypoints(self, car_position):
        return

        

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
    
    def __init__(self, wp_topic = "/mpc/local_waypoints"):
        self.number_of_waypoints = Settings.LOOK_AHEAD_STEPS
        
        self.waypoint_positions = np.zeros((self.number_of_waypoints,2), dtype=np.float32)
        self.next_waypoints = np.zeros((self.number_of_waypoints, 7), dtype=np.float32)
        self.next_waypoint_positions = np.zeros((self.number_of_waypoints,2), dtype=np.float32)
        
        rospy.Subscriber(wp_topic, WpntArray, self.local_waypoints_cb)
        rospy.wait_for_message("local_waypoints", WpntArray, timeout=30)
        rospy.loginfo("Waypointutils: Got local waypoints")
        rospy.sleep(0.5)


    @staticmethod
    def get_interpolated_waypoints(waypoints, interpolation_steps):
        if waypoints is None: return None
        assert(interpolation_steps >= 1)
        waypoints_interpolated = []
        
        for j in range(len(waypoints) - 1):
            for i in range(interpolation_steps):
                interpolated_waypoint = waypoints[j] + (float(i)/interpolation_steps)*(waypoints[j+1]-waypoints[j])
                waypoints_interpolated.append(interpolated_waypoint)
        waypoints_interpolated.append(waypoints[-1])
        return np.array(waypoints_interpolated)
    

    def local_waypoints_cb(self, data):
        # Check length of waypoints array for initialization
        if(self.number_of_waypoints is None):
            self.number_of_waypoints = len(data.wpnts)
            
        # Save waypoint array
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

        

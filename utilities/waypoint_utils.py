
from utilities.Settings import Settings
from utilities.state_utilities import *

import os.path
import numpy as np
import pandas as pd
import yaml

        
class WaypointUtils:
    
    def __init__(self, waypoint_file = None):
        config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
        
        self.interpolation_steps = config['waypoints']['INTERPOLATION_STEPS']
        self.look_ahead_steps = self.interpolation_steps * config['waypoints']['LOOK_AHEAD_STEPS']
        
        self.original_waypoints = WaypointUtils.load_waypoints(waypoint_file)
        self.waypoints = np.array(WaypointUtils.get_interpolated_waypoints(self.original_waypoints, self.interpolation_steps))
        
        self.next_waypoints = np.zeros((self.look_ahead_steps, 2), dtype=np.float32)
        self.nearest_waypoint_index = None
        
        
    def update_next_waypoints(self, s):
        self.next_waypoints, self.nearest_waypoint_index = WaypointUtils.get_next_waypoints(s, self.waypoints, self.next_waypoints, self.nearest_waypoint_index, self.look_ahead_steps)
    
        
    @staticmethod
    def load_waypoints(map_waypoint_file = None):
        
        path = Settings.MAP_WAYPOINT_FILE
        if map_waypoint_file is not None:
            path = map_waypoint_file

        file_path = path + '.csv'
        assert os.path.isfile(file_path), "Waypoint file (" + path+  ") does not exist"
        
        waypoints = pd.read_csv(file_path, header=None).to_numpy()
        waypoints = np.array(waypoints[0:-1:1, 1:3])
        return waypoints
    
    @staticmethod
    def get_interpolated_waypoints(waypoints, interpolation_steps):
        assert(interpolation_steps >= 1)
        waypoints_interpolated = []
        
        for j in range(len(waypoints) - 1):
            for i in range(interpolation_steps):
                interpolated_waypoint = waypoints[j] + (float(i)/interpolation_steps)*(waypoints[j+1]-waypoints[j])
                waypoints_interpolated.append(interpolated_waypoint)
        waypoints_interpolated.append(waypoints[-1])
        return waypoints_interpolated
    
    @staticmethod
    def get_next_waypoints(s, all_wpts, current_wpts, nearest_waypoint_index, look_ahead):
        
        if nearest_waypoint_index is None:
            nearest_waypoint_index = WaypointUtils.get_nearest_waypoint_index(s, all_wpts)  # Run initial search of starting waypoint
        else:
            nearest_waypoint_index += WaypointUtils.get_nearest_waypoint_index(s, current_wpts)

        nearest_waypoints = []
        for j in range(look_ahead):
            next_waypoint = all_wpts[(nearest_waypoint_index + j) % len(all_wpts)]
            nearest_waypoints.append(next_waypoint)
            
        return nearest_waypoints, nearest_waypoint_index

    @staticmethod
    def get_nearest_waypoint_index(s, wpts):

        min_dist = 10000
        min_dist_index = 0
        position = [s[POSE_X_IDX], s[POSE_Y_IDX]]

        def squared_distance(p1, p2):
            squared_distance = abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2
            return squared_distance

        for i in range(len(wpts)):

            dist = squared_distance(wpts[i], position)
            if (dist) < min_dist:
                min_dist = dist
                min_dist_index = i

        return min_dist_index

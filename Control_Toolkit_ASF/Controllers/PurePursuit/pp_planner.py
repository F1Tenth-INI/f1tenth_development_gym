import sys
sys.path.insert(1, 'FollowtheGap')

import numpy as np
import math

from utilities.Settings import Settings




from Control_Toolkit_ASF.Controllers.PurePursuit.pp_helpers import *
from utilities.state_utilities import *

'''
Example PP planner, adapted to our system
'''
class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self):
    
        print("Initializing PP Planner")
    
        self.lidar_points = 1080 * [[0,0]]
        self.lidar_scan_angles = np.linspace(-2.35,2.35, 1080)
        
        self.waypoints = None
        self.speed = 1

        
        # Controller settings
        self.waypoint_velocity_factor = 0.56
        self.lookahead_distance = 1.82461887897713965
        self.wheelbase = 0.17145+0.15875  
        self.max_reacquire = 20.
        
        self.simulation_index = 0

        self.current_position = None
        self.curvature_integral = 0
        self.translational_control = None
        self.angular_control = None
        
        self.angular_control = 0.
        self.translational_control = 0.
        
        print('Initialization done.')
        # Original values 
        # self.wheelbase = 0.17145+0.15875        
        # self.max_reacquire = 20.
        # self.lookahead_distance = 1.82461887897713965
        # self.waypoint_velocity_factor = 0.80338203837889
    
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """    
        wpts = waypoints[:, 1:3]
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] =  waypoints[i, 5]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, 5])
        else:
            return None
    
    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        
    def set_car_state(self, car_state):
        self.car_state = car_state
        
        
    def process_observation(self, ranges=None, ego_odom=None):
        """
        gives actuation given observation
        @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
        @ ego_odom: A dict with following indices:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
            'linear_vel_x': float,
            'linear_vel_y': float,
            'angular_vel_z': float,
        }
        """
        pose_x = ego_odom['pose_x']
        pose_y = ego_odom['pose_y']
        pose_theta = ego_odom['pose_theta']
        v_x = ego_odom['linear_vel_x']
        position = np.array([pose_x, pose_y])
                
        # Dynamic Lookahead distance
        # self.lookahead_distance = 0.7 * self.speed
        
        lookahead_point = self._get_current_waypoint(self.waypoints, self.lookahead_distance, position, pose_theta)
        # print ("lookaheadpoints", lookahead_point)
        if lookahead_point is None:
            print("warning no lookahead point")
            lookahead_point = self.waypoints[3]
            lookahead_point = [lookahead_point[X_M_IDX],lookahead_point[Y_M_IDX],lookahead_point[V_X_IDX]]
            # self.angular_control = 0.
            # self.translational_control = 1.
            # return 1.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, self.lookahead_distance, self.wheelbase)
        speed = self.waypoint_velocity_factor * speed
        self.speed = speed

        self.angular_control = steering_angle
        self.translational_control = speed
        return steering_angle, speed



        


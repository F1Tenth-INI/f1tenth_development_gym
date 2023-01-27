import sys
sys.path.insert(1, 'FollowtheGap')

import numpy as np
import math
import matplotlib.pyplot as plt
import pyglet.gl as gl

from utilities.waypoint_utils import WaypointUtils
from utilities.render_utilities import RenderUtils


'''
Pure Pursuit planner from the phyaical car
'''
class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self):
    
        print("Controller initialized")
    
        self.lidar_points = 1080 * [[0,0]]
        self.lidar_scan_angles = np.linspace(-2.35,2.35, 1080)
        
        self.waypoint_utils = WaypointUtils()
        self.Render = RenderUtils()
        self.Render.waypoints = self.waypoint_utils.waypoint_positions
        
        
        self.simulation_index = 0

        self.current_position = None
        self.curvature_integral = 0
        self.translational_control = None
        self.angular_control = None
        

    def render(self, e):
        self.Render.render(e)
        
        
    def get_curvatur_integral(self):
        
        waypoints = np.array(self.waypoint_utils.next_waypoint_positions)
        gradient = np.gradient(waypoints, axis=0)
        angles = []
        for i in range(len(gradient)-1):
            vector_1 = gradient[i]
            vector_1[vector_1 == 0] = 0.01
            vector_2 = gradient[i+1]
            vector_2[vector_2 == 0] = 0.01

            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            angles.append(1/(i+1)*angle)

       
        angles = np.array(angles)
        angles[np.isnan(angles)] = 0
        curvature_integral = np.sum(angles)
        return curvature_integral

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
        
        # x, y, yaw
        self.current_position = [pose_x, pose_y, pose_theta] 
        self.curvature_integral = self.get_curvatur_integral()
         
        target_point_index = 4  + int(0.5* v_x)
        target_point =  self.waypoint_utils.next_waypoints[target_point_index][1:3]
        diff_x = target_point[0] - self.current_position[0]
        diff_y = target_point[1] - self.current_position[1]
        
        alpha=np.arctan(diff_y/diff_x)-self.car_state[2]

        if alpha > np.pi/2:
            alpha -= np.pi
        if alpha < - np.pi/2:
            alpha += np.pi 

        # print("alpha", alpha)
        kpp = 2.5
        desired_angle=np.arctan(2*3*np.sin(alpha)/(kpp*v_x))
        
        speed = 7.0 # -5. * self.curvature_integral- 1. * abs(desired_angle) # - 2.0 * abs(self.car_state[8])
        if(speed < 0.0): speed = 0.0

        speed = np.clip(speed, -2.0, 8.0)
        desired_angle = np.clip(desired_angle, -1.2, 1.2)
        

        # speed = 3
        
        self.waypoint_utils.update_next_waypoints([pose_x, pose_y])
        
        self.Render.update(
            next_waypoints= self.waypoint_utils.next_waypoint_positions,
        )
        
        self.translational_control = speed
        self.angular_control = desired_angle

        return speed, desired_angle


        


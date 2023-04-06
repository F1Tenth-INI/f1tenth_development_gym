import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from tqdm import trange
from utilities.Settings import Settings
from utilities.Recorder import Recorder
import pandas as pd

from f110_gym.envs.dynamic_models import pid

# Utilities
from utilities.state_utilities import *
from utilities.random_obstacle_creator import RandomObstacleCreator # Obstacle creation

if(Settings.ROS_BRIDGE):
    from utilities.waypoint_utils_ros import WaypointUtils
    from utilities.render_utilities_ros import RenderUtils
else:
    from utilities.waypoint_utils import WaypointUtils
    from utilities.render_utilities import RenderUtils



class CarSystem:
    
    def __init__(self):
        
        # Settings
        self.plot_lidar_data = False
        self.draw_lidar_data = True
        self.lidar_visualization_color = (255, 0, 255)
        
        # Initial values
        self.car_state = [1,1,1,1,1,1,1,1,1]
        car_index = 1
        
        
        # Utilities 
        self.waypoint_utils = WaypointUtils()
        self.render_utils = RenderUtils()
        self.render_utils.waypoints = self.waypoint_utils.waypoint_positions
        self.recorder = Recorder(controller_name='Blank-MPPI-{}'.format(str(car_index)), dt=Settings.TIMESTEP_CONTROL)

        
        # Planner
        self.planner = None
        if Settings.CONTROLLER == 'mpc':
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
            self.planner = mpc_planner()
        elif Settings.CONTROLLER =='ftg':
            from Control_Toolkit_ASF.Controllers.FollowTheGap.ftg_planner import FollowTheGapPlanner
            self.planner =  FollowTheGapPlanner()
        elif Settings.CONTROLLER == 'neural':
            from Control_Toolkit_ASF.Controllers.NeuralNetImitator.nni_planner import NeuralNetImitatorPlanner
            self.planner =  NeuralNetImitatorPlanner()
        elif Settings.CONTROLLER == 'pp':
            from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
            self.planner = PurePursuitPlanner()
        else:
            NotImplementedError('{} is not a valid controller name for f1t'.format(Settings.CONTROLLER))
            
        self.planner.render_utils = self.render_utils
            

    
    def set_car_state(self, car_state):
        self.car_state = car_state
    
    def render(self, e):
        self.render_utils.render(e)
        
    
    """
        returns actuation given observation
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
    def process_observation(self, ranges=None, ego_odom=None):
        
        # if Settings.ONLY_ODOMETRY_AVAILABLE:
        #     car_state = odometry_dict_to_state(ego_odom)
        # else:
        #     car_state = self.car_state
        
        car_state = self.car_state
        ranges = np.array(ranges)
        self.waypoint_utils.update_next_waypoints(car_state)
        
        # Pass data to the planner
        if hasattr(self.planner, 'set_waypoints'):
            self.planner.set_waypoints(self.waypoint_utils.next_waypoints)
            
        if hasattr(self.planner, 'set_car_state'):
            self.planner.set_car_state(car_state)
            
       
                
        self.angular_control, self.translational_control = self.planner.process_observation(ranges, ego_odom)
        
        self.render_utils.update(
            next_waypoints= self.waypoint_utils.next_waypoint_positions,
            car_state = car_state
        )
        
        if (Settings.SAVE_RECORDINGS):
            self.recorder.get_data(
                control_inputs_calculated=(self.translational_control, self.angular_control),
                odometry=ego_odom, 
                ranges=ranges, 
                state=self.car_state,
                next_waypoints=self.waypoint_utils.next_waypoint_positions,
                next_waypoints_relative=self.waypoint_utils.next_waypoint_positions_relative,
                time=0 # TODO
            )     
        
                
        return self.angular_control, self.translational_control

            
            

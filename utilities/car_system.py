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
from utilities.obstacle_detector import ObstacleDetector
from utilities.lidar_utils import LidarHelper

if(Settings.ROS_BRIDGE):
    from utilities.waypoint_utils_ros import WaypointUtils
    from utilities.render_utilities_ros import RenderUtils
else:
    from utilities.waypoint_utils import WaypointUtils
    from utilities.render_utilities import RenderUtils



class CarSystem:
    
    def __init__(self, controller=None, save_recording = True):

        self.time = 0.0
        self.time_increment = Settings.TIMESTEP_CONTROL

        # Settings
        self.plot_lidar_data = False
        self.draw_lidar_data = True
        self.lidar_visualization_color = (255, 0, 255)
        self.LIDAR = LidarHelper()

        # TODO: Move to a config file ( which one tho?)
        self.control_average_window = Settings.CONTROL_AVERAGE_WINDOW # Window for averaging control input for smoother control [angular, translational]
        self.angular_control_history = np.zeros(self.control_average_window[0], dtype=np.int32)
        self.translational_control_history = np.zeros(self.control_average_window[1], dtype=np.int32)
        
        # Initial values
        self.car_state = np.ones(len(STATE_VARIABLES))
        car_index = 1
        self.scans = None
        self.control_index = 0
        
        
        # Utilities 
        self.waypoint_utils = WaypointUtils()
        self.render_utils = RenderUtils()
        self.render_utils.waypoints = self.waypoint_utils.waypoint_positions
        self.save_recording = save_recording
        if save_recording:
            self.recorder = Recorder(controller_name='Blank-MPPI-{}'.format(str(car_index)), dt=Settings.TIMESTEP_CONTROL, lidar=self.LIDAR)
        self.obstacle_detector = ObstacleDetector()
        
        # Planner
        self.planner = None
        if(controller is None):
            controller = Settings.CONTROLLER
        if controller == 'mpc':
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
            self.planner = mpc_planner()
        elif controller =='ftg':
            from Control_Toolkit_ASF.Controllers.FollowTheGap.ftg_planner import FollowTheGapPlanner
            self.planner =  FollowTheGapPlanner()
        elif controller == 'neural':
            from Control_Toolkit_ASF.Controllers.NeuralNetImitator.nni_planner import NeuralNetImitatorPlanner
            self.planner =  NeuralNetImitatorPlanner()
        elif controller == 'pp':
            from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
            self.planner = PurePursuitPlanner()
        elif controller == 'manual':
            from Control_Toolkit_ASF.Controllers.Manual.manual_planner import manual_planner
            self.planner = manual_planner()
        else:
            NotImplementedError('{} is not a valid controller name for f1t'.format(controller))
            
        self.planner.render_utils = self.render_utils
        self.planner.waypoint_utils = self.waypoint_utils
            

    
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
        self.LIDAR.load_lidar_measurement(ranges)
        lidar_points = self.LIDAR.get_all_lidar_points_in_map_coordinates(
            car_state[POSE_X_IDX], car_state[POSE_Y_IDX], car_state[POSE_THETA_IDX])
        self.waypoint_utils.update_next_waypoints(car_state)
        obstacles = self.obstacle_detector.get_obstacles(ranges, car_state)

        # Pass data to the planner
        if hasattr(self.planner, 'set_waypoints'):
            self.planner.set_waypoints(self.waypoint_utils.next_waypoints)
        if hasattr(self.planner, 'set_car_state'):
            self.planner.set_car_state(car_state)
        if hasattr(self.planner, 'set_obstacles'):
            self.planner.set_obstacles(obstacles)


        # Control step 
        if(self.control_index % Settings.OPTIMIZE_EVERY_N_STEPS == 0 or not hasattr(self.planner, 'optimal_control_sequence') ):
            self.angular_control, self.translational_control = self.planner.process_observation(ranges, ego_odom)

        # Control Queue if exists
        if hasattr(self.planner, 'optimal_control_sequence'):
            self.optimal_control_sequence = self.planner.optimal_control_sequence
            next_control_step = self.optimal_control_sequence[self.control_index % Settings.OPTIMIZE_EVERY_N_STEPS]
            self.angular_control = next_control_step[0]
            self.translational_control = next_control_step[1]
            
        # Average filter
        self.angular_control_history = np.append(self.angular_control_history, self.angular_control)[1:]
        self.translational_control_history = np.append(self.translational_control_history, self.translational_control)[1:]
        self.angular_control = np.average(self.angular_control_history)
        self.translational_control = np.average(self.translational_control_history)

        
        # Rendering and recording
        self.render_utils.update(
            lidar_points= lidar_points,
            next_waypoints= self.waypoint_utils.next_waypoint_positions,
            car_state = car_state
        )
        self.render_utils.update_obstacles(obstacles)
        self.time = self.control_index*self.time_increment
        if (Settings.SAVE_RECORDINGS):
            self.recorder.get_data(
                control_inputs_calculated=(self.translational_control, self.angular_control),
                odometry=ego_odom,
                state=self.car_state,
                next_waypoints=self.waypoint_utils.next_waypoints,
                next_waypoints_relative=self.waypoint_utils.next_waypoint_positions_relative,
                time=self.time
            )     
        
        self.control_index += 1
        return self.angular_control, self.translational_control

            
            

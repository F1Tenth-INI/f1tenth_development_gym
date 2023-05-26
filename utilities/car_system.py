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

from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX
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
        self.save_recordings = save_recording
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

        self.waypoints_for_controller = None

        self.waypoints_planner = None
        self.waypoints_from_mpc = np.zeros((Settings.LOOK_AHEAD_STEPS, 7))
        if Settings.WAYPOINTS_FROM_MPC:
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
            self.waypoints_planner = mpc_planner()
            self.waypoints_planner.waypoint_utils = self.waypoint_utils
        
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


        if Settings.WAYPOINTS_FROM_MPC:
            if self.control_index % Settings.PLAN_EVERY_N_STEPS == 0:
                pass_data_to_planner(self.waypoints_planner, self.waypoint_utils.next_waypoints, car_state, obstacles)
                self.waypoints_planner.process_observation(ranges, ego_odom)
                optimal_trajectory = self.waypoints_planner.mpc.optimizer.optimal_trajectory
                if optimal_trajectory is not None:
                    self.waypoints_from_mpc[:, WP_X_IDX] = optimal_trajectory[0, -len(self.waypoints_from_mpc):, POSE_X_IDX]
                    self.waypoints_from_mpc[:, WP_Y_IDX] = optimal_trajectory[0, -len(self.waypoints_from_mpc):, POSE_Y_IDX]
                    self.waypoints_from_mpc[:, WP_VX_IDX] = optimal_trajectory[0, -len(self.waypoints_from_mpc):, LINEAR_VEL_X_IDX]
                    # self.waypoints_from_mpc[:, WP_VX_IDX] = self.waypoint_utils.next_waypoints[:, WP_VX_IDX]
                    self.waypoints_for_controller = self.waypoints_from_mpc
                else:
                    self.waypoints_for_controller = self.waypoint_utils.next_waypoints

        else:
            self.waypoints_for_controller = self.waypoint_utils.next_waypoints

        pass_data_to_planner(self.planner, self.waypoints_for_controller, car_state, obstacles)

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
            next_waypoints= self.waypoints_for_controller[:, (WP_X_IDX, WP_Y_IDX)],
            car_state = car_state
        )
        self.render_utils.update_obstacles(obstacles)
        self.time = self.control_index*self.time_increment
        if (Settings.SAVE_RECORDINGS and self.save_recordings):
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

            
            
def pass_data_to_planner(planner, next_waypoints=None, car_state=None, obstacles=None):
    # Pass data to the planner
    if hasattr(planner, 'set_waypoints'):
        planner.set_waypoints(next_waypoints)
    if hasattr(planner, 'set_car_state'):
        planner.set_car_state(car_state)
    if hasattr(planner, 'set_obstacles'):
        planner.set_obstacles(obstacles)
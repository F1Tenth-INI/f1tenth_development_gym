import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from tqdm import trange
from utilities.Settings import Settings
from utilities.Recorder import Recorder, get_basic_data_dict
from utilities.imu_simulator import IMUSimulator
import pandas as pd
                
import os

from f110_gym.envs.dynamic_models import pid

# Utilities
from utilities.state_utilities import *
from utilities.random_obstacle_creator import RandomObstacleCreator # Obstacle creation
from utilities.obstacle_detector import ObstacleDetector
from utilities.lidar_utils import LidarHelper

from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX, WP_KAPPA_IDX
from utilities.render_utilities import RenderUtils
if(Settings.ROS_BRIDGE):
    from utilities.waypoint_utils_ros import WaypointUtils
else:
    from utilities.waypoint_utils import WaypointUtils
# from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
# from SI_Toolkit.computation_library import TensorFlowLibrary

# from TrainingLite.slip_prediction import predict



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
        self.imu_simulator = IMUSimulator()
        self.current_imu_dict = self.imu_simulator.array_to_dict(np.zeros(3))
    
        self.angular_control_dict = {}
        self.translational_control_dict = {}
      
        
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
        self.controller_name = controller
        if(controller is None):
            controller = Settings.CONTROLLER
        if controller == 'mpc':
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
            self.planner = mpc_planner()
            horizon = self.planner.mpc.predictor.horizon
            self.angular_control_dict = {"cs_a_{}".format(i): 0 for i in range(horizon)}
            self.translational_control_dict = {"cs_t_{}".format(i): 0 for i in range(horizon)}
        elif controller =='ftg':
            from Control_Toolkit_ASF.Controllers.FollowTheGap.ftg_planner import FollowTheGapPlanner
            self.planner =  FollowTheGapPlanner()
        elif controller == 'neural':
            from Control_Toolkit_ASF.Controllers.NeuralNetImitator.nni_planner import NeuralNetImitatorPlanner
            self.planner =  NeuralNetImitatorPlanner()
        elif controller == 'nni-lite':
            from Control_Toolkit_ASF.Controllers.NNLite.nni_lite_planner import NNLitePlanner
            self.planner =  NNLitePlanner()
        elif controller == 'pp':
            from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
            self.planner = PurePursuitPlanner()
        elif controller == 'stanley':
            from Control_Toolkit_ASF.Controllers.Stanley.stanley_planner import StanleyPlanner
            self.planner = StanleyPlanner()
        elif controller == 'manual':
            from Control_Toolkit_ASF.Controllers.Manual.manual_planner import manual_planner
            self.planner = manual_planner()
        elif controller == 'random':
            from Control_Toolkit_ASF.Controllers.Random.random_planner import random_planner
            self.planner = random_planner()
        else:
            NotImplementedError('{} is not a valid controller name for f1t'.format(controller))
            exit()
            
        self.planner.render_utils = self.render_utils
        self.planner.waypoint_utils = self.waypoint_utils

        self.use_waypoints_from_mpc = Settings.WAYPOINTS_FROM_MPC

        self.savse_recording = save_recording
        if save_recording:
            self.recorder = Recorder(driver=self)
        
        
        self.config_onlinelearning = yaml.load(
                open(os.path.join("SI_Toolkit_ASF", "config_onlinelearning.yml")),
                Loader=yaml.FullLoader
            )
        self.online_learning_activated = self.config_onlinelearning.get('activated', False)
        
        
        if self.online_learning_activated:
            from SI_Toolkit.Training.OnlineLearning import OnlineLearning

            if Settings.CONTROLLER == 'mpc':    
                    self.predictor = self.planner.mpc.predictor
            # else:
            #     self.predictor = PredictorWrapper()
            #     self.predictor.configure(
            #         batch_size=1,
            #         horizon=1,
            #         dt=Settings.TIMESTEP_CONTROL,
            #         computation_library=TensorFlowLibrary,
            #         predictor_specification="neural_parameter_determination"
            #     )
                
            self.online_learning = OnlineLearning(self.predictor, Settings.TIMESTEP_CONTROL, self.config_onlinelearning)

            
                
    
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
        
        
        if Settings.LIDAR_PLOT_SCANS:
            self.LIDAR.plot_lidar_data()
            
        car_state = self.car_state
        
        
        # print("Car state: ", car_state)
        # s = car_state
        # input = [s[ANGULAR_VEL_Z_IDX], s[LINEAR_VEL_X_IDX],s[POSE_THETA_IDX],s[STEERING_ANGLE_IDX]]
        # # print("input", input)
        # output_true = s[SLIP_ANGLE_IDX]
        # print("output as should be", output_true)
        # output = predict.predict_slip_angle_from_car_state(car_state)
        # print("output", output)
        
                
        imu_array = self.imu_simulator.update_car_state(car_state)
        self.planner.imu_data = imu_array
        self.current_imu_dict = self.imu_simulator.array_to_dict(imu_array)
        
        # if hasattr(self.planner, 'mu_predicted'):
        #     imu_dict['mu_predicted'] = self.planner.mu_predicted
        
        
        ranges = np.array(ranges)
        self.LIDAR.load_lidar_measurement(ranges)
        lidar_points = self.LIDAR.get_all_lidar_points_in_map_coordinates(
            car_state[POSE_X_IDX], car_state[POSE_Y_IDX], car_state[POSE_THETA_IDX])
        self.waypoint_utils.update_next_waypoints(car_state)
        obstacles = self.obstacle_detector.get_obstacles(ranges, car_state)          
                
        if self.use_waypoints_from_mpc:
            if self.control_index % Settings.PLAN_EVERY_N_STEPS == 0:
                pass_data_to_planner(self.waypoints_planner, self.waypoint_utils.next_waypoints, car_state, obstacles)
                self.waypoints_planner.process_observation(ranges, ego_odom)
                optimal_trajectory = self.waypoints_planner.mpc.optimizer.optimal_trajectory
                if optimal_trajectory is not None:
                    self.waypoints_from_mpc[:, WP_X_IDX] = optimal_trajectory[0, -len(self.waypoints_from_mpc):, POSE_X_IDX]
                    self.waypoints_from_mpc[:, WP_Y_IDX] = optimal_trajectory[0, -len(self.waypoints_from_mpc):, POSE_Y_IDX]
                    self.waypoints_from_mpc[:, WP_VX_IDX] = optimal_trajectory[0, -len(self.waypoints_from_mpc):, LINEAR_VEL_X_IDX]
                    angular_vel = optimal_trajectory[0, :, ANGULAR_VEL_Z_IDX]
                    linear_vel = optimal_trajectory[0, :, LINEAR_VEL_X_IDX]
                    curvature = np.divide(angular_vel, linear_vel, out=np.zeros_like(angular_vel), where=linear_vel != 0)
                    self.waypoints_from_mpc[:, WP_KAPPA_IDX] = curvature[-len(self.waypoints_from_mpc):]
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
        
        control_sequence_dict = None
        if hasattr(self.planner, 'optimal_control_sequence'):
            optimal_control_sequence = self.planner.optimal_control_sequence
            optimal_control_sequence = np.array(optimal_control_sequence)
            angular_control_sequence = optimal_control_sequence[:, 0]
            translational_control_sequence = optimal_control_sequence[:, 1]
            
            # Convert MPC's control sequence to dictionary for recording
            self.angular_control_dict = {"cs_a_{}".format(i): control for i, control in enumerate(angular_control_sequence)}
            self.translational_control_dict = {"cs_t_{}".format(i): control for i, control in enumerate(translational_control_sequence)}
            
            # if controller gives an optimal sequence (MPC), extract the N'th step with delay or the 0th step without delay
            self.angular_control, self.translational_control = optimal_control_sequence[Settings.EXECUTE_NTH_STEP_OF_CONTROL_SEQUENCE]
            
        
        # Rendering and recording
        label_dict = {
            '2: slip_angle': car_state[SLIP_ANGLE_IDX],
            '0: angular_control': self.angular_control,
            '1: translational_control': self.translational_control,
            '4: Surface Friction': Settings.SURFACE_FRICITON,
        }
        self.render_utils.set_label_dict(label_dict)
        
        self.render_utils.update(
            lidar_points= lidar_points,
            next_waypoints= WaypointUtils.get_interpolated_waypoints(self.waypoints_for_controller[:, (WP_X_IDX, WP_Y_IDX)], Settings.INTERPOLATE_LOCA_WP),
            car_state = car_state
        )
        self.render_utils.update_obstacles(obstacles)
        self.time = self.control_index*self.time_increment
        
        basic_dict = get_basic_data_dict(self)
        self.recorder.dict_data_to_save_basic.update(basic_dict)
        
        self.control_index += 1
        # print('angular control:', self.angular_control, 'translational control:', self.translational_control)
        return self.angular_control, self.translational_control

            
            
def pass_data_to_planner(planner, next_waypoints=None, car_state=None, obstacles=None):
    # Pass data to the planner
    if hasattr(planner, 'set_waypoints'):
        next_waypoints = WaypointUtils.get_interpolated_waypoints(next_waypoints, Settings.INTERPOLATE_LOCA_WP)
        planner.set_waypoints(next_waypoints)
    if hasattr(planner, 'set_car_state'):
        planner.set_car_state(car_state)
    if hasattr(planner, 'set_obstacles'):
        planner.set_obstacles(obstacles)
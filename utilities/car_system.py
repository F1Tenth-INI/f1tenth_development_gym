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
from utilities.waypoint_utils import WaypointUtils
# from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
# from SI_Toolkit.computation_library import TensorFlowLibrary

# from TrainingLite.slip_prediction import predict

from RaceTuner.TunerConnectorSim import TunerConnectorSim
from utilities.EmergencySlowdown import EmergencySlowdown
from utilities.LapAnalyzer import LapAnalyzer
from utilities.HistoryForger import HistoryForger
from utilities.StateMetricCalculator import StateMetricCalculator

class CarSystem:
    
    def __init__(self, controller=None, save_recording = Settings.SAVE_RECORDINGS):

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

        if(Settings.ALLOW_ALTERNATIVE_RACELINE):
            self.waypoint_utils_alternative = WaypointUtils(waypoint_file_name=f'{Settings.MAP_NAME}_wp_alternative', speed_scaling_file_name=f'{Settings.MAP_NAME}_speed_scaling_alternative.csv')
        else:
            self.waypoint_utils_alternative = None

        self.alternative_raceline = False
        self.timesteps_on_current_raceline = 0

        self.render_utils = RenderUtils()
        self.render_utils.waypoints = self.waypoint_utils.waypoint_positions

        if self.waypoint_utils_alternative is not None:
            self.render_utils.waypoints_alternative = self.waypoint_utils_alternative.waypoint_positions

        self.obstacle_detector = ObstacleDetector()

        self.waypoints_for_controller = None

        self.allow_rendering = True

        self.waypoints_planner = None
        self.waypoints_from_mpc = np.zeros((Settings.LOOK_AHEAD_STEPS, 7))
        if Settings.WAYPOINTS_FROM_MPC:
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
            self.waypoints_planner = mpc_planner()
            self.waypoints_planner.waypoint_utils = self.waypoint_utils


        # Planner
        self.controller_name = controller
        self.planner = initialize_planner(self.controller_name)
        self.angular_control_dict, self.translational_control_dict = if_mpc_define_cs_variables(self.planner)

        if hasattr(self.planner, 'waypoint_utils'):
            self.planner.waypoint_utils = self.waypoint_utils
        if hasattr(self.planner, 'LIDAR'):
            self.planner.LIDAR = self.LIDAR


        if Settings.FRICTION_FOR_CONTROLLER is not None:
            has_mpc = hasattr(self.planner, 'mpc')
            if has_mpc:
                predictor = self.planner.mpc.predictor.predictor
                if hasattr(predictor, 'next_step_predictor') and hasattr(predictor.next_step_predictor, 'env'):
                    predictor.next_step_predictor.env.change_friction_coefficient(Settings.FRICTION_FOR_CONTROLLER)


        if(hasattr(self.planner, 'render_utils')):
            self.planner.render_utils = self.render_utils

        self.use_waypoints_from_mpc = Settings.WAYPOINTS_FROM_MPC

        self.savse_recording = save_recording
        if save_recording:
            self.recorder = Recorder(driver=self)

        self.tuner_connector = None

        self.emergency_slowdown = EmergencySlowdown()

        self.config_onlinelearning = yaml.load(
                open(os.path.join("SI_Toolkit_ASF", "config_onlinelearning.yml")),
                Loader=yaml.FullLoader
            )
        self.online_learning_activated = self.config_onlinelearning.get('activated', False)

        self.lap_analyzer = LapAnalyzer(
            total_waypoints=len(self.waypoint_utils.waypoints),
            lap_finished_callback=self.lap_complete_cb
        )

        if Settings.FORGE_HISTORY:
            self.history_forger = HistoryForger()

        self.state_metric_calculator = StateMetricCalculator(
            environment_name="Car",
            initial_environment_attributes={
                "next_waypoints": self.waypoint_utils.next_waypoints,
            },
            recorder_base_dict=self.recorder.dict_data_to_save_basic
        )

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

    def launch_tuner_connector(self):
        try:
            self.tuner_connector = TunerConnectorSim()
        except OSError:
            print("Tunner connection not possible.")
    
    def set_car_state(self, car_state):
        self.car_state = car_state
    
    def render(self, e):
        self.render_utils.render(e)

    def process_observation(self, ranges=None, ego_odom=None):
        
        
        if Settings.LIDAR_PLOT_SCANS:
            self.LIDAR.plot_lidar_data()
            
        car_state = self.car_state

                
        # imu_array = self.imu_simulator.update_car_state(car_state)
        # self.planner.imu_data = imu_array
        # self.current_imu_dict = self.imu_simulator.array_to_dict(imu_array)
        
        # if hasattr(self.planner, 'mu_predicted'):
        #     imu_dict['mu_predicted'] = self.planner.mu_predicted

        ranges = np.array(ranges)
        self.LIDAR.update_ranges(ranges, car_state)
        processed_lidar_points = self.LIDAR.processed_points_map_coordinates

        self.waypoint_utils.update_next_waypoints(car_state)
        self.waypoint_utils.check_if_obstacle_on_my_raceline(processed_lidar_points)

        if self.waypoint_utils_alternative is not None:
            self.waypoint_utils_alternative.update_next_waypoints(car_state)
            self.waypoint_utils_alternative.check_if_obstacle_on_my_raceline(processed_lidar_points)



        if Settings.STOP_IF_OBSTACLE_IN_FRONT:
            corrected_next_waypoints_vx, use_alternative_waypoints_for_control_flag = self.emergency_slowdown.stop_if_obstacle_in_front(
                ranges,
                np.linspace(-2.35, 2.35, 1080),
                self.waypoint_utils.next_waypoints[:, WP_VX_IDX],
                car_state[STEERING_ANGLE_IDX]
            )
            self.waypoint_utils.next_waypoints[:, WP_VX_IDX] = corrected_next_waypoints_vx
            self.waypoint_utils.use_alternative_waypoints_for_control_flag = use_alternative_waypoints_for_control_flag

        obstacles = self.obstacle_detector.get_obstacles(ranges, car_state)

        if self.use_waypoints_from_mpc:
            if self.control_index % Settings.PLAN_EVERY_N_STEPS == 0:
                next_interpolated_waypoints = WaypointUtils.get_interpolated_waypoints(self.waypoint_utils.next_waypoints, Settings.INTERPOLATE_LOCA_WP)
                self.waypoints_planner.pass_data_to_planner(next_interpolated_waypoints, car_state, obstacles)
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

            # Decide between primary and alternative raceline

            if(not self.alternative_raceline and self.waypoint_utils.obstacle_on_raceline and self.timesteps_on_current_raceline > 150 and self.waypoint_utils_alternative is not None):
                # Check distance of raceline to alternative raceline
                distance_to_alternative_raceline = self.waypoint_utils_alternative.current_distance_to_raceline
                if(distance_to_alternative_raceline < 0.3):
                    self.alternative_raceline = True
                    self.timesteps_on_current_raceline = 0
                    print('Switching to alternative raceline')

            if(self.alternative_raceline and not self.waypoint_utils.obstacle_on_raceline and self.timesteps_on_current_raceline > 150):
                # Check distance of raceline to alternative raceline
                distance_to_raceline = self.waypoint_utils.current_distance_to_raceline
                if(distance_to_raceline < 0.3):
                    self.alternative_raceline = False
                    self.timesteps_on_current_raceline = 0
                    print('Switching to primary raceline')


            # Decide which raceline to use
            if(not self.alternative_raceline or self.waypoint_utils_alternative is None): #Primary raceline
                self.waypoints_for_controller = self.waypoint_utils.next_waypoints
            else:
                self.waypoints_for_controller = self.waypoint_utils_alternative.next_waypoints

            self.timesteps_on_current_raceline += 1

        if self.planner is None: # Planer not initialized
            return 0, 0

        if Settings.FORGE_HISTORY:
            self.history_forger.feed_planner_forged_history(car_state, ranges, self.waypoint_utils, self.planner, self.render_utils, Settings.INTERPOLATE_LOCA_WP)

        next_interpolated_waypoints_for_controller = WaypointUtils.get_interpolated_waypoints(self.waypoints_for_controller, Settings.INTERPOLATE_LOCA_WP)
        self.planner.pass_data_to_planner(next_interpolated_waypoints_for_controller, car_state, obstacles)


        # Control step
        if(self.control_index % Settings.OPTIMIZE_EVERY_N_STEPS == 0 or not hasattr(self.planner, 'optimal_control_sequence') ):
            self.angular_control, self.translational_control = self.planner.process_observation(ranges, ego_odom)


        self.state_metric_calculator.calculate_metrics(
            current_state=car_state,
            current_control=np.array([self.angular_control, self.translational_control]),
            updated_attributes={"next_waypoints": self.waypoint_utils.next_waypoints},
        )
        # Control Queue if exists
        if hasattr(self.planner, 'optimal_control_sequence'):
            self.optimal_control_sequence = self.planner.optimal_control_sequence
            next_control_step = self.optimal_control_sequence[self.control_index % Settings.OPTIMIZE_EVERY_N_STEPS + Settings.EXECUTE_NTH_STEP_OF_CONTROL_SEQUENCE]
            self.angular_control, self.translational_control = next_control_step

            
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

        if self.render_utils is not None:
            self.render_utils.set_label_dict(label_dict)
            self.render_utils.update(
                lidar_points= self.LIDAR.processed_points_map_coordinates,
                # next_waypoints= self.waypoints_for_controller[:, (WP_X_IDX, WP_Y_IDX)], # Might be more convenient to see what the controller actually gets
                next_waypoints= self.waypoint_utils.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)],
                next_waypoints_alternative=self.waypoint_utils_alternative.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)] if self.waypoint_utils_alternative is not None else None,
                car_state = car_state,
            )
            self.render_utils.update_obstacles(obstacles)


        if Settings.STOP_IF_OBSTACLE_IN_FRONT:
            self.emergency_slowdown.update_emergency_slowdown_sprites(
            car_x=car_state[POSE_X_IDX], car_y=car_state[POSE_Y_IDX], car_yaw=car_state[POSE_THETA_IDX],
            )
            self.render_utils.update(
                emergency_slowdown_sprites=self.emergency_slowdown.emergency_slowdown_sprites,
            )

        # self.render_utils.update_obstacles(obstacles)
        self.time = self.control_index*self.time_increment
                        
        # Update Lap Analyzer
        nearest_waypoint_index = self.waypoint_utils.nearest_waypoint_index
        distance_to_raceline = self.waypoint_utils.current_distance_to_raceline
        self.lap_analyzer.update(nearest_waypoint_index, self.time,distance_to_raceline)

        
        basic_dict = get_basic_data_dict(self)
        if Settings.FORGE_HISTORY:
            basic_dict.update({'forged_history_applied': lambda: self.history_forger.forged_history_applied})




        if(hasattr(self, 'recorder') and self.recorder is not None):
            self.recorder.dict_data_to_save_basic.update(basic_dict)
        
        self.control_index += 1
        # print('angular control:', self.angular_control, 'translational control:', self.translational_control)

        return self.angular_control, self.translational_control

    def lap_complete_cb(self,lap_time, mean_distance, std_distance, max_distance):
        print(f"Lap time: {lap_time}, Error: Mean: {mean_distance}, std: {std_distance}, max: {max_distance}")


def initialize_planner(controller: str):

    if controller is None:
        planner = None
    elif controller == 'mpc':
        from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
        planner = mpc_planner()
    elif controller == 'ftg':
        from Control_Toolkit_ASF.Controllers.FollowTheGap.ftg_planner import FollowTheGapPlanner
        planner = FollowTheGapPlanner()
    elif controller == 'neural':
        from Control_Toolkit_ASF.Controllers.NeuralNetImitator.nni_planner import NeuralNetImitatorPlanner
        planner = NeuralNetImitatorPlanner()
    elif controller == 'nni-lite':
        from Control_Toolkit_ASF.Controllers.NNLite.nni_lite_planner import NNLitePlanner
        planner = NNLitePlanner()
    elif controller == 'pp':
        from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
        planner = PurePursuitPlanner()
    elif controller == 'stanley':
        from Control_Toolkit_ASF.Controllers.Stanley.stanley_planner import StanleyPlanner
        planner = StanleyPlanner()
    elif controller == 'manual':
        from Control_Toolkit_ASF.Controllers.Manual.manual_planner import manual_planner
        planner = manual_planner()
    elif controller == 'random':
        from Control_Toolkit_ASF.Controllers.Random.random_planner import random_planner
        planner = random_planner()
    else:
        NotImplementedError('{} is not a valid controller name for f1t'.format(controller))
        exit()

    return planner


def if_mpc_define_cs_variables(planner):
    if hasattr(planner, 'mpc'):
        horizon = planner.mpc.predictor.horizon
        angular_control_dict = {"cs_a_{}".format(i): 0 for i in range(horizon)}
        translational_control_dict = {"cs_t_{}".format(i): 0 for i in range(horizon)}
        return angular_control_dict, translational_control_dict
    else:
        return {}, {}

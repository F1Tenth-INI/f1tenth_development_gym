import yaml
import numpy as np
from tqdm import trange
from typing import Optional


                
import os


# Utilities
from utilities.Settings import Settings

from utilities.state_utilities import *
from utilities.obstacle_detector import ObstacleDetector
from utilities.lidar_utils import LidarHelper

from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX, WP_KAPPA_IDX
from utilities.render_utilities import RenderUtils
from utilities.waypoint_utils import WaypointUtils

from utilities.imu_simulator import IMUSimulator
from utilities.Recorder import Recorder, get_basic_data_dict
from utilities.csv_logger import augment_csv_header_with_laptime
from utilities.saving_helpers import save_experiment_data, move_csv_to_crash_folder

from TrainingLite.rl_racing.train_model import *


# from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
# from SI_Toolkit.computation_library import TensorFlowLibrary

# from TrainingLite.slip_prediction import predict

from utilities.EmergencySlowdown import EmergencySlowdown
from utilities.LapAnalyzer import LapAnalyzer

if Settings.CONNECT_RACETUNER_TO_MAIN_CAR:
    from RaceTuner.TunerConnectorSim import TunerConnectorSim

if Settings.FORGE_HISTORY: # will import TF
    from utilities.HistoryForger import HistoryForger

class CarSystem:
    
    def __init__(self, controller=None, save_recording = Settings.SAVE_RECORDINGS, recorder_dict={}):

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
        self.laptimes = []

        self.angular_control = 0
        self.translational_control = 0

        
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
        
        if self.online_learning_activated:
            from SI_Toolkit.Training.OnlineLearning import OnlineLearning

            if Settings.CONTROLLER == 'mpc':
                    self.predictor = self.planner.mpc.predictor
                    
            self.online_learning = OnlineLearning(self.predictor, Settings.TIMESTEP_CONTROL, self.config_onlinelearning)

        if Settings.FORGE_HISTORY:
            self.history_forger = HistoryForger()

       
        # Recorder
        self.init_recorder_and_start(recorder_dict=recorder_dict)
        
        self.spin_counter = 0
        self.stuck_counter = 0
        self.last_steering = 0
        self.last_progress = 0
        self.rewards = []

           
            
    def launch_tuner_connector(self):
        try:
            self.tuner_connector = TunerConnectorSim()
        except OSError:
            print("Tunner connection not possible.")
    
    def set_car_state(self, car_state):
        self.car_state = car_state

    def set_scans(self, ranges):
        ranges = np.array(ranges)
        self.LIDAR.update_ranges(ranges, self.car_state)

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

        obstacles = self.obstacle_detector.get_obstacles(ranges, car_state)

        if(Settings.OPTIMIZE_FOR_RL):
            return 0,0


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

        if Settings.SAVE_STATE_METRICS and hasattr(self, 'state_metric_calculator'):
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
            
        


        if self.render_utils is not None:
            self.update_render_utils()


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
        self.lap_analyzer.update(nearest_waypoint_index, self.time, distance_to_raceline)

        
        basic_dict = get_basic_data_dict(self)
        if Settings.FORGE_HISTORY:
            basic_dict.update({'forged_history_applied': lambda: self.history_forger.forged_history_applied})

        if(hasattr(self, 'recorder') and self.recorder is not None):
            self.recorder.dict_data_to_save_basic.update(basic_dict)
            self.recorder.step()
        
        self.control_index += 1
        # print('angular control:', self.angular_control, 'translational control:', self.translational_control)
        
        reward = self._calculate_reward()
        self.rewards.append(reward)
        
        # print('Reward:', reward)

        return self.angular_control, self.translational_control


    def update_render_utils(self):

        car_state = self.car_state


             # Rendering and recording
        label_dict = {
            '2: slip_angle': car_state[SLIP_ANGLE_IDX],
            '0: angular_control': self.angular_control,
            '1: translational_control': self.translational_control,
            '4: Surface Friction': Settings.SURFACE_FRICTION,
        }

        self.render_utils.set_label_dict(label_dict)
        self.render_utils.update(
            lidar_points= self.LIDAR.processed_points_map_coordinates,
            # next_waypoints= self.waypoints_for_controller[:, (WP_X_IDX, WP_Y_IDX)], # Might be more convenient to see what the controller actually gets
            next_waypoints= self.waypoint_utils.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)],
            next_waypoints_alternative=self.waypoint_utils_alternative.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)] if self.waypoint_utils_alternative is not None else None,
            car_state = car_state,
        )
        # self.render_utils.update_obstacles(obstacles)

    def lap_complete_cb(self,lap_time, mean_distance, std_distance, max_distance):
        self.laptimes.append(lap_time)
        print(f"Lap time: {lap_time}, Error: Mean: {mean_distance}, std: {std_distance}, max: {max_distance}")

     
    '''
    Called on the end of the simulation before terminating the program
    '''   
    def on_simulation_end(self, collision=False):
        
        # import matplotlib.pyplot as plt
        # plt.clf()
        # plt.plot(self.rewards)
        # plt.show()
        
        if self.recorder is not None:    
            if self.recorder.recording_mode == 'offline':  # As adding lines to header needs saving whole file once again
                self.recorder.finish_csv_recording()            
            augment_csv_header_with_laptime(self.laptimes, self.recorder.csv_filepath)

            path_to_plots = None
            if Settings.SAVE_PLOTS:
                path_to_plots = save_experiment_data(self.recorder.csv_filepath)

            if collision:
                move_csv_to_crash_folder(self.recorder.csv_filepath, path_to_plots)
                
    def init_recorder_and_start(self, recorder_dict={}):
        self.recorder: Optional[Recorder] = None
        
        if Settings.SAVE_RECORDINGS and self.save_recordings:
            self.recorder = Recorder(driver=self)
            
            # Add more internal data to recording dict:
            self.recorder.dict_data_to_save_basic.update(
                {   
                    'nearest_wpt_idx': lambda: self.waypoint_utils.nearest_waypoint_index,
                }
            )
            # Add data from outside the car stysem
            self.recorder.dict_data_to_save_basic.update(recorder_dict)
       
            if Settings.FORGE_HISTORY:
                self.recorder.dict_data_to_save_basic.update(
                    {
                        'forged_history_applied': lambda: self.history_forger.forged_history_applied,
                    }
                )
            if Settings.SAVE_STATE_METRICS:
                from utilities.StateMetricCalculator import StateMetricCalculator
                self.state_metric_calculator = StateMetricCalculator(
                    environment_name="Car",
                    initial_environment_attributes={
                        "next_waypoints": self.waypoint_utils.next_waypoints,
                    },
                    recorder_base_dict=self.recorder.dict_data_to_save_basic
                )
            
            # Start Recording
            self.recorder.start_csv_recording()
            
    def _calculate_reward(self):
        waypoint_utils = self.waypoint_utils
        car_state = self.car_state

        reward = 0

        # Reward Track Progress (Incentivize Moving Forward)
        progress = waypoint_utils.get_cumulative_progress()
        delta_progress = progress - self.last_progress
        progress_reward = delta_progress * 100.0 + progress * 0.25
        if progress_reward < 0:
            progress_reward *= 3.0  # Increase penalty for going backwards
            
        if delta_progress < -0.9: # Lap complete
            progress_reward = 5
            # print("Lap complete")
            
        
        reward += progress_reward
        # print(f"Progress: {progress}, Reward: {progress_reward}")
        

        # ✅ 2. Reward Maintaining Speed (Prevent Stops and Stalls)
        speed = car_state[LINEAR_VEL_X_IDX]
        speed_reward = speed * 0.5    
        reward += speed_reward
        

        # # ✅ 3. Penalize Sudden Steering Changes
        steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        reward -= steering_diff * 0.05  # Increased penalty to discourage aggressive corrections
        

        
        # Penalize slip
        vy = abs(car_state[LINEAR_VEL_Y_IDX])
        reward -= vy * 0.1
        
        # Penalize Steering Angle
        steering_penalty = abs(car_state[STEERING_ANGLE_IDX]) * 0.01  # Scale penalty
        reward -= steering_penalty

        # Penalize Collisions
        # if self.simulation.obs["collisions"][0] == 1:
        #     reward -= 100  # Keep high penalty for crashes

        # ✅ 5. Penalize Distance from Raceline
        # nearest_waypoint_index, nearest_waypoint_dist = get_nearest_waypoint(car_state, waypoint_utils.next_waypoints)
        # if nearest_waypoint_dist < 0.05:
        #     nearest_waypoint_dist = 0
        # wp_penality = nearest_waypoint_dist * 1
        # reward -= wp_penality
        # if print:
            # print(f"Nearest waypoint distance: {nearest_waypoint_dist}, penality: {-wp_penality}")

        # Penalize if lidar scans are < 0.5 and penalize more for even less ranges
        # Find all ranges < 0.5
        lidar_penality = 0
        for i, range in enumerate(self.LIDAR.processed_ranges):
            angle = self.LIDAR.processed_angles_rad[i]
            if range < 0.5 and range != 0:
                lidar_penality += min(100, np.cos(angle) * 1 / range)
                    
        reward -= lidar_penality * 0.1

        #  Penalize Spinning (Fixing Instability)
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 15.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                reward -= self.spin_counter * 0.5
            
            if self.spin_counter >= 200:
                reward -= 100

        else:
            self.spin_counter = 0
            
            
        # Penalize beeing stuck
        if abs(car_state[LINEAR_VEL_X_IDX]) < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 10:
                reward -= 5
            
            if self.stuck_counter >= 200:
                reward -= 100

       
        self.last_steering = car_state[STEERING_ANGLE_IDX]
        self.last_progress = progress
        
        if(print_info):
            print(f"Reward: {reward}")
        return reward
        


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
        print(f"controller {controller} not recognized")
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

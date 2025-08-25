import yaml
import numpy as np
from tqdm import trange
from typing import Optional
import importlib
                
import os


# Utilities
from utilities.Settings import Settings

if not Settings.ROS_BRIDGE and Settings.RENDER_MODE is not None:
    from pynput import keyboard

from utilities.state_utilities import *
from utilities.obstacle_detector import ObstacleDetector
from utilities.lidar_utils import LidarHelper

from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX, WP_KAPPA_IDX # 35MB
from utilities.render_utilities import RenderUtils
from utilities.waypoint_utils import WaypointUtils

from utilities.imu_simulator import IMUSimulator
from utilities.Recorder import Recorder, get_basic_data_dict
from utilities.csv_logger import augment_csv_header_with_laptime
from utilities.saving_helpers import save_experiment_data, move_csv_to_crash_folder # 25MB

from TrainingLite.rl_racing.RewardCalculator import RewardCalculator


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

        # Pure control without noise
        self.angular_control_calculated = 0
        self.translational_control_calculated = 0
        
        # Control with added control noise
        self.angular_control = 0
        self.translational_control = 0
        
        self.control_noise = None
        
        # Initial values
        self.car_state = np.ones(len(STATE_VARIABLES))
        self.car_state_history = []
        self.car_state_noiseless = np.ones(len(STATE_VARIABLES))
        car_index = 1
        self.scans = None
        self.control_index = 0
        
        
        ### Utilities 
        
        # Waypoints
        self.waypoint_utils = WaypointUtils()
        if(Settings.ALLOW_ALTERNATIVE_RACELINE): # Second instance of waypoints for alternative raceline
            self.waypoint_utils_alternative = WaypointUtils(waypoint_file_name=f'{Settings.MAP_NAME}_wp_alternative', speed_scaling_file_name=f'{Settings.MAP_NAME}_speed_scaling_alternative.csv')
        else:
            self.waypoint_utils_alternative = None

        self.alternative_raceline = False
        self.timesteps_on_current_raceline = 0
        self.waypoints_for_controller = self.waypoint_utils.next_waypoints

        # Rendering
        self.render_utils = RenderUtils()
        self.render_utils.waypoints = self.waypoint_utils.waypoint_positions

        if self.waypoint_utils_alternative is not None:
            self.render_utils.waypoints_alternative = self.waypoint_utils_alternative.waypoint_positions
        self.allow_rendering = True
        
        # Obstacles
        self.obstacle_detector = ObstacleDetector()
        

        # Waypoints from MPC
        self.use_waypoints_from_mpc = Settings.WAYPOINTS_FROM_MPC

        self.waypoints_planner = None
        self.waypoints_from_mpc = np.zeros((Settings.LOOK_AHEAD_STEPS, 7))
        if Settings.WAYPOINTS_FROM_MPC:
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
            self.waypoints_planner = mpc_planner()
            self.waypoints_planner.waypoint_utils = self.waypoint_utils

    
        # Rewards
        self.reward_calculator = RewardCalculator()
        self.reward = 0

        ### Planner
        self.controller_name = controller
        self.initialize_controller(self.controller_name)
        self.angular_control_dict, self.translational_control_dict = if_mpc_define_cs_variables(self.planner)


            
        if Settings.FRICTION_FOR_CONTROLLER is not None:
            has_mpc = hasattr(self.planner, 'mpc')
            if has_mpc:
                predictor = self.planner.mpc.predictor.predictor
                if hasattr(predictor, 'next_step_predictor') and hasattr(predictor.next_step_predictor, 'env'):
                    predictor.next_step_predictor.env.change_friction_coefficient(Settings.FRICTION_FOR_CONTROLLER)


        


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
        # self.init_recorder_and_start(recorder_dict=recorder_dict)
        self.init_recorder(recorder_dict=recorder_dict)
        
        if(not Settings.ROS_BRIDGE):
            self.start_recorder()

           
    def initialize_controller(self, controller_name):
        
        self.planner = initialize_planner(controller_name)
        
        if(hasattr(self.planner, 'render_utils')):
            self.planner.render_utils = self.render_utils
        if(hasattr(self.planner, 'waypoint_utils')):
            self.planner.waypoint_utils = self.waypoint_utils
        
        
    def launch_tuner_connector(self):
        try:
            self.tuner_connector = TunerConnectorSim()
        except OSError:
            print("Tunner connection not possible.")
    
    def set_car_state(self, car_state):
        self.car_state = car_state
        self.car_state_history.append(car_state)

    def set_scans(self, ranges):
        ranges = np.array(ranges)
        self.LIDAR.update_ranges(ranges, self.car_state)

    def render(self, e):
        if Settings.RENDER_MODE is not None:
            self.render_utils.render(e)

    def process_observation(self, ranges=None, ego_odom=None):
        
        #Car state and Lidar are updated by parent
        
        car_state = self.car_state
        
        self.update_waypoints()
        obstacles = self.obstacle_detector.get_obstacles(self.LIDAR.processed_ranges, car_state)


        # Pass data to planner
        if hasattr(self.planner, 'pass_data_to_planner'):
            self.planner.pass_data_to_planner(self.waypoints_for_controller, car_state, obstacles)


        # Control step
        if self.planner is not None:
            if(self.control_index % Settings.OPTIMIZE_EVERY_N_STEPS == 0):
                self.angular_control, self.translational_control = self.planner.process_observation(ranges, ego_odom)
                
            # Extract from mpc control sequence
            if hasattr(self.planner, 'optimal_control_sequence'):
                self.angular_control, self.translational_control = self.extract_control_from_control_sequence()
        else: # planner = none
            self.angular_control = 0
            self.translational_control = 0
        
        
        self.process_data_post_control()
        
        

        self.control_index += 1
        self.time += self.time_increment
        
        self.angular_control_calculated = self.angular_control
        self.translational_control_calculated = self.translational_control
        
        # Add noise to control
        self.angular_control, self.translational_control = self.add_control_noise(np.array([self.angular_control, self.translational_control]))

        return self.angular_control, self.translational_control


    '''
    Update waypoints, check for obstacles and adjust waypoints / suggested speed
    '''
    def update_waypoints(self):
        # Update waypoints
        self.waypoint_utils.update_next_waypoints(self.car_state)
        self.waypoint_utils.check_if_obstacle_on_my_raceline(self.LIDAR.processed_points_map_coordinates)
        if self.waypoint_utils_alternative is not None:
            self.waypoint_utils_alternative.update_next_waypoints(self.car_state)
            self.waypoint_utils_alternative.check_if_obstacle_on_my_raceline(self.LIDAR.processed_points_map_coordinates)
        
        # Adjust waypoints
        if self.use_waypoints_from_mpc:
            self.waypoints_for_controller = self.get_mpc_waypoints_from_mpc()
        else:
            self.waypoints_for_controller = self.chose_raceline_from_wpts()
        self.handle_emergency_slowdown() # overwrite self.waypoint_utils.next_waypoints if necessary
        
        
    def update_render_utils(self):

        car_state = self.car_state


             # Rendering and recording
        label_dict = {
            '0: angular_control': self.angular_control,
            '1: translational_control': self.translational_control,
            'yaw': car_state[POSE_THETA_IDX],
            '4: Surface Friction': Settings.SURFACE_FRICTION,
            '5: Laptimes:': ', '.join(f'{lt:.2f}' for lt in self.laptimes),
            '6: Reward': self.reward,
            'Distance to raceline': self.waypoint_utils.current_distance_to_raceline,
            'speed': car_state[LINEAR_VEL_X_IDX],
            'Wp_idx': self.waypoint_utils.nearest_waypoint_index,
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
    
    # The MPC returns a control seqwuence instead of a single control: extract for delay compensation
    def extract_control_from_control_sequence(self) -> tuple:
        
        optimal_control_sequence = self.planner.optimal_control_sequence
        optimal_control_sequence = np.array(optimal_control_sequence)
        angular_control_sequence = optimal_control_sequence[:, 0]
        translational_control_sequence = optimal_control_sequence[:, 1]
        
        # Convert MPC's control sequence to dictionary for recording
        self.angular_control_dict = {"cs_a_{}".format(i): control for i, control in enumerate(angular_control_sequence)}
        self.translational_control_dict = {"cs_t_{}".format(i): control for i, control in enumerate(translational_control_sequence)}
        
        # if controller gives an optimal sequence (MPC), extract the N'th step with delay or the 0th step without delay
        mpc_execution_step = (int)(Settings.CONTROL_DELAY / self.planner.config_optimizer["mpc_timestep"])
        angular_control, translational_control = optimal_control_sequence[mpc_execution_step]
        
        return angular_control, translational_control
        
    # Decide between primary and alternative raceline
    def chose_raceline_from_wpts(self) -> np.ndarray:
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
            waypoints_for_controller = self.waypoint_utils.next_waypoints
        else:
            waypoints_for_controller = self.waypoint_utils_alternative.next_waypoints

        self.timesteps_on_current_raceline += 1
        
        return waypoints_for_controller

    # Get waypoints from MPC in case they are generated with it
    def get_mpc_waypoints_from_mpc(self) -> np.ndarray:
        
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
                return self.waypoints_from_mpc
            else:
                return self.waypoint_utils.next_waypoints
        else:
            return self.waypoints_for_controller

    
    def handle_emergency_slowdown(self):
        if Settings.STOP_IF_OBSTACLE_IN_FRONT:
            car_state = self.car_state
            corrected_next_waypoints_vx, use_alternative_waypoints_for_control_flag = self.emergency_slowdown.stop_if_obstacle_in_front(
                self.LIDAR.processed_ranges,
                self.LIDAR.processed_angles_rad,
                self.waypoint_utils.next_waypoints[:, WP_VX_IDX],
                car_state[STEERING_ANGLE_IDX]
            )
            self.waypoint_utils.next_waypoints[:, WP_VX_IDX] = corrected_next_waypoints_vx

            self.emergency_slowdown.update_emergency_slowdown_sprites(
                car_x=car_state[POSE_X_IDX], car_y=car_state[POSE_Y_IDX], car_yaw=car_state[POSE_THETA_IDX],
            )
            self.render_utils.update(
                emergency_slowdown_sprites=self.emergency_slowdown.emergency_slowdown_sprites,
            )
    
    def process_data_post_control(self):
    
        # Update data post control
        if self.render_utils is not None:
            self.update_render_utils()
        self.lap_analyzer.update(nearest_waypoint_index = self.waypoint_utils.nearest_waypoint_index, time_now = self.time, distance_to_raceline = self.waypoint_utils.current_distance_to_raceline)

        if Settings.FORGE_HISTORY:
            self.history_forger.feed_planner_forged_history(self.car_state, self.LIDAR.all_lidar_ranges, self.waypoint_utils, self.planner, self.render_utils, Settings.INTERPOLATE_LOCA_WP)
        if Settings.SAVE_STATE_METRICS and hasattr(self, 'state_metric_calculator'):
            self.state_metric_calculator.calculate_metrics(
                current_state=self.car_state,
                current_control=np.array([self.angular_control, self.translational_control]),
                updated_attributes={"next_waypoints": self.waypoint_utils.next_waypoints},
            )
            
        if Settings.FORGE_HISTORY:
            basic_dict = get_basic_data_dict(self)
            basic_dict.update({'forged_history_applied': lambda: self.history_forger.forged_history_applied})

        if(hasattr(self, 'recorder') and self.recorder is not None):
            basic_dict = get_basic_data_dict(self)
            self.recorder.dict_data_to_save_basic.update(basic_dict)
            self.recorder.step()
        
        self.reward = self.reward_calculator._calculate_reward(self)        
        # print('Reward:', self.reward)
    
    '''
    Called by LapAnalyser when a lap is completed
    '''
    def lap_complete_cb(self,lap_time, mean_distance, std_distance, max_distance):
        self.laptimes.append(lap_time)
        print(f"Lap time: {lap_time}, Error: Mean: {mean_distance}, std: {std_distance}, max: {max_distance}")

     
    
    '''
    Initialize the recorder, add basic dict active dictionary and start recording
    '''        
    def init_recorder(self,recorder_dict={}):
        self.recorder: Optional[Recorder] = None
        
        if Settings.SAVE_RECORDINGS and self.save_recordings:
            self.recorder = Recorder(driver=self)
            
            # Add more internal data to recording dict:
            self.recorder.dict_data_to_save_basic.update(
                {   
                    'nearest_wpt_idx': lambda: self.waypoint_utils.nearest_waypoint_index,
                    'reward': lambda: self.reward,
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
            

    def on_press(self,key):
        try:
            if key.char == 'r':  # Press 'r' to start recording
                print("Start recording...")
                self.start_recorder()  # Replace 'car' with your object instance
        except AttributeError:
            pass  # For special keys like shift, ctrl, etc.

    def start_keyboard_listener(self):
        if Settings.RENDER_MODE is None:
            print("Keyboard listener not started, starting recording automatically")
            self.start_recorder()
            return
        if Settings.KEYBOARD_LISTENER_ON:
            listener = keyboard.Listener(on_press=self.on_press)
            listener.start()

    def start_recorder(self):
        self.recorder.start_csv_recording()
    


    # def init_recorder_and_start(self, recorder_dict={}):
    #     self.recorder: Optional[Recorder] = None
        
    #     if Settings.SAVE_RECORDINGS and self.save_recordings:
    #         self.recorder = Recorder(driver=self)
            
    #         # Add more internal data to recording dict:
    #         self.recorder.dict_data_to_save_basic.update(
    #             {   
    #                 'nearest_wpt_idx': lambda: self.waypoint_utils.nearest_waypoint_index,
    #                 'reward': lambda: self.reward,
    #             }
    #         )
    #         # Add data from outside the car stysem
    #         self.recorder.dict_data_to_save_basic.update(recorder_dict)
       
    #         if Settings.FORGE_HISTORY:
    #             self.recorder.dict_data_to_save_basic.update(
    #                 {
    #                     'forged_history_applied': lambda: self.history_forger.forged_history_applied,
    #                 }
    #             )
    #         if Settings.SAVE_STATE_METRICS:
    #             from utilities.StateMetricCalculator import StateMetricCalculator
    #             self.state_metric_calculator = StateMetricCalculator(
    #                 environment_name="Car",
    #                 initial_environment_attributes={
    #                     "next_waypoints": self.waypoint_utils.next_waypoints,
    #                 },
    #                 recorder_base_dict=self.recorder.dict_data_to_save_basic
    #             )
            
            # # Start Recording
            # self.recorder.start_csv_recording()

    
    def add_control_noise(self, control):
        if self.control_noise is None or self.control_index % Settings.CONTROL_NOISE_DURATION == 0:
            noise_level = Settings.NOISE_LEVEL_CONTROL
            noise_array = np.array(noise_level) * np.random.uniform(-1, 1, len(noise_level))
            self.control_noise = noise_array
        control_with_noise = control + self.control_noise
        return control_with_noise


    
    '''
    Called by parent on the end of the simulation before terminating the program
    Plotting and saving data
    '''   
    def on_simulation_end(self, collision=False):
        
        # import matplotlib.pyplot as plt
        # plt.clf()
        # plt.plot(self.reward_calculator.reward_history)
        # plt.savefig('rewards.png')
        
        # # Also plot the accumulated rewards:
        # accumulated_rewards = np.cumsum(self.reward_calculator.reward_history)
        # plt.clf()
        # plt.plot(accumulated_rewards)
        # plt.savefig('accumulated_rewards.png')
        # plt.clf()
        
       
        
        if self.recorder is not None:    
            
            if Settings.SAVE_REWARDS:
                self.reward_calculator.plot_history(save_path=self.recorder.recording_path)
            
            
            if self.recorder.recording_mode == 'offline':  # As adding lines to header needs saving whole file once again
                self.recorder.finish_csv_recording()            
            augment_csv_header_with_laptime(self.laptimes, self.recorder.csv_filepath)

            path_to_plots = None
            if Settings.SAVE_PLOTS:
                path_to_plots = save_experiment_data(self.recorder.csv_filepath)

            if collision:
                index = min(len(self.car_state_history), 200)
                
                # Save or append self.control_index to csv file
                # if a csv file called survival.csv already exists, just add a new line, otherwise cfreate the file
                with open('survival.csv', 'a') as f:
                    f.write(f"{self.control_index}\n")
          
                
                print('Collision detected, moving csv to crash folder')
                print('Car State at crtash:', self.car_state)
                print('Car State at -index steps:', self.car_state_history[-index])
                
                # Save to csv file
                np.savetxt("Test.csv", [self.car_state_history[-index]], delimiter=",")
                move_csv_to_crash_folder(self.recorder.csv_filepath, path_to_plots)
                
def initialize_planner(controller: str):

    if controller is None:
            planner = None
    elif controller == 'mpc':
        from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
        planner = mpc_planner()
    elif controller == 'mppi-lite':
        from Control_Toolkit_ASF.Controllers.MPPILite.mppi_lite_planner import MPPILitePlanner
        planner = MPPILitePlanner() 
    elif controller == 'mppi-lite-jax':
        from Control_Toolkit_ASF.Controllers.MPPILite.mppi_lite_jax_planner import MPPILitePlanner
        planner = MPPILitePlanner()
    elif controller == 'ftg':
        from Control_Toolkit_ASF.Controllers.FollowTheGap import ftg_planner
        importlib.reload(ftg_planner)
        planner = ftg_planner.FollowTheGapPlanner()
    elif controller == 'neural':
        from Control_Toolkit_ASF.Controllers.NeuralNetImitator import nni_planner
        importlib.reload(nni_planner)
        planner = nni_planner.NeuralNetImitatorPlanner()
    elif controller == 'nni-lite':
        from Control_Toolkit_ASF.Controllers.NNLite import nni_lite_planner
        importlib.reload(nni_lite_planner)
        planner = nni_lite_planner.NNLitePlanner()
    elif controller == 'pp':
        from Control_Toolkit_ASF.Controllers.PurePursuit import pp_planner
        importlib.reload(pp_planner)
        planner = pp_planner.PurePursuitPlanner()
    elif controller == 'stanley':
        from Control_Toolkit_ASF.Controllers.Stanley import stanley_planner
        importlib.reload(stanley_planner)
        planner = stanley_planner.StanleyPlanner()
    elif controller == 'sysid':
        from Control_Toolkit_ASF.Controllers.SysId import sysid_planner
        importlib.reload(sysid_planner)
        planner = sysid_planner.SysIdPlanner()
    elif controller == 'manual':
        from Control_Toolkit_ASF.Controllers.Manual import manual_planner
        importlib.reload(manual_planner)
        planner = manual_planner.manual_planner()
    elif controller == 'random':
        from Control_Toolkit_ASF.Controllers.Random import random_planner
        importlib.reload(random_planner)
        planner = random_planner.random_planner()
    else:
        print(f"controller {controller} not recognized")
        raise NotImplementedError('{} is not a valid controller name for f1t'.format(controller))
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

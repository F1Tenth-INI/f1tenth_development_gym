import yaml
import numpy as np
from tqdm import trange
from typing import Any, Optional
import importlib
from collections import deque
                
import os


# Utilities
from utilities.Settings import Settings

from utilities.state_utilities import *
from utilities.obstacle_detector import ObstacleDetector
from utilities.lidar_utils import LidarHelper

from utilities.waypoint_utils import WP_D_LEFT_IDX, WP_D_RIGHT_IDX, WP_X_IDX, WP_Y_IDX, WP_VX_IDX, WP_KAPPA_IDX # 35MB
from utilities.render_utilities import RenderUtils
from utilities.waypoint_utils import WaypointUtils

from utilities.Recorder import Recorder, get_basic_data_dict
from utilities.csv_logger import augment_csv_header_with_laptime
from utilities.saving_helpers import save_experiment_data, move_csv_to_crash_folder, experiment_analysis_path # 25MB
from utilities.imu_utilities import IMUUtilities

# Bounded history for controller observations (SAC/NNI need ~25; we append up to twice per control step).
CAR_STATE_HISTORY_MAXLEN = 128
CONTROL_HISTORY_MAXLEN = 128

try:
    from TrainingLite.rl_racing.RewardCalculator import RewardCalculator
except ModuleNotFoundError:
    from f1tenth_development_gym.TrainingLite.rl_racing.RewardCalculator import RewardCalculator


# from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
# from SI_Toolkit.computation_library import TensorFlowLibrary

# from TrainingLite.slip_prediction import predict

from utilities.EmergencySlowdown import EmergencySlowdown
from utilities.LapAnalyzer import LapAnalyzer
from utilities.episode_termination import EpisodeTerminator
from utilities.opponent_tracker import OpponentTracker
from utilities.recording_replay import get_virtual_opponent_poses_for_render
from utilities.virtual_opponents import VirtualOpponents

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
        self.lidar_utils = LidarHelper()
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
        self.car_state_history = deque(maxlen=CAR_STATE_HISTORY_MAXLEN)
        self.control_history = deque(maxlen=CONTROL_HISTORY_MAXLEN)
        car_index = 1
        self.scans = None
        self.control_index = 0


        self.driver_observation = None
        self.controller_observation = None
        self.imu = IMUUtilities.zeros_dict()
        self.motor_sensors = {}
        
        
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
        self.render_utils.waypoints_full = self.waypoint_utils.waypoints

        if self.waypoint_utils_alternative is not None:
            self.render_utils.waypoints_alternative = self.waypoint_utils_alternative.waypoint_positions
        self.allow_rendering = True
        

        # Waypoints from MPC
        self.use_waypoints_from_mpc = Settings.WAYPOINTS_FROM_MPC

        self.waypoints_planner = None
        self.waypoints_from_mpc = np.zeros((Settings.LOOK_AHEAD_STEPS, 7))
        if Settings.WAYPOINTS_FROM_MPC:
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
            self.waypoints_planner = mpc_planner()
            self.waypoints_planner.waypoint_utils = self.waypoint_utils

    
        self.obstacle_detector = ObstacleDetector()
        self.reward_calculator = RewardCalculator()
        self.episode_terminator = EpisodeTerminator()
        self.virtual_opponents = VirtualOpponents.from_settings()
        if bool(getattr(Settings, "OPPONENT_TRACKER_ENABLED", False)):
            self.opponent_tracker = OpponentTracker.from_settings()
        else:
            self.opponent_tracker = None
        self.reward = 0
        self.reward_components = {}

        ### Planner
        self.controller_name = controller
        self.initialize_controller(self.controller_name)
        self.angular_control_dict, self.translational_control_dict = if_mpc_define_cs_variables(self.planner)

        # Other utilities
        self.tuner_connector = None  # Initialize before potential assignment
        if Settings.CONNECT_RACETUNER_TO_MAIN_CAR:
            self.launch_tuner_connector()

        if Settings.FRICTION_FOR_CONTROLLER is not None:
            has_mpc = hasattr(self.planner, 'mpc')
            if has_mpc:
                predictor = self.planner.mpc.predictor.predictor
                if hasattr(predictor, 'next_step_predictor') and hasattr(predictor.next_step_predictor, 'env'):
                    predictor.next_step_predictor.env.change_friction_coefficient(Settings.FRICTION_FOR_CONTROLLER)


        


        self.savse_recording = save_recording

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

        self.backward_predictor = None
        
        self.backward_predictor = None
        
        if(not Settings.ROS_BRIDGE):
            self.start_recorder()

    def reset(self):
        self.car_state = None
        self.driver_observation = None
        self.imu = IMUUtilities.zeros_dict()
        self.motor_sensors = {}
        self.laptimes = []
        self.lap_limit_reached = False
        self._virtual_opponent_collision = False
        self.virtual_opponents = VirtualOpponents.from_settings()

        self.control_index = 0
        self.control_history = deque(maxlen=CONTROL_HISTORY_MAXLEN)
        self.car_state_history = deque(maxlen=CAR_STATE_HISTORY_MAXLEN)
        self.lidar_utils.reset()
        self.waypoint_utils.reset()
        if self.reward_calculator is not None:
            self.reward_calculator.reset()
        if self.episode_terminator is not None:
            self.episode_terminator.reset()
        if self.lap_analyzer is not None:
            self.lap_analyzer.reset()
        self.render_utils.reset()
        if self.virtual_opponents is not None:
            self.virtual_opponents.reset()
        if self.opponent_tracker is not None:
            self.opponent_tracker.reset()
        self.planner.reset()
        self.waypoint_utils.reset_frenet_progress()
        # self.waypoint_utils.get_frenet_coordinates(self.car_state)

    def initialize_controller(self, controller_name):
        
        self.planner = initialize_planner(controller_name)
        self.planner.render_utils = self.render_utils
        self.planner.waypoint_utils = self.waypoint_utils
        self.planner.lidar_utils = self.lidar_utils
        self.planner.obstacle_detector = self.obstacle_detector
        
        
    def launch_tuner_connector(self):
        try:
            self.tuner_connector = TunerConnectorSim()
        except OSError:
            print("Tunner connection not possible.")
    
    def set_car_state(self, car_state):
        self.car_state = np.asarray(car_state, dtype=np.float32)

    def _append_car_state_history(self, car_state=None):
        """Append one car state snapshot to history (post-step in on_step_end)."""
        if car_state is None:
            car_state = self.car_state
        self.car_state_history.append(np.asarray(car_state, dtype=np.float32).copy())

    def set_scans(self, ranges):
        ranges = np.array(ranges)
        if self.virtual_opponents is not None and self.car_state is not None:
            ranges = self.virtual_opponents.apply_to_scan(self.car_state, ranges)
        self.lidar_utils.update_ranges(ranges, self.car_state)

    def set_sensors(self, sensors):
        """Store raw sensor readings from driver_obs['sensors']."""
        self.imu = sensors["imu"]
        self.motor_sensors = sensors["motor_sensors"]

    def render(self, e):
        if Settings.RENDER_MODE is not None:
            self.render_utils.render(e)

    def _apply_observation(self, observation):
        self.driver_observation = observation
        # `driver_obs` is expected to always contain these fields.
        scans = observation["scans"]
        sensors = observation["sensors"]
        self.set_car_state(observation["car_state"])
        self.set_sensors(sensors)
        self._update_waypoint_indices()
        if self.virtual_opponents is not None:
            env_time = observation.get("env", {}).get("time", 0.0)
            self.virtual_opponents.set_state(
                self.waypoint_utils.nearest_waypoint_index,
                float(env_time),
            )
        self.set_scans(scans)
        self._finalize_waypoints_for_control()

    def _build_planner_step_end_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Build post-step controller observation for planner transition logging."""
        controller_observation = self._build_controller_observation(observation)
        controller_observation.update({
            "reward": observation["reward"],
            "done": observation["done"],
            "info": observation.get("info", {}),
            "truncated": observation.get("truncated"),
        })
        return controller_observation

    def _build_controller_observation(self, driver_observation: dict[str, Any]) -> dict[str, Any]:
        """Enrich raw driver observation with CarSystem-computed planner fields."""
        controller_observation = {
            **driver_observation,
            "next_waypoints": np.asarray(self.waypoint_utils.next_waypoints, dtype=np.float32),
            "state_history": np.asarray(self.car_state_history, dtype=np.float32),
            "control_history": np.asarray(self.control_history, dtype=np.float32),
            "frenet_coordinates": np.asarray(
                self.waypoint_utils.frenet_coordinates, dtype=np.float32
            ),
            # Flatten sensors for planners / observation builders (also under driver_obs["sensors"]).
            "imu": self.imu,
            "motor_sensors": self.motor_sensors,
        }
        controller_observation["processed_ranges"] = np.asarray(
            self.lidar_utils.processed_ranges, dtype=np.float32
        )
        controller_observation["lidar_points"] = self.lidar_utils.processed_points_map_coordinates
        if self.virtual_opponents is not None:
            from utilities.virtual_opponents import get_ego_car_dimensions

            ego_length, ego_width = get_ego_car_dimensions()
            controller_observation["virtual_opponent_poses"] = self.virtual_opponents.get_poses()
            controller_observation["min_virtual_opponent_distance"] = (
                self.virtual_opponents.min_clearance_to_ego(
                    self.car_state, ego_length, ego_width
                )
            )
            controller_observation["virtual_opponent_collision"] = bool(
                getattr(self, "_virtual_opponent_collision", False)
            )
        controller_observation.update(
            self.opponent_tracker.to_controller_observation(self.car_state)
            if self.opponent_tracker is not None
            else OpponentTracker.empty_controller_observation()
        )
        return controller_observation

    def process_observation(self, driver_observation):
                
        # Control step
        # observation: car_state, scans/lidar, sensors (imu + drivetrain), from sim or ROS bridge.
        # env_state (optional): full-environment snapshot in driver_observation for multi-agent sim.
        self.env_state = driver_observation.get("env_state")
        self._apply_observation(driver_observation)
        # self._update_opponent_tracker() // done on_step_end
        if not self.car_state_history:
            self._append_car_state_history()

        if self.planner is not None:
            controller_observation = self._build_controller_observation(driver_observation)
            self.angular_control, self.translational_control = self.planner.process_observation(
                controller_observation
            )

            # MPC delay compensation: only when a control sequence exists (not during startup ramp).
            if getattr(self.planner, 'optimal_control_sequence', None) is not None:
                self.angular_control, self.translational_control = self.extract_control_from_control_sequence()
            
        else: # planner == None
            self.angular_control = 0
            self.translational_control = 0
        

        self.angular_control_calculated = self.angular_control
        self.translational_control_calculated = self.translational_control
        
        # Add noise to control
        self.angular_control, self.translational_control = self.add_control_noise(np.array([self.angular_control, self.translational_control]))
        self.control_history.append(
            np.array([self.angular_control, self.translational_control], dtype=np.float32)
        )
        
        self.process_data_post_control()
        
        self.control_index += 1
        self.time += self.time_increment
        
        return self.angular_control, self.translational_control

    def on_step_end(self, observation=None):
        if observation is None:
            return

        self._apply_observation(observation)
        self._update_opponent_tracker()
        self._append_car_state_history()

        virtual_opponent_collision = False
        if (
            self.virtual_opponents is not None
            and self.car_state is not None
            and bool(getattr(Settings, "TERMINATE_ON_VIRTUAL_OPPONENT_COLLISION", False))
        ):
            from utilities.virtual_opponents import get_ego_car_dimensions

            ego_length, ego_width = get_ego_car_dimensions()
            virtual_opponent_collision = self.virtual_opponents.collides_with_ego(
                self.car_state, ego_length, ego_width
            )
        self._virtual_opponent_collision = virtual_opponent_collision
        if virtual_opponent_collision:
            observation["collision"] = True
        
        # TODO: Recording
        controller_observation = self._build_controller_observation(observation)
        episode_termination = self.episode_terminator.evaluate(
            controller_observation, observation
        )
        controller_observation["episode_termination"] = episode_termination

        reward_result = self.reward_calculator._calculate_reward(controller_observation)
        self.reward = float(reward_result["total_reward"])
        self.reward_components = dict(reward_result.get("components") or {})

        info = {
            # Snapshot current lap history to avoid sharing a mutable list
            # reference across stored transitions.
            "lap_times": list(self.laptimes),
            "reward_components": dict(self.reward_components),
            **episode_termination,
        }

        observation.update({
            "reward": self.reward,
            "info": info,
            "truncated": bool(episode_termination["truncated"]),
            "done": bool(episode_termination["done"]),
            "episode_termination": episode_termination,
        })
        if self.render_utils is not None:
            self.update_render_utils()

        if self.planner is not None and hasattr(self.planner, 'on_step_end'):
            self.planner.on_step_end(
                self._build_planner_step_end_observation(observation)
            )

        # Do not reset the reward calculator here: render_env() runs after on_step_end
        # and must still publish the terminal-step reward. driver.reset() clears it.



    def _update_waypoint_indices(self):
        """Refresh nearest waypoint and look-ahead windows from the current car pose."""
        car_state = self.car_state
        self.waypoint_utils.update_next_waypoints(car_state)
        if self.waypoint_utils_alternative is not None:
            self.waypoint_utils_alternative.update_next_waypoints(car_state)

    def _finalize_waypoints_for_control(self):
        """Obstacle checks and raceline selection (requires up-to-date lidar)."""
        lidar_points = self.lidar_utils.processed_points_map_coordinates
        self.waypoint_utils.check_if_obstacle_on_my_raceline(lidar_points)
        if self.waypoint_utils_alternative is not None:
            self.waypoint_utils_alternative.check_if_obstacle_on_my_raceline(lidar_points)

        if self.use_waypoints_from_mpc:
            self.waypoints_for_controller = self.get_mpc_waypoints_from_mpc()
        else:
            self.waypoints_for_controller = self.chose_raceline_from_wpts()
        self.handle_emergency_slowdown()
        self.waypoint_utils.get_frenet_coordinates(self.car_state)

    def _update_opponent_tracker(self):
        """Detect/track opponents from the ego lidar (once per control step)."""
        if self.opponent_tracker is None or self.car_state is None:
            return
        self.opponent_tracker.update(
            self.car_state,
            self.lidar_utils.all_lidar_ranges,
            self.lidar_utils.all_angles_rad,
            self.waypoint_utils.get_corridor_waypoints(self.opponent_tracker.max_range),
        )

    def set_waypoints(self):
        """Backward-compatible entry point for waypoint refresh."""
        self._update_waypoint_indices()
        self._finalize_waypoints_for_control()

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
        label_dict.update(IMUUtilities.overlay_label_dict(self.imu))
        for name, value in (self.reward_components or {}).items():
            label_dict[f'reward: {name}'] = float(value)
        label_dict['reward: total'] = float(self.reward)
        self.render_utils.set_label_dict(label_dict)
        virtual_opponent_poses = get_virtual_opponent_poses_for_render(self)
        if virtual_opponent_poses is None:
            virtual_opponent_poses = np.empty((0, 3), dtype=np.float32)
        self.render_utils.update(
            lidar_points= self.lidar_utils.processed_points_map_coordinates,
            # next_waypoints= self.waypoints_for_controller[:, (WP_X_IDX, WP_Y_IDX)], # Might be more convenient to see what the controller actually gets
            next_waypoints= self.waypoint_utils.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)],
            next_waypoints_alternative=self.waypoint_utils_alternative.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)] if self.waypoint_utils_alternative is not None else None,
            car_state = car_state,
            track_border_points = self.waypoint_utils.get_track_border_positions(self.waypoint_utils.next_waypoints),
            virtual_opponents=virtual_opponent_poses,
            detected_opponents=(
                self.opponent_tracker.get_render_points()
                if self.opponent_tracker is not None
                else None
            ),
        )
        # self.render_utils.update_obstacles(obstacles)

    # The MPC returns a control sequence instead of a single control: extract for delay compensation
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
    def _primary_raceline_blocked(self) -> bool:
        if self.waypoint_utils.obstacle_on_raceline:
            return True
        if self.virtual_opponents is None or self.car_state is None:
            return False
        from utilities.virtual_opponents import get_ego_car_dimensions

        ego_length, ego_width = get_ego_car_dimensions()
        clearance = self.virtual_opponents.min_clearance_to_ego(
            self.car_state, ego_length, ego_width
        )
        return clearance < 4.0

    def chose_raceline_from_wpts(self) -> np.ndarray:
        alt = self.waypoint_utils_alternative
        min_dwell_steps = 50  # ~1 s at 50 Hz; avoids raceline chatter

        primary_blocked = self._primary_raceline_blocked()
        alt_blocked = alt.obstacle_on_raceline if alt is not None else False

        if (
            not self.alternative_raceline
            and alt is not None
            and primary_blocked
            and not alt_blocked
            and self.timesteps_on_current_raceline > min_dwell_steps
        ):
            self.alternative_raceline = True
            self.timesteps_on_current_raceline = 0
            print('Switching to alternative raceline')

        if (
            self.alternative_raceline
            and not primary_blocked
            and self.timesteps_on_current_raceline > min_dwell_steps
            and self.waypoint_utils.current_distance_to_raceline < 0.3
        ):
            self.alternative_raceline = False
            self.timesteps_on_current_raceline = 0
            print('Switching to primary raceline')

        if not self.alternative_raceline or alt is None:
            waypoints_for_controller = self.waypoint_utils.next_waypoints
        else:
            waypoints_for_controller = alt.next_waypoints

        self.timesteps_on_current_raceline += 1

        return waypoints_for_controller

    # Get waypoints from MPC in case they are generated with it
    def get_mpc_waypoints_from_mpc(self) -> np.ndarray:
        
        if self.control_index % Settings.PLAN_EVERY_N_STEPS == 0:
            next_interpolated_waypoints = WaypointUtils.get_interpolated_waypoints(self.waypoint_utils.next_waypoints, Settings.INTERPOLATE_LOCA_WP)
            self.waypoints_planner.lidar_utils = self.lidar_utils
            driver_obs = self.driver_observation if self.driver_observation is not None else {"car_state": self.car_state}
            controller_observation = {
                **driver_obs,
                "next_waypoints": next_interpolated_waypoints,
                "processed_ranges": np.asarray(self.lidar_utils.processed_ranges, dtype=np.float32),
                "lidar_points": self.lidar_utils.processed_points_map_coordinates,
            }
            self.waypoints_planner.process_observation(controller_observation)
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
                self.lidar_utils.processed_ranges,
                self.lidar_utils.processed_angles_rad,
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

        # Send car state to RaceTuner if connected
        if self.tuner_connector is not None:
            tuner_state = {
                'car_x': float(self.car_state[POSE_X_IDX]),
                'car_y': float(self.car_state[POSE_Y_IDX]),
                'car_v': float(self.car_state[LINEAR_VEL_X_IDX]),
                'idx_global': int(self.waypoint_utils.nearest_waypoint_index) if self.waypoint_utils.nearest_waypoint_index is not None else 0,
                'time': float(self.time),
            }
            self.tuner_connector.update_car_state(tuner_state)
        if self.backward_predictor is not None:
            self.backward_predictor.feed_planner_forged_history(
                self.car_state,
                self.lidar_utils.all_lidar_ranges,
                self.waypoint_utils,
                self.planner,
                self.render_utils,
                Settings.INTERPOLATE_LOCA_WP,
            )
        if Settings.SAVE_STATE_METRICS and hasattr(self, 'state_metric_calculator'):
            self.state_metric_calculator.calculate_metrics(
                current_state=self.car_state,
                current_control=np.array([self.angular_control, self.translational_control]),
                updated_attributes={"next_waypoints": self.waypoint_utils.next_waypoints},
            )
            
        if Settings.FORGE_HISTORY and hasattr(self, 'history_forger'):
            basic_dict = get_basic_data_dict(self)
            basic_dict.update({'forged_history_applied': lambda: self.history_forger.forged_history_applied})

        if(hasattr(self, 'recorder') and self.recorder is not None):
            # Get simulation observations if available
            sim_obs = getattr(self, 'sim_obs', None)
            basic_dict = get_basic_data_dict(self, sim_obs)
            self.recorder.dict_data_to_save_basic.update(basic_dict)
            self.recorder.step()
        
        # print('Reward:', self.reward)
    
    '''
    Called by LapAnalyser when a lap is completed
    '''
    def lap_complete_cb(self,lap_time, mean_distance, std_distance, max_distance):
        self.laptimes.append(lap_time)
        print(f"Lap time: {lap_time}, Error: Mean: {mean_distance}, std: {std_distance}, max: {max_distance}")

        stop_recording_after = getattr(Settings, "STOP_RECORDING_AFTER_N_LAPS", None)
        if (
            stop_recording_after is not None
            and len(self.laptimes) >= int(stop_recording_after)
            and self.recorder is not None
            and self.recorder.recording_running
        ):
            print(f"Stopping recording after {len(self.laptimes)} lap(s).")
            self.recorder.finish_csv_recording()

        stop_after = getattr(Settings, "STOP_AFTER_N_LAPS", None)
        if stop_after is not None and len(self.laptimes) >= int(stop_after):
            print(f"Stopping simulation after {len(self.laptimes)} lap(s).")
            self.lap_limit_reached = True
     
    
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

            if self.virtual_opponents is not None:
                from utilities.recording_replay import get_virtual_opponent_recording_dict

                self.recorder.dict_data_to_save_basic.update(
                    get_virtual_opponent_recording_dict(
                        self, len(self.virtual_opponents.opponents)
                    )
                )
       
            if Settings.FORGE_HISTORY and hasattr(self, 'history_forger'):
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
            if key.char == 'r':  # Press 'r' to toggle recording
                if self.recorder is not None:
                    recording_started = self.recorder.toggle_recording()
                    if recording_started is True:
                        print("Recording STARTED (r key pressed)")
                    elif recording_started is False:
                        print("Recording STOPPED (r key pressed)")
                    else:
                        print("Recording toggle requested but recorder is starting up...")
                else:
                    print("No recorder available to toggle")
        except AttributeError:
            pass  # For special keys like shift, ctrl, etc.

    def start_keyboard_listener(self):
        if Settings.RENDER_MODE is None:
            # print("Keyboard listener not started, starting recording automatically")
            self.start_recorder()
            return
        # If keyboard exists
        try:
            from pynput import keyboard
            listener = keyboard.Listener(on_press=self.on_press)
            listener.start()
        except ImportError:
            self.start_recorder()
        
           

    def start_recorder(self):
        if self.recorder is not None:
            # print(f"Starting recorder for {self.controller_name}")
            self.recorder.start_csv_recording()
        else:
            pass
            # print("No recorder to start - recorder is None")

    
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
        # Prevent further recording after simulation end
        if hasattr(self, '_simulation_ended') and self._simulation_ended:
            print("Simulation already ended, skipping recorder finalization.")
            return

        if hasattr(self.planner, 'on_simulation_end'):
            self.planner.on_simulation_end(collision=collision)

        self._simulation_ended = True

        if self.recorder is not None:    

            if self.recorder.recording_mode == 'offline':  # As adding lines to header needs saving whole file once again
                self.recorder.finish_csv_recording()            
            augment_csv_header_with_laptime(self.laptimes, self.recorder.csv_filepath)

            path_to_plots = None
            if Settings.SAVE_PLOTS:
                path_to_plots = save_experiment_data(self.recorder.csv_filepath)

            if collision:
                index = min(len(self.car_state_history), 200)
                # Save or append self.control_index to csv file
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
    elif controller == 'mppi-lite-jax':
        from Control_Toolkit_ASF.Controllers.MPPILite.mppi_lite_jax_planner import MPPILitePlanner
        planner = MPPILitePlanner()
    elif controller == 'rpgd-lite-jax':
        from Control_Toolkit_ASF.Controllers.MPPILite.rpgd_jax_planner import RPGDPlanner
        planner = RPGDPlanner()
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
    elif controller == 'sac_agent':
        from TrainingLite.rl_racing.sac_agent_planner import RLAgentPlanner
        planner = RLAgentPlanner()
    elif controller == 'manual':
        from Control_Toolkit_ASF.Controllers.Manual import manual_planner
        importlib.reload(manual_planner)
        planner = manual_planner.manual_planner()
    elif controller == 'example':
        from Control_Toolkit_ASF.Controllers.ExamplePlanner import example_planner
        importlib.reload(example_planner)
        planner = example_planner.ExamplePlanner()
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
    if hasattr(planner, 'mpc'): # MPC planner from Control_Toolkit_ASF
        horizon = planner.mpc.optimizer.mpc_horizon
        angular_control_dict = {"cs_a_{}".format(i): 0 for i in range(horizon)}
        translational_control_dict = {"cs_t_{}".format(i): 0 for i in range(horizon)}
        return angular_control_dict, translational_control_dict
    
    if hasattr(planner, 'optimal_control_sequence'): # MPC planner lite
        horizon = len(planner.optimal_control_sequence)
        angular_control_dict = {"cs_a_{}".format(i): 0 for i in range(horizon)}
        translational_control_dict = {"cs_t_{}".format(i): 0 for i in range(horizon)}
        return angular_control_dict, translational_control_dict 
    else:
        return {}, {}
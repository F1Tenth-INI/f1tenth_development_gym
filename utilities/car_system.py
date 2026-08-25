"""
CarSystem bridges sim/ROS driver observations to planner control.

Control loop (one timestep)::

    process_observation(driver_obs)   # pre-step: plan control from s
    ... physics sim ...
    on_step_end(driver_obs)           # post-step: reward, termination, logging

Observation naming:

- ``driver_observation``: raw dict from sim/ROS (car_state, scans, sensors, …).
- ``controller_observation``: driver obs enriched with waypoints, history, lidar, …;
  built at control time and stored on ``self.controller_observation``.
- ``step_end_observation``: post-step bundle for reward / RL / ``planner.on_step_end``;
  top-level fields are the post-step controller obs; nested ``controller_observation``
  is the pre-control snapshot from the same timestep.
"""

import os
from collections import deque
from typing import Any, Optional

import numpy as np
import yaml

from utilities.Settings import Settings
from utilities.EmergencySlowdown import EmergencySlowdown
from utilities.LapAnalyzer import LapAnalyzer
from utilities.Recorder import get_basic_data_dict
from utilities.cbf_safety_filter import CBFSafetyFilter
from utilities.mpc_safety_filter import MPCSafetyFilter
from utilities.csv_logger import augment_csv_header_with_laptime
from utilities.episode_termination import EpisodeTerminator
from utilities.imu_utilities import IMUUtilities
from utilities.lidar_utils import LidarHelper
from utilities.obstacle_detector import ObstacleDetector
from utilities.opponent_tracker import OpponentTracker
from utilities.planner_factory import if_mpc_define_cs_variables, initialize_planner
from utilities.recorder_setup import init_car_recorder
from utilities.recording_keyboard_listener import start_recording_keyboard_listener
from utilities.recording_replay import get_virtual_opponent_poses_for_render
from utilities.render_utilities import RenderUtils
from utilities.saving_helpers import move_csv_to_crash_folder, save_experiment_data
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    STATE_VARIABLES,
    STEERING_ANGLE_IDX,
)
from utilities.virtual_opponents import VirtualOpponents, get_ego_car_dimensions
from utilities.waypoint_utils import (
    WP_KAPPA_IDX,
    WP_S_IDX,
    WP_VX_IDX,
    WP_X_IDX,
    WP_Y_IDX,
    WaypointUtils,
    fit_local_raceline_polynomial,
    fit_local_raceline_polynomial_parametric,
    transform_from_car_coordinates,
)

try:
    from TrainingLite.rl_racing.RewardCalculator import RewardCalculator
except ModuleNotFoundError:
    from f1tenth_development_gym.TrainingLite.rl_racing.RewardCalculator import RewardCalculator

if Settings.CONNECT_RACETUNER_TO_MAIN_CAR:
    from RaceTuner.TunerConnectorSim import TunerConnectorSim

if Settings.FORGE_HISTORY:
    from utilities.HistoryForger import HistoryForger

# Bounded history for controller observations (SAC/NNI need ~25; we append up to twice per control step).
CAR_STATE_HISTORY_MAXLEN = 128
CONTROL_HISTORY_MAXLEN = 128


class CarSystem:
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def __init__(
        self,
        controller=None,
        save_recording=Settings.SAVE_RECORDINGS,
        recorder_dict=None,
    ):
        if recorder_dict is None:
            recorder_dict = {}

        self._init_timing_and_control_state()
        self._init_observation_state()
        self._init_waypoints_and_rendering()
        self._init_race_utilities()
        self._init_planner(controller)
        self._init_optional_features()
        self._init_lap_analyzer()
        self.save_recordings = save_recording
        init_car_recorder(self, recorder_dict=recorder_dict)
        self.backward_predictor = None
        if not Settings.ROS_BRIDGE:
            self.start_recorder()

    def _init_timing_and_control_state(self) -> None:
        self.time = 0.0
        self.time_increment = Settings.TIMESTEP_CONTROL
        self.control_index = 0
        self.angular_control_calculated = 0.0
        self.translational_control_calculated = 0.0
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.control_noise = None
        self.laptimes = []
        self._lap_finished = False
        self._last_lap_time = 0.0
        self.reward = 0.0
        self.reward_components = {}
        self.episode_done = False
        self.episode_truncated = False

    def _init_observation_state(self) -> None:
        self.car_state = np.ones(len(STATE_VARIABLES))
        self.car_state_history = deque(maxlen=CAR_STATE_HISTORY_MAXLEN)
        self.control_history = deque(maxlen=CONTROL_HISTORY_MAXLEN)
        self.driver_observation = None
        self.controller_observation = None
        self.imu = IMUUtilities.zeros_dict()
        self.motor_sensors = {}
        self.cbf_info = None
        self.mpc_info = None
        self._virtual_opponent_collision = False

    def _init_waypoints_and_rendering(self) -> None:
        self.waypoint_utils = WaypointUtils()
        if Settings.ALLOW_ALTERNATIVE_RACELINE:
            self.waypoint_utils_alternative = WaypointUtils(
                waypoint_file_name=f"{Settings.MAP_NAME}_wp_alternative",
                speed_scaling_file_name=f"{Settings.MAP_NAME}_speed_scaling_alternative.csv",
            )
        else:
            self.waypoint_utils_alternative = None

        self.alternative_raceline = False
        self.timesteps_on_current_raceline = 0
        self.waypoints_for_controller = self.waypoint_utils.next_waypoints
        self.use_waypoints_from_mpc = Settings.WAYPOINTS_FROM_MPC
        self.waypoints_planner = None
        self.waypoints_from_mpc = np.zeros((Settings.LOOK_AHEAD_STEPS, 7))
        if Settings.WAYPOINTS_FROM_MPC:
            from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner

            self.waypoints_planner = mpc_planner()
            self.waypoints_planner.waypoint_utils = self.waypoint_utils

        self.render_utils = RenderUtils()
        self.render_utils.waypoints = self.waypoint_utils.waypoint_positions
        self.render_utils.waypoints_full = self.waypoint_utils.waypoints
        if self.waypoint_utils_alternative is not None:
            self.render_utils.waypoints_alternative = (
                self.waypoint_utils_alternative.waypoint_positions
            )
        self.allow_rendering = True
        self.lidar_utils = LidarHelper()

    def _init_race_utilities(self) -> None:
        self.obstacle_detector = ObstacleDetector()
        self.reward_calculator = RewardCalculator()
        self.episode_terminator = EpisodeTerminator()
        self.virtual_opponents = VirtualOpponents.from_settings()
        if bool(getattr(Settings, "OPPONENT_TRACKER_ENABLED", False)):
            self.opponent_tracker = OpponentTracker.from_settings()
        else:
            self.opponent_tracker = None
        self.emergency_slowdown = EmergencySlowdown()
        self.cbf_safety_filter = (
            CBFSafetyFilter() if getattr(Settings, "CBF_SAFETY_FILTER", False) else None
        )
        self.mpc_safety_filter = (
            MPCSafetyFilter() if getattr(Settings, "MPC_SAFETY_FILTER", False) else None
        )
        if self.mpc_safety_filter is not None:
            self.mpc_safety_filter.attach_render_utils(self.render_utils)
        self.tuner_connector = None
        self.mpc_info = None

    def _init_planner(self, controller) -> None:
        self.controller_name = controller
        self.initialize_controller(self.controller_name)
        self.angular_control_dict, self.translational_control_dict = if_mpc_define_cs_variables(
            self.planner
        )

    def _init_optional_features(self) -> None:
        if Settings.CONNECT_RACETUNER_TO_MAIN_CAR:
            self.launch_tuner_connector()

        if Settings.FRICTION_FOR_CONTROLLER is not None and hasattr(self.planner, "mpc"):
            predictor = self.planner.mpc.predictor.predictor
            if hasattr(predictor, "next_step_predictor") and hasattr(
                predictor.next_step_predictor, "env"
            ):
                predictor.next_step_predictor.env.change_friction_coefficient(
                    Settings.FRICTION_FOR_CONTROLLER
                )

        self.config_onlinelearning = yaml.load(
            open(os.path.join("SI_Toolkit_ASF", "config_onlinelearning.yml")),
            Loader=yaml.FullLoader,
        )
        self.online_learning_activated = self.config_onlinelearning.get("activated", False)
        if self.online_learning_activated:
            from SI_Toolkit.Training.OnlineLearning import OnlineLearning

            if Settings.CONTROLLER == "mpc":
                self.predictor = self.planner.mpc.predictor
            self.online_learning = OnlineLearning(
                self.predictor, Settings.TIMESTEP_CONTROL, self.config_onlinelearning
            )

        if Settings.FORGE_HISTORY:
            self.history_forger = HistoryForger()

    def _init_lap_analyzer(self) -> None:
        self.lap_analyzer = LapAnalyzer(
            total_waypoints=len(self.waypoint_utils.waypoints),
            lap_finished_callback=self.lap_complete_cb,
        )

    def reset(self):
        self.car_state = None
        self.driver_observation = None
        self.controller_observation = None
        self.imu = IMUUtilities.zeros_dict()
        self.motor_sensors = {}
        self.laptimes = []
        self._lap_finished = False
        self._last_lap_time = 0.0
        self.lap_limit_reached = False
        self._virtual_opponent_collision = False
        self.episode_done = False
        self.episode_truncated = False
        self.virtual_opponents = VirtualOpponents.from_settings()

        self.control_index = 0
        self.cbf_info = None
        self.mpc_info = None
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

    def initialize_controller(self, controller_name):
        self.planner = initialize_planner(controller_name)
        if self.planner is None:
            return
        self.planner.render_utils = self.render_utils
        self.planner.waypoint_utils = self.waypoint_utils
        self.planner.lidar_utils = self.lidar_utils
        self.planner.obstacle_detector = self.obstacle_detector

    def launch_tuner_connector(self):
        try:
            self.tuner_connector = TunerConnectorSim()
        except OSError:
            print("Tunner connection not possible.")

    # -------------------------------------------------------------------------
    # Driver observation ingestion
    # -------------------------------------------------------------------------

    def set_car_state(self, car_state):
        self.car_state = np.asarray(car_state, dtype=np.float32)

    def set_scans(self, ranges):
        ranges = np.array(ranges)
        if self.virtual_opponents is not None and self.car_state is not None:
            ranges = self.virtual_opponents.apply_to_scan(self.car_state, ranges)
        self.lidar_utils.update_ranges(ranges, self.car_state)

    def set_sensors(self, sensors):
        """Store raw sensor readings from driver_obs['sensors']."""
        self.imu = sensors["imu"]
        self.motor_sensors = sensors["motor_sensors"]

    def _append_car_state_history(self, car_state=None):
        """Append one car state snapshot to history (post-step in on_step_end)."""
        if car_state is None:
            car_state = self.car_state
        self.car_state_history.append(np.asarray(car_state, dtype=np.float32).copy())

    def _ingest_driver_observation(self, observation):
        """Apply a raw driver observation dict to internal CarSystem state."""
        self.driver_observation = observation
        self.set_car_state(observation["car_state"])
        self.set_sensors(observation["sensors"])
        self._update_waypoint_indices()
        if self.virtual_opponents is not None:
            env_time = observation.get("env", {}).get("time", 0.0)
            self.virtual_opponents.set_state(
                self.waypoint_utils.nearest_waypoint_index,
                float(env_time),
            )
        self.set_scans(observation["scans"])
        self._finalize_waypoints_for_control()

    # -------------------------------------------------------------------------
    # Observation builders
    # -------------------------------------------------------------------------

    def _lap_metrics(self) -> dict[str, Any]:
        frenet = self.waypoint_utils.frenet_coordinates
        along_track_s = float(frenet[0]) if frenet is not None else 0.0
        waypoints = self.waypoint_utils.waypoints
        if waypoints is not None and len(waypoints) > 0:
            track_length = float(waypoints[-1][WP_S_IDX])
            if track_length > 0.0:
                wrapped_s = along_track_s % track_length
                if wrapped_s < 0.0:
                    wrapped_s += track_length
                along_track_progress = wrapped_s / track_length
            else:
                along_track_progress = 0.0
        else:
            along_track_progress = 0.0

        lap_finished = bool(self._lap_finished)
        if lap_finished:
            lap_time = float(self._last_lap_time)
        elif (
            self.lap_analyzer is not None
            and self.lap_analyzer.single_measurement_point_time is not None
        ):
            lap_time = float(self.time - self.lap_analyzer.single_measurement_point_time)
        else:
            lap_time = 0.0

        return {
            "along_track_progress": along_track_progress,
            "lap_fraction": float(self.waypoint_utils.cumulative_progress % 1.0),
            "lap_finished": lap_finished,
            "lap_time": lap_time,
            "lap_count": len(self.laptimes),
        }

    def _virtual_opponent_observation_fields(self) -> dict[str, Any]:
        if self.virtual_opponents is None:
            return {"virtual_opponent_poses": np.zeros((0, 3), dtype=np.float32)}
        ego_length, ego_width = get_ego_car_dimensions()
        return {
            "virtual_opponent_poses": self.virtual_opponents.get_poses(),
            "min_virtual_opponent_distance": self.virtual_opponents.min_clearance_to_ego(
                self.car_state, ego_length, ego_width
            ),
            "virtual_opponent_collision": bool(self._virtual_opponent_collision),
        }

    def _build_controller_observation(self, driver_observation: dict[str, Any]) -> dict[str, Any]:
        """Enrich raw driver observation with CarSystem-computed planner fields."""
        controller_observation = {
            **driver_observation,
            "next_waypoints": np.asarray(self.waypoint_utils.next_waypoints, dtype=np.float32),
            "waypoints": np.asarray(
                self.waypoint_utils.waypoints
                if self.waypoint_utils.waypoints is not None
                else self.waypoint_utils.next_waypoints,
                dtype=np.float32,
            ),
            "state_history": np.asarray(self.car_state_history, dtype=np.float32),
            "control_history": np.asarray(self.control_history, dtype=np.float32),
            "frenet_coordinates": np.asarray(
                self.waypoint_utils.frenet_coordinates, dtype=np.float32
            ),
            "control_index": int(self.control_index),
            "imu": self.imu,
            "motor_sensors": self.motor_sensors,
            "processed_ranges": np.asarray(self.lidar_utils.processed_ranges, dtype=np.float32),
            "lidar_points": self.lidar_utils.processed_points_map_coordinates,
            **self._lap_metrics(),
            **self._virtual_opponent_observation_fields(),
        }
        controller_observation.update(
            self.opponent_tracker.to_controller_observation(self.car_state)
            if self.opponent_tracker is not None
            else OpponentTracker.empty_controller_observation()
        )
        return controller_observation

    def _evaluate_episode(
        self, post_step_controller_observation: dict[str, Any], post_step_driver_observation: dict
    ) -> tuple[dict, float, dict[str, Any], dict[str, Any]]:
        episode_termination = self.episode_terminator.evaluate(
            post_step_controller_observation, post_step_driver_observation
        )
        post_step_controller_observation["episode_termination"] = episode_termination
        post_step_controller_observation["cbf_info"] = self.cbf_info
        post_step_controller_observation["mpc_info"] = self.mpc_info

        reward_result = self.reward_calculator._calculate_reward(post_step_controller_observation)
        reward = float(reward_result["total_reward"])
        reward_components = dict(reward_result.get("components") or {})
        info = {
            "lap_times": list(self.laptimes),
            "reward_components": reward_components,
            **episode_termination,
        }
        return episode_termination, reward, reward_components, info

    def _build_step_end_observation(
        self, post_step_driver_observation: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the full step-end observation for reward, logging, and planner transitions."""
        post_step_controller_observation = self._build_controller_observation(
            post_step_driver_observation
        )
        post_step_controller_observation["virtual_opponent_collision"] = (
            self._virtual_opponent_collision
        )

        episode_termination, reward, _reward_components, info = self._evaluate_episode(
            post_step_controller_observation, post_step_driver_observation
        )

        pre_control_controller_observation = self.controller_observation
        if pre_control_controller_observation is None:
            pre_control_controller_observation = post_step_controller_observation

        return {
            **post_step_controller_observation,
            "controller_observation": pre_control_controller_observation,
            "post_step_driver_observation": post_step_driver_observation,
            "cbf_info": self.cbf_info,
            "mpc_info": self.mpc_info,
            "episode_termination": episode_termination,
            "reward": reward,
            "done": bool(episode_termination["done"]),
            "truncated": bool(episode_termination["truncated"]),
            "info": info,
        }

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

    def _check_virtual_opponent_collision(self, driver_observation: dict) -> bool:
        if (
            self.virtual_opponents is None
            or self.car_state is None
            or not bool(getattr(Settings, "TERMINATE_ON_VIRTUAL_OPPONENT_COLLISION", False))
        ):
            return False
        ego_length, ego_width = get_ego_car_dimensions()
        collision = self.virtual_opponents.collides_with_ego(
            self.car_state, ego_length, ego_width
        )
        if collision:
            driver_observation["collision"] = True
        return collision
    
    def _sync_planner_recording_dicts(self) -> None:
        """Copy optional planner sequence dictionaries for recorder compatibility."""
        if self.planner is None:
            return
        if hasattr(self.planner, "angular_control_dict"):
            self.angular_control_dict = dict(self.planner.angular_control_dict)
        if hasattr(self.planner, "translational_control_dict"):
            self.translational_control_dict = dict(self.planner.translational_control_dict)

    def _apply_cbf_safety_filter(self) -> None:
        """Apply CBF-QP safety filter to current control command."""
        if self.cbf_safety_filter is None:
            return
        u_safe, self.cbf_info = self.cbf_safety_filter.filter_from_observation(
            np.array([self.angular_control, self.translational_control], dtype=np.float64),
            self.controller_observation,
        )
        self.angular_control, self.translational_control = float(u_safe[0]), float(
            u_safe[1]
        )

    def _apply_mpc_safety_filter(self) -> None:
        """Apply Tearle et al. predictive safety filter to the current command.

        Runs after the CBF filter (if any). Solves a short-horizon SLSQP backup
        MPC that either certifies ``u_d`` or returns a minimally invasive ``u_0*``
        ending in the configured terminal set ``S_f``.
        """
        if self.mpc_safety_filter is None:
            return
        u_safe, self.mpc_info = self.mpc_safety_filter.filter_from_observation(
            np.array([self.angular_control, self.translational_control], dtype=np.float64),
            self.controller_observation,
        )
        self.angular_control, self.translational_control = float(u_safe[0]), float(
            u_safe[1]
        )

    def process_observation(self, driver_observation):
        self._lap_finished = False
        self.env_state = driver_observation.get("env_state")
        self._ingest_driver_observation(driver_observation)
        if not self.car_state_history:
            self._append_car_state_history()

        self.controller_observation = self._build_controller_observation(driver_observation)

        if self.planner is not None:
            # CONTROL STEP
            self.angular_control, self.translational_control = self.planner.process_observation(
                self.controller_observation
            )
            self._sync_planner_recording_dicts()
            self._apply_cbf_safety_filter()
            self._apply_mpc_safety_filter()
        else: # Controller = None
            self.angular_control = 0.0
            self.translational_control = 0.0

        # Add control noise and append to control history
        self.angular_control_calculated = self.angular_control
        self.translational_control_calculated = self.translational_control
        self.angular_control, self.translational_control = self.add_control_noise(
            np.array([self.angular_control, self.translational_control])
        )
        self.control_history.append(
            np.array([self.angular_control, self.translational_control], dtype=np.float32)
        )

        self._post_control_side_effects()
        self.control_index += 1
        self.time += self.time_increment
        return self.angular_control, self.translational_control



    def on_step_end(self, observation=None):
        if observation is None:
            return

        self._ingest_driver_observation(observation)
        self._update_opponent_tracker()
        self._append_car_state_history()
        self._virtual_opponent_collision = self._check_virtual_opponent_collision(observation)

        step_end_observation = self._build_step_end_observation(observation)
        self.reward = step_end_observation["reward"]
        self.reward_components = step_end_observation["info"]["reward_components"]
        self.episode_done = step_end_observation["done"]
        self.episode_truncated = step_end_observation["truncated"]
        observation.update(
            {
                "reward": self.reward,
                "info": step_end_observation["info"],
                "truncated": self.episode_truncated,
                "done": self.episode_done,
                "episode_termination": step_end_observation["episode_termination"],
            }
        )

        if self.render_utils is not None:
            self.update_render_utils()

        if self.planner is not None and hasattr(self.planner, "on_step_end"):
            self.planner.on_step_end(
                self._build_planner_step_end_observation(observation)
            )

    # -------------------------------------------------------------------------
    # Waypoints and raceline
    # -------------------------------------------------------------------------

    def _update_waypoint_indices(self):
        car_state = self.car_state
        self.waypoint_utils.update_next_waypoints(car_state)
        if self.waypoint_utils_alternative is not None:
            self.waypoint_utils_alternative.update_next_waypoints(car_state)

    def _finalize_waypoints_for_control(self):
        lidar_points = self.lidar_utils.processed_points_map_coordinates
        self.waypoint_utils.check_if_obstacle_on_my_raceline(lidar_points)
        if self.waypoint_utils_alternative is not None:
            self.waypoint_utils_alternative.check_if_obstacle_on_my_raceline(lidar_points)

        if self.use_waypoints_from_mpc:
            self.waypoints_for_controller = self.get_mpc_waypoints_from_mpc()
        else:
            self.waypoints_for_controller = self.choose_raceline_from_waypoints()
        self.handle_emergency_slowdown()
        self.waypoint_utils.get_frenet_coordinates(self.car_state)
        self.waypoint_utils.get_cumulative_lap_progress()

    def _update_opponent_tracker(self):
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

    def _primary_raceline_blocked(self) -> bool:
        if self.waypoint_utils.obstacle_on_raceline:
            return True
        if self.virtual_opponents is None or self.car_state is None:
            return False
        ego_length, ego_width = get_ego_car_dimensions()
        clearance = self.virtual_opponents.min_clearance_to_ego(
            self.car_state, ego_length, ego_width
        )
        return clearance < 4.0

    def choose_raceline_from_waypoints(self) -> np.ndarray:
        alt = self.waypoint_utils_alternative
        min_dwell_steps = 50

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
            print("Switching to alternative raceline")

        if (
            self.alternative_raceline
            and not primary_blocked
            and self.timesteps_on_current_raceline > min_dwell_steps
            and self.waypoint_utils.current_distance_to_raceline < 0.3
        ):
            self.alternative_raceline = False
            self.timesteps_on_current_raceline = 0
            print("Switching to primary raceline")

        if not self.alternative_raceline or alt is None:
            waypoints_for_controller = self.waypoint_utils.next_waypoints
        else:
            waypoints_for_controller = alt.next_waypoints

        self.timesteps_on_current_raceline += 1
        return waypoints_for_controller

    def chose_raceline_from_wpts(self) -> np.ndarray:
        """Deprecated alias for :meth:`choose_raceline_from_waypoints`."""
        return self.choose_raceline_from_waypoints()

    def get_mpc_waypoints_from_mpc(self) -> np.ndarray:
        if self.control_index % Settings.PLAN_EVERY_N_STEPS != 0:
            return self.waypoints_for_controller

        next_interpolated_waypoints = WaypointUtils.get_interpolated_waypoints(
            self.waypoint_utils.next_waypoints, Settings.INTERPOLATE_LOCA_WP
        )
        self.waypoints_planner.lidar_utils = self.lidar_utils
        driver_obs = (
            self.driver_observation
            if self.driver_observation is not None
            else {"car_state": self.car_state}
        )
        controller_observation = self._build_controller_observation(driver_obs)
        controller_observation["next_waypoints"] = next_interpolated_waypoints
        self.waypoints_planner.process_observation(controller_observation)
        optimal_trajectory = self.waypoints_planner.mpc.optimizer.optimal_trajectory
        if optimal_trajectory is None:
            return self.waypoint_utils.next_waypoints

        self.waypoints_from_mpc[:, WP_X_IDX] = optimal_trajectory[
            0, -len(self.waypoints_from_mpc) :, POSE_X_IDX
        ]
        self.waypoints_from_mpc[:, WP_Y_IDX] = optimal_trajectory[
            0, -len(self.waypoints_from_mpc) :, POSE_Y_IDX
        ]
        self.waypoints_from_mpc[:, WP_VX_IDX] = optimal_trajectory[
            0, -len(self.waypoints_from_mpc) :, LINEAR_VEL_X_IDX
        ]
        angular_vel = optimal_trajectory[0, :, ANGULAR_VEL_Z_IDX]
        linear_vel = optimal_trajectory[0, :, LINEAR_VEL_X_IDX]
        curvature = np.divide(
            angular_vel, linear_vel, out=np.zeros_like(angular_vel), where=linear_vel != 0
        )
        self.waypoints_from_mpc[:, WP_KAPPA_IDX] = curvature[-len(self.waypoints_from_mpc) :]
        return self.waypoints_from_mpc

    def handle_emergency_slowdown(self):
        if not Settings.STOP_IF_OBSTACLE_IN_FRONT:
            return
        car_state = self.car_state
        corrected_next_waypoints_vx, _ = self.emergency_slowdown.stop_if_obstacle_in_front(
            self.lidar_utils.processed_ranges,
            self.lidar_utils.processed_angles_rad,
            self.waypoint_utils.next_waypoints[:, WP_VX_IDX],
            car_state[STEERING_ANGLE_IDX],
        )
        self.waypoint_utils.next_waypoints[:, WP_VX_IDX] = corrected_next_waypoints_vx
        self.emergency_slowdown.update_emergency_slowdown_sprites(
            car_x=car_state[POSE_X_IDX],
            car_y=car_state[POSE_Y_IDX],
            car_yaw=car_state[POSE_THETA_IDX],
        )
        self.render_utils.update(
            emergency_slowdown_sprites=self.emergency_slowdown.emergency_slowdown_sprites,
        )

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def render(self, e):
        if Settings.RENDER_MODE is not None:
            self.render_utils.render(e)

    def _compute_polynomial_raceline_global(
        self,
        relative: np.ndarray,
    ) -> np.ndarray | None:
        # Parametric x(t)/y(t) fit handles hairpins where y=f(x) breaks down.
        car_frame, info = fit_local_raceline_polynomial_parametric(relative)
        if car_frame is None or not info.get("ok"):
            return None
        return transform_from_car_coordinates(car_frame, self.car_state).astype(np.float32)

    def update_render_utils(self):
        car_state = self.car_state
        relative = self.waypoint_utils.next_waypoint_positions_relative
        _, poly_fit_info = fit_local_raceline_polynomial(relative)
        poly_info = {}
        if poly_fit_info.get("ok"):
            poly_info = {
                "poly d_lat": poly_fit_info["d_lat"],
                "poly e": poly_fit_info["e"],
                "poly kappa": poly_fit_info["kappa"],
            }
        polynomial_raceline = self._compute_polynomial_raceline_global(relative)

        label_dict = {
            "0: angular_control": self.angular_control,
            "1: translational_control": self.translational_control,
            "yaw": car_state[POSE_THETA_IDX],
            "4: Surface Friction": Settings.SURFACE_FRICTION,
            "5: Laptimes:": ", ".join(f"{lt:.2f}" for lt in self.laptimes),
            "6: Reward": self.reward,
            "Distance to raceline": self.waypoint_utils.current_distance_to_raceline,
            "speed": car_state[LINEAR_VEL_X_IDX],
            "Wp_idx": self.waypoint_utils.nearest_waypoint_index,
            **poly_info,
        }
        label_dict.update(IMUUtilities.overlay_label_dict(self.imu))
        for name, value in (self.reward_components or {}).items():
            label_dict[f"reward: {name}"] = float(value)
        label_dict["reward: total"] = float(self.reward)
        self.render_utils.set_label_dict(label_dict)

        virtual_opponent_poses = get_virtual_opponent_poses_for_render(self)
        if virtual_opponent_poses is None:
            virtual_opponent_poses = np.empty((0, 3), dtype=np.float32)
        self.render_utils.update(
            lidar_points=self.lidar_utils.processed_points_map_coordinates,
            next_waypoints=self.waypoint_utils.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)],
            next_waypoints_polynomial=polynomial_raceline,
            next_waypoints_alternative=(
                self.waypoint_utils_alternative.next_waypoints[:, (WP_X_IDX, WP_Y_IDX)]
                if self.waypoint_utils_alternative is not None
                else None
            ),
            car_state=car_state,
            track_border_points=self.waypoint_utils.get_track_border_positions(
                self.waypoint_utils.next_waypoints
            ),
            virtual_opponents=virtual_opponent_poses,
            detected_opponents=(
                self.opponent_tracker.get_render_points()
                if self.opponent_tracker is not None
                else None
            ),
        )

    # -------------------------------------------------------------------------
    # Post-control logging and integrations
    # -------------------------------------------------------------------------

    def _post_control_side_effects(self):
        self._update_lap_and_integrations()
        self._record_control_step()

    def _update_lap_and_integrations(self):
        self.lap_analyzer.update(
            nearest_waypoint_index=self.waypoint_utils.nearest_waypoint_index,
            time_now=self.time,
            distance_to_raceline=self.waypoint_utils.current_distance_to_raceline,
        )
        if self.tuner_connector is not None:
            self.tuner_connector.update_car_state(
                {
                    "car_x": float(self.car_state[POSE_X_IDX]),
                    "car_y": float(self.car_state[POSE_Y_IDX]),
                    "car_v": float(self.car_state[LINEAR_VEL_X_IDX]),
                    "idx_global": (
                        int(self.waypoint_utils.nearest_waypoint_index)
                        if self.waypoint_utils.nearest_waypoint_index is not None
                        else 0
                    ),
                    "time": float(self.time),
                }
            )
        if self.backward_predictor is not None:
            self.backward_predictor.feed_planner_forged_history(
                self.car_state,
                self.lidar_utils.all_lidar_ranges,
                self.waypoint_utils,
                self.planner,
                self.render_utils,
                Settings.INTERPOLATE_LOCA_WP,
            )
        if Settings.SAVE_STATE_METRICS and hasattr(self, "state_metric_calculator"):
            self.state_metric_calculator.calculate_metrics(
                current_state=self.car_state,
                current_control=np.array([self.angular_control, self.translational_control]),
                updated_attributes={"next_waypoints": self.waypoint_utils.next_waypoints},
            )

    def _record_control_step(self):
        if not (hasattr(self, "recorder") and self.recorder is not None):
            return
        sim_obs = getattr(self, "sim_obs", None)
        self.recorder.dict_data_to_save_basic.update(get_basic_data_dict(self, sim_obs))
        self.recorder.step()

    def lap_complete_cb(self, lap_time, mean_distance, std_distance, max_distance):
        self._lap_finished = True
        self._last_lap_time = float(lap_time)
        self.laptimes.append(lap_time)
        print(
            f"Lap time: {lap_time}, Error: Mean: {mean_distance}, "
            f"std: {std_distance}, max: {max_distance}"
        )

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

    # -------------------------------------------------------------------------
    # Recording and simulation end
    # -------------------------------------------------------------------------

    def init_recorder(self, recorder_dict=None):
        init_car_recorder(self, recorder_dict=recorder_dict)

    def on_press(self, key):
        from utilities.recording_keyboard_listener import on_recording_key_press

        on_recording_key_press(self, key)

    def start_keyboard_listener(self):
        start_recording_keyboard_listener(self)

    def start_recorder(self):
        if self.recorder is not None:
            self.recorder.start_csv_recording()

    def add_control_noise(self, control):
        if self.control_noise is None or self.control_index % Settings.CONTROL_NOISE_DURATION == 0:
            noise_level = Settings.NOISE_LEVEL_CONTROL
            noise_array = np.array(noise_level) * np.random.uniform(-1, 1, len(noise_level))
            self.control_noise = noise_array
        return control + self.control_noise

    def on_simulation_end(self, collision=False):
        if hasattr(self, "_simulation_ended") and self._simulation_ended:
            print("Simulation already ended, skipping recorder finalization.")
            return

        if self.planner is not None and hasattr(self.planner, "on_simulation_end"):
            self.planner.on_simulation_end(collision=collision)

        self._simulation_ended = True
        if self.recorder is None:
            return

        if self.recorder.recording_mode == "offline":
            self.recorder.finish_csv_recording()
        augment_csv_header_with_laptime(self.laptimes, self.recorder.csv_filepath)

        path_to_plots = None
        if Settings.SAVE_PLOTS:
            path_to_plots = save_experiment_data(self.recorder.csv_filepath)

        if not collision:
            return

        index = min(len(self.car_state_history), 200)
        with open("survival.csv", "a") as f:
            f.write(f"{self.control_index}\n")
        print("Collision detected, moving csv to crash folder")
        print("Car State at crash:", self.car_state)
        print("Car State at -index steps:", self.car_state_history[-index])
        np.savetxt("Test.csv", [self.car_state_history[-index]], delimiter=",")
        move_csv_to_crash_folder(self.recorder.csv_filepath, path_to_plots)


# Re-export for backward compatibility.
__all__ = ["CarSystem", "initialize_planner", "if_mpc_define_cs_variables"]

import os
import random
import time
from collections import deque
import yaml

import numpy as np

from tqdm import trange
from argparse import Namespace

from sim.f110_sim.envs.base_classes import Simulator, wrap_angle_rad

from typing import Optional
from utilities.Settings import Settings
from utilities.car_system import CarSystem
from utilities.random_obstacle_creator import RandomObstacleCreator
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_PSI_IDX
from utilities.state_utilities import (
    STATE_VARIABLES, POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX, LINEAR_VEL_X_IDX, ANGULAR_VEL_Z_IDX,
    )
from utilities.Exceptions import CarCrashException
from utilities.screen_utils import ScreenUtils
from utilities.imu_simulator import IMUSimulator
from utilities.lidar_simulator import LidarSimulator
from utilities.map_scale import scale_positions
from utilities.motor_sensor_simulator import MotorSensorSimulator
from utilities.episode_randomization import apply_episode_randomization
from sim.f110_sim.envs.rendering.WebRenderer.overlay_builder import build_web_overlay
if Settings.DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
Settings.ROS_BRIDGE = False  # No ros bridge if this script is running





class RacingSimulation:
    RANDOM_START_MAIN_JITTER_XY = 0.2
    RANDOM_START_MAIN_JITTER_YAW = 0.1
    RANDOM_START_OPPONENT_WP_GAP_MIN = 10
    RANDOM_START_OPPONENT_WP_GAP_MAX = 40

    def __init__(self):
        self.crash_repetition = 0
        self.drivers = []
        self.number_of_drivers = 0
        self.starting_positions = []

        self.start_time = time.time()
        self.sim_time = 0.0
        self.episode_index = 0

        self.sim_index = 0

        self.sim_obs = {}
        # Full-environment snapshot at the current control timestep: car states
        # (from world_sim), controls, and world-sim outputs (scans, collisions).
        # Updated in `_update_state_history()`.
        self.env_state = {}
        self.step_reward = 0
        self.done = False
        self.info = None

        self.agent_controls_calculated = []
        self.control_delay_buffer = deque()

        self.env = None
        self.world_sim: Optional[Simulator] = None
        self.lidar_simulator: Optional[LidarSimulator] = None
        self.laptime = 0.0
        self.initial_states = None

        self.step_end_time = 0

        
        self.renderer = None
        self.renderer_backend = None

        self.vehicle_parameters_instance = VehicleParameters( param_file_name = Settings.CONTROLLER_CAR_PARAMETER_FILE)
        self.env_car_parameters = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)

        # State history for respawn functionality
        self.RESPAWN_HISTORY_LENGTH = Settings.RESPAWN_SETBACK_TIMESTEPS
        self.state_history = deque(maxlen=self.RESPAWN_HISTORY_LENGTH)
        self.control_history = deque(maxlen=self.RESPAWN_HISTORY_LENGTH)
        self.sim_obs_history = deque(maxlen=self.RESPAWN_HISTORY_LENGTH)
        self.sim_time_history = deque(maxlen=self.RESPAWN_HISTORY_LENGTH)
        self.sim_index_history = deque(maxlen=self.RESPAWN_HISTORY_LENGTH)

    

    """Run experiments including crash retries per settings."""
    def run_experiments(self, initial_states=None):
        self.initial_states = initial_states
            
        number_of_experiments = Settings.NUMBER_OF_EXPERIMENTS
        i = 0
        while i < number_of_experiments:
            try:
                print(f'Experiment nr.: {i + 1}')
                self.prepare_simulation()
                self.run_simulation()
            except CarCrashException as e:
                print("the car crashed.")
                if(Settings.RESET_ON_DONE):
                    
                    if self.crash_repetition < Settings.MAX_CRASH_REPETITIONS:
                        self.crash_repetition += 1
                        number_of_experiments += 1
                        print("Repeating experiment", self.crash_repetition)
                    else:
                        print(f"Max number of crash repetitions ({Settings.MAX_CRASH_REPETITIONS}) reached. Exiting.")
                        raise Exception("Max number of crash repetitions reached.")
                else:
                    print(f"Controller {Settings.CONTROLLER} crashed the car.")
                    print("Crash repetition disabled. Exiting.")
                    raise Exception("Crash repetition disabled.")
            i += 1
                
    
    def prepare_simulation(self):
        self.renderer = None
        self.renderer_backend = None
        self._init_renderer()
        
        
        self.init_drivers()
        self.get_starting_positions()
        self.setup_gym_environment()
        
        


        self.sim_time = 0.0
        self.episode_index = 0

        print("initializing environment with", self.number_of_drivers, "drivers")

    def _init_renderer(self):
        """Initialize renderer backend and load map when rendering is enabled."""
        if Settings.RENDER_MODE is None:
            return

        map_name = Settings.MAP_NAME
        map_ext = ".png"
        map_path = os.path.join(Settings.MAP_PATH, map_name)
        self.renderer_backend = str(getattr(Settings, "RENDER_BACKEND", "pyglet")).lower()

        if self.renderer_backend == "web":
            from sim.f110_sim.envs.rendering.WebRenderer.web_renderer import WebEnvRenderer

            web_host = str(getattr(Settings, "WEB_RENDER_HOST", "127.0.0.1"))
            actor_id = int(getattr(Settings, "ACTOR_ID", 0))
            web_port = int(getattr(Settings, "WEB_RENDER_PORT", 8765)) + actor_id
            auto_open = bool(getattr(Settings, "WEB_RENDER_AUTO_OPEN", True))
            self.renderer = WebEnvRenderer(
                host=web_host,
                port=web_port,
                actor_id=actor_id,
                auto_open_browser=auto_open,
            )
        elif self.renderer_backend == "pygame":
            from sim.f110_sim.envs.rendering.pygame_rendering import EnvRenderer

            window_width, _ = ScreenUtils.get_scaled_window_size(0.7)
            window_height = int(window_width / 1.5)
            self.renderer = EnvRenderer(window_width, window_height)
        else:
            from sim.f110_sim.envs.rendering.pyglet_rendering import EnvRenderer

            window_width, _ = ScreenUtils.get_scaled_window_size(0.7)
            window_height = int(window_width / 1.5)
            self.renderer = EnvRenderer(window_width, window_height)

        if not Settings.BLANK_MAP:
            self.renderer.update_map(map_path, map_ext)


    def setup_gym_environment(self):
        racetrack = os.path.join(Settings.MAP_PATH,Settings.MAP_NAME)

        # Tobi: Place random obstacles on the track
        if(Settings.PLACE_RANDOM_OBSTACLES):
            random_obstacle_creator = RandomObstacleCreator()
            racetrack=random_obstacle_creator.add_random_obstacles(racetrack, self.starting_positions) # uses its own yaml, sets racetrack to the resulting new map in temp folder

        car_parameter_file = Settings.ENV_CAR_PARAMETER_FILE
        path = 'utilities/car_files/'
        env_car_parameters = yaml.load(open(os.path.join(path, car_parameter_file), "r"), Loader=yaml.FullLoader)

        # Simulation settings
        num_agents = 1 + Settings.NUMBER_OF_OPPONENTS
        seed = 12345

        # Initialize lidar simulator
        self.lidar_simulator = LidarSimulator(seed=seed)
        self.lidar_simulator.set_map(Settings.MAP_CONFIG_FILE, ".png")

        # Initialize physics world simulator
        self.world_sim = Simulator(env_car_parameters, num_agents, seed)
        self.world_sim.set_map_collision_checker(self.lidar_simulator.scan_simulator)
        
    """Initialize driver instances for the ego and opponents."""
    def init_drivers(self):
        
        # Init recording active dict with all data from the environment that should be recorded in the car system
        recording_dict = {
                    'time': lambda: self.sim_time,
                    'sim_index': lambda: self.episode_index,
                    'mu': lambda: np.float32(self.vehicle_parameters_instance.mu),
        }
        
        # First planner settings
        driver = CarSystem(Settings.CONTROLLER, recorder_dict=recording_dict)
        
        # Explicitly start recorder since ROS_BRIDGE might be True by default
        if driver.recorder is not None:
            driver.start_recorder()


        #Start looking for keyboard press
        # driver.start_keyboard_listener()

        opponents = []
        waypoint_velocity_factor = (np.random.uniform(-0.05, 0.05) + Settings.OPPONENTS_VEL_FACTOR )
        for _ in range(Settings.NUMBER_OF_OPPONENTS):
            opponent = CarSystem(
                Settings.OPPONENTS_CONTROLLER,
                save_recording=False,
            )
            opponent.planner.waypoint_velocity_factor = waypoint_velocity_factor
            opponent.save_recordings = False
            opponent.use_waypoints_from_mpc = Settings.OPPONENTS_GET_WAYPOINTS_FROM_MPC
            opponents.append(opponent)
            
        self.drivers = [driver] + opponents
        self.number_of_drivers = len(self.drivers)
       
       
  
    def reset(self, poses = None):
        self._last_episode_randomization = apply_episode_randomization(self)

        # Check if respawn is enabled and we have enough history
        if Settings.RESPAWN_ON_RESET and len(self.state_history) >= self.RESPAWN_HISTORY_LENGTH:
            self.respawn()
            return
        
        # Normal reset
        self.episode_index = 0
        self.sim_time = 0.0

        self._reset_control_delay_buffer()

        initial_states = self.get_initial_states()

        self.sim_obs = self.world_sim.reset(initial_states=initial_states)
        self.lidar_simulator.reset_rng(seed=12345)
        self._reset_all_drivers()

        # Clear state history on full reset
        self._clear_respawn_history()

        self.on_step_end()
        self.render_env()

    def run_simulation(self):

        self.reset()
    
        # Main loop
        experiment_length = Settings.SIMULATION_LENGTH
        for self.sim_index in trange(experiment_length):
            self.simulation_step()
            if getattr(self.drivers[0], "lap_limit_reached", False):
                break

        self.on_simulation_end(collision=False)

        print('Sim elapsed time:', self.sim_time, 'Real elapsed time:', time.time()-self.start_time)
        print('laptimes:', str(self.drivers[0].laptimes), 's')
        # End of similation

    def build_driver_observation(self, driver_index, car_state=None, env_state=None):
        """
        Build the observation dict passed to CarSystem.process_observation().

        Same structure as the external ROS bridge: car_state, scans,
        scalar sensors (imu, drivetrain, ...), and env parameters.
        """
        agent = self.world_sim.agents[driver_index]
        env_state = env_state or self.env_state
        if not env_state.get("car_states"):
            env_state = {**env_state, "car_states": self._get_car_states()}

        if car_state is None:
            car_state = agent.state.copy()

        state = agent.state_history[-1]
        prev_state = agent.state_history[-2]
        control = agent.control_history[-1]

        # Simulate sensors from physics state history (not noisy car_state).
        imu = IMUSimulator.from_states(state, prev_state, Settings.TIMESTEP_SIM, self.env_car_parameters)
        motor_sensors = MotorSensorSimulator.from_states(
            state, prev_state, control, self.env_car_parameters, dt=Settings.TIMESTEP_SIM
        )
        simulate_lidar = (
            driver_index == self.world_sim.ego_idx or Settings.OPPONENTS_SIMULATE_LIDAR
        )
        scans = self.lidar_simulator.from_env_state(driver_index, env_state, simulate=simulate_lidar)

        collision = bool(agent.in_collision) or bool(self.sim_obs['collisions'][driver_index])
        terminated = self.sim_obs['terminated']
        interrupted = False # Only happens on manual driving

        done = collision or terminated or interrupted
        # Build observation dict
        observation = {
            'car_state': car_state,
            'scans': scans,
            'sensors': {
                'imu': imu,
                'motor_sensors': motor_sensors,
            },
            'env': {
                'time': float(self.sim_time),
                'sim_index': int(self.episode_index),
                'surface_friction': float(self.vehicle_parameters_instance.mu),
            },
            'env_state': env_state,
            'collision': collision,
            'terminated': terminated,
            'interrupted': False,
            'done': done,
            'info': {},
        }
        return observation

    def simulation_step(self):

        step_start_time = time.time()

        # Build an up-to-date snapshot before control so planners can read the
        # current environment through `env_state`.
        self.env_state = self._build_env_state_snapshot()
        agent_controls = self.get_agent_controls()
        self._run_physics_substeps(agent_controls)

        # Reward/labels are computed in on_step_end; render after so plots include crash penalties.
        self._finalize_control_step()

       
        
        # Store state history for respawn functionality
        self._update_state_history()
        self._apply_step_pacing(step_start_time)

    def _run_physics_substeps(self, agent_controls):
        """Advance world simulation for control timestep, honoring delay buffer."""
        intermediate_steps = int(Settings.TIMESTEP_CONTROL / Settings.TIMESTEP_SIM)
        for _ in range(intermediate_steps):
            self.control_delay_buffer.append(agent_controls)
            agent_controls_execute = self.control_delay_buffer.popleft()
            self.sim_obs = self.world_sim.step(np.array(agent_controls_execute))
            self.sim_time += Settings.TIMESTEP_SIM
            self.episode_index += 1

    def _finalize_control_step(self):
        """Run post-physics step-end hooks for drivers, rendering and done handling."""
        self.on_step_end()
        self.render_env()
        self.check_done()

    def _apply_step_pacing(self, step_start_time):
        """Throttle loop according to render mode and MAX_SIM_FREQUENCY."""
        time_taken = time.time() - step_start_time
        sleep_time = 0.0

        if (
            Settings.RENDER_MODE == "human_fast"
            and self.renderer_backend == "pyglet"
            and time_taken < 0.25 * Settings.TIMESTEP_CONTROL
        ):
            sleep_time = max(sleep_time, 0.25 * Settings.TIMESTEP_CONTROL - time_taken)

        if Settings.MAX_SIM_FREQUENCY is not None:
            min_step_time = 1.0 / Settings.MAX_SIM_FREQUENCY
            if time_taken < min_step_time:
                sleep_time = max(sleep_time, min_step_time - time_taken)

        if sleep_time > 0:
            time.sleep(sleep_time)
        self.step_end_time = time.time()

    def _get_car_states(self):
        """Car poses from the physics world (single source of truth)."""
        return [self.world_sim.agents[i].state.copy() for i in range(self.number_of_drivers)]

    def _build_env_state_snapshot(self):
        """Collect a single snapshot of the full race environment."""
        sim_obs_copy = self.sim_obs.copy() if self.sim_obs is not None else {}
        controls = []
        for i in range(self.number_of_drivers):
            controls.append(
                [
                    float(getattr(self.drivers[i], "angular_control", 0.0)),
                    float(getattr(self.drivers[i], "translational_control", 0.0)),
                ]
            )
        return {
            "time": float(self.sim_time),
            "sim_index": int(self.episode_index),
            "car_states": self._get_car_states(),
            "controls": controls,
            "sim_obs": sim_obs_copy,
        }

    def _update_state_history(self):
        """Update state history for respawn functionality"""
        self.env_state = self._build_env_state_snapshot()
        current_states = self.env_state["car_states"]
        current_controls = self.env_state["controls"]
        current_sim_obs = self.env_state["sim_obs"]
        
        # Add to history
        self.state_history.append(current_states)
        self.control_history.append(current_controls)
        self.sim_obs_history.append(current_sim_obs)
        self.sim_time_history.append(self.sim_time)
        self.sim_index_history.append(self.episode_index)
        
    def _clear_respawn_history(self):
        """Clear stored history used for respawn snapshots."""
        self.state_history.clear()
        self.control_history.clear()
        self.sim_obs_history.clear()
        self.sim_time_history.clear()
        self.sim_index_history.clear()

    def _reset_control_delay_buffer(self):
        """Recreate delay buffer according to current settings and driver count."""
        control_delay_steps = int(Settings.CONTROL_DELAY / Settings.TIMESTEP_SIM)
        self.control_delay_buffer.clear()
        self.control_delay_buffer = deque(
            [
                [np.zeros(2) for _ in range(self.number_of_drivers)]
                for _ in range(control_delay_steps)
            ]
        )

    def _reset_all_drivers(self):
        """Reset every driver instance."""
        for i in range(self.number_of_drivers):
            driver: CarSystem = self.drivers[i]
            driver.reset()

    def respawn(self):
        """Respawn the environment to a state from N timesteps ago (configurable via Settings.RESPAWN_SETBACK_TIMESTEPS)"""
        if len(self.state_history) < self.RESPAWN_HISTORY_LENGTH:
            print("Warning: Not enough state history for respawn. Falling back to full reset.")
            self.reset()
            return
        
        # Get state from N timesteps ago (first entry in history)
        respawn_states = self.state_history[0]
        respawn_controls = self.control_history[0]
        respawn_sim_time = self.sim_time_history[0]
        respawn_sim_index = self.sim_index_history[0]
        
        # Reset simulation index and time
        self.episode_index = respawn_sim_index
        self.sim_time = respawn_sim_time
        
        # Reset simulator to respawn state
        self.sim_obs = self.world_sim.reset(initial_states=np.array(respawn_states))
        
        # Reset drivers
        for i in range(self.number_of_drivers):
            driver = self.drivers[i]
            driver.reset()
            # Set car state to respawn state
            driver.set_car_state(respawn_states[i])
        
        # Clear control delay buffer and repopulate
        self._reset_control_delay_buffer()
        
        # Clear state history to prevent respawn loops
        self._clear_respawn_history()
        
        self.on_step_end()
        self.render_env()

    def manual_respawn(self):
        """Manually trigger respawn - useful for testing or external control"""
        if len(self.state_history) < self.RESPAWN_HISTORY_LENGTH:
            print(f"Warning: Not enough state history for respawn. Need {self.RESPAWN_HISTORY_LENGTH}, have {len(self.state_history)}. Falling back to full reset.")
            self.reset()
            return
        
        print(f"Respawn triggered: Going back {self.RESPAWN_HISTORY_LENGTH} timesteps from sim_index {self.episode_index} to {self.sim_index_history[0]}")
        self.respawn()

    def can_respawn(self):
        """Check if respawn is available (enough state history)"""
        return len(self.state_history) >= self.RESPAWN_HISTORY_LENGTH

    def get_agent_controls(self):
        self._update_history_forger_pre_control()

        self.agent_controls = []

        #Process observations and get control actions
        for index, driver in enumerate(self.drivers):
            driver : CarSystem = driver

            car_state_clean = self.world_sim.agents[index].state
            car_state = self.add_state_noise(car_state_clean)
            driver.sim_obs = self.sim_obs
            driver.car_state_noiseless = car_state_clean

            observation = self.build_driver_observation(index, car_state=car_state)
            angular_control, translational_control = driver.process_observation(observation)
            self.agent_controls.append([angular_control, translational_control ])

        self._update_history_forger_post_control()

        # shape: [number_of_drivers, 2]
        return self.agent_controls

    def _update_history_forger_pre_control(self):
        """Feed control history to history-forger before planner updates."""
        if not Settings.FORGE_HISTORY:
            return
        if self.episode_index > 0:
            for index, driver in enumerate(self.drivers):
                if hasattr(driver, 'history_forger'):
                    driver.history_forger.update_control_history(self.world_sim.agents[index].u_pid_with_constrains)

    def _update_history_forger_post_control(self):
        """Feed state history to history-forger after planner updates."""
        if not Settings.FORGE_HISTORY:
            return
        for index, driver in enumerate(self.drivers):
            if hasattr(driver, 'history_forger'):
                driver.history_forger.update_state_history(self.world_sim.agents[index].state)

    
    def on_step_end(self):
        post_step_env = self._build_env_state_snapshot()
        for i in range(self.number_of_drivers):
            driver : CarSystem = self.drivers[i]
            driver.on_step_end(self.build_driver_observation(i, env_state=post_step_env))
        
    

    def _build_renderer_obs(self):
        """Merge world-sim car states with latest sim_obs for render backends."""
        render_obs = self.sim_obs.copy() if self.sim_obs else {}
        render_obs['car_states'] = np.array(self._get_car_states())
        render_obs['simulation_time'] = self.sim_time
        return render_obs

    def render_env(self):
        if Settings.RENDER_MODE == "human":
            time.sleep(0.001)
            
        if self.renderer is not None:
            render_obs = self._build_renderer_obs()
            if self.renderer_backend in ("web", "pygame"):
                render_obs["web_overlay"] = build_web_overlay(self.drivers)

            self.renderer.render(render_obs)

            # render_callback uses pyglet-specific window attributes (left/right/top/
            # bottom/zoomed_*). Only invoke it when the legacy pyglet backend was
            # actually loaded; if pyglet was unavailable we fell back to pygame
            # under the same `pyglet` name and the callback would crash.
            if (
                self.renderer_backend == "pyglet"
                and Settings.RENDER_MODE in ("human", "human_fast")
                and hasattr(self.renderer, "zoomed_height")
            ):
                self.render_callback(self.renderer)


    """Render extra overlays in pyglet backend."""
    def render_callback(self, env_renderer):
        e = env_renderer
        margin = 0.875 * e.zoomed_height  # ≈ previous 700 when default scaling

        if Settings.CAMERA_AUTO_FOLLOW:
            # update camera to follow car
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)

            # --- Critical change: preserve current zoom (field of view),
            #     only re-center the camera on the car's center.
            cx = 0.5 * (left + right)
            cy = 0.5 * (top + bottom)

            # Keep whatever zoom the user set via mouse scroll:
            half_w = 0.5 * e.zoomed_width
            half_h = 0.5 * e.zoomed_height

            e.left   = cx - half_w
            e.right  = cx + half_w
            e.bottom = cy - half_h
            e.top    = cy + half_h

            # Place labels relative to the current view so they don't drift.
            # Using margins tied to the current view height keeps positions sensible under zoom.
            # ------------------------------------------------------------------

        

        # Keep score label centered at the top of the current camera view
        e.score_label.x = e.left + 0.5 * e.zoomed_width
        e.score_label.y = e.top - margin

        # Place info label at the top-left corner of the current view regardless of camera mode
        padding_x = 0.05 * e.zoomed_width
        padding_y = 0.05 * e.zoomed_height
        e.info_label.x = e.left + padding_x
        e.info_label.y = e.top - padding_y

        # Let the main driver draw its overlays even if the camera is static
        main_driver = self.drivers[0]
        if hasattr(main_driver, 'render'):
            main_driver.render(env_renderer)

    
    """Get starting positions from config/settings with optional randomization."""
    def _load_config_starting_positions(self):
        """Load start positions from map config or fallback settings."""
        with open(Settings.MAP_CONFIG_FILE) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
            conf = Namespace(**conf_dict)
        if hasattr(conf, "starting_positions"):
            starting_positions = conf.starting_positions[0 : self.number_of_drivers]
        else:
            starting_positions = Settings.STARTING_POSITION
        return scale_positions(starting_positions)

    def _apply_reverse_direction_start_positions(self, starting_positions):
        """Apply reverse-direction start behavior (kept backward-compatible)."""
        if not Settings.REVERSE_DIRECTION:
            return starting_positions
        # Preserve existing behavior: use a fixed reverse start anchor.
        starting_positions = [[0, 0, -3.0]]
        new_starting_positions = []
        for starting_position in starting_positions:
            starting_theta = wrap_angle_rad(starting_position[2] + np.pi)
            new_starting_positions.append(
                [starting_position[0], starting_position[1], starting_theta]
            )
        return new_starting_positions

    def _expand_starting_positions(self, starting_positions):
        """Ensure positions list has one slot per driver."""
        if len(starting_positions) >= self.number_of_drivers:
            return starting_positions
        starting_positions = [list(p) for p in starting_positions]
        base_fallback = (
            list(starting_positions[0]) if len(starting_positions) > 0 else [0.0, 0.0, 0.0]
        )
        while len(starting_positions) < self.number_of_drivers:
            starting_positions.append(list(base_fallback))
        return starting_positions

    def _randomize_starting_positions(self, starting_positions):
        """Randomize main and opponent starts around waypoint positions."""
        random_wp_source = None
        if self.drivers and hasattr(self.drivers[0], "waypoint_utils"):
            random_wp_source = self.drivers[0].waypoint_utils.waypoints
        if random_wp_source is None or len(random_wp_source) == 0:
            print("Warning: Could not sample random waypoint; falling back to configured start.")
            return starting_positions

        n_waypoints = len(random_wp_source)
        starting_positions = self._expand_starting_positions(starting_positions)

        base_idx = random.randint(0, n_waypoints - 1)
        random_wp = np.array(random_wp_source[base_idx], copy=True)
        random_wp[WP_X_IDX] += random.uniform(0.0, self.RANDOM_START_MAIN_JITTER_XY)
        random_wp[WP_Y_IDX] += random.uniform(0.0, self.RANDOM_START_MAIN_JITTER_XY)
        random_wp[WP_PSI_IDX] += random.uniform(0.0, self.RANDOM_START_MAIN_JITTER_YAW)
        starting_positions[0] = random_wp[1:4]

        current_idx = base_idx
        for i in range(1, self.number_of_drivers):
            offset = random.randint(
                self.RANDOM_START_OPPONENT_WP_GAP_MIN,
                self.RANDOM_START_OPPONENT_WP_GAP_MAX,
            )
            current_idx = (current_idx + offset) % n_waypoints
            opp_wp = np.array(random_wp_source[current_idx], copy=True)
            opp_wp[WP_X_IDX] += random.uniform(0.0, self.RANDOM_START_MAIN_JITTER_XY)
            opp_wp[WP_Y_IDX] += random.uniform(0.0, self.RANDOM_START_MAIN_JITTER_XY)
            opp_wp[WP_PSI_IDX] += random.uniform(0.0, self.RANDOM_START_MAIN_JITTER_YAW)
            starting_positions[i] = opp_wp[1:4]
        return starting_positions

    def _validate_starting_positions(self, starting_positions):
        """Validate that there are enough start positions for all drivers."""
        if len(starting_positions) >= self.number_of_drivers:
            return
        raise RuntimeError(
            "No starting positions found for all drivers. "
            f"Please configure starts in {Settings.MAP_NAME}.yaml or enable random starts."
        )

    def get_starting_positions(self):
        starting_positions = self._load_config_starting_positions()
        starting_positions = self._apply_reverse_direction_start_positions(starting_positions)

        if Settings.START_FROM_RANDOM_POSITION:
            starting_positions = self._randomize_starting_positions(starting_positions)

        self._validate_starting_positions(starting_positions)
        self.starting_positions = starting_positions
        Settings.STARTING_POSITION = starting_positions
        return starting_positions


    def get_initial_states(self):
        
        if self.initial_states is not None:
            initial_states = np.array(self.initial_states)
        
       
        else:
            starting_positions = self.get_starting_positions()
            initial_states = np.zeros((self.number_of_drivers, len(STATE_VARIABLES)))            
            for i in range(len(starting_positions)):
                initial_states[i][POSE_X_IDX] = starting_positions[i][0]
                initial_states[i][POSE_Y_IDX] = starting_positions[i][1]
                initial_states[i][POSE_THETA_IDX] = starting_positions[i][2]
                initial_states[i][POSE_THETA_COS_IDX] = np.cos(initial_states[i][POSE_THETA_IDX])
                initial_states[i][POSE_THETA_SIN_IDX] = np.sin(initial_states[i][POSE_THETA_IDX])
                initial_states[i][LINEAR_VEL_X_IDX] = 0.0
                initial_states[i][ANGULAR_VEL_Z_IDX] = 0.0
        return initial_states

    # Noise Level can now be set in Settings.py
    def add_state_noise(self, state):

        noise_level = Settings.NOISE_LEVEL_CAR_STATE
        noise_array = np.array(noise_level) * np.random.uniform(-1, 1, len(noise_level))
        state_with_noise = state + noise_array
        
        # Recalculate sin and cos of theta
        state_with_noise[POSE_THETA_COS_IDX] = np.cos(state_with_noise[POSE_THETA_IDX])
        state_with_noise[POSE_THETA_SIN_IDX] = np.sin(state_with_noise[POSE_THETA_IDX])
                
        return state_with_noise

 
    def check_done(self):
        driver = self.drivers[0]
        if driver.driver_observation and driver.driver_observation.get("done"):
            self.handle_done()


    def handle_done(self):
        if Settings.RESET_ON_DONE:
            self.reset()
        else:
            self.on_simulation_end()
            raise CarCrashException("episode done")

                
    """Called at the end of experiment."""
    def on_simulation_end(self, collision=False):
        for driver in self.drivers:
            driver.on_simulation_end(collision=collision)
        if self.renderer is not None:
            self.renderer.close()

    
   

if __name__ == '__main__':

    simulation = RacingSimulation()
    simulation.run_experiments()
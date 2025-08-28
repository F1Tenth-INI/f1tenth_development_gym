from operator import index
import psutil
import os
import time
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
from utilities.waypoint_utils import WaypointUtils, WP_X_IDX, WP_Y_IDX, WP_PSI_IDX
from utilities.state_utilities import (
    STATE_VARIABLES, POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX, LINEAR_VEL_X_IDX, ANGULAR_VEL_Z_IDX,
    )
from utilities.Exceptions import CarCrashException
from utilities.screen_utils import ScreenUtils
if Settings.DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
Settings.ROS_BRIDGE = False  # No ros bridge if this script is running



class RacingSimulation:

    def __init__(self):
        self.crash_repetition = 0
        self.drivers = []
        self.number_of_drivers = 0
        self.starting_positions = []

        self.start_time = time.time()
        self.sim_time = 0.0
        self.sim_index = 0

        self.obs = {}
        self.step_reward = 0
        self.done = False
        self.info = None

        self.agent_controls_calculated = []
        self.control_delay_buffer = []

        self.env = None
        self.sim : Optional[Simulator] = None
        self.laptime = 0.0
        self.initial_states = None

        
        self.renderer = None

        self.vehicle_parameters_instance = VehicleParameters( param_file_name = Settings.CONTROLLER_CAR_PARAMETER_FILE)

        self.state_recording = None
        if Settings.REPLAY_RECORDING:
            import pandas as pd

            self.state_recording = pd.read_csv(Settings.RECORDING_PATH, delimiter=',', comment='#')
            self.time_axis = self.state_recording['time'].to_numpy()
            self.state_recording = self.state_recording[STATE_VARIABLES].to_numpy()
    

    '''
    Run a number of experiments including repetitions on crash as defined in Settings
    '''
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
                if(Settings.REPEAT_IF_CRASHED):
                    
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
        
        # Init renderer
        
        if Settings.RENDER_MODE is not None:        
            from sim.f110_sim.envs.rendering import EnvRenderer

            map_name = Settings.MAP_NAME
            map_ext = ".png"
            map_path = os.path.join(Settings.MAP_PATH, map_name)


            # screen size is 40% of the actual screen size
            # Determine screen size
            window_width, _ = ScreenUtils.get_scaled_window_size(0.7)
            window_height = int(window_width / 1.5)
            self.renderer = EnvRenderer(window_width, window_height)
            self.renderer.update_map(map_path, map_ext)
        
        
        self.init_drivers()
        self.get_starting_positions()
        self.setup_gym_environment()
        
        


        self.sim_time = 0.0
        self.sim_index = 0

        print("initializing environment with", self.number_of_drivers, "drivers")


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
        timestep = 0.01
        seed = 12345

        # Initialize Simulator
        self.sim = Simulator(env_car_parameters, num_agents, seed)

        # Set the map
        map_file = os.path.join(Settings.MAP_PATH, Settings.MAP_NAME + ".png")  
        self.sim.set_map(Settings.MAP_CONFIG_FILE, ".png")
        
    '''
    Initialize the drivers (car_systems) for the simulation:
    First driver is the main car, the others are opponents as defined in Settings.NUMBER_OF_OPPONENTS
    '''
    def init_drivers(self):
        
        # Init recording active dict with all data from the environment that should be recorded in the car system
        recording_dict = {
                    'time': lambda: self.sim_time,
                    'sim_index': lambda: self.sim_index,
                    'mu': lambda: self.vehicle_parameters_instance.mu,
        }
        
        # First planner settings
        driver = CarSystem(Settings.CONTROLLER, recorder_dict=recording_dict)
        
        # Explicitly start recorder since ROS_BRIDGE might be True by default
        if driver.recorder is not None:
            driver.start_recorder()

        if Settings.CONNECT_RACETUNER_TO_MAIN_CAR:
            driver.launch_tuner_connector()
        
        #Start looking for keyboard press
        # driver.start_keyboard_listener()

        opponents = []
        waypoint_velocity_factor = (np.random.uniform(-0.05, 0.05) + Settings.OPPONENTS_VEL_FACTOR )
        for i in range(Settings.NUMBER_OF_OPPONENTS):
            opponent = CarSystem(Settings.OPPONENTS_CONTROLLER)
            opponent.planner.waypoint_velocity_factor = waypoint_velocity_factor
            opponent.save_recordings = False
            opponent.use_waypoints_from_mpc = Settings.OPPONENTS_GET_WAYPOINTS_FROM_MPC
            opponents.append(opponent)


        self.drivers = [driver] + opponents
        self.number_of_drivers = len(self.drivers)
       


        # Populate control delay buffer
        control_delay_steps = int(Settings.CONTROL_DELAY / Settings.TIMESTEP_SIM)
        self.control_delay_buffer = [[np.zeros(2) for j in range(self.number_of_drivers)] for i in range(control_delay_steps)] 
  
    def reset(self, poses = None):

        self.get_starting_positions()

        self.sim_index = 0
        if self.initial_states is not None:
            initial_states = np.array(self.initial_states)
        else:
            initial_states = np.zeros((self.number_of_drivers, len(STATE_VARIABLES)))            
            for i in range(len(self.starting_positions)):
                initial_states[i][POSE_X_IDX] = self.starting_positions[i][0]
                initial_states[i][POSE_Y_IDX] = self.starting_positions[i][1]
                initial_states[i][POSE_THETA_IDX] = self.starting_positions[i][2]
                initial_states[i][POSE_THETA_COS_IDX] = np.cos(initial_states[i][POSE_THETA_IDX])
                initial_states[i][POSE_THETA_SIN_IDX] = np.sin(initial_states[i][POSE_THETA_IDX])
                initial_states[i][LINEAR_VEL_X_IDX] = 0.0
                initial_states[i][ANGULAR_VEL_Z_IDX] = 0.0
                
                
        self.obs = self.sim.reset(initial_states=initial_states)
        time.sleep(0.05) # wait a bit to let the sim initialize properly
    
        for i in range(self.number_of_drivers):
            driver = self.drivers[i]
            driver.reset()
            driver_obs = self.get_driver_obs()
            driver.on_step_end(next_obs=driver_obs)

        # Reset the planner if it has a reset method (for RL agents)
        if hasattr(self.drivers[0].planner, 'reset'):
            self.drivers[0].planner.reset()
        

    def run_simulation(self):

        self.reset()
    
        # Main loop
        experiment_length = len(self.state_recording) if Settings.REPLAY_RECORDING else Settings.EXPERIMENT_LENGTH
        for _ in trange(experiment_length):

            self.simulation_step()
            if(self.obs['collisions'][0]):
                pass


        self.on_simulation_end(collision=False)

        print('Sim elapsed time:', self.laptime, 'Real elapsed time:', time.time()-self.start_time)
        print('laptimes:', str(self.drivers[0].laptimes), 's')
        # End of similation

    def get_driver_obs(self):
        driver_obs = {}
        driver_obs['car_state'] = self.sim.agents[0].state
        driver_obs['scans'] = self.obs['scans'][0]
        driver_obs['truncated'] = self.sim_index >= 8000
        driver_obs['collision'] = True if self.obs['collisions'][0] else False
        driver_obs['done'] = driver_obs['collision'] or driver_obs['truncated']
        driver_obs['info'] = {}
        return driver_obs

    def simulation_step(self, agent_controls=None):
        
        self.update_driver_state(self.drivers[0], 0)
        if(agent_controls is None):
            agent_controls = self.get_agent_controls()
        else:
            agent_controls = agent_controls


        intermediate_steps = int(Settings.TIMESTEP_CONTROL/Settings.TIMESTEP_SIM)
        for _ in range(intermediate_steps):

            # Control delay buffer
            self.control_delay_buffer.append(agent_controls)        
            agent_controls_execute  = self.control_delay_buffer.pop(0)


            self.obs = self.sim.step(np.array(agent_controls_execute))
            self.laptime += self.step_reward
            self.sim_time += Settings.TIMESTEP_SIM
            self.sim_index += 1
            if Settings.RENDER_MODE == "human":
                time.sleep(0.001)

        for i in range(self.number_of_drivers):
            driver : CarSystem = self.drivers[i]
            car_state = self.sim.agents[i].state
            driver_obs = self.get_driver_obs()
            driver.set_car_state(car_state)
            driver.on_step_end(next_obs=driver_obs)

            if(driver_obs['done']):
                self.reset()


        self.update_driver_state(self.drivers[0], 0)

        self.render_env()
        self.check_and_handle_collisions()

        # End of controller time step


    def get_agent_controls(self):
        ranges = self.obs['scans']
        self.get_control_for_history_forger()

        self.agent_controls = []

        #Process observations and get control actions
        for index, driver in enumerate(self.drivers):
            driver : CarSystem = driver
            self.update_driver_state(driver, index)
            # observation = {key: value[index] for key, value in self.obs.items()}
            # Get control actions from driver 
            driver_obs = self.get_driver_obs()
            angular_control, translational_control = driver.process_observation(driver_obs)
            self.agent_controls.append([angular_control, translational_control ])

        self.get_state_for_history_forger()

        # shape: [number_of_drivers, 2]
        return self.agent_controls

    def get_control_for_history_forger(self):
        if not Settings.FORGE_HISTORY: return
        if self.sim_index > 0:
            for index, driver in enumerate(self.drivers):
                if hasattr(driver, 'history_forger'):
                    driver.history_forger.update_control_history(self.sim.agents[index].u_pid_with_constrains)

    def get_state_for_history_forger(self):
        if not Settings.FORGE_HISTORY: return
        for index, driver in enumerate(self.drivers):
            if hasattr(driver, 'history_forger'):
                driver.history_forger.update_state_history(self.sim.agents[index].state)

    def render_env(self):
        
        if self.renderer is not None:
            render_obs = self.obs.copy()
            render_obs.update({
                'simulation_time': self.sim_time,
            })

            self.renderer.render(render_obs)
            
            self.render_callback(self.renderer)


    '''
    This function is called by the environment renderer to render additional information on the screen
    env_renderer: pyglet env_renderer object
    '''
    def render_callback(self, env_renderer):
        e = env_renderer
        if Settings.CAMERA_AUTO_FOLLOW:
            # update camera to follow car
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)
            
            e.score_label.x = left
            e.score_label.y = top - 700
            e.left = left - 800
            e.right = right + 800
            e.top = top + 800
            e.bottom = bottom - 800
            
            e.info_label.x = left - 150 
            e.info_label.y = top +750
            main_driver = self.drivers[0]
            if hasattr(main_driver, 'render'):
                main_driver.render(env_renderer)


    
    '''
    Get starting positions from map config file
    or Settings
    or random waypoint
    '''
    def get_starting_positions(self):
        map_config_file = Settings.MAP_CONFIG_FILE

        with open(map_config_file) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
            conf = Namespace(**conf_dict)


        # Determine Starting positions
        if hasattr(conf, 'starting_positions'):
            starting_positions =  conf.starting_positions[0:self.number_of_drivers]
        else:
            # print("No starting positions in INI.yaml. Taking value from settings.py")
            starting_positions = Settings.STARTING_POSITION

        if(len(starting_positions) < self.number_of_drivers):
            print("No starting positions found.")
            print("For multiple cars please specify starting postions in " + Settings.MAP_NAME + ".yaml")
            print("You can also let oponents start at random waypoint positions")
            exit()
        
        # Reverse direction of map initial positions
        if Settings.REVERSE_DIRECTION:
            starting_positions = [[0,0,-3.0]]
            new_starting_positions = []
            for starting_position in starting_positions:
                starting_theta = wrap_angle_rad(starting_position[2]+np.pi)
                new_starting_positions.append([starting_position[0], starting_position[1], starting_theta])
        
        
        # Starting from random position near a waypoint (overwrite)
        if Settings.START_FROM_RANDOM_POSITION:
            import random
            
            wu = WaypointUtils()
            random_wp = random.choice(wu.waypoints)
            random_wp[WP_X_IDX] += random.uniform(0., 0.2)
            random_wp[WP_Y_IDX] += random.uniform(0., 0.2)
            random_wp[WP_PSI_IDX] += random.uniform(0.0, 0.1)
            
            starting_positions[0] = random_wp[1:4]
            # print("Starting position: ", random_wp[1:4])
            
        
        self.starting_positions = starting_positions
        Settings.STARTING_POSITION = starting_positions


    
    '''
    Update the driver state with the current car state
    Either from gym env or recording
    '''
    def update_driver_state(self, driver, agent_index):
        if Settings.REPLAY_RECORDING:
            driver.set_car_state(self.state_recording[self.sim_index])
            self.env.sim.agents[agent_index].state = driver.car_state
        else:
            car_state = self.sim.agents[agent_index].state 
            car_state_with_noise = self.add_state_noise(car_state)
            driver.set_car_state(car_state_with_noise)
            driver.set_scans(self.obs['scans'][agent_index])


    # Noise Level can now be set in Settings.py
    def add_state_noise(self, state):

        noise_level = Settings.NOISE_LEVEL_CAR_STATE
        noise_array = np.array(noise_level) * np.random.uniform(-1, 1, len(noise_level))
        state_with_noise = state + noise_array
        
        # Recalculate sin and cos of theta
        state_with_noise[POSE_THETA_COS_IDX] = np.cos(state_with_noise[POSE_THETA_IDX])
        state_with_noise[POSE_THETA_SIN_IDX] = np.sin(state_with_noise[POSE_THETA_IDX])
                
        return state_with_noise

 
    def check_and_handle_collisions(self):
        # Collision ends simulation
        if Settings.CRASH_DETECTION:
            if self.obs['collisions'][0] == 1:
                self.on_simulation_end(collision=True)
                if not Settings.OPTIMIZE_FOR_RL:
                    raise CarCrashException('car crashed')

                
    '''
    Called at the end of experiment
    '''
    def on_simulation_end(self, collision=False):
        for driver in self.drivers:
            driver.on_simulation_end(collision=collision)
        if self.renderer is not None:
            self.renderer.close()

    
   

if __name__ == '__main__':

    simulation = RacingSimulation()
    simulation.run_experiments()
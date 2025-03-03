import os
import gym
import time
import yaml

import numpy as np
import pandas as pd

from tqdm import trange
from argparse import Namespace

from f110_gym.envs.base_classes import wrap_angle_rad

from utilities.Settings import Settings
from utilities.car_system import CarSystem
from utilities.random_obstacle_creator import RandomObstacleCreator
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.csv_logger import augment_csv_header_with_laptime
from utilities.waypoint_utils import WaypointUtils, WP_X_IDX, WP_Y_IDX, WP_PSI_IDX
from utilities.saving_helpers import save_experiment_data, move_csv_to_crash_folder
from utilities.state_utilities import (
    STATE_VARIABLES, POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX, LINEAR_VEL_X_IDX, ANGULAR_VEL_Z_IDX,
    full_state_alphabetical_to_original, full_state_original_to_alphabetical)
from utilities.Exceptions import CarCrashException
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
        self.laptime = 0.0

        self.env = None
        self.laptime = 0.0

        self.vehicle_parameters_instance = VehicleParameters( param_file_name = Settings.CONTROLLER_CAR_PARAMETER_FILE)

        self.state_recording = None
        if Settings.REPLAY_RECORDING:
            self.state_recording = pd.read_csv(Settings.RECORDING_PATH, delimiter=',', comment='#')
            self.time_axis = self.state_recording['time'].to_numpy()
            self.state_recording = self.state_recording[STATE_VARIABLES].to_numpy()
    

    '''
    Run a number of experiments including repetitions on crash as defined in Settings
    '''
    def run_experiments(self):

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
                    print("Crash repetition disabled. Exiting.")
                    raise Exception("Crash repetition disabled.")
            i += 1
                
    
    def prepare_simulation(self):
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

        self.env = gym.make('f110-v0', map=racetrack, map_ext=".png", num_agents=len(self.drivers))
        self.env.add_render_callback(self.render_callback)
        assert(self.env.timestep == 0.01)

    '''
    Initialize the drivers (car_systems) for the simulation:
    First driver is the main car, the others are opponents as defined in Settings.NUMBER_OF_OPPONENTS
    '''
    def init_drivers(self):
        # First planner settings
        driver = CarSystem(Settings.CONTROLLER)

        if Settings.CONNECT_RACETUNER_TO_MAIN_CAR:
            driver.launch_tuner_connector()

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

        # Init recorder
        main_driver = self.drivers[0]
        if Settings.SAVE_RECORDINGS and main_driver.save_recordings:
            if Settings.FORGE_HISTORY:
                main_driver.recorder.dict_data_to_save_basic.update(
                    {
                        'forged_history_applied': lambda: main_driver.history_forger.forged_history_applied,
                    }
                )
            main_driver.recorder.dict_data_to_save_basic.update(
                {
                    'lap_times': lambda: self.obs['lap_times'][0],
                    'time': lambda: self.sim_time,
                    'sim_index': lambda: self.sim_index,
                    'nearest_wpt_idx': lambda: main_driver.waypoint_utils.nearest_waypoint_index,
                }
            )
            main_driver.recorder.start_csv_recording()


        # Populate control delay buffer
        control_delay_steps = int(Settings.CONTROL_DELAY / 0.01)
        self.control_delay_buffer = [[np.zeros(2) for j in range(self.number_of_drivers)] for i in range(control_delay_steps)] 


    def run_simulation(self):

        # Reset env
        self.obs, self.step_reward, self.done, self.info = self.env.reset(poses=np.array(self.starting_positions) )
    
        # Main loop
        experiment_length = len(self.state_recording) if Settings.REPLAY_RECORDING else Settings.EXPERIMENT_LENGTH
        for _ in trange(experiment_length):

            self.simulation_step()

        self.env.close()

        self.handle_recording_end()

        print('Sim elapsed time:', self.laptime, 'Real elapsed time:', time.time()-self.start_time)
        print(Settings.STOP_TIMER_AFTER_N_LAPS, ' laptime:', str(self.obs['lap_times']), 's')
        # End of similation

    def simulation_step(self):

        agent_controls_execute = self.get_agent_controls()

        # From here on, controls have to be in [steering angle, speed ]
        self.obs, self.step_reward, self.done, self.info = self.env.step(np.array(agent_controls_execute))


        self.laptime += self.step_reward
        self.sim_time += Settings.TIMESTEP_CONTROL
        self.sim_index += 1

    
        self.check_and_handle_collisions()
        self.handle_recording_step()
        self.render_env()

        

        # End of controller time step


    def get_agent_controls(self):
        ranges = self.obs['scans']
        self.get_control_for_history_forger()

        # Recalculate control every Nth timestep (N = Settings.TIMESTEP_CONTROL)
        intermediate_steps = int(Settings.TIMESTEP_CONTROL/self.env.timestep)
        if self.sim_index % intermediate_steps == 0:

            self.agent_controls_calculated = []

            #Process observations and get control actions
            for index, driver in enumerate(self.drivers):
                self.update_driver_state(driver, index)

                # Get control actions from driver 
                angular_control, translational_control = driver.process_observation(ranges[index], None)

                control_with_noise = self.add_control_noise([angular_control, translational_control])
                self.agent_controls_calculated.append(control_with_noise)

        # Control delay buffer
        self.control_delay_buffer.append(self.agent_controls_calculated)        
        agent_controls_execute  = self.control_delay_buffer.pop(0)

        self.get_state_for_history_forger()

        # shape: [number_of_drivers, 2]
        return agent_controls_execute

    def get_control_for_history_forger(self):
        if not Settings.FORGE_HISTORY: return
        if self.sim_index > 0:
            for index, driver in enumerate(self.drivers):
                if hasattr(driver, 'history_forger'):
                    driver.history_forger.update_control_history(self.env.sim.agents[index].u_pid_with_constrains)

    def get_state_for_history_forger(self):
        if not Settings.FORGE_HISTORY: return
        for index, driver in enumerate(self.drivers):
            if hasattr(driver, 'history_forger'):
                driver.history_forger.update_state_history(full_state_original_to_alphabetical(self.env.sim.agents[index].state))

    def render_env(self):
        # Render the environment
        if Settings.RENDER_MODE is not None:
            self.env.render(mode=Settings.RENDER_MODE)


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
            print("No starting positions in INI.yaml. Taking value from settings.py")
            starting_positions = Settings.STARTING_POSITION

        if(len(starting_positions) < self.number_of_drivers):
            print("No starting positions found.")
            print("For multiple cars please specify starting postions in " + Settings.MAP_NAME + ".yaml")
            print("You can also let oponents start at random waypoint positions")
            exit()
        
            
        
        # Starting from random position near a waypoint
        if Settings.START_FROM_RANDOM_POSITION:
            import random
            
            wu = WaypointUtils()
            random_wp = random.choice(wu.waypoints)
            random_wp[WP_X_IDX] += random.uniform(0., 0.2)
            random_wp[WP_Y_IDX] += random.uniform(0., 0.2)
            random_wp[WP_PSI_IDX] += random.uniform(0.0, 0.1)
            
            starting_positions[0] = random_wp[1:4]
            print("Starting position: ", random_wp[1:4])
            
            if Settings.REVERSE_DIRECTION:
                starting_positions[0][2] = wrap_angle_rad(starting_positions[0][2] + np.pi)
                
        if Settings.REVERSE_DIRECTION:
            starting_positions = [[0,0,-3.0]]
            new_starting_positions = []
            # starting_positions = conf_dict['starting_positions']
            for starting_position in starting_positions:
                starting_theta = wrap_angle_rad(starting_position[2]+np.pi)
                new_starting_positions.append([starting_position[0], starting_position[1], starting_theta])
                print(new_starting_positions)
            conf.starting_positions = new_starting_positions
        
        self.starting_positions = starting_positions
        Settings.STARTING_POSITION = starting_positions


    
    '''
    Update the driver state with the current car state
    Either from gym env or recording
    '''
    def update_driver_state(self, driver, agent_index):
        if Settings.REPLAY_RECORDING:
            driver.set_car_state(self.state_recording[self.sim_index])
            self.env.sim.agents[agent_index].state = full_state_alphabetical_to_original(driver.car_state)
        else:
            car_state = full_state_original_to_alphabetical(self.env.sim.agents[agent_index].state) 
            car_state_with_noise = self.add_state_noise(car_state)
            driver.set_car_state(car_state_with_noise)


    # Noise Level can now be set in Settings.py
    def add_state_noise(self, state):

        noise_level = Settings.NOISE_LEVEL_CAR_STATE
        noise_array = np.array(noise_level) * np.random.uniform(-1, 1, len(noise_level))
        state_with_noise = state + noise_array
        
        # Recalculate sin and cos of theta
        state_with_noise[POSE_THETA_COS_IDX] = np.cos(state_with_noise[POSE_THETA_IDX])
        state_with_noise[POSE_THETA_SIN_IDX] = np.sin(state_with_noise[POSE_THETA_IDX])
                
        return state_with_noise

    def add_control_noise(self, control):
        noise_level = Settings.NOISE_LEVEL_CONTROL
        noise_array = np.array(noise_level) * np.random.uniform(-1, 1, len(noise_level))
        control_with_noise = control + noise_array
        return control_with_noise

    '''
    Recorder function that is called at every step
    '''
    def handle_recording_step(self):
        if Settings.SAVE_RECORDINGS and self.sim_index % Settings.SAVE_REVORDING_EVERY_NTH_STEP == 0:
                for index, driver in enumerate(self.drivers):
                    if driver.save_recordings:
                        driver.recorder.step()

    '''
    Recorder function that is called at the end of experiment
    '''
    def handle_recording_end(self):
        if Settings.SAVE_RECORDINGS:
            for index, driver in enumerate(self.drivers):
                if driver.save_recordings:
                    if driver.recorder.recording_mode == 'offline':  # As adding lines to header needs saving whole file once again
                        augment_csv_header_with_laptime(self.laptime, self.obs, Settings, driver.recorder.csv_filepath)
                    driver.recorder.finish_csv_recording()
                    if Settings.SAVE_PLOTS:
                        save_experiment_data(driver.recorder.csv_filepath)


    def check_and_handle_collisions(self):
        # Collision ends simulation
        if Settings.CRASH_DETECTION:
            if self.obs['collisions'][0] == 1:
                for index, driver in enumerate(self.drivers):
                    if Settings.SAVE_RECORDINGS and driver.save_recordings:
                        driver.recorder.finish_csv_recording()
                        if Settings.SAVE_PLOTS:
                            path_to_plots = save_experiment_data(driver.recorder.csv_filepath)
                        else:
                            path_to_plots = None
                        move_csv_to_crash_folder(driver.recorder.csv_filepath, path_to_plots)

                raise CarCrashException('car crashed')

   

if __name__ == '__main__':

    simulation = RacingSimulation()
    simulation.run_experiments()
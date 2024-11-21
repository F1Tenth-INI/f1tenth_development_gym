import os
import gym
import time
import yaml

import numpy as np
import pandas as pd

from tqdm import trange
from argparse import Namespace

from f110_gym.envs.base_classes import wrap_angle_rad
from f110_gym.envs.dynamic_models_pacejka import StateIndices

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



if Settings.DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

Settings.ROS_BRIDGE = False  # No ros bridge if this script is running

control_noise = [0, 0]


# Noise Level can now be set in Settings.py
def add_noise(x, noise_level=1.0):
    return x+noise_level*np.random.uniform(-1.0, 1.0)


control_with_noise = np.zeros(2)

repeat_because_of_crash = None

def main():
    """
    main entry point
    """
    global repeat_because_of_crash
    repeat_because_of_crash = False
    if Settings.REPLAY_RECORDING:
        state_recording = pd.read_csv(Settings.RECORDING_PATH, delimiter=',', comment='#')
        time_axis = state_recording['time'].to_numpy()
        state_recording = state_recording[STATE_VARIABLES].to_numpy()
    else:
        state_recording = None

    # Config
    # overwrite config.yaml files 
    map_config_file = Settings.MAP_CONFIG_FILE
    
    vehicle_parameters_instance = VehicleParameters( param_file_name = Settings.CONTROLLER_CAR_PARAMETER_FILE)


    # First planner settings
    driver = CarSystem(Settings.CONTROLLER)

    opponents = []
    waypoint_velocity_factor = np.min((np.random.uniform(-0.05, 0.05) + Settings.OPPONENTS_VEL_FACTOR / driver.waypoint_utils.global_limit, 0.5))
    for i in range(Settings.NUMBER_OF_OPPONENTS):
        opponent = CarSystem(Settings.OPPONENTS_CONTROLLER)
        opponent.planner.waypoint_velocity_factor = waypoint_velocity_factor
        opponent.save_recordings = False
        opponent.use_waypoints_from_mpc = Settings.OPPONENTS_GET_WAYPOINTS_FROM_MPC
        opponents.append(opponent)

    ##################### DEFINE DRIVERS HERE #####################

    drivers = [driver] + opponents

    ###############################################################

    number_of_drivers = len(drivers)
    
    # Control can be executed delays ( as on physical car )
    # Control steps are first have to go throuh the control delay buffer before they are executed
    control_delay_steps = int(Settings.CONTROL_DELAY / 0.01)
    
    
    
    # control_delay_buffer =  control_delay_steps * [[[0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1]]]
    control_delay_buffer = [[[0.0, 0.1] for j in range(number_of_drivers)] for i in range(control_delay_steps)] 
    
    print("initializing environment with", number_of_drivers, "drivers")

    with open(map_config_file) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    conf = Namespace(**conf_dict)
        
    # Determine Starting positions [pos_x, pos_y, absolut_angle_car]
    if hasattr(conf, 'starting_positions'):
        starting_positions =  conf.starting_positions[0:number_of_drivers]
    else:
        print("No starting positions in INI.yaml. Taking 0, 0, 0 as default value")
        starting_positions = [[0,0,0]]

    if(len(starting_positions) < number_of_drivers):
        print("No starting positions found")
        print("For multiple cars please specify starting postions in " + Settings.MAP_NAME + ".yaml")
        print("You can also let oponents start at random waypoint positions")
        exit()

    conf = Namespace(**conf_dict)
    
    # Determine Starting positions
    
    if hasattr(conf, 'starting_positions'):
        starting_positions =  conf.starting_positions[0:number_of_drivers]
    else:
        print("No starting positions in INI.yaml. Taking value from settings.py")
        starting_positions = Settings.STARTING_POSITION

    if(len(starting_positions) < number_of_drivers):
        print("No starting positions found")
        print("For multiple cars please specify starting postions in " + Settings.MAP_NAME + ".yaml")
        print("You can also let oponents start at random waypoint positions")
        exit()
        
    if Settings.REVERSE_DIRECTION:
        starting_positions = [[0,0,-3.0]]
        new_starting_positions = []
        # starting_positions = conf_dict['starting_positions']
        for starting_position in starting_positions:
            starting_theta = wrap_angle_rad(starting_position[2]+np.pi)
            new_starting_positions.append([starting_position[0], starting_position[1], starting_theta])
            print(new_starting_positions)
        conf_dict['starting_positions'] = new_starting_positions
        # conf_dict['starting_positions'] = starting_positions
        # print(conf_dict['starting_positions'])
        


    def get_odom(obs, agent_index, drivers):
        
        odom = {
            'pose_x': drivers[agent_index].car_state[POSE_X_IDX],
            'pose_y': drivers[agent_index].car_state[POSE_Y_IDX],
            'pose_theta': drivers[agent_index].car_state[POSE_THETA_IDX],
            'linear_vel_x': drivers[agent_index].car_state[LINEAR_VEL_X_IDX],
            'linear_vel_y': 0, #drivers[agent_index].car_state[LINEAR_VEL_Y_IDX],
            'angular_vel_z': drivers[agent_index].car_state[ANGULAR_VEL_Z_IDX],
        }
        return odom


    def render_callback(env_renderer):
        # custom extra drawing function

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

        for driver in drivers:
            if hasattr(driver, 'render'):
                driver.render(env_renderer)
        
    racetrack = os.path.join(Settings.MAP_PATH,Settings.MAP_NAME)
    
        
            
    
    # Starting from random position near a waypoint
    if Settings.START_FROM_RANDOM_POSITION:
        import random
        
        wu = WaypointUtils()
        random_wp = random.choice(wu.waypoints)
        random_wp[WP_PSI_IDX] -= 0.5 * np.pi
        # random_wp[2] += 0.5 * np.pi
        random_wp[WP_X_IDX] += random.uniform(0., 0.2)
        random_wp[WP_Y_IDX] += random.uniform(0., 0.2)
        random_wp[WP_PSI_IDX] += random.uniform(0.0, 0.1)
        
        starting_positions[0] = random_wp[1:4]
        print("Starting position: ", random_wp[1:4])
    
        
    # Tobi: Place random obstacles on the track
    # For obstacle settings, look @ random_obstacles.yaml
    if(Settings.PLACE_RANDOM_OBSTACLES):
        random_obstacle_creator = RandomObstacleCreator()
        racetrack=random_obstacle_creator.add_random_obstacles(racetrack, starting_positions) # uses its own yaml, sets racetrack to the resulting new map in temp folder


    env = gym.make('f110-v0', map=racetrack, map_ext=".png", num_agents=number_of_drivers)
    env.add_render_callback(render_callback)
    assert(env.timestep == 0.01)
    current_time_in_simulation = 0.0
    cars = [env.sim.agents[i] for i in range(number_of_drivers)]
    obs, step_reward, done, info = env.reset(poses=np.array(starting_positions) )

    for index, driver in enumerate(drivers):
        if Settings.SAVE_RECORDINGS and driver.save_recordings:
            driver.recorder.dict_data_to_save_basic.update(
                {
                    'translational_control_applied': lambda: control_with_noise[0],
                    'angular_control_applied': lambda: control_with_noise[1],
                    'v_y': lambda: env.sim.agents[index].state[StateIndices.v_y],
                    'mu': lambda: vehicle_parameters_instance.mu,
                    # 'estimated_mu': lambda: driver.estimated_mu,
                }
            )
            driver.recorder.start_csv_recording()
    
    # Set Starting position for Settings
    
    Settings.STARTING_POSITION = starting_positions

    if Settings.RENDER_MODE is not None:
        env.render()

    # Add Keyboard event listener

    if Settings.KEYBOARD_INPUT_ENABLE:
        # Run it from pycharm terminal first to receive a prompt to allow for keyboard input
        from pynput import keyboard
        def on_press(key):
            if key == keyboard.Key.space:
                Settings.CAMERA_AUTO_FOLLOW = not Settings.CAMERA_AUTO_FOLLOW
        listener = keyboard.Listener(on_press=on_press)
        listener.start()  # start to listen on a separate thread

    laptime = 0.0
    start = time.time()

    render_index = 0

    real_slip_vec = []
    real_steer_vec = []

    est_slip_vec = []
    est_steer_vec = []

    if Settings.REPLAY_RECORDING:
        experiment_length = len(state_recording)
    else:
        experiment_length = Settings.EXPERIMENT_LENGTH

    for simulation_index in trange(experiment_length):
        if done:
            break

        ranges = obs['scans']

        for index, driver in enumerate(drivers):
            if Settings.REPLAY_RECORDING:
                # sleep(0.05)
                driver.set_car_state(state_recording[simulation_index])
                odom = get_odom(obs, index, drivers)
                env.sim.agents[index].state = full_state_alphabetical_to_original(driver.car_state)
            else:
                odom = get_odom(obs, index, drivers)
                odom.update({'pose_theta_cos': np.cos(odom['pose_theta'])})
                odom.update({'pose_theta_sin': np.sin(odom['pose_theta'])})
                # Add Noise to the car state
                car_state_without_noise = full_state_original_to_alphabetical(env.sim.agents[index].state)  # Get the driver's true car state in case it is needed
                car_state_with_noise = np.zeros(len(STATE_VARIABLES))
                for state_index in range(len(car_state_with_noise)):
                    car_state_with_noise[state_index] = add_noise(car_state_without_noise[state_index], Settings.NOISE_LEVEL_CAR_STATE[state_index])
                # Do not add noise to cosine and sine values, since they are calculated anyways from the noisy pose_theta
                car_state_with_noise[POSE_THETA_COS_IDX] = np.cos(car_state_with_noise[POSE_THETA_IDX])
                car_state_with_noise[POSE_THETA_SIN_IDX] = np.sin(car_state_with_noise[POSE_THETA_IDX])
                
                driver.set_car_state(car_state_with_noise)

            ### GOES TO MPC PLANNER PROCESS OBSERVATION
            start_control = time.time()
            angular_control, translational_control = driver.process_observation(ranges[index], odom)
            end = time.time()-start_control
            # print("time for 1 step:", end)
        

        if Settings.RENDER_MODE is not None:
            env.render(mode=Settings.RENDER_MODE)
            render_index += 1

        # Get noisy control for each driver:
        agent_control_with_noise = []
        
        if(simulation_index % Settings.CONTROL_NOISE_DURATION == 0):
             control_noise =  np.array(Settings.NOISE_LEVEL_CONTROL) *  np.random.uniform(-1, 1, 2)
            
        for index, driver in enumerate(drivers):

            control_with_noise = np.array([driver.angular_control, driver.translational_control]) + control_noise
            # control_with_noise[:] = np.array([driver.angular_control, driver.translational_control]) + control_noise

            agent_control_with_noise.append(control_with_noise)

        # Recalculate control every Nth timestep (N = Settings.TIMESTEP_CONTROL)
        # Add zero angle offset to the steering angle
        if Settings.ZERO_ANGLE_OFFSET is not None:
            env.sim.agents[index].state[StateIndices.steering_angle] = env.sim.agents[index].state[StateIndices.steering_angle] + Settings.ZERO_ANGLE_OFFSET
            
        intermediate_steps = int(Settings.TIMESTEP_CONTROL/env.timestep)
        for i in range(int(intermediate_steps)):
            
            control_delay_buffer.append(agent_control_with_noise)        
            agent_control_executed  = control_delay_buffer.pop(0)
            
            # Check if PID controller needs to be applied
            controlls = []
            for index, driver in enumerate(drivers):
                angular_control_ecexuted, translational_control_executed = agent_control_executed[index]

                controlls.append([angular_control_ecexuted , translational_control_executed]) # Steering velocity, acceleration

            # From here on, controls have to be in [steering angle, speed ]
            obs, step_reward, done, info = env.step(np.array(controlls))
            laptime += step_reward
            
            # Collision ends simulation
            if Settings.CRASH_DETECTION:
                if obs['collisions'][0] == 1:
                    for index, driver in enumerate(drivers):
                        if Settings.SAVE_RECORDINGS and driver.save_recordings:
                            driver.recorder.finish_csv_recording()
                            if Settings.SAVE_PLOTS:
                                path_to_plots = save_experiment_data(driver.recorder.csv_filepath)
                            else:
                                path_to_plots = None
                            move_csv_to_crash_folder(driver.recorder.csv_filepath, path_to_plots)

                    if Settings.REPEAT_IF_CRASHED:
                        repeat_because_of_crash = True
                        break
                    else:
                        raise Exception("The car has crashed.")

        if repeat_because_of_crash:
            break
        # End of controller time step
        if Settings.SAVE_RECORDINGS:
            for index, driver in enumerate(drivers):
                if driver.save_recordings:
                    driver.recorder.step()

        current_time_in_simulation += Settings.TIMESTEP_CONTROL

    if repeat_because_of_crash:
        env.close()
        return
    # End of similation
    if Settings.SAVE_RECORDINGS:
        for index, driver in enumerate(drivers):
            if driver.save_recordings:
                if driver.recorder.recording_mode == 'offline':  # As adding lines to header needs saving whole file once again
                    augment_csv_header_with_laptime(laptime, obs, Settings, driver.recorder.csv_filepath)
                driver.recorder.finish_csv_recording()
                if Settings.SAVE_PLOTS:
                    save_experiment_data(driver.recorder.csv_filepath)
    
    env.close()

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    print(Settings.STOP_TIMER_AFTER_N_LAPS, ' laptime:', str(obs['lap_times']), 's')

def run_experiments():
    for i in range(Settings.NUMBER_OF_EXPERIMENTS):
        print('Experiment nr.: {}'.format(i+1))

        global repeat_because_of_crash
        while repeat_because_of_crash in (True, None):
            if Settings.EXPERIMENTS_IN_SEPARATE_PROGRAMS:
                import subprocess
                import sys
                program = '''
    from run.run_simulation import main
    main()
    '''
                result = subprocess.run([sys.executable, "-c", program])
            else:
                main()

        repeat_because_of_crash = None

if __name__ == '__main__':
    run_experiments()
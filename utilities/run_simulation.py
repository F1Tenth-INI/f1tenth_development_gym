

import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from tqdm import trange
from utilities.Settings import Settings
from utilities.Recorder import Recorder
from utilities.car_system import CarSystem

import pandas as pd

from f110_gym.envs.dynamic_models import pid

# Utilities
from utilities.state_utilities import full_state_original_to_alphabetical, full_state_alphabetical_to_original, FULL_STATE_VARIABLES
from utilities.random_obstacle_creator import RandomObstacleCreator # Obstacle creation
from utilities.setting_utilities import overwrite_from_master_settings

from time import sleep

Settings.ROS_BRIDGE = False # No ros bridge if this script is running

# Noise Level can now be set in Settings.py
def add_noise(x, noise_level=1.0):
    return x+noise_level*np.random.uniform(-1.0, 1.0)



def main():
    """
    main entry point
    """
    if Settings.FROM_RECORDING:
        state_recording = pd.read_csv(Settings.RECORDING_PATH, delimiter=',', comment='#')
        time_axis = state_recording['time'].to_numpy()
        state_recording = state_recording[FULL_STATE_VARIABLES].to_numpy()
    else:
        state_recording = None

    # Config
    # overwrite config.yaml files 
    overwrite_from_master_settings()
    map_config_file = Settings.MAP_CONFIG_FILE

    # First planner settings
    driver1 = CarSystem()


    # second planner
    # driver2 = CarSystem()
   
    ##################### DEFINE DRIVERS HERE #####################
    drivers = [driver1]
    ###############################################################

    number_of_drivers = len(drivers)
    print("initializing environment with", number_of_drivers, "drivers")

    with open(map_config_file) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    ###Loading neural network for slip steer estimation -> specify net name in Settings
    if Settings.SLIP_STEER_PREDICTION:
        from SI_Toolkit_ASF.nn_loader_race import NeuralNetImitatorPlannerNV
        import matplotlib.pyplot as plt
        steer_estimator = NeuralNetImitatorPlannerNV(Settings.NET_NAME_STEER)
        slip_estimator = NeuralNetImitatorPlannerNV(Settings.NET_NAME_SLIP)
        translational_control = None
        angular_control = None
    

    def get_odom(obs, agent_index):
        odom = {
            'pose_x': obs['poses_x'][agent_index],
            'pose_y': obs['poses_y'][agent_index],
            'pose_theta': obs['poses_theta'][agent_index],
            'linear_vel_x': obs['linear_vels_x'][agent_index],
            'linear_vel_y': obs['linear_vels_y'][agent_index],
            'angular_vel_z': obs['ang_vels_z'][agent_index]
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

        for driver in drivers:
            if hasattr(driver, 'render'):
                driver.render(env_renderer)

    racetrack = conf.map_path
    starting_positions =  conf.starting_positions[0:number_of_drivers]
    
    # Tobi: Place random obstacles on the track
    # For obstacle settings, look @ random_obstacles.yaml
    if(Settings.PLACE_RANDOM_OBSTACLES):
        random_obstacle_creator = RandomObstacleCreator()
        racetrack=random_obstacle_creator.add_random_obstacles(racetrack, starting_positions) # uses its own yaml, sets racetrack to the resulting new map in temp folder


    env = gym.make('f110_gym:f110-v0', map=racetrack,
                   map_ext=conf.map_ext, num_agents=number_of_drivers)
    env.add_render_callback(render_callback)
    assert(env.timestep == 0.01)
    current_time_in_simulation = 0.0
    cars = [env.sim.agents[i] for i in range(number_of_drivers)]
  
    obs, step_reward, done, info = env.reset(
        np.array(starting_positions) )

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

    if Settings.FROM_RECORDING:
        experiment_length = len(state_recording)
    else:
        experiment_length = Settings.EXPERIMENT_LENGTH

    for simulation_index in trange(experiment_length):
        if done:
            break

        ranges = obs['scans']

        for index, driver in enumerate(drivers):
            if Settings.FROM_RECORDING:
                sleep(0.05)
                driver.set_car_state(state_recording[simulation_index])
                odom = {} # FIXME: MPC uses just the car state
                env.sim.agents[index].state = full_state_alphabetical_to_original(driver.car_state)
            else:
                odom = get_odom(obs, index)
                odom.update({'pose_theta_cos': np.cos(odom['pose_theta'])})
                odom.update({'pose_theta_sin': np.sin(odom['pose_theta'])})
                # Add Noise to the car state
                car_state_without_noise = full_state_original_to_alphabetical(env.sim.agents[index].state)  # Get the driver's true car state in case it is needed
                car_state_with_noise = np.zeros(9)
                for state_index in range(9):
                    car_state_with_noise[state_index] = add_noise(car_state_without_noise[state_index], Settings.NOISE_LEVEL_CAR_STATE[state_index])
                
                driver.set_car_state(car_state_with_noise)

            real_slip_vec.append(driver.car_state[7])
            real_steer_vec.append(driver.car_state[8])

            if Settings.SLIP_STEER_PREDICTION:
                print("state before: ", driver.car_state)
                driver.car_state = steer_estimator.get_slip_steer_car_state(slip_estimator, odom, translational_control, angular_control)
                print("state after: ", driver.car_state)
                if _ < 20:
                    driver.car_state[7] = 0.0
                est_slip_vec.append(driver.car_state[7])
                est_steer_vec.append(driver.car_state[8])

            ### GOES TO MPC PLANNER PROCESS OBSERVATION
            start_control = time.time()
            angular_control, translational_control = driver.process_observation(ranges[index], odom)
            end = time.time()-start_control
            # print("time for 1 step:", end)
        

        if Settings.RENDER_MODE is not None:
            env.render(mode=Settings.RENDER_MODE)
            render_index += 1

        # Get noisy control for each driver:
        noisy_control = []
        for index, driver in enumerate(drivers):
            translational_control_with_noise = add_noise(driver.translational_control,
                                                         noise_level=Settings.NOISE_LEVEL_TRANSLATIONAL_CONTROL)
            angular_control_with_noise = add_noise(driver.angular_control, noise_level=Settings.NOISE_LEVEL_ANGULAR_CONTROL)
            noisy_control.append([translational_control_with_noise, angular_control_with_noise])
            if (Settings.SAVE_RECORDINGS):
                driver.recorder.get_data(control_inputs_applied=(translational_control_with_noise, angular_control_with_noise))


        for i in range(int(Settings.TIMESTEP_CONTROL/env.timestep)):
            controlls = []

            for index, driver in enumerate(drivers):
                translational_control_with_noise, angular_control_with_noise = noisy_control[index]
                if Settings.WITH_PID:
                    accl, sv = pid(translational_control_with_noise, angular_control_with_noise,
                                   cars[index].state[3], cars[index].state[2], cars[index].params['sv_max'],
                                   cars[index].params['a_max'], cars[index].params['v_max'], cars[index].params['v_min'])
                else:
                    accl, sv = translational_control_with_noise, angular_control_with_noise
                
                controlls.append([sv, accl]) # Steering velocity, acceleration

            obs, step_reward, done, info = env.step(np.array(controlls))
            laptime += step_reward

        if (Settings.SAVE_RECORDINGS):
            for index, driver in enumerate(drivers):
                driver.recorder.save_data()

        current_time_in_simulation += Settings.TIMESTEP_CONTROL
    if Settings.SAVE_RECORDINGS and Settings.SAVE_PLOTS:
        for index, driver in enumerate(drivers):
            driver.recorder.plot_data()
    
    env.close()

    ###PRINT RESULTS FOR ESTIMATION
    if Settings.SLIP_STEER_PREDICTION:
        if Settings.RENDER_MODE is None:
            x = range(Settings.EXPERIMENT_LENGTH)
        else:
            x = range(render_index)

        steer_estimator.show_slip_steer_results(x, real_slip_vec, est_slip_vec, real_steer_vec, est_steer_vec)


    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


def run_experiments():
    for i in range(Settings.NUMBER_OF_EXPERIMENTS):
        print('Experiment nr.: {}'.format(i+1))
        if Settings.EXPERIMENTS_IN_SEPARATE_PROGRAMS:
            import subprocess
            import sys
            program = '''
from utilities.run import main
main()
'''
            result = subprocess.run([sys.executable, "-c", program])
        else:
            main()


if __name__ == '__main__':
    run_experiments()

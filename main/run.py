# Import Planner Classes
from MPPI_Marcin.mppi_planner import MPPI_F1TENTH
from xiang.ftg_planner_freespace import FollowTheGapPlanner as FollowTheGapPlannerXiang
from xiang.ftg_planner_postqualification import FollowTheGapPlanner as FollowTheGapPlannerXiang2
from examples.pure_pursuit_planner import PurePursuitPlanner
from NeuralNetImitator.nni_planner import NeuralNetImitatorPlanner
from FollowTheGap.ftg_planner import FollowTheGapPlanner
# from MPPI.mppi_planner import MppiPlanner

# Obstacle creation
from tobi.random_obstacle_creator import RandomObstacleCreator

import time

import yaml
import gym
import numpy as np
from argparse import Namespace

from tqdm import trange
from main.Settings import Settings

from Recorder import Recorder

from f110_gym.envs.dynamic_models import pid

from main.state_utilities import full_state_original_to_alphabetical


def add_noise(x, noise_level=1.0):
    return x+noise_level*np.random.uniform(-1.0, 1.0)


noise_level_translational_control = 0.0  # ftg: 0.5  # mppi: 2.0
noise_level_angular_control = 0.0  # ftg: 0.05  # mppi: 3.0

def main():
    """
    main entry point
    """

    # Config
    map_config_file = Settings.MAP_CONFIG_FILE

    # First planner settings
    planner1 = FollowTheGapPlanner()
    # planner1 = MPPI_F1TENTH()
    # planner1 = FollowTheGapPlannerXiang2()
    # planner1 = FollowTheGapPlannerIcra()
    # planner1 = NeuralNetImitatorPlanner()
    planner1.plot_lidar_data = False
    planner1.draw_lidar_data = True
    planner1.lidar_visualization_color = (255, 0, 255)

    # second planner
    # planner2 = PurePursuitPlanner(map_config_file = map_config_file)

    # Old MPPI Planner without TF
    # planner2 = MppiPlanner()

    ##################### DEFINE DRIVERS HERE #####################
    drivers = [planner1]
    ###############################################################

    number_of_drivers = len(drivers)
    print("initializing environment with", number_of_drivers, "drivers")
    with open(map_config_file) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    

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
    recorders = [Recorder(controller_name='Blank-MPPI-{}'.format(str(i)), dt=Settings.TIMESTEP_CONTROL) for i in range(number_of_drivers)]
  
    obs, step_reward, done, info = env.reset(
        np.array(starting_positions) )

    if Settings.RENDER_MODE is not None:
        env.render()

    laptime = 0.0
    start = time.time()

    render_index = 0

    for _ in trange(Settings.EXPERIMENT_LENGTH):
        if done:
            break
        ranges = obs['scans']

        for index, driver in enumerate(drivers):
            odom = get_odom(obs, index)
            odom.update({'pose_theta_cos': np.cos(odom['pose_theta'])})
            odom.update({'pose_theta_sin': np.sin(odom['pose_theta'])})
            driver.car_state = full_state_original_to_alphabetical(env.sim.agents[index].state)  # Get the driver's true car state in case it is needed
            translational_control, angular_control = driver.process_observation(ranges[index], odom)

            if (Settings.SAVE_RECORDINGS):
                recorders[index].save_data(control_inputs=(translational_control, angular_control),
                                           odometry=odom, ranges=ranges[index], state=driver.car_state,
                                           time=current_time_in_simulation)

        if Settings.RENDER_MODE is not None:
            env.render(mode=Settings.RENDER_MODE)
            render_index += 1

        for i in range(int(Settings.TIMESTEP_CONTROL/env.timestep)):
            controlls = []

            for index, driver in enumerate(drivers):
                translational_control_with_noise = add_noise(driver.translational_control, noise_level=noise_level_translational_control)
                angular_control_with_noise = add_noise(driver.angular_control, noise_level=noise_level_angular_control)
                if Settings.WITH_PID:
                    accl, sv = pid(translational_control_with_noise, angular_control_with_noise,
                                   cars[index].state[3], cars[index].state[2], cars[index].params['sv_max'],
                                   cars[index].params['a_max'], cars[index].params['v_max'], cars[index].params['v_min'])
                else:
                    accl, sv = translational_control_with_noise, angular_control_with_noise


                controlls.append([sv, accl])

            obs, step_reward, done, info = env.step(np.array(controlls))
            laptime += step_reward

        current_time_in_simulation += Settings.TIMESTEP_CONTROL

    env.close()
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


def run_experiments():
    for i in range(Settings.NUMBER_OF_EXPERIMENTS):
        print('Experiment nr.: {}'.format(i+1))
        if Settings.EXPERIMENTS_IN_SEPARATE_PROGRAMS:
            import subprocess
            import sys
            program = '''
from main.run import main
main()
'''
            result = subprocess.run([sys.executable, "-c", program])
        else:
            main()


if __name__ == '__main__':
    run_experiments()


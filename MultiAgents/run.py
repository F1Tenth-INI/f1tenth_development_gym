

# Import Planner Classes
from FollowTheGap.ftg_planner import FollowTheGapPlanner as FollowTheGapPlannerFlo
from xiang.ftg_planner_freespace import FollowTheGapPlanner as FollowTheGapPlannerXiang
from examples.pure_pursuit_planner import PurePursuitPlanner

# Obstacle creation
from tobi.random_obstacle_creator import RandomObstacleCreator

import time

from matplotlib.font_manager import json_dump
from matplotlib.pyplot import close, sca
import yaml
import gym
import numpy as np
from argparse import Namespace
import json
from Settings import Settings

from OpenGL.GL import *
from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid

# Config
map_config_file = Settings.MAP_CONFIG_FILE


# First planner settings
planner1 = FollowTheGapPlannerFlo()
planner1.speed_fraction = 1.5
planner1.plot_lidar_data =False
planner1.draw_lidar_data = True
planner1.lidar_visualization_color = (255, 0, 255)


# 2nd Car
planner2 = FollowTheGapPlannerXiang()
planner2.speed_fraction = 2.1
planner2.plot_lidar_data = False
planner2.draw_lidar_data = True
planner2.lidar_visualization_color = (255, 255, 255)

# second planner
# planner2 = PurePursuitPlanner(map_config_file = map_config_file)



##################### DEFINE DRIVERS HERE #####################    
# drivers = [ planner1, planner2]
drivers = [ planner2, planner1]
###############################################################    


number_of_drivers = len(drivers)
print("initializing environment with", number_of_drivers, "drivers")


"""
Planner Helpers
"""

def main():
    """
    main entry point
    """
    

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
    cars = [env.sim.agents[i] for i in range(number_of_drivers)]
  
    obs, step_reward, done, info = env.reset(
        np.array(starting_positions) )

    env.render()

    laptime = 0.0
    start = time.time()

    render_index = 0
    while not done:


        ranges = obs['scans']

        # First car
        controlls = []
        
        for index, driver in enumerate(drivers):
            odom = get_odom(obs, index)
            speed, steer =  driver.process_observation(ranges[index], odom)
            accl, sv = pid(speed, steer, cars[index].state[3], cars[index].state[2], cars[index].params['sv_max'], cars[index].params['a_max'], cars[index].params['v_max'], cars[index].params['v_min'])
            controlls.append([accl, sv])

        obs, step_reward, done, info = env.step(np.array(controlls))

        laptime += step_reward
        env.render(mode=Settings.RENDER_MODE)
        render_index += 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


if __name__ == '__main__':
    main()

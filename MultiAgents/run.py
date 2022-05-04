import sys
# Insert every folder where driver classes are in
# like this we can start the project from the root folder
# sys.path.insert(1, 'FollowTheGap')
# sys.path.insert(1, 'examples')

# Import Planner Classes
from FollowTheGap.ftg_planner import FollowTheGapPlanner
from examples.pure_pursuit_planner import PurePursuitPlanner

import time

from matplotlib.font_manager import json_dump
from matplotlib.pyplot import close, sca
import yaml
import gym
import numpy as np
from argparse import Namespace
import json


from OpenGL.GL import *
from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid


config_file = "MultiAgents/config_Oschersleben.yaml"



# First planner settings
planner1 = FollowTheGapPlanner()
planner1.speed_fraction = 1.3
planner1.plot_lidar_data =False
planner1.draw_lidar_data = True
planner1.lidar_visualization_color = (255, 0, 255)


# 2nd Car
planner2 = FollowTheGapPlanner()
planner1.speed_fraction = 1.1
planner2.plot_lidar_data = False
planner2.draw_lidar_data = True
planner2.lidar_visualization_color = (255, 255, 255)

# second planner
# planner2 = PurePursuitPlanner(config_file = config_file)



##################### DEFINE DRIVERS HERE #####################    
drivers = [ planner1, planner2]
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
    

    with open(config_file) as file:
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
        
        
        if False:
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

    env = gym.make('f110_gym:f110-v0', map=conf.map_path,
                   map_ext=conf.map_ext, num_agents=number_of_drivers)
    env.add_render_callback(render_callback)
    
    
    cars = [env.sim.agents[i] for i in range(number_of_drivers)]
    starting_positions =  conf.starting_positions[0:number_of_drivers]
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
        env.render(mode='human_fast')
        render_index += 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


if __name__ == '__main__':
    main()

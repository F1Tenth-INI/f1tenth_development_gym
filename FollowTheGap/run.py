import time
from matplotlib.pyplot import close, sca
import yaml
import gym
import numpy as np
from argparse import Namespace
import json

from ftg_planner import FollowTheGapPlanner

from OpenGL.GL import *
from numba import njit

from pyglet.gl import GL_POINTS
import pyglet

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid

def main():
    """
    main entry point
    """

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # First car
    planner = FollowTheGapPlanner()
    planner.speed_fraction = 1.5
    planner.plot_lidar_data = False
    planner.draw_lidar_data = True
    planner.lidar_visualization_color = (255, 0, 255)



    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

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

        planner.render(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path,
                   map_ext=conf.map_ext, num_agents=1)
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]]))

    car = env.sim.agents[0] 

    env.render()

    laptime = 0.0
    start = time.time()

    render_index = 0
    while not done:


        ranges = obs['scans'][0]

        # print("scan_angles", car.scan_angles)
        # print("side_distances", car.side_distances)
        # print("Scans",  obs['scans'][0])
        # print("obs", obs)
        # print("Car state", car_state)

        # First car
        odom_1 = {
            'pose_x': obs['poses_x'][0],
            'pose_y': obs['poses_y'][0],
            'pose_theta': obs['poses_theta'][0],
            'linear_vel_x': obs['linear_vels_x'][0],
            'linear_vel_y': obs['linear_vels_y'][0],
            'angular_vel_z': obs['ang_vels_z'][0]
        }

        speed, steer =  planner.process_observation(ranges, odom_1)
        accl, sv = pid(speed, steer, car.state[3], car.state[2], car.params['sv_max'], car.params['a_max'], car.params['v_max'], car.params['v_min'])

    
        obs, step_reward, done, info = env.step(np.array([[ accl, sv]]))

        laptime += step_reward
        env.render(mode='human')
        render_index += 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


if __name__ == '__main__':
    main()

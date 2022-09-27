import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from numba import njit

from pyglet.gl import GL_POINTS


from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid
from pure_pursuit_planner import PurePursuitPlanner



def main():
    """
    main entry point
    """

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 1.82461887897713965, 'vgain': 0.90338203837889}
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, 0.17145+0.15875)

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

        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()
    car = env.sim.agents[0] 


    while not done:
        #  Old function
        # speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])

        ranges = ranges = obs['scans'][0]
        odom_1 = {
            'pose_x': obs['poses_x'][0],
            'pose_y': obs['poses_y'][0],
            'pose_theta': obs['poses_theta'][0],
            'linear_vel_x': obs['linear_vels_x'][0],
            'linear_vel_y': obs['linear_vels_y'][0],
            'angular_vel_z': obs['ang_vels_z'][0]
        }
        
        speed, steer = planner.process_observation(ranges, odom_1)

        # PID controller
        accl, sv = pid(speed, steer, car.state[3], car.state[2], car.params['sv_max'], car.params['a_max'], car.params['v_max'], car.params['v_min'])

        obs, step_reward, done, info = env.step(np.array([[ accl, sv]]))

        laptime += step_reward
        env.render(mode='human')
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()

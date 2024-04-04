import subprocess
import os
import time

working_dir_quad_sim = '/home/marcin/mpac_a1'
python_executable_for_vis = '/usr/bin/python3'
path_to_vis = '/home/marcin/mpac_a1/analysis/live/vis.py'

def start_quad_vis():
    working_dir = os.getcwd()
    os.chdir(working_dir_quad_sim)
    vis = subprocess.Popen([python_executable_for_vis, path_to_vis])
    time.sleep(3)
    print('Quadruped visualization started')
    os.chdir(working_dir)
    return vis

def stop_quad_vis(vis):
    working_dir = os.getcwd()
    os.chdir(working_dir_quad_sim)
    vis.kill()
    print('Quadruped visualization stopped')
    os.chdir(working_dir)


if __name__ == '__main__':
    vis = start_quad_vis()
    input("Press Enter to continue...")
    stop_quad_vis(vis)


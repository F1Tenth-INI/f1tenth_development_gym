import subprocess
import signal
import os
import time

"""
If something goes wrong with closing ctrl try in console
pidof ctrl
sudo kill -9 00000
(replace 00000 with the number obtained from first command)
"""

working_dir_quad_sim = '/home/marcin/mpac_a1'
path_to_ctrl = '/home/marcin/mpac_a1/build/ctrl'
path_to_tlm = '/home/marcin/mpac_a1/build/tlm'


def start_quad_sim():
    working_dir = os.getcwd()
    os.chdir(working_dir_quad_sim)
    ctrl = subprocess.Popen([path_to_ctrl], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid, text=True)
    tlm = subprocess.Popen([path_to_tlm], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid, text=True)
    time.sleep(3)
    print("Quadruped simulation started")
    os.chdir(working_dir)
    return ctrl, tlm


def stop_quad(ctrl, tlm):
    working_dir = os.getcwd()
    os.chdir(working_dir_quad_sim)
    os.killpg(os.getpgid(tlm.pid), signal.SIGTERM)
    os.killpg(os.getpgid(ctrl.pid), signal.SIGTERM)
    print("Quadruped simulation stopped")
    os.chdir(working_dir)


if __name__ == '__main__':
    ctrl, tlm = start_quad_sim()
    input('Press Enter to continue')
    stop_quad(ctrl, tlm)


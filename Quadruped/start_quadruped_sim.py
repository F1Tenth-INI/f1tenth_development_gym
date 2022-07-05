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

working_dir = '/home/marcin/mpac_a1'
os.chdir(working_dir)

path_to_ctrl = '/home/marcin/mpac_a1/build/ctrl'
path_to_tlm = '/home/marcin/mpac_a1/build/tlm'

ctrl = subprocess.Popen([path_to_ctrl], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid, text=True)
tlm = subprocess.Popen([path_to_tlm], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid, text=True)

time.sleep(3)

input("Press Enter to continue...")

os.killpg(os.getpgid(tlm.pid), signal.SIGTERM)
os.killpg(os.getpgid(ctrl.pid), signal.SIGTERM)


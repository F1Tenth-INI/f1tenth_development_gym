import subprocess
import os
import time


python_executable_for_vis = '/usr/bin/python3'
path_to_vis = '/home/marcin/mpac_a1/analysis/live/vis.py'

working_dir = '/home/marcin/mpac_a1'
os.chdir(working_dir)

vis = subprocess.Popen([python_executable_for_vis, path_to_vis])
time.sleep(3)
input("Press Enter to continue...")
vis.kill()

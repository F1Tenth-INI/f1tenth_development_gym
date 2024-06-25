
import os
import sys
import time

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utilities.Settings import Settings
from run.run_simulation import run_experiments

# Test with simple PP controller
Settings.MAP_NAME = "RCA2"

Settings.RENDER_MODE = None
Settings.CONTROLLER = 'pp'
Settings.START_FROM_RANDOM_POSITION = False # Start from random position (randomly selected waypoint + delta)
# Settings.STARTING_POSITION = [[3.62, 6.26, 0.378]] # Starting position [x, y, yaw] in case of START_FROM_RANDOM_POSITION = False

time.sleep(1)
run_experiments()
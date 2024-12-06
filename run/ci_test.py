# This script is used to test the CI/CD pipeline. It runs the simulation with the PP controller on the RCA2 map.


import os
import sys
import time

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utilities.Settings import Settings 

time.sleep(1)

# Global Settings
Settings.RENDER_MODE = None
Settings.REVERSE_DIRECTION = False # Drive reverse waypoints
Settings.GLOBAL_WAYPOINT_VEL_FACTOR = 0.6 
Settings.APPLY_SPEED_SCALING_FROM_CSV = False # Speed scaling from speed_scaling.yaml are multiplied with GLOBAL_WAYPOINT_VEL_FACTOR
    
Settings.MAP_NAME = "RCA2"
Settings.START_FROM_RANDOM_POSITION = False # Start from random position (randomly selected waypoint + delta)


time.sleep(1)


# Test: Run the simulation with the PP controller on the RCA2 map (without delay)
Settings.CONTROLLER = 'pp'
Settings.GLOBAL_WAYPOINT_VEL_FACTOR = 0.6 
Settings.CONTROL_DELAY = 0.00

from run.run_simulation import run_experiments
time.sleep(1)
run_experiments()



time.sleep(1)

# Test: Run the simulation with the PP controller on the RCA2 map (with delay)
Settings.CONTROLLER = 'mpc'
Settings.GLOBAL_WAYPOINT_VEL_FACTOR = 1.0 
Settings.CONTROL_DELAY = 0.08
Settings.EXECUTE_NTH_STEP_OF_CONTROL_SEQUENCE = 2


time.sleep(1)

from run.run_simulation import run_experiments
time.sleep(1)

if __name__ == "__main__":
    run_experiments()
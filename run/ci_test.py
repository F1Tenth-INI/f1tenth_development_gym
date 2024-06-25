
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
Settings.RENDER_MODE = None
Settings.CONTROLLER = 'pp'

time.sleep(1)
run_experiments()
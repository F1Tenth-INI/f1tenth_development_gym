
import os

# Must be set BEFORE importing numpy, torch, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


""" Run the race! """
from run.run_simulation import RacingSimulation
if __name__ == '__main__':
    simulation = RacingSimulation()
    simulation.run_experiments()
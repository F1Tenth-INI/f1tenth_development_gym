# This script is used to test the CI/CD pipeline. It runs the simulation with the PP controller on the RCA2 map.


if __name__ == "__main__":
    
    import importlib
    import os
    import sys
    import time

    # Add the parent directory to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from utilities.Settings import Settings # Settings need to be imported first, so that they can be overwritten
    time.sleep(1)
    

    # Global Settings
    Settings.EXPERIMENT_LENGTH = 3000
    Settings.MAP_NAME = "RCA2"
    Settings.MAP_PATH = os.path.join("utilities", "maps", Settings.MAP_NAME)
    Settings.MAP_CONFIG_FILE = os.path.join(Settings.MAP_PATH, Settings.MAP_NAME+".yaml")

    Settings.RENDER_MODE = None 
    Settings.REVERSE_DIRECTION = False # Drive reverse waypoints
    Settings.APPLY_SPEED_SCALING_FROM_CSV = False # Speed scaling from speed_scaling.yaml are multiplied with GLOBAL_WAYPOINT_VEL_FACTOR
    Settings.START_FROM_RANDOM_POSITION = False # Start from random position (randomly selected waypoint + delta)

    Settings.SURFACE_FRICTION = 0.75




    # Test: Run the simulation with the PP controller on the RCA2 map (without delay)
    Settings.CONTROLLER = 'pp'
    Settings.GLOBAL_WAYPOINT_VEL_FACTOR = 0.5 
    Settings.CONTROL_DELAY = 0.00
    time.sleep(1)

    from run_simulation import RacingSimulation
    import run_simulation
    importlib.reload(run_simulation)

    simulation = RacingSimulation()
    simulation.run_experiments()



    time.sleep(1)

    # Test: Run the simulation with the PP controller on the RCA2 map (with delay)
    Settings.CONTROLLER = 'mpc'
    Settings.GLOBAL_WAYPOINT_VEL_FACTOR = 1.0 
    
    Settings.CONTROL_DELAY = 0.08

    importlib.reload(run_simulation)    
    time.sleep(1)

    simulation = RacingSimulation()
    simulation.run_experiments()

    # Assert at least one lap was completed
    laptimes = simulation.drivers[0].laptimes
    assert len(laptimes) > 0, "No lap times recorded"


    time.sleep(1)
     # Test: Run the simulation with the PP controller on the RCA2 map (with delay)
    Settings.CONTROLLER = 'mppi-lite'
    Settings.GLOBAL_WAYPOINT_VEL_FACTOR = 0.9 
    
    Settings.CONTROL_DELAY = 0.08

    importlib.reload(run_simulation)    
    time.sleep(1)


    simulation = RacingSimulation()
    simulation.run_experiments()

    # Assert at least one lap was completed
    laptimes = simulation.drivers[0].laptimes
    assert len(laptimes) > 0, "No lap times recorded"


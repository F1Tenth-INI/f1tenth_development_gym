import os

# Must be set BEFORE importing numpy, torch, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

if __name__ == '__main__':
    # Parse command-line arguments and override Settings
    from utilities.parser_utilities import parse_settings_args
    parse_settings_args(description='Run F1TENTH simulation with configurable settings')

    # Import and run simulation after settings are configured
    from run.run_simulation import RacingSimulation
    simulation = RacingSimulation()
    simulation.run_experiments()
import os

# Must be set BEFORE importing numpy, torch, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Read input arguments from terminal
import argparse
# python run.py --simulation_length 1000 --render_mode human_fast --save_recordings 1 --inference_model SAC_CTest3
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation_length', type=str, default=None)
    parser.add_argument('--render_mode', type=str, default=None)
    parser.add_argument('--save_recordings', type=str, default=None)
    parser.add_argument('--inference_model', type=str, default=None,
                       help='SAC model name for inference mode (e.g., SAC0)')
    
    args = parser.parse_args()

    # Import Settings after parsing arguments
    from utilities.Settings import Settings


    # Override settings with command line arguments
    if args.simulation_length is not None:
        Settings.SIMULATION_LENGTH = int(args.simulation_length)
    if args.save_recordings is not None:
        Settings.SAVE_RECORDINGS = args.save_recordings.lower() in ['true', '1', 'yes']
    if args.render_mode is not None:
        Settings.RENDER_MODE = args.render_mode
    

    # Import and run simulation after settings are configured
    from run.run_simulation import RacingSimulation
    simulation = RacingSimulation()
    simulation.run_experiments()
import os

# Must be set BEFORE importing numpy, torch, jax, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Limit JAX/XLA thread pool (avoids 10s–100s of threads showing in htop)
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=4 "
        "inter_op_parallelism_threads=1"
    )

if __name__ == '__main__':
    # Parse command-line arguments and override Settings
    from utilities.parser_utilities import parse_settings_args
    parse_settings_args(description='Run F1TENTH simulation with configurable settings')

    # Import and run simulation after settings are configured
    from run.run_simulation import RacingSimulation
    simulation = RacingSimulation()
    simulation.run_experiments()
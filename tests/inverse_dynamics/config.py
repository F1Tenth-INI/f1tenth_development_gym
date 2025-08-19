# tests/inverse_dynamics/config.py
# Hard-coded settings for the inverse dynamics test battery (no CLI, no env).
# Adjust paths and lists below to taste.

# If empty, the tests synthesize a small CSV so they always run.
DATA_DIR: str = "/Users/marcinpaluch/PycharmProjects/f1tenth_development_gym/SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_tiny/Recordings/Test"                  # e.g., "/abs/path/to/FOLDER_WITH_DATA"
PROCESS_ALL_FILES: bool = False     # True → ignore MAX_FILES and process all CSVs in DATA_DIR
MAX_FILES: int = 1                  # quick smoke: first file only

# Trajectory lengths to test (number of control steps).
# Keep these short by default so the automatic run is fast.
TRAJECTORY_LENGTHS = [10, 30, 60, 120]

# Which solvers to run
SOLVERS = ["fast", "refine", "hybrid"]

# Inits to sweep for fast/refine (hybrid does its own warm start)
INITS = ["none", "gt", "noisy"]

# Noise scale for 'noisy' init (× state std)
NOISE_SCALE: float = 0.2

# Optional CSV to save results ("" to disable)
SAVE_CSV: str = "/Users/marcinpaluch/PycharmProjects/f1tenth_development_gym/tests/inverse_dynamics/"  # e.g., "tests/_out/inverse_eval_results.csv"

# Synthesis parameters (used when DATA_DIR is empty or missing)
SYNTH_N: int = 128
SYNTH_DT: float = 0.02
SYNTH_MU: float = 0.7

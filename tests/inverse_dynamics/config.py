# tests/inverse_dynamics/config.py
# Hard-coded settings for the inverse dynamics test battery (no CLI, no env).
# Adjust paths and lists below to taste.

# If empty, the tests synthesize a small CSV so they always run.
DATA_DIR: str = "/Users/marcinpaluch/PycharmProjects/f1tenth_development_gym/SI_Toolkit_ASF/test_inverse_dynamics"
PROCESS_ALL_FILES: bool = False   # True → ignore MAX_FILES and process all CSVs in DATA_DIR
MAX_FILES: int = 1                # quick smoke: first file only

# Trajectory lengths to test (number of control steps).
# Keep these *short* by default so the automatic run is snappy; extend when benchmarking.
TRAJECTORY_LENGTHS = [20]     # you can add 60, 120 for deeper sweeps

# === Clear, non-legacy solver names ===
SOLVERS = ["single_pass", "progressive_window"]

# Inits to sweep for fast/refine (we keep 'none' and 'noisy' only; 'gt' is trivial)
INITS = ["none"]

# Noise scale for 'noisy' init (× state std)
NOISE_SCALE: float = 0.2

# Window enumeration over the file for each T
WINDOW_MODE: str = "tail"          # "tail" or "full"
WINDOW_STRIDE: int = 1             # step between end indices
MAX_WINDOWS_PER_T: int = 40         # only used in "tail" mode

# Hybrid windowing parameters (progressive grow)
HYB_WINDOW: int = 30
HYB_OVERLAP: int = 10

# Collect per-step series (enables rmse curves and optional plots)
COLLECT_SERIES: bool = True
PLOT_SERIES: bool = False          # set True to save plots
PLOTS_DIR: str = "tests/inverse_dynamics/_out/plots"
MAX_SERIES_PLOTS: int = 600          # limit number of plots written
# Aggregate per-step RMSE plots (mean ± std) per {file, solver, T, init} bundle
PLOT_AGG_SERIES: bool = True
PLOTS_AGG_DIR: str = "tests/inverse_dynamics/_out/agg_plots"
# Also save the aggregated curve data (mean/std) next to the PNG
SAVE_AGG_SERIES_CSV: bool = True


# Optional CSV to save results ("" to disable). If directory, an auto filename is created.
SAVE_CSV: str = "tests/inverse_dynamics/_out"

# Synthesis parameters (used when DATA_DIR is empty or missing)
SYNTH_N: int = 128
SYNTH_DT: float = 0.02
SYNTH_MU: float = 0.7

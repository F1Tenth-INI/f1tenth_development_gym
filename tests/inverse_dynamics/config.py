# tests/inverse_dynamics/config.py
# Hard-coded settings for the inverse dynamics test battery (no CLI, no env).
# Adjust paths and lists below to taste.

# If empty, the tests synthesize a small CSV so they always run.
DATA_DIR: str = "/Users/marcinpaluch/PycharmProjects/f1tenth_development_gym/SI_Toolkit_ASF/test_inverse_dynamics"
PROCESS_ALL_FILES: bool = False   # True → ignore MAX_FILES and process all CSVs in DATA_DIR
MAX_FILES: int = 1                # quick smoke: first file only

# Trajectory lengths to test (number of control steps).
TRAJECTORY_LENGTHS = [50]     # add 60, 120 for deeper sweeps

# === Clear, non-legacy solver names ===
SOLVERS = ["single_pass", "progressive_window"]

# Inits to sweep for fast/refine ('none' and 'noisy'; 'gt' is trivial)
INITS = ["none"]

# Noise scale for 'noisy' init (× state std)
NOISE_SCALE: float = 0.2

# Window enumeration over the file for each T
WINDOW_MODE: str = "tail"      # "tail" or "full"
WINDOW_STRIDE: int = 20
MAX_WINDOWS_PER_T: int = 40

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

# Overlay GT vs. predicted segments on per-feature time-series and XY path plots
PLOT_OVERLAY: bool = True
PLOT_OVERLAY_FEATURES: list[str] = ["pose_theta", "linear_vel_x", "pose_x", "pose_y"]
PLOTS_OVERLAY_DIR: str = "tests/inverse_dynamics/_out/overlay_plots"

# Optional CSV to save results ("" to disable). If directory, auto filename is created.
SAVE_CSV: str = "tests/inverse_dynamics/_out"

# Synthesis parameters (used when DATA_DIR is empty or missing)
SYNTH_N: int = 128
SYNTH_DT: float = 0.02
SYNTH_MU: float = 0.7

# ==== NEW: combine with offline diagnostics ====
# Enable the diagnostics bundle (time-shift, per-dim residuals, dt alignment, dynamics consistency, prior audit, etc.)
ENABLE_DIAG: bool = True

# Optional heavier sweeps (beware runtime)
DIAG_SWEEP_WINDOWS: bool = True   # ← default ON now
DIAG_SWEEP_MODEL: bool = True     # ← default ON now


# Choose the seed prior kind for warm-starts (depends on utilities.prior_trajectories)
# Common values in your codebase are often "nn" or "kinematic"; default matches online refiner.
KIN_SEED_PRIOR_KIND: str = "nn"

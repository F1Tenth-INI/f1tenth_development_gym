"""Tunables for the safe acados MPC (``Settings.CONTROLLER = 'mpc-acados'``).

Every upper-case number / string can be overridden with an environment variable
``SAFE_MPC_<NAME>`` (e.g. ``SAFE_MPC_SQP_MAX_ITER=5``); the solver is rebuilt automatically.
"""

import os

ACADOS_SOURCE_DIR = os.environ.get("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))
CODEGEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_generated_code")

# --- Horizon / solver -----------------------------------------------------------
N_HORIZON = 30                 # stages; stage length = Settings.TIMESTEP_CONTROL
SQP_MAX_ITER = 2               # SQP iterations per control step
GLOBALIZATION = "MERIT_BACKTRACKING"  # full steps (FIXED_STEP) diverge on this model
HESSIAN_APPROX = "EXACT"       # Gauss-Newton ignores the curvature of the tyre model and stalls
LEVENBERG_MARQUARDT = 0.1
QP_COND_BLOCK = 4              # HPIPM partial condensing: stages per condensed block
NLP_TOL = 1.0e-4               # SQP stops early once converged (eq / ineq / complementarity)
NLP_TOL_STAT = 1.0e-2
FEASIBILITY_TOL = 1.0e-3       # max OCP constraint violation of a rolled-out input sequence
SOLVER_TIGHTENING = 0.10       # solver limits are this fraction tighter than the checked ones

# --- Track constraints ------------------------------------------------------------
BORDER_MARGIN = 0.10           # [m] every car corner keeps this distance to the border
MAX_SLIP_FRONT = 0.25          # [rad] soft tyre slip-angle limits (keeps the NLP well conditioned)
MAX_SLIP_REAR = 0.15
KAPPA_SMOOTHING_WINDOW = 5     # waypoints, circular moving average on geometric curvature

# --- Terminal set: inside the track, v_x <= factor * waypoint speed, heading close to the raceline ---
TERMINAL_SPEED_FACTOR = 0.9     # waypoint speeds are the maximum the raceline allows
TERMINAL_MAX_HEADING_DEG = 5.0  # |mu_N| bound

# --- Speed reference ---------------------------------------------------------------
LAT_ACC_FACTOR_REF = 0.85      # cost reference: v_ref = min(v_wp, sqrt(f * mu g / |kappa|))

# Smoothing of the simulator's near-singular resistance terms in the OCP model
OCP_V_DEAD = 0.3               # [m/s] sign deadband of rolling/cornering resistance
OCP_LAT_FORCE_SMOOTH = 0.3     # [N] |F_y| ~ sqrt(F_y^2 + eps^2) - eps

# --- Cost weights -----------------------------------------------------------------
W_LATERAL = 15.0
W_HEADING = 0.0
W_PROGRESS = 3.0               # reward on progress rate s_dot (m/s); the MPC picks its own speed
W_SPEED = 0.0                  # optional tracking of the relative error 1 - vx / v_ref
W_YAW_ERR = 0.00
W_VY = 0.0
W_STEER_RATE = 0.05            # on servo rate (rad/s)
W_ACCEL = 0.0
W_ACCEL_RATE = 0.5             # on accel change per stage (m/s^2)
W_TERMINAL_SCALE = 5.0
SLACK_L1 = 1.0e4               # exact penalty: soft == hard solution whenever the hard problem is feasible
SLACK_L2 = 1.0e2
SLIP_SLACK_L1 = 1.0e2          # slip limits are not safety constraints, only a mild penalty
SLIP_SLACK_L2 = 1.0e2

# --- Raceline-tracking policy --------------------------------------------------------------
# Used as terminal input of the shifted candidate, as initial candidate, and as SQP warm start
# when the shifted plan is slower than GUESS_STOPPED_SPEED at mid-horizon (standing still is
# a local minimum of the OCP).
GUESS_STOPPED_SPEED = 0.5      # [m/s]
GUESS_SPEED_GAIN = 4.0         # [1/s]
GUESS_K_LATERAL = 1.0
GUESS_K_HEADING = 1.0

for _name, _value in list(globals().items()):
    if _name.isupper() and isinstance(_value, (int, float, str)) and f"SAFE_MPC_{_name}" in os.environ:
        globals()[_name] = type(_value)(os.environ[f"SAFE_MPC_{_name}"])

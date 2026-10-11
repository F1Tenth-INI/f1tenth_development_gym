import math
import os

from numpy import False_
class Settings():

    ## Environment ##
    ENVIRONMENT_NAME = 'Car'  # Car or Quadruped
    ENV_CAR_PARAMETER_FILE = "yokomo1_car_parameters.yml" # Car parameters for simulated car
    SIM_ODE_IMPLEMENTATION = "jax_pacejka"  # Use the implementation  'jax_pacejka' or 'jit_Pacejka' or 'residual': For fast simulation / 'ODE_TF': For SI_Toolkit batch model thats also used in mpc
    CAR_NAME = "yokomo1"

    ## Map ##
    MAP_NAME = "RCA1"  # hangar3, hangar9, hangar12, hangar14, hangar16, london3_small, london3_large, ETF1, ini10, icra2022, RCA1, RCA2, IPZ2
    MAP_PATH = os.path.join("utilities", "maps", MAP_NAME)
    MAP_CONFIG_FILE = os.path.join(MAP_PATH, MAP_NAME+".yaml")
    MAP_SCALE = 1.0  # Uniform scale for map origin/resolution and waypoint positions (1.0 = unchanged)
    WAYPOINT_SPACING = None  # Resample spacing [m] after MAP_SCALE; None = median spacing from CSV

    BLANK_MAP = False  # If True, skip setting map for all sensors (no borders, no scans, no crashes possible)

    # Controller Settings
    CONTROLLER = 'sac_MidlineReturner' # Options: 'manual','mpc','ftg',neural,'pp','stanley', 'mppi-lite', 'mppi-lite-jax', 'sac_agent', 'rpgd-lite-jax', 'example', 'sac_MidlineReturner', 'HirarchicalPlanner'
    MOTOR_PID_IN_CAR_MODEL = False  # If True: control[1] is desired speed and PI is used. If False: control[1] is direct acceleration.

    TIMESTEP_CONTROL = 0.04    # Multiple of 0.01; how often to recalculate control input
    TIMESTEP_SIM = 0.01       # Dont touch.
    MAX_SIM_FREQUENCY = None   # Max simulation frequency in Hz (e.g. 250). None = no limit. If step is faster, waits so it takes exactly 1/freq.
    ACCELERATION_TIME = 20                   #nni 50, mpc 10 (necessary to overcome initial velocity of 0 m/s)
    ACCELERATION_AMPLITUDE = 10           #nni 2, mpc 10 [Float!]

    # Zero Angle offset
    ZERO_ANGLE_OFFSET = 0.00  # Angle offset for the car (left drift is positive, right drift is negative) absolut max steeringangle = 0.4186
    
    ## driving behaviour ## 
    START_FROM_RANDOM_POSITION = True # Start from random position (randomly selected waypoint + delta)
    STARTING_POSITION = [[3.62, 6.26, 0.378]] # Starting position [x, y, yaw] in case of START_FROM_RANDOM_POSITION = False
    
    REVERSE_DIRECTION = False # Drive reverse waypoints
    GLOBAL_WAYPOINT_VEL_FACTOR = 1
    
    RANDOM_WAYPOINT_VEL_FACTOR = False

    
    GLOBAL_SPEED_LIMIT = 15.0
    APPLY_SPEED_SCALING_FROM_CSV = False # Speed scaling from speed_scaling.yaml are multiplied with GLOBAL_WAYPOINT_VEL_FACTOR

    ## Recordings ##
    SAVE_RECORDINGS = True
    SAVE_PLOTS = True # Only possible when SAVE_RECORDINGS is True
    SAVE_REWARDS = True
    SAVE_VIDEOS = False
    
    RECORDING_INDEX = 0
    RECORDING_NAME = '2026-06-19_08-49-32_.csv'
    RECORDING_FOLDER = './ExperimentRecordings/'
    RECORDING_PATH = os.path.join(RECORDING_FOLDER, RECORDING_NAME)
    DATASET_NAME = ""
    RECORDING_MODE = 'online'  # 'online' or 'offline', also 'disable' - partly redundant with SAVE_RECORDINGS
    TIME_LIMITED_RECORDING_LENGTH = None  # FIXME: Not yet working in F1T

    CONNECT_RACETUNER_TO_MAIN_CAR = False

    # Oponents
    NUMBER_OF_OPPONENTS = 0
    OPPONENTS_CONTROLLER = 'pp'
    OPPONENTS_VEL_FACTOR = 0.6
    OPPONENTS_GET_WAYPOINTS_FROM_MPC = False
    OPPONENTS_SIMULATE_LIDAR = False  # If False, only ego runs lidar; opponents get max-range placeholder scans.

    # Virtual opponents: lightweight lidar occluders replaying recorded trajectories (no physics).
    # Per-opponent arrays may hold more entries than NUMBER_OF_VIRTUAL_OPPONENTS; the first N are used.
    NUMBER_OF_VIRTUAL_OPPONENTS = 0
    VIRTUAL_OPPONENT_TRAJECTORY_FOLDER = os.path.join("utilities", "virtual_opponent_trajectories")
    # CSV filenames must contain Settings.MAP_NAME (e.g. sac_opponent_trajectory_RCA1.csv).
    VIRTUAL_OPPONENT_RECORDINGS = [
        "sac_opponent_trajectory_RCA1.csv",
        "sac_opponent_trajectory_RCA1.csv",
        "sac_opponent_trajectory_RCA1.csv",
        "sac_opponent_trajectory_RCA1.csv",
    ]
    VIRTUAL_OPPONENT_DISTANCE_AHEAD_WAYPOINTS = [30, 100, 170, 240]
    VIRTUAL_OPPONENT_DISTANCE_AHEAD_WAYPOINTS_RANDOM_MAX = 20  # Extra waypoints added per opponent at spawn; 0 = off.
    VIRTUAL_OPPONENT_VEL_FACTORS = [0.85, 0.1, 0.1, 0.1]
    VIRTUAL_OPPONENT_START_OFFSET_S = [0.0, 0.0, 0.0, 0.0]
    VIRTUAL_OPPONENT_TRIM_TO_SINGLE_LAP = True  # Use one clean lap from recording (no loop jump).
    VIRTUAL_OPPONENT_SIZE = [0.25, 0.38]  # [width, length] in meters; None uses ego car dimensions.
    VIRTUAL_OPPONENT_LOOP = True  # Loop single-lap recording when playback time exceeds lap duration.
    TERMINATE_ON_VIRTUAL_OPPONENT_COLLISION = True  # End episode with crash penalty on body overlap.

    # Opponent tracker: detect/track opponents (real or virtual) from the ego lidar.
    # Lidar returns inside the track corridor (walls removed via waypoint borders)
    # are clustered into opponent bodies and tracked frame-to-frame for rel. pos/vel.
    OPPONENT_TRACKER_ENABLED = False
    OPPONENT_TRACKER_WALL_MARGIN = 0.30       # m inside the border still counted as wall
    OPPONENT_TRACKER_MAX_RANGE = 8.0          # m, ignore lidar hits beyond this for opponents
    OPPONENT_TRACKER_CLUSTER_GAP = 0.40       # m, max gap between adjacent points in one cluster
    OPPONENT_TRACKER_CLUSTER_ANGULAR_GAP_DEG = 3.0  # deg, max angular gap between beams in one cluster
    OPPONENT_TRACKER_MIN_POINTS = 1           # min lidar points to accept an opponent cluster
    OPPONENT_TRACKER_MIN_SIZE = 0.25          # m, min cluster extent (typical opponent body)
    OPPONENT_TRACKER_MAX_SIZE = 1.5           # m, max cluster extent (rejects wall fragments)
    OPPONENT_TRACKER_GATING_DISTANCE = 1.5    # m, max frame-to-frame association distance
    OPPONENT_TRACKER_MAX_MISSED = 12          # drop a track after this many missed updates
    OPPONENT_TRACKER_HISTORY_LEN = 25         # closest-opponent frames remembered for the policy
    
    # Stop sim / recording after N completed laps (None = use SIMULATION_LENGTH only).
    STOP_AFTER_N_LAPS = None
    STOP_RECORDING_AFTER_N_LAPS = None
    
    # Head2Head Settings
    STOP_IF_OBSTACLE_IN_FRONT = False # Stop if obstacle is immediately in front of the car
    SLOW_DOWN_IF_OBSTACLE_ON_RACELINE = False # Slow down if obstacle is close to the next waypoints
    ALLOW_ALTERNATIVE_RACELINE = False # TODO: check and automatically generate file

    # CBF-QP safety filter (controller-agnostic, applied in CarSystem after the planner).
    # Tunables live in utilities/cbf_safety_filter.py; this is only the on/off switch.
    CBF_SAFETY_FILTER = False         # Master switch for the safety filter

    # MPC predictive safety filter (Tearle et al.). Tunables live in
    # utilities/mpc_safety_filter/; this is only the CarSystem on/off switch.
    MPC_SAFETY_FILTER = False          # Master switch (lazy-loads Pacejka PSF)

    # Random Obstacles
    PLACE_RANDOM_OBSTACLES = False  # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    DELETE_MAP_WITH_OBSTACLES_IF_CRASHED = False
    
    MAX_CRASH_REPETITIONS = 10000000
    
    TRUNCATE_ON_LEAVE_TRACK = True
    LEAVE_TRACK_SLACK = 0.08  # [m] Frenet overshoot past border before truncate
    RESET_ON_DONE = True  # Reset the environment when done
    RESPAWN_ON_RESET = False  # If True, respawn to state N timesteps ago instead of complete reset
    RESPAWN_SETBACK_TIMESTEPS = 50  # Number of timesteps to go back when respawning

    # Experiment Settings
    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENT_MAX_LENGTH = 4000  # In sim timesteps: Length until the simulation is reset
    SIMULATION_LENGTH = 10**6 # In sim timesteps: Length until the simulation is terminated
    MAX_EPISODE_LENGTH = 2048 


    ## Noise ##
    CONTROL_DELAY = 0.08 # Delay between control calculated and control applied to the car, multiple of 0.01 [s]
    # Delay on physical car is about 0.06s (Baseline right now is 0.1s)
    
    #NOISE_LEVEL_CAR_STATE = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    NOISE_LEVEL_CAR_STATE = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    #NOISE_LEVEL_CAR_STATE = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    NOISE_LEVEL_CONTROL = [0.0, 0.0] # noise level [angular, translational]
    # NOISE_EVEL_CONTROL = [0.05, 0.1] # noise level [angular, translational]
    #NOISE_LEVEL_CONTROL = [0.1, 0.7] # noise level [angular, translational]

    
    FACTOR_APPLIED_TRANSLATIONAL_CONTROL = 1.0
    CONTROL_NOISE_DURATION = 10 # Number of timesteps for which the control noise is applied

    ## waypoints generation ##
    MIN_CURV_SAFETY_WIDTH = 0.6             # Safety width [m] incliding car width for the Waypoint generation /utilities/run_create_min_curve_waypoints.py  
    LOOK_AHEAD_STEPS = 30                    # Number of original waypoints that are considered for cost
    INTERPOLATION_STEPS = 1                  # >= 1 Interpolation steps to increase waypoint resolution
    DECREASE_RESOLUTION_FACTOR = 4           # >= 1 Only take every n^th waypoint to decrease resolution
    IGNORE_STEPS = 1                         # Number of interpolated waypoints to ignore starting at the closest one
    INTERPOLATE_LOCA_WP = 1
    GLOBAL_WAYPOINTS_SEARCH_THRESHOLD = 10.0  # If there is a waypoint in cache with a distance to the car position smaller than this, only cache is searched for nearest waypoints, set None to always use global search
    WAYPOINT_LOCAL_SEARCH_BACK = 10   # Local nearest-wpt search behind last index (was 3)
    WAYPOINT_LOCAL_SEARCH_AHEAD = 15  # Local nearest-wpt search ahead of last index (was 5)
    

    ##Lidar Settings ##
    LIDAR_COVERED_ANGLE_DEG = 270
    LIDAR_NUM_SCANS = 1080

    LIDAR_MODE = 'decimation'  # possible: 'decimation', 'custom indices'
    LIDAR_PROCESSED_ANGLE_DEG = 250  # number or 'max'; 170 corresponds to old "LOOK_FORWARD" option
    LIDAR_DECIMATION = 25  # Only taken into account if LIDAR_MODE is 'decimation'

    LIDAR_CORRUPT = False
    LIDAR_MAX_CORRUPTED_RATIO = 0.5

    LIDAR_PLOT_SCANS = False


    ## Pure Pursuit Controller ##
    PP_USE_CURVATURE_CORRECTION = False
    PP_WAYPOINT_VELOCITY_FACTOR = 0.7
    PP_LOOKAHEAD_DISTANCE = 1.62461887897713965  # lookahead distance [m], Seems not used
    PP_VEL2LOOKAHEAD = 0.6  # None for fixed lookahead distance (PP_LOOKAHEAD_DISTANCE)
    PP_FIXPOINT_FOR_CURVATURE_FACTOR = (0.2, 0.3)  # Second number big - big shortening of the lookahead distance, you can change from 0.2+ (no hyperbolic effect) to 1.0 (lookahead minimal already at minimal curvature)
    PP_NORMING_V_FOR_CURRVATURE = 10.0  # Bigger number - higher velocity required to have effect on shortening of lookahead horizon
    PP_BACKUP_LOOKAHEAD_POINT_INDEX = 1  # Backup should be obsolete after new change
    PP_MINIMAL_LOOKAHEAD_DISTANCE = 0.5

    RELOAD_WP_IN_BACKGROUND = False  # If True, waypoints are reloaded in a separate thread

    
    ## MPC Controller ##
    CONTROLLER_CAR_PARAMETER_FILE = "yokomo1_car_parameters.yml"  # Car parameters for future state estimation (might derrive from the GYM_CAR_PARAMETER_FILE) for simulationg "wrong" model
    ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:ks_pacejka'  # Its the model that the predictor uses. Only used for mpc predictions, if ODE predictor chosen
    
    NUM_TRAJECTORIES_TO_PLOT = 20
    OPTIMIZE_EVERY_N_STEPS = 1
    
    ANALYZE_COST = False # Analyze and plot diufferent parts of the MPC cost
    ANALYZE_COST_PERIOD = 100 # Period for analyzing the cost
    

    WAYPOINTS_FROM_MPC = False # Use waypoints generated from MPC instead of the map
    PLAN_EVERY_N_STEPS = 4 # in case of waypoints from MPC, plan the waypoints every Nth step
    
    
    ## Visualization ##
    KEYBOARD_INPUT_ENABLE = False  # Allows for keyboard input during experiment. Causes silent crash on some computers
    RENDER_MODE = 'human_fast'    #selects rendering cadence only: None, 'human', or 'human_fast'
    # RENDER_MODE = None          # no rendering
    RENDER_BACKEND = 'web'               # backend: 'web', 'pyglet' (deprecated), or 'pygame' (experimental)
    WEB_RENDER_HOST = '0.0.0.0'          # web renderer bind host (0.0.0.0 exposes to LAN/VPN)
    WEB_RENDER_PORT = 8765               # base TCP port for actor 0; actor N uses WEB_RENDER_PORT + N
    WEB_RENDER_AUTO_OPEN = True          # open a browser tab per sim on startup (one tab per renderer)
    WEB_RENDER_DRAW_POLYNOMIAL = False    # show fitted local raceline polynomial in the web renderer
    # Live display delay in *simulation* seconds. Same-PC pygame parity ≈ 1 control step (~0.04–0.08).
    # Raise (e.g. 1.5–2.5) only for remote/VPN viewing where you prefer smoothness over lag.
    WEB_RENDER_LIVE_DELAY_S = 0.06
    WEB_RENDER_BUFFER_WINDOW_S = 1.5
    # Dense overlays each frame (0 = every published pose). Lower only if remote bandwidth is tight.
    WEB_RENDER_OVERLAY_HZ = 0.0
    WEB_RENDER_LIDAR_POINTS = 220
    WEB_RENDER_MAX_ROLLOUTS = 6
    WEB_RENDER_MAX_ROLLOUT_POINTS = 24
    PYGAME_RENDER_FPS = 60               # cap pygame draw rate; sim still runs at full speed, only screen flips are throttled

    CAMERA_AUTO_FOLLOW = False  # Automatically follow the first car on the map
    RENDER_INFO = True  # Render additional information on the screen
    PRINTING_ON = False
    FLOAT_ON_TOP = False  # Float the rendering window on top of all other windows, implemented for Mac only

    KEYBOARD_LISTENER_ON = False
    
    ## RL Settings
    RL_CRASH_REWQARD_RAMPING = False
    
    
    ## Experiment Analysis Settings ##
    
    ## Forged history settings 
    FORGE_HISTORY = False # Forge history of friction values
    SAVE_STATE_METRICS = False # Save state metrics for analysis
    FRICTION_FOR_CONTROLLER = None # Friction value for the controller. If None, controller will use the friction value from the car params / Settings.SURFACE_FRICTION

    
    ### Other Settings ###
    ROS_BRIDGE = True # Automatically determined on program start
    GLOBALLY_DISABLE_COMPILATION = False # Disable TF Compilation
    DISABLE_GPU = True # Disable GPU usage for TF

    ## SAC Agent planner
    ACTOR_ID = 0                         # RL actor index (TCP learner + web renderer port offset)
    LEARNER_TCP_HOST = "127.0.0.1"       # learner server address for sac_agent training client
    LEARNER_TCP_PORT = 5555              # learner server port (match run_training.py --port)
    SAC_INFERENCE_MODEL_NAME = None  # Model name to be used for inference. If None, the agent will be in training mode
    # SAC_INFERENCE_MODEL_NAME = 'Example-3b'  # Model name to be used for inference. If None, the agent will be in training mode
    # If set (seconds), training actor sends learner terminate once at least two completed laps
    # are strictly faster than this (so metrics can include the first sub-threshold lap).
    SAC_TERMINATE_BELOW_LAPTIME = None
    SAC_AGENT_DEBUG = False
    LEARNER_SERVER_DEBUG = False
    # Log every N streamed batches to stdout (1 = every batch).
    LEARNER_INGEST_LOG_EVERY_BATCHES = 5
    # Min seconds between train status updates when not using run_training combined display.
    LEARNER_SERVER_TRAIN_LOG_INTERVAL_S = 1.0

    SAC_SPEED_CURRICULUM_LEARNING = False
    SAC_CURRICULUM_DEBUG = False

    SAC_CURRICULUM_ENABLED = False

    ## start to t1 -> starting difficulty | t1 to t2 -> linear increase to 1.0 | t2 to end -> 1.0
    ## t1=0 ensures difficulty rises from the first boost; t1=0.3 required 6+ boosts before any visible change
    SAC_CURRICULUM_STARTING_DIFFICULTY = 0.0 
    SAC_CURRICULUM_T1 = 0.0        # progress threshold: difficulty stays at initial until progress > t1
    SAC_CURRICULUM_T2 = 0.9

    ## Translational control clipping: curriculum increases clip from min to max as difficulty increases
    SAC_TRANSLATIONAL_CLIP_MIN = 2.5   # clip at low difficulty (conservative)
    SAC_TRANSLATIONAL_CLIP_MAX = 6.0   # clip at high difficulty (full range)
    SAC_TRANSLATIONAL_CONTROL_CLIP = 2.5  # runtime value, updated by curriculum (default for inference)

    ## Curriculum speed limit: v_max in car model increases with difficulty (works with acceleration control)
    SAC_CURRICULUM_V_MAX_ENABLED = True
    SAC_CURRICULUM_V_MAX_MIN = 3.0   # v_max at low difficulty [m/s]
    SAC_CURRICULUM_V_MAX_MAX = 10.0  # v_max at high difficulty [m/s], or use vehicle default
    SAC_CURRICULUM_V_MAX = None      # runtime value, set by curriculum (None = use vehicle default)

    ## Adaptive curriculum: when avg reward over last N episodes > threshold, boost progress
    SAC_CURRICULUM_ADAPTIVE = True
    SAC_CURRICULUM_REWARD_THRESHOLD = 10.0  # avg reward above this triggers difficulty boost (0 = break-even)
    SAC_CURRICULUM_REWARD_WINDOW = 4       # number of episodes for rolling average (smaller = more responsive)
    SAC_CURRICULUM_FAST_TRACK_BOOST = 0.05  # progress increment when threshold exceeded
    ## Episode-length curriculum: when X% of last N episodes reach max length, increase difficulty
    SAC_CURRICULUM_MAX_LENGTH_PERCENTAGE = 0.6  # fraction of episodes at max length to trigger boost (e.g. 0.6 = 60%)

    SAC_SAVE_MODEL_CHECKPOINTS = True
    SAC_CHECKPOINT_FREQUENCY = 5000 #in timesteps
    # Replay CSV during background checkpoints (can be slow; main model save is always kept).
    SAC_CHECKPOINT_SAVE_REPLAY = False
    # UDT = learner total_weight_updates / total_actor_timesteps. When set, SAC agent adjusts
    # MAX_SIM_FREQUENCY after each training_info update (see learner_server + sac_agent_planner).
    SAC_TARGET_UTD = None
    SAC_MAX_UTD = 4 
    SAC_UDT_FREQ_ADJUST_STEP_RATIO = 0.05
    SAC_MIN_SIM_FREQUENCY = 20.0
    SAC_UDT_REF_SIM_FREQUENCY = 200.0
    # SAC replay minibatch size (samples per gradient step on the learner).
    SAC_BATCH_SIZE = 256
    # Multi-step TD: critic target uses an n-step return instead of 1-step Bellman.
    # 1 = classic SAC. Typical values 3–5; larger n reduces bias, increases variance.
    SAC_N_STEP = 1
    # Stream transitions to the learner every N env steps (episode tail is flushed on done).
    SAC_STREAM_BATCH_SIZE = 32
    # Skip episode-end TCP batches shorter than this (guards crash-only spam).
    SAC_MIN_EPISODE_END_BATCH_SIZE = 2
    # Metrics PNG (training_metrics.png):
    # SAC_METRICS_PLOT_INTERVAL_S > 0: refresh every N seconds (wall clock).
    # SAC_METRICS_PLOT_EVERY (when interval is 0): positive int = every N CSV rows; "end" = only when training stops.
    SAC_METRICS_PLOT_INTERVAL_S = 0
    SAC_METRICS_PLOT_EVERY = "end"
    # TrainingPlot dashboard (run TrainingPlot/run_training_plot.py separately).
    TRAINING_PLOT_PORT = 8070
    TRAINING_PLOT_POLL_S = 2.0
    # Legacy aliases kept for older scripts / web renderer configs.
    LEARNER_METRICS_HTTP_ENABLED = True
    LEARNER_METRICS_HTTP_PORT = TRAINING_PLOT_PORT
    LEARNER_METRICS_HTTP_POLL_S = TRAINING_PLOT_POLL_S
    # Open SAC metrics side panel in the web renderer on load (sac_agent only).
    SAC_METRICS_PANEL_OPEN_DEFAULT = False
    # Matplotlib PNG (training_metrics.png) written to the model directory.
    SAC_MODEL_AUTOSAVE_INTERVAL_S = 60.0
    # When UDT is below target, allow larger grad bursts (updates per train round).
    SAC_GRAD_BURST_MULTIPLIER = 4
    SAC_MAX_GRAD_BURST = 2048

    # Saves full obs and action for each transition, so that for analysis, models can be called on all transitions explored during training directly
    SAC_STAT_TRACKER = False
    SAC_STAT_TRACKER_FULL_OBS_ACTION_SAVE = True 

    USE_CUSTOM_SAC_SAMPLING = False

    ## Hierarchical world-model planner (TrainingLite/hwm) ## temporary, some will beredundant later
    HWM_INFERENCE_MODEL_NAME = None      # None -> training mode (weights from learner over TCP); str -> load models/<name>/ from disk, no TCP
    HWM_AGENT_DEBUG = False
    HWM_STREAM_BATCH_SIZE = 16 #128           # stream [state, action] rows to the learner every N control steps
    HWM_MIN_EPISODE_END_BATCH_SIZE = 2   # skip episode-end batches shorter than this
    HWM_MAX_TRANSLATIONAL_CONTROL = 12.0  # clip on physical translational control
    HWM_WALL_POINTS = 40                 # relative wall points per side fed to the observation builder
    HWM_CONTEXT_LEN = 16                 # recent [state, action] rows available to the observation builder
    HWM_SUPER_STATE_SIZE = 3             # [state, action] steps in one LLD superstate
    HWM_LLD_BATCH_SIZE = 32              # independent random transitions per LLD update
    HWM_MEMORY_DROP_FINAL_STEPS = 5   # the last this many rows of every finished episode (the crash) are excluded from the LLD's neighbor bank and from every training batch and root. They stay stored. 0 -> keep everything
    HWM_SAMPLE_ACTIONS = True            # planner samples from the actor; False -> tanh of the Gaussian mean
    HWM_TARGET_POSITION_WEIGHT = 1.7     # progress-reward distance to the target state: weight per m of position error
    HWM_TARGET_HEADING_WEIGHT = 0.5#1.0 / math.pi  # ... per rad of heading error (wrapped)
    HWM_TARGET_SPEED_WEIGHT = 0.5        # ... per m/s of forward-speed error
    HWM_TARGET_SPEED = 1.0               # forward speed of the waypoint target states, m/s
    HWM_WAYPOINT_LOOKAHEAD = 20           # tracked target: the n-th waypoint ahead of the car, 1 = next one (about 0.1 m apart)
    HWM_PP_DRIVES = True                # Pure Pursuit drives instead of the actor; its rows still train the learner
    HWM_PP_UNTIL_WEIGHTS = True         # Pure Pursuit until the learner's first weight broadcast; False -> drive immediately with the local weights, including the random init
    HWM_LLD_PREVIEW = True               # render the driving controller's plan (actor, GD coefficients, or Pure Pursuit) in the simulator's dynamics (black) next to the LLD's prediction of it (orange mean, 2σ band)
    HWM_SIM_USE_LLD = False              # the car is stepped by the LLD once per control step, instead of the Pacejka model. The action is the one just commanded, matching how the LLD is trained
    HWM_SIM_LLD_SAMPLE = False           # with HWM_SIM_USE_LLD: False -> the LLD mean (expected next state). True -> one sample from the LLD
    HWM_LLD_PREVIEW_SAMPLES = 10         # sampled LLD rollouts for the mean path and its uncertainty band
    HWM_LLD_COSINE_ATTENTION = False     # False: attention is softmax of minus L2 distance over temperature, and n_soft is 1 + sum of exp(-distance/temperature), which scales the covariance and the uncertainty reward. True: attention is softmax of the raw inner product <k, q> over temperature, and n_soft is ignored (covariance is unscaled, uncertainty term is zero)
    HWM_LLD_SEPARATE_ENCODERS = False   # False: one encoder for the query and the memory keys. True: a second encoder of the same shape encodes keys only; the query keeps attention_norm_projector. Works with inner product and L2. Needs a new model: the key encoder is extra weights
    HWM_LLD_ROLLOUT_WINDOW = None       # None: a rollout attends over every memory key, same as LLD training. An integer: at every imagined step of a rollout, keep that many keys with the highest attention for that step's query and use only those. Training the LLD still uses the whole memory
    HWM_LLD_STUDENT_T = True            # False: the LLD predicts a Gaussian. True: one multivariate Student-t with the same weighted mean and weighted covariance (no (1 + 1/n) factor, that is Gaussian only) and nu = max(HWM_LLD_T_MIN_DOF, n - d), d the delta dimension, n = n_soft (L2 attention) or the effective sample size 1 / sum(w^2) (dot-product attention, where n_soft is ignored). Training maximizes its log likelihood. The predicted mean, scale_tril and noise inversion keep their meaning; sample() draws from the t
    HWM_LLD_CORRECTION_NET = True       # True: a 2-layer network g of the flat superstate corrects the LLD mean. Residual mode (HWM_LLD_CORRECTION_RESIDUAL True): mean = g(q) + sum_j w_j (delta_j - g(k_j)), covariance of the residuals. Additive mode: mean = attention mean + g([q, attention mean]), stored deltas and covariance untouched. The last layer starts at zero. g extrapolates where memory has no close neighbor. Needs a new model: g is extra weights
    HWM_LLD_CORRECTION_RESIDUAL = True   # see HWM_LLD_CORRECTION_NET. Residual and additive models have different weight shapes
    HWM_LLD_DUMMY_DISTANCE_INIT = 2    # initial distance of the dummy token (its attention logit is -distance / HWM_LLD_ATTENTION_TEMPERATURE, then learned). With a small temperature the dummy starts far below every key (T=1/60: logit -120) and gets no gradient, so it never competes. Set it near the typical distance of the nearest key (about 12 T)
    HWM_LLD_CORRECTION_INPUT_RESIDUAL = False  # residual mode only: the attention mean of the retrieved residuals (detached) is an extra input of g at the query (zero at the keys, whose residuals must not depend on it)
    HWM_LLD_CORRECTION_INPUT_COV = False       # residual mode only: the retrieved covariance (its lower-triangular Cholesky factor, detached) is an extra input of g at the query, same as above
    HWM_LLD_CORRECTION_HIDDEN = 64       # hidden width of that network
    HWM_LLD_CORRECTION_DEEP = False      # residual mode only: one more hidden layer of HWM_LLD_CORRECTION_HIDDEN, added onto the previous activation (h + Linear(ReLU(h)), that linear starting at zero). The output layer still starts at zero. Needs a new model: the extra layer is extra weights
    HWM_LLD_DENSITY_WEIGHTING = True    # True: the attention weight of every memory key is also divided by that key's own n_soft, the kernel mass sum_k exp(-||e_j - e_k|| / T) of the keys around it (a key in a dense region counts less). Computed from the stored key embeddings and kept in memory; refreshed with the embeddings. n_soft of the query (Student-t nu, Gaussian scale) is not changed. Ignored with HWM_LLD_COSINE_ATTENTION. Needs HWM_LLD_WHOLE_MEMORY_ATTENTION
    HWM_LLD_GRAD_CLIP = 1.0             # max gradient norm of an LLD update
    HWM_LLD_TRAIN_FUTURE_MARGIN = 25     # LLD training only: a sample may retrieve memory windows from the past and from the far future, but not from the rows right after it. Windows that start fewer than this many rows after the sample's next row are hidden (25 control steps = 1 s, about one curve). Needs HWM_LLD_WHOLE_MEMORY_ATTENTION. Inference still sees only the past
    HWM_LLD_SKIP_START_ROWS = 10         # LLD training ignores steps from the first N rows of each episode (padded history, delayed first commands)
    HWM_LLD_KINEMATIC_RESIDUAL = False   # position is the average-velocity integration, as without HWM_LLD_PREDICT_DISPLACEMENT, plus a correction the model predicts: the delta gets two more channels, the true car-frame displacement minus that integral (forward, left), and the prediction is added to the position. Yaw rate, vx, vy, steering and heading are predicted and reconstructed as without it (yaw rate = yaw rate + the predicted change). Takes the place of HWM_LLD_PREDICT_DISPLACEMENT if both are on. Needs a new model: two more delta channels
    HWM_PLAN_MODE = "traj"             # "actor": the stochastic actor's own action, trained by imagination; "gd": gradient descent on the coefficient feedback law, the actor is trained on the optimized coefficients; "traj": a deterministic actor drives as is, trained on action sequences optimized by gradient descent under fixed LLD noise (HWM_TRAJ_*)
    HWM_TRAJ_HORIZON = 20               # control steps n of every training trajectory
    HWM_TRAJ_SAMPLES = 8                # trajectories K per root, each with its own noise vector held for all n steps and its own action sequence
    HWM_TRAJ_ROOTS = 4                  # rows of each sampled LLD batch used as roots; None -> all HWM_LLD_BATCH_SIZE rows. Cost scales with roots * samples * (gd steps + 2)
    HWM_TRAJ_GD_STEPS = 5               # gradient steps on the action sequences per training step
    HWM_TRAJ_GD_LR = 1.0                # multiplies d(reward)/d(action); the step is also capped by HWM_TRAJ_ACTION_STEP
    HWM_TRAJ_ACTION_STEP = 0.3          # max |Δaction| per gradient step, network units (actions live in [-1, 1])
    HWM_TRAJ_NOISE_CLIP = 10.0           # trajectory noise and the inferred transition noise are clamped to ±this many standard deviations
    HWM_TRAJ_NOISE_EMA = 0.95            # inference: actor noise input = this * previous + (1 - this) * noise inferred from the last transition; 0 -> last transition only. Zeros at every episode start. Training feeds each trajectory's noise as a constant
    HWM_TRAJ_WALL_WEIGHT = 10.0         # multiplies the trajectory's summed wall punishment (meters, zero or negative) against its summed waypoint progress
    HWM_TRAJ_ACTION_REG = 0.01          # subtracted from the reward: this times the summed squared distance from the actor's actions
    HWM_TRAJ_DRIVE_OPTIMIZED = True    # the car plays one gradient-descent action sequence open-loop under the LLD mean (noise zero), and plans the next when it runs out. The actor is not used while this is on
    HWM_TRAJ_TARGET_STEPS = 5        # actor regression uses only the first this many optimized actions of each trajectory. None -> the whole HWM_TRAJ_HORIZON
    HWM_GD_HORIZON = 20                  # control steps one set of coefficients is rolled out; the car drives the action the law gives at the current state
    HWM_GD_LR = 1.0                     # multiplies d(reward)/d(coefficient). The step is also capped by HWM_GD_COEFF_STEP, so a huge slope cannot flip a coefficient in one update
    HWM_GD_COEFF_STEP = 0.1             # max |Δcoefficient| per gradient step. Coefficients live in [-1, 1]
    HWM_GD_STEPS = 8                     # gradient steps per plan while training
    HWM_GD_TIME_BUDGET_S = 0.4          # inference: no retrieval or gradient step starts after this many seconds; the coefficients so far are driven if time runs out first. None -> HWM_GD_STEPS
    HWM_GD_ASYNC = True                 # inference: plans run back to back in a background thread, each from the newest state; every control step drives the feedback law with the latest finished coefficients instead of waiting. False -> one blocking plan per control step
    HWM_GD_ASYNC_THREADS = 1            # torch intra-op threads once the background thread starts. Torch keeps one count per process, so the control thread uses it too
    HWM_GD_PRINT_COEFFS = True          # print the optimized coefficients by name whenever a plan finishes
    HWM_GD_SHOW_PLAN = True             # draw the plan's expected path (noise-free LLD rollout of the optimized coefficients) in blue; costs one extra rollout per plan
    
    
    # ----------- Important settings -----------

    HWM_SIM_SPEED = 1                    # max simulation speed as a multiple of real time (1.0: one control step per TIMESTEP_CONTROL wall seconds, 0.5: half speed). Sets MAX_SIM_FREQUENCY = this / TIMESTEP_CONTROL. None -> MAX_SIM_FREQUENCY untouched
    HWM_DISPLAY_PATH_STEPS = 25          # control steps of every drawn path: the preview's sim plan and LLD replay, and the blue GD expected path
    HWM_LLD_PREVIEW_EVERY = 1            # recompute the preview every N control steps
    HWM_LLD_ATTENTION_TEMPERATURE = 1 / 5     # softmax/kernel temperature for LLD memory distances. Below 1 makes attention sharper; must be > 0
    HWM_LLD_T_MIN_DOF = 10             # floor of nu. None: 2, the smallest nu with a finite covariance. Must be > 0
    HWM_LLD_ALONE_LIKELIHOOD = True    # with HWM_LLD_CORRECTION_ALONE. False: g(q) is trained by the squared error against the recorded delta. True: by the log-likelihood of the delta with g(q) alone as the mean (Student-t / Gaussian as the LLD), the scale and df being those of the LLD prediction, detached. Bounded gradient for large errors, channels weighted by the covariance
    HWM_LLD_CORRECTION_ALONE = True    # residual mode only. True: g is trained so g(q) equals the recorded delta, with no gradient from the retrieved samples. The retrieval encoder is trained on the full residual prediction g(q) + sum_j w_j (delta_j - g(k_j)) with g held fixed, so it learns to help g. At inference that sum is used, and delta - g(key) is evaluated with the current g, not stored. False: g and the memory residuals are trained together
    HWM_LLD_REFLECT = False              # TEST. Works with or without HWM_LLD_DIRECT_DELTA; it makes LLD and actor superstates pose-free by itself (no x, y, heading, sin, cos). Assume dynamics are symmetric about the car's x-axis: every LLD superstate row is mirrored on its own so its steering angle is not negative, the value tokens and training target are mirrored by the steering sign of the row they leave, and the predicted delta (mean, covariance, noise) is mirrored back. Actor inputs stay in the true frame. Needs a new model: the memory embeddings and LLD weights change meaning
    HWM_LLD_T_GRAD_THROUGH_N = False     # False: n (hence nu) is detached, so the encoder is trained only through the attention weights of the mean and covariance. True: gradients also flow through n into the attention
    HWM_LLD_STANDALONE_ENCODER = True   # True: a single encoder (overrides HWM_LLD_SEPARATE_ENCODERS and HWM_LLD_COV_HEAD) is trained only on its own likelihood: the attention-weighted mean and covariance of the stored deltas (no g, no residuals, own dummy value), Student-t / Gaussian as set, on the recorded delta. The rest of the model (residual tokens, covariance, g's training, inference) uses that encoder and its dummy distance detached. n_soft and keys50 of this encoder are the ones logged. Needs a new model: one extra dummy value
    HWM_LLD_COV_CENTER_TRANSITIONS = False  # with HWM_LLD_COV_HEAD. True (a): the covariance is the spread of the stored transitions delta_j about the final prediction g(q) + mu, so it also contains how much g differs between the neighbor and the query, g(k_j) - g(q). False (b): the spread of the residual tokens delta_j - g(k_j) (g(k_j) detached) about their weighted mean mu, the error scatter of g at the neighbors only. Residual mode; without it both are the same
    HWM_LLD_COV_DETACH_TOKENS = True     # residual mode only. True: the residual tokens delta_j - g(k_j) are constants everywhere (mean and covariance), so g is trained only through g(q) of the current state and the retrieval encoders only through the attention weights. Without HWM_LLD_COV_HEAD the covariance (of the retrieved tokens around their weighted mean) is not trained at all. False: g also gets gradient through the tokens, and the covariance is trained
    HWM_LLD_COV_HEAD = True             # the covariance uses its own attention weights, from its own encoder and dummy distance, trained on the same likelihood. The residual tokens are centered on the main head's detached weighted mean (every neighbor's transition is compared with g(k_j) plus that correction). The main head retrieves what corrects the mean, the covariance head what shows the error spread. Needs a new model: extra weights
    HWM_LLD_WHOLE_MEMORY_ATTENTION = True  # True -> pass every valid memory window to the LLD's differentiable attention instead of hard top-50 retrieval. Keep HWM_MEMORY_MAX_ROWS around 1000: training expands this bank over the whole batch
    HWM_LLD_DIRECT_DELTA = True         # TEST. False restores the previous LLD. True: the value token is the delta of every state channel, and the next state is the current state plus that delta (no integration of velocity into position). LLD superstates drop pose_x, pose_y, heading, and sin/cos of heading. Needs a new model: the LLD and the actor noise input change width
    HWM_LLD_PREDICT_DISPLACEMENT = False  # the delta also predicts the step's displacement in the car frame (forward, then left), instead of integrating position from velocity. Rotated back to the world by the heading at the start of the step. Lateral is odd under HWM_LLD_REFLECT. Needs a new model: two more delta channels, so the LLD and the actor noise input change width

    HWM_MEMORY_OVERLAP_TRAIN = True  # learner memory: True -> every row that ends a full window (S rows plus the step it predicts, one episode) is a key, so keys overlap. False -> keys are tiled per episode, one every S+1 rows, and never share a row. Whole-memory attention and the density use this key set; top-50 retrieval also stops excluding neighbours within S+1 rows of an earlier pick
    HWM_MEMORY_OVERLAP_INFERENCE = True  # same, for the planner's memory (driving, test track, previews). True gives a repeated lap a twin key at every step instead of every S+1th

    HWM_MEMORY_BANK = True           # TEST. True: the memory never evicts rows. The LLD's keys (superstates) are a separate array of HWM_MEMORY_BANK_SIZE indices into the rows; every row that completes a window (S rows plus the step it predicts) is offered, and when the bank is full it replaces the key with the highest n_soft, but only if its own n_soft is lower, so dense regions are thinned first. Embeddings and all n_soft are recomputed when the encoder changes. Ignores HWM_MEMORY_OVERLAP_*. Learner and planner each keep their own bank
    HWM_MEMORY_MAX_ROWS = 3_000       # ring-buffer capacity of the memory; new rows overwrite the oldest. With HWM_MEMORY_BANK it is only the initial allocation: rows are never evicted

    HWM_MEMORY_BANK_SIZE = 1000       # number of superstates (keys) kept with HWM_MEMORY_BANK. Whole-memory attention expands this bank over the whole LLD batch
    HWM_MEMORY_BANK_EXCLUDE_OVERLAP = True  # n_soft of a key ignores keys that share a row with it (|Δ row| <= S), which are near copies of it
    # Rectangular arena, 40 m long and 16 m wide: walls only on the border, inside free.
    # The car starts at the left end, centered, heading along the length, from rest.
    # It keeps accelerating, and the steering is (slope * t) * sin(2π t / 2 s), clipped at ±0.6 rad.
    TEST_TRACK = True                # True: drive this arena instead of MAP_NAME, with the open-loop steer and accel below. The LLD is not trained and its weights stay as they were at the start
    TEST_TRACK_ACCEL = 1.5             # constant acceleration [m/s^2], held from the start
    TEST_TRACK_STEER_SLOPE = 0.2      # [rad/s]. Steering amplitude is this times the time since the episode start
    TEST_TRACK_STORE_MEMORY = True    # True: the drive is still written into memory and streamed. The LLD weights are not updated either way


    #--------------- end of important settings ---------------

    HWM_GD_RESAMPLE_NEIGHBORS = True    # LLD re-retrieves its memory windows after every gradient step; False -> once, from the initial rollout, one window set per step
    HWM_GD_WARM_START = True            # True: each plan starts from the previous plan's coefficients (zeros at the start of an episode) and gradient descent updates those. False: each plan starts from the actor's coefficients. The actor is not trained while this is True
    HWM_GD_CRASH_SMOOTHING_M = 1.0      # sigmoid width (m) of the smooth wall test the gradient sees; the hard crash test has no gradient. Smaller is sharper and a weaker gradient
    HWM_GD_WALL_WEIGHT = 10             # multiplies the plan's average wall punishment (meters, zero or negative). 1 = one meter of penetration counts like one meter of distance to the target
    HWM_GD_COEFF_REG = 0.02             # subtracted from the reward: this times the sum of squared distance from the coefficients this plan started at
    HWM_GD_INFERENCE_SAMPLES = 4        # LLD trajectories averaged per inference plan, one of them the mean (no noise)
    HWM_GD_TRAIN_SAMPLES = 8            # same, while training the actor on the optimized coefficients
    HWM_GD_NOISE_CLIP = 2.0             # LLD sample noise ε is clamped to ±this many standard deviations, and held fixed for every gradient step of a plan
    HWM_STEERING_MULTIPLIER = 1.6       # car's steering_multiplier: wheel angle = network steering command * HWM_ACTION_DENORM[0] * this. 0.375 * 1.6 = 0.6 = s_max
    # Gain on each order-1 feature. Steering then acceleration, same order as search/coefficients.py.
    # Steering gains shift the desired wheel angle relative to the measured wheel; the simulator's servo enforces its physical rate limit.
    HWM_COEFF_DENORM = [
        1.0, 3.0, 2.0, 1.0, 1.5, 1.0, 1.5, 2.0, 1.0,  # steer: bias, lateral, heading, yaw rate, wheel, lat vel, wall side, wall push, clearance
        1.0, 2.0, 1.0, 1.0, 2.0, 1.0,                 # accel: bias, speed error, speed, turn, urgency, clearance
    ]
    HWM_LEARNING_RATE = 3e-4             # Adam learning rate for each HWM module optimizer
    # Divisors for raw state channels, STATE_VARIABLES order. Observation uses state / scale.
    HWM_STATE_SCALE = [
        10.0 / 3.0,  # angular_vel_z
        10.0,        # linear_vel_x
        1.0,         # linear_vel_y
        math.pi,     # pose_theta
        1.0,         # pose_theta_cos
        1.0,         # pose_theta_sin
        10.0,        # pose_x
        10.0,        # pose_y
        1.0,         # slip_angle
        0.4,         # steering_angle
    ]
    # Divisors for consecutive state differences after HWM_STATE_SCALE, same channel order.
    # Chosen so a strong normal step (about the 95th percentile of |Δ| on a Pure Pursuit
    # rollout, dt = TIMESTEP_CONTROL) is near 1. pose_x and pose_y share one divisor.
    HWM_STATE_DIFF_SCALE = [
        0.1,   # angular_vel_z
        0.02,  # linear_vel_x
        0.05,  # linear_vel_y
        0.05,  # pose_theta
        0.1,   # pose_theta_cos
        0.1,   # pose_theta_sin
        0.04,  # pose_x
        0.04,  # pose_y
        0.02,  # slip_angle
        0.2,   # steering_angle
    ]
    HWM_STATE_DIM = 10                   # car-state width stored in memory and predicted by the dynamics
    HWM_ACTION_DIM = 2                   # [angular, translational] action width
    HWM_HIDDEN_DIM = 128                 # MLP width for actor, value, and the placeholder dynamics
    HWM_ACTION_DENORM = [0.375, 5.0]     # network output ±1 -> physical [steering rad, translational]; 0.375 = s_max 0.6 / steering_multiplier 1.6 of the car, so ±1 is full lock
    HWM_CLIENT_MODULES = ["actor", "LLD"]       # modules the learner broadcasts to the planner after each train round
    

    
    ## Speed cap ##
    ## Curriculum speed cap: GLOBAL_SPEED_LIMIT increases with difficulty (clips car state in base_classes)
    GLOBAL_SPEED_LIMIT_CURRICULUM_ENABLED = False
    GLOBAL_SPEED_LIMIT_MIN = 3.0   # at low difficulty [m/s]
    GLOBAL_SPEED_LIMIT_MAX = 15.0  # at high difficulty [m/s]

    ## Friction ##
    SURFACE_FRICTION = None # Surface friction coefficient

    ## Per-episode domain randomization (see utilities/episode_randomization.py) ##
    EPISODE_RANDOMIZATION_FILE = None # YAML spec; sampled values applied on each episode reset
    # EPISODE_RANDOMIZATION_FILE = "utilities/episode_randomization.yaml"  # YAML spec; sampled values applied on each episode reset
    EPISODE_RANDOMIZATION_VERBOSE = True  # Print sampled values at each episode reset

    # Deprecated
    AVERAGE_WINDOW = 200  # Window for avg filter [friction]
    WALL_CLOCK_LAPTIMES = True

    @classmethod
    def recalculate_paths(cls) -> None:
        """Recompute path dependent settings after attribute overrides."""
        cls.MAP_PATH = os.path.join("utilities", "maps", cls.MAP_NAME)
        cls.MAP_CONFIG_FILE = os.path.join(cls.MAP_PATH, cls.MAP_NAME + ".yaml")
        cls.RECORDING_PATH = os.path.join(cls.RECORDING_FOLDER, cls.RECORDING_NAME)


    def save_snapshot(self, path: str) -> None:
        """
        Save the current settings into a YAML file located at <path>/settings_snapshot.yml.
        """

        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, "settings_snapshot.yml")
        with open(out_path, "w") as f:
            for attr, value in Settings.__dict__.items():
                if not attr.startswith("__") and not callable(getattr(Settings, attr)):
                    f.write(f"{attr}: {value}\n")
        print(f"Settings snapshot saved to {out_path}")


from utilities.test_track import prepare as _prepare_test_track

_prepare_test_track()



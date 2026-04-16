import os
class Settings():

    ## Environment ##
    ENVIRONMENT_NAME = 'Car'  # Car or Quadruped
    ENV_CAR_PARAMETER_FILE = "gym_car_parameters.yml" # Car parameters for simulated car
    SIM_ODE_IMPLEMENTATION = "ODE_TF"  # Use the implementation  'jax_pacejka' or 'jit_Pacejka': For fast simulation / 'ODE_TF': For SI_Toolkit batch model thats also used in mpc
    
    ## Map ##
    MAP_NAME = "RCA1"  # hangar3, hangar9, hangar12, hangar14, hangar16, london3_small, london3_large, ETF1, ini10, icra2022, RCA1, RCA2, IPZ2
    MAP_PATH = os.path.join("utilities", "maps", MAP_NAME)
    MAP_CONFIG_FILE = os.path.join(MAP_PATH, MAP_NAME+".yaml")
    
    BLANK_MAP = False  # If True, skip setting map for all sensors (no borders, no scans, no crashes possible)

    # Controller Settings
    CONTROLLER = 'sac_agent' # Options: 'manual','mpc','ftg',neural,'pp','stanley', 'mppi-lite', 'mppi-lite-jax', 'rpgd-lite-jax', 'example'
    MOTOR_PID_IN_CAR_MODEL = False  # If True: control[1] is desired speed and PI is used. If False: control[1] is direct acceleration.

    TIMESTEP_CONTROL = 0.04    # Multiple of 0.01; how often to recalculate control input
    TIMESTEP_SIM = 0.01       # Dont touch.
    MAX_SIM_FREQUENCY = 250   # Max simulation frequency in Hz (e.g. 250). None = no limit. If step is faster, waits so it takes exactly 1/freq.
    ACCELERATION_TIME = 20                   #nni 50, mpc 10 (necessary to overcome initial velocity of 0 m/s)
    ACCELERATION_AMPLITUDE = 10           #nni 2, mpc 10 [Float!]

    # Zero Angle offset
    ZERO_ANGLE_OFFSET = 0.00  # Angle offset for the car (left drift is positive, right drift is negative) absolut max steeringangle = 0.4186
    
    ## driving behaviour ## 
    START_FROM_RANDOM_POSITION = True # Start from random position (randomly selected waypoint + delta)
    STARTING_POSITION = [[3.62, 6.26, 0.378]] # Starting position [x, y, yaw] in case of START_FROM_RANDOM_POSITION = False
    
    REVERSE_DIRECTION = False # Drive reverse waypoints
    GLOBAL_WAYPOINT_VEL_FACTOR = 1.0
    RANDOM_WAYPOINT_VEL_FACTOR = False

    
    GLOBAL_SPEED_LIMIT = 15.0
    APPLY_SPEED_SCALING_FROM_CSV = False # Speed scaling from speed_scaling.yaml are multiplied with GLOBAL_WAYPOINT_VEL_FACTOR

    ## Recordings ##
    REPLAY_RECORDING = False

    SAVE_RECORDINGS = False
    SAVE_PLOTS = True # Only possible when SAVE_RECORDINGS is True
    SAVE_REWARDS = True
    SAVE_VIDEOS = False
    
    RECORDING_INDEX = 0
    RECORDING_NAME = 'F1TENTH_ETF1_NNI__2023-11-23_15-54-27.csv'
    RECORDING_FOLDER = './ExperimentRecordings/'
    RECORDING_PATH = os.path.join(RECORDING_FOLDER, RECORDING_NAME)
    DATASET_NAME = "Recording1"
    RECORDING_MODE = 'online'  # 'online' or 'offline', also 'disable' - partly redundant with SAVE_RECORDINGS
    TIME_LIMITED_RECORDING_LENGTH = None  # FIXME: Not yet working in F1T

    CONNECT_RACETUNER_TO_MAIN_CAR = False

    # Oponents
    NUMBER_OF_OPPONENTS = 0
    OPPONENTS_CONTROLLER = 'pp'
    OPPONENTS_VEL_FACTOR = 0.3
    OPPONENTS_GET_WAYPOINTS_FROM_MPC = False
    OPPONENTS_SIMULATE_LIDAR = False  # If False, only ego runs lidar; opponents get max-range placeholder scans.
    
    # Head2Head Settings
    STOP_IF_OBSTACLE_IN_FRONT = False # Stop if obstacle is immediately in front of the car
    SLOW_DOWN_IF_OBSTACLE_ON_RACELINE = False # Slow down if obstacle is close to the next waypoints
    ALLOW_ALTERNATIVE_RACELINE = False # TODO: check and automatically generate file
    
    # Random Obstacles
    PLACE_RANDOM_OBSTACLES = False  # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    DELETE_MAP_WITH_OBSTACLES_IF_CRASHED = False
    
    MAX_CRASH_REPETITIONS = 10000000
    
    TRUNCATE_ON_LEAVE_TRACK = True
    RESET_ON_DONE = True  # Reset the environment when done
    RESPAWN_ON_RESET = False  # If True, respawn to state N timesteps ago instead of complete reset
    RESPAWN_SETBACK_TIMESTEPS = 125  # Number of timesteps to go back when respawning
    RESPAWN_PROBABILITY = 0.5 #% chance (0 to 1) to respawn on crash

    # Experiment Settings
    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENT_MAX_LENGTH = 8000  # In sim timesteps: Length until the simulation is reset
    SIMULATION_LENGTH = 2000 # In sim timesteps: Length until the simulation is terminated
    MAX_EPISODE_LENGTH = 2000 


    ## Noise ##
    CONTROL_DELAY = 0.08 # Delay between control calculated and control applied to the car, multiple of 0.01 [s]
    # Delay on physical car is about 0.06s (Baseline right now is 0.1s)
    
    # NOISE_LEVEL_CAR_STATE = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    NOISE_LEVEL_CAR_STATE = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    NOISE_LEVEL_CONTROL = [0.0, 0.0] # noise level [angular, translational]
    # NOISE_EVEL_CONTROL = [0.05, 0.1] # noise level [angular, translational]
    # NOISE_LEVEL_CONTROL = [0.1, 0.7] # noise level [angular, translational]

    
    FACTOR_APPLIED_TRANSLATIONAL_CONTROL = 1.0
    CONTROL_NOISE_DURATION = 10 # Number of timesteps for which the control noise is applied

    ## waypoints generation ##
    MIN_CURV_SAFETY_WIDTH = 1.0             # Safety width [m] incliding car width for the Waypoint generation /utilities/run_create_min_curve_waypoints.py  
    LOOK_AHEAD_STEPS = 30                    # Number of original waypoints that are considered for cost
    INTERPOLATION_STEPS = 1                  # >= 1 Interpolation steps to increase waypoint resolution
    DECREASE_RESOLUTION_FACTOR = 4           # >= 1 Only take every n^th waypoint to decrease resolution
    IGNORE_STEPS = 1                         # Number of interpolated waypoints to ignore starting at the closest one
    INTERPOLATE_LOCA_WP = 1
    GLOBAL_WAYPOINTS_SEARCH_THRESHOLD = 10.0  # If there is a waypoint in cache with a distance to the car position smaller than this, only cache is searched for nearest waypoints, set None to always use global search
    

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
    PP_WAYPOINT_VELOCITY_FACTOR = 0.6
    PP_LOOKAHEAD_DISTANCE = 1.62461887897713965  # lookahead distance [m], Seems not used
    PP_VEL2LOOKAHEAD = 0.6  # None for fixed lookahead distance (PP_LOOKAHEAD_DISTANCE)
    PP_FIXPOINT_FOR_CURVATURE_FACTOR = (0.2, 0.3)  # Second number big - big shortening of the lookahead distance, you can change from 0.2+ (no hyperbolic effect) to 1.0 (lookahead minimal already at minimal curvature)
    PP_NORMING_V_FOR_CURRVATURE = 10.0  # Bigger number - higher velocity required to have effect on shortening of lookahead horizon
    PP_BACKUP_LOOKAHEAD_POINT_INDEX = 1  # Backup should be obsolete after new change
    PP_MINIMAL_LOOKAHEAD_DISTANCE = 0.5

    RELOAD_WP_IN_BACKGROUND = False  # If True, waypoints are reloaded in a separate thread

    
    ## MPC Controller ##
    CONTROLLER_CAR_PARAMETER_FILE = "gym_car_parameters.yml"  # Car parameters for future state estimation (might derrive from the GYM_CAR_PARAMETER_FILE) for simulationg "wrong" model
    ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:ks_pacejka'  # Its the model that the predictor uses. Only used for mpc predictions, if ODE predictor chosen
    
    NUM_TRAJECTORIES_TO_PLOT = 20
    OPTIMIZE_EVERY_N_STEPS = 1
    
    ANALYZE_COST = False # Analyze and plot diufferent parts of the MPC cost
    ANALYZE_COST_PERIOD = 100 # Period for analyzing the cost
    

    WAYPOINTS_FROM_MPC = False # Use waypoints generated from MPC instead of the map
    PLAN_EVERY_N_STEPS = 4 # in case of waypoints from MPC, plan the waypoints every Nth step
    
    
    ## Visualization ##
    KEYBOARD_INPUT_ENABLE = False  # Allows for keyboard input during experiment. Causes silent crash on some computers
    # RENDER_MODE = 'human' # slow rendering ('human') and fast rendering ('human_fast') an no rendering (None)
    # RENDER_MODE = 'human_fast' # slow rendering ('human') and fast rendering ('human_fast') an no rendering (None)
    RENDER_MODE = None # slow rendering ('human') and fast rendering ('human_fast') an no rendering (None)

    CAMERA_AUTO_FOLLOW = True  # Automatically follow the first car on the map
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
    ROS_BRIDGE = False # Automatically determined on program start
    GLOBALLY_DISABLE_COMPILATION = False # Disable TF Compilation
    DISABLE_GPU = False #True # Disable GPU usage for TF

    ## SAC Agent planner
    SAC_INFERENCE_MODEL_NAME = None  # Model name to be used for inference. If None, the agent will be in training mode
    
    #SAC Sampling Weights
    USE_CUSTOM_SAC_SAMPLING = True

    SAC_CUSTOM_UNIFORM_CRITIC = True #if true forces uniform critic sampling anyways

    SAC_CRITIC_PURE_TD = False

    #True: weight by inverse of TD, False: weight by TD error
    SAC_CUSTOM_CRITIC_INVERT_TD = False 
    SAC_CUSTOM_ACTOR_INVERT_TD = True

    SAC_LOG_SQUISH = True

    SAC_CUSTOM_SAMPLING_REPLACE = True #True -> means same sample can be drawn multiple times, this is default

    SAC_WP_OFFSET_WEIGHT = 3.0
    SAC_WP_HEADING_ERROR_WEIGHT = 5.0
    SAC_REWARD_WEIGHT = 3.0
    SAC_VELOCITY_WEIGHT = 0.0

    SAC_PRIORITY_FACTOR = 0.8   #(alpha) 0: full uniform, 1: full priority -> p = SAC_PRIORITY_FACTOR * w_vec + (1.0 - SAC_PRIORITY_FACTOR) * uniform_p
    SAC_IMPORANCE_SAMPLING_CORRECTOR = 0.6 #(beta), corrects the introduced bias from prioritized sampling
    
    SAC_BETA_ANNEALING_RATIO = 0.4 #at how much % of total agent timesteps should beta have grown to 1.0
    SAC_STATE_TO_TD_RATIO = 0.8 #if 0, only TD error based priorities

    SAC_DYNAMIC_IS_CORRECTOR = True
    SAC_USE_IS_WEIGHTS_FOR_ACTOR = False #seems to be pretty bad if i turn this on

    SAC_N_STEP = 1 #lookahead steps for reward calculations

    # Debug logging for SAC training internals
    SAC_DEBUG_LOGGING = False
    SAC_CLIP_WEIGHTS = False

    EXTENDED_AUTO_STOP = True

    SAC_RANK_BASED_SAMPLING = False

    SAC_CURRICULUM_DEBUG = False
    SAC_AGENT_DEBUG = True
    LEARNER_SERVER_DEBUG = True

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
    # UDT = learner total_weight_updates / total_actor_timesteps. When set, SAC agent adjusts
    # MAX_SIM_FREQUENCY after each training_info update (see learner_server + sac_agent_planner).
    SAC_TARGET_UDT = 1
    SAC_MAX_UTD = 2 
    SAC_UDT_DEADBAND_RATIO = 0.1
    SAC_UDT_FREQ_ADJUST_STEP_RATIO = 0.05
    SAC_MIN_SIM_FREQUENCY = 20.0

    # Saves full obs and action for each transition, so that for analysis, models can be called on all transitions explored during training directly
    SAC_STAT_TRACKER = True
    SAC_STAT_TRACKER_FULL_OBS_ACTION_SAVE = True 
    SAC_STAT_TRACKER_CSV_FLOAT_DECIMALS = 4
    
    ## start to t1 -> starting difficulty | t1 to t2 -> linear increase to 1.0 | t2 to end -> 1.0
    SAC_CURRICULUM_STARTING_DIFFICULTY = 0.2
    SAC_CURRICULUM_T1 = 0.05        # in % of total learning progress
    SAC_CURRICULUM_T2 = 0.6
    SAC_CURRICULUM_MAX_DIFFICULTY = 1.0

    SAC_CURRICULUM_SPEED = False
    SAC_CURRICULUM_SPEED_ADJUST_MODE = 'speed_cap' #'speed cap' or 'vel_factor'

    SAC_ACCEL_CAP_MAX = 3.0 #3.0 is the max, and this can be scaled down based on difficulty
    SAC_ACCEL_CAP = 3.0

    SAC_CURRICULUM_SPEED_LIMIT_MAX = 15 #absolute max speed limit during curriculum learning
    SAC_CURRICULUM_SPEED_LIMIT = 15

    SAC_CURRICULUM_TRACK_WIDTH_SCALING = False
    SAC_CURRICULUM_TRACK_WIDTH_FACTOR = 1.0

    SAC_CURRICULUM_NOISE_SCALING = False
    SAC_NOISE_LEVEL_CAR_STATE_MAX = [0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    # SAC_NOISE_LEVEL_CAR_STATE_MAX = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    SAC_NOISE_LEVEL_CONTROL_MAX = [0.35, 0.7] # noise level [angular, translational]
    # NOISE_LEVEL_CONTROL = [0.0, 0.0] # noise level [angular, translational]

    SAC_AMPLIFY_NEGATIVE_PROGRESS_REWARD = False
    SAC_NEGATIVE_PROGRESS_REWARD_AMPLIFICATION_FACTOR = 20

    SAC_PREFILL_BUFFER_WITH_PP = False
    SAC_PREFILL_BUFFER_WITH_PP_AMOUNT = 30000 #number of transitions to prefill
    SAC_PREFILL_BEHAVIOR_CLONING_EPOCHS = 20


    
    ## Speed cap ##
    ## Curriculum speed cap: GLOBAL_SPEED_LIMIT increases with difficulty (clips car state in base_classes)
    GLOBAL_SPEED_LIMIT_CURRICULUM_ENABLED = False
    GLOBAL_SPEED_LIMIT_MIN = 3.0   # at low difficulty [m/s]
    GLOBAL_SPEED_LIMIT_MAX = 15.0  # at high difficulty [m/s]

    ## Friction ##
    SURFACE_FRICTION = None # Surface friction coefficient
    
    # Deprecated
    AVERAGE_WINDOW = 200  # Window for avg filter [friction]


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



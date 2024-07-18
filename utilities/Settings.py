import os
class Settings():

    ## Environment ##
    ENVIRONMENT_NAME = 'Car'  # Car or Quadruped
    ENV_CAR_PARAMETER_FILE = "utilities/car_files/gym_car_parameters.yml" # Car parameters for simulated car
    SIM_ODE_IMPLEMENTATION = "pacejka"  # Use the implementation 'pacejka' or implement a new one

    ## Map ##
    MAP_NAME = "RCA2"  # hangar3, hangar9, hangar12, hangar14, hangar16, london3_small, london3_large, ETF1, ini10, icra2022, RCA1, RCA2
    MAP_PATH = os.path.join("utilities", "maps", MAP_NAME)
    MAP_CONFIG_FILE = os.path.join(MAP_PATH, MAP_NAME+".yaml")
    SURFACE_FRICITON = 0.7  # Surface friction coefficient

    # Controller Settings
    CONTROLLER = 'mpc'  # Options: 'manual' (requires connected joystick) ,'mpc', 'ftg' (follow the gap), neural (neural network),  'pp' (pure pursuit), 'stanley' (stanley controller)

    TIMESTEP_CONTROL = 0.02    # Multiple of 0.01; how often to recalculate control input
    TIMESTEP_PLANNER = 0.02      # For model based planner (MPC) timestep of simulation, can be arbitrary number

    ACCELERATION_TIME = 5                   #nni 50, mpc 10 (necessary to overcome initial velocity of 0 m/s)
    ACCELERATION_AMPLITUDE = 10           #nni 2, mpc 10 [Float!]

    
    ## driving behaviour ## 
    START_FROM_RANDOM_POSITION = True # Start from random position (randomly selected waypoint + delta)
    STARTING_POSITION = [[3.62, 6.26, 0.378]] # Starting position [x, y, yaw] in case of START_FROM_RANDOM_POSITION = False
    
    REVERSE_DIRECTION = False # Drive reverse waypoints
    GLOBAL_WAYPOINT_VEL_FACTOR = 0.8
    APPLY_SPEED_SCALING_FROM_YAML = False # Speed scaling from speed_scaling.yaml are multiplied with GLOBAL_WAYPOINT_VEL_FACTOR
    
    
    ## Recordings ##
    REPLAY_RECORDING = False
    
    SAVE_RECORDINGS = True
    SAVE_REVORDING_EVERY_NTH_STEP = None # Save recording file also during the simulation (slow down, every Nth step, None for no saving during sim)
    SAVE_PLOTS = False # Only possible when SAVE_RECORDINGS is True
    
    RECORDING_NAME = 'F1TENTH_ETF1_NNI__2023-11-23_15-54-27.csv'
    RECORDING_FOLDER = './ExperimentRecordings/'
    RECORDING_PATH = os.path.join(RECORDING_FOLDER, RECORDING_NAME)
    DATASET_NAME = "Recording1"


    # Oponents
    NUMBER_OF_OPPONENTS = 0
    OPPONENTS_CONTROLLER = 'pp'
    OPPONENTS_VEL_FACTOR = 0.2
    OPPONENTS_GET_WAYPOINTS_FROM_MPC = False
    
    # Random Obstacles
    PLACE_RANDOM_OBSTACLES = False  # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    DELETE_MAP_WITH_OBSTACLES_IF_CRASHED = False
    CRASH_DETECTION = True


    # Experiment Settings
    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENTS_IN_SEPARATE_PROGRAMS = False
    EXPERIMENT_LENGTH = 100  # in timesteps, only valid if DISABLE_AUTOMATIC_TIMEOUT is True.
    STOP_TIMER_AFTER_N_LAPS = 2                 # Timer stops after N laps for competition 
    DISABLE_AUTOMATIC_TERMINATION = False
    DISABLE_AUTOMATIC_TIMEOUT = True


    ## Noise ##
    CONTROL_DELAY = 0.00 # Delay between control calculated and control applied to the car, multiple of 0.01 [s]
    # Delay on physical car is about 0.06s (Baseline right now is 0.1s)
    
    NOISE_LEVEL_CAR_STATE = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    NOISE_LEVEL_CONTROL = [0.0, 0.0] # noise level [angular, translational]
    
    FACTOR_APPLIED_TRANSLATIONAL_CONTROL = 1.0
    CONTROL_NOISE_DURATION = 10 # Number of timesteps for which the control noise is applied

    CONTROL_AVERAGE_WINDOW = (1,1)     # Window for avg filter [angular, translational]


    ## waypoints generation ##
    MIN_CURV_SAFETY_WIDTH = 1.0              # Safety width [m] incliding car width for the Waypoint generation /utilities/run_create_min_curve_waypoints.py  
    LOOK_AHEAD_STEPS = 20                    # Number of original waypoints that are considered for cost
    INTERPOLATION_STEPS = 1                  # >= 1 Interpolation steps to increase waypoint resolution
    DECREASE_RESOLUTION_FACTOR = 4           # >= 1 Only take every n^th waypoint to decrease resolution
    IGNORE_STEPS = 1                         # Number of interpolated waypoints to ignore starting at the closest one
    INTERPOLATE_LOCA_WP = 1
    
    AUTOMATIC_SECTOR_TUNING = False
    

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
    PP_WAYPOINT_VELOCITY_FACTOR = 1.0
    PP_LOOKAHEAD_DISTANCE = 1.62461887897713965  # lookahead distance [m], Seems not used
    PP_VEL2LOOKAHEAD = None  # None for fixed lookahead distance (PP_LOOKAHEAD_DISTANCE)
    PP_FIXPOINT_FOR_CURVATURE_FACTOR = (0.2, 0.3)  # Second number big - big shortening of the lookahead distance, you can change from 0.2+ (no hyperbolic effect) to 1.0 (lookahead minimal already at minimal curvature)
    PP_NORMING_V_FOR_CURRVATURE = 10.0  # Bigger number - higher velocity required to have effect on shortening of lookahead horizon
    PP_BACKUP_LOOKAHEAD_POINT_INDEX = 1  # Backup should be obsolete after new change
    PP_MINIMAL_LOOKAHEAD_DISTANCE = 0.1

    ## MPC Controller ##
    CONTROLLER_CAR_PARAMETER_FILE = "utilities/car_files/gym_car_parameters.yml"  # Car parameters for future state estimation (might derrive from the GYM_CAR_PARAMETER_FILE) for simulationg "wrong" model
    ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:pacejka'  # Its the model that the predictor uses. Only used for mpc predictions, if ODE predictor chosen
    
    NUM_TRAJECTORIES_TO_PLOT = 20
    OPTIMIZE_EVERY_N_STEPS = 1
    
    ANALYZE_COST = False # Analyze and plot diufferent parts of the MPC cost
    ANALYZE_COST_PERIOD = 100 # Period for analyzing the cost
    
    EXECUTE_NTH_STEP_OF_CONTROL_SEQUENCE = 0 # Make sure you match with Control delay: Nth step = contol delay / timestep control

    WAYPOINTS_FROM_MPC = False # Use waypoints generated from MPC instead of the map
    PLAN_EVERY_N_STEPS = 4 # in case of waypoints from MPC, plan the waypoints every Nth step
    
    
    ## Visualization ##
    KEYBOARD_INPUT_ENABLE = False  # Allows for keyboard input during experiment. Causes silent crash on some computers
    RENDER_MODE = 'human_fast' # slow rendering ('human') and fast rendering ('human_fast') an no rendering (None)
    # RENDER_MODE = None # slow rendering ('human') and fast rendering ('human_fast') an no rendering (None)
    CAMERA_AUTO_FOLLOW = True  # Automatically follow the first car on the map
    PRINTING_ON = False
    
    
    ### Other Settings ###
    ROS_BRIDGE = None # Automatically determined on program start
    GLOBALLY_DISABLE_COMPILATION = False # Disable TF Compilation
    DISABLE_GPU = True # Disable GPU usage for TF



    # if os.getenv('CI_TEST', 'false').lower() == 'true':
    #     RENDER_MODE = None
    #     CONTROLLER = 'pp'
    #     START_FROM_RANDOM_POSITION = False

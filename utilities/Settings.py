import os
class Settings():

    ### Environment ###

    ENVIRONMENT_NAME = 'Car'  # Car or Quadruped

    FROM_RECORDING = False
    RECORDING_NAME = 'shit_behaviour.csv'
    RECORDING_FOLDER = './'
    RECORDING_PATH = os.path.join(RECORDING_FOLDER, RECORDING_NAME)

    MAP_NAME = "london3_small"  # hangar3, hangar9, hangar11, hangar12, icra2022, ini1, Oschersleben
    MAP_PATH = os.path.join("utilities", "maps", MAP_NAME)
    MAP_CONFIG_FILE = os.path.join(MAP_PATH, "config_map_gym.yaml")
    REVERSE_DIRECTION = False

    ENV_CAR_PARAMETER_FILE = "utilities/car_files/gym_car_parameters.yml" # Car parameters for simulated car

    NUMBER_OF_OPPONENTS = 0
    OPPONENTS_CONTROLLER = 'pp'
    OPPONENTS_VEL_FACTOR = 0.2
    OPPONENTS_GET_WAYPOINTS_FROM_MPC = False

    DISABLE_AUTOMATIC_TERMINATION = True
    DISABLE_AUTOMATIC_TIMEOUT = True
    PLACE_RANDOM_OBSTACLES = False  # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    DELETE_MAP_WITH_OBSTACLES_IF_CRASHED = False

    # Decide if to use PID as in the original F1TENTH implementation [angle, speed] Or bypass it [angular_vel, acceleration]
    WITH_PID = True # Warning: The planner classes that can not handle both (pp, ftg) will overwrite this setting

    KEYBOARD_INPUT_ENABLE = False  # Allows for keyboard input during experiment. Causes silent crash on some computers
    RENDER_MODE = 'human_fast' # slow rendering (human) and fast rendering (human_fast) an no rendering (None)
    # RENDER_MODE = None  # slow rendering (human) and fast rendering (human_fast) an no rendering (None)
    CAMERA_AUTO_FOLLOW = True  # Automatically follow the first car on the map

    DRAW_POSITION_HISTORY = True
    QUAD_VIZ = True  # Visualization, only for Quadruped


    ### Experiment Settings ###
    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENTS_IN_SEPARATE_PROGRAMS = False
    EXPERIMENT_LENGTH = 10000  # in timesteps, only valid if DISABLE_AUTOMATIC_TIMEOUT is True.

    SAVE_RECORDINGS = True
    SAVE_PLOTS = True # Only possible when SAVE_RECORDINGS is True

    ### State Estimation ###

    # Options for ODE_MODEL_OF_CAR_DYNAMICS: 'ODE:simple', 'ODE:ks', 'ODE:st' # TODO: Currently only st discerns correctly between scenario with and without PID
    ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:ks'  # Its the model that the predictor uses. Only used for mpc predictions, if ODE predictor chosen



    ONLY_ODOMETRY_AVAILABLE = False     # Decide if available state consists of full car state or only of odometry

    # Noise Level for the controller's state estimation
    # NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.5 # ftg: 0.5  # mppi: 2.0
    # NOISE_LEVEL_ANGULAR_CONTROL = 0.30  # ftg: 0.05  # mppi: 3.0
    NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.0 # ftg: 0.5  # mppi: 2.0
    NOISE_LEVEL_ANGULAR_CONTROL = 0.0  # ftg: 0.05  # mppi: 3.0
    FACTOR_APPLIED_TRANSLATIONAL_CONTROL = 1.0
    # NOISE_LEVEL_CAR_STATE = [ 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
    NOISE_LEVEL_CAR_STATE = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    # Nikita's Slip/Steer Predictor
    SLIP_STEER_PREDICTION = False
    NET_NAME_SLIP = 'GRU-15IN-64H1-64H2-1OUT-0'
    NET_NAME_STEER = 'GRU-14IN-64H1-64H2-1OUT-0'

    ###################################################################################
    ### Driver Settings
    
    # waypoints:
    LOOK_AHEAD_STEPS = 15                    # Number of original waypoints that are considered for cost
    INTERPOLATION_STEPS = 1                  # >= 1 Interpolation steps to increase waypoint resolution
    DECREASE_RESOLUTION_FACTOR = 4           # >= 1 Only take every n^th waypoint to decrease resolution
    IGNORE_STEPS = 1                         # Number of interpolated waypoints to ignore starting at the closest one
    INTERPOLATE_LOCA_WP = 4


    CONTROL_AVERAGE_WINDOW = (2, 2)     # Window for avg filter [angular, translational]

    ###################################################################################
    ### Controller Settings

    CONTROLLER = 'pp'  # Options: 'manual' (requires connected joystick) ,'mpc', 'ftg' (follow the gap), neural (neural network),  'pp' (pure pursuit), 'stanley' (stanley controller)

    TIMESTEP_CONTROL = 0.04    # Multiple of 0.01; how often to recalculate control input
    TIMESTEP_PLANNER = 0.1      # For model based planner (MPC) timestep of simulation, can be arbitrary number

    ACCELERATION_TIME = 1                   #nni 50, mpc 10 (necessary to overcome initial velocity of 0 m/s)
    ACCELERATION_AMPLITUDE = 10           #nni 2, mpc 10 [Float!]

    FOLLOW_RANDOM_TARGETS = False

    LIDAR_COVERED_ANGLE_DEG = 270
    LIDAR_NUM_SCANS = 1080

    LIDAR_MODE = 'decimation'  # possible: 'decimation', 'custom indices'
    LIDAR_PROCESSED_ANGLE_DEG = 250  # number or 'max'; 170 corresponds to old "LOOK_FORWARD" option
    LIDAR_DECIMATION = 25  # Only taken into account if LIDAR_MODE is 'decimation'

    LIDAR_CORRUPT = False
    LIDAR_MAX_CORRUPTED_RATIO = 0.5

    LIDAR_PLOT_SCANS = False

    ## Pure Pursuit Controller ##
    PP_USE_CURVATURE_CORRECTION = True
    PP_WAYPOINT_VELOCITY_FACTOR = 1.0
    PP_LOOKAHEAD_DISTANCE = 1.82461887897713965  # lookahead distance [m], Seems not used
    PP_VEL2LOOKAHEAD = 0.7
    PP_FIXPOINT_FOR_CURVATURE_FACTOR = (0.8, 0.7)  # Second number big - big shortening of the lookahead distance, you can change from 0.2+ (no hyperbolic effect) to 1.0 (lookahead minimal already at minimal curvature)
    PP_NORMING_V_FOR_CURRVATURE = 10.0  # Bigger number - higher velocity required to have effect on shortening of lookahead horizon
    PP_BACKUP_LOOKAHEAD_POINT_INDEX = 1  # Backup should be obsolete after new change
    PP_MINIMAL_LOOKAHEAD_DISTANCE = 0.1

    ## Neural Controller ##
    #Network to be used for Neural control in nni_planner   -> Path to model can be adapted in nni_planner (controller=neursl)
    PATH_TO_MODELS = './SI_Toolkit_ASF/Experiments/Obstacle_v1/Models/'
    NET_NAME = 'Dense-89IN-64H1-64H2-2OUT-0'

    ## MPC Controller ##
    # Car parameters for future state estimation (might derrive from the GYM_CAR_PARAMETER_FILE) for simulationg "wrong" model
    MPC_CAR_PARAMETER_FILE = "utilities/car_files/ini_car_parameters.yml"
    NUM_TRAJECTORIES_TO_PLOT = 20
    OPTIMIZE_EVERY_N_STEPS = 1
    
    ### Other Settings ###
    GLOBALLY_DISABLE_COMPILATION = False # Disable TF Compilation
    ROS_BRIDGE = None # Automatically determined on program start
    DISABLE_GPU = True

    # Settings for data collection
    GLOBAL_WAYPOINT_VEL_FACTOR = 1.0
    START_FROM_RANDOM_POSITION = False

    WAYPOINTS_FROM_MPC = True
    PLAN_EVERY_N_STEPS = 12

    PRINTING_ON = False


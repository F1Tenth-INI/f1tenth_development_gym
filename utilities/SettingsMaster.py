import os
class SettingsMaster:

    ### Environment ###
    
    ENVIRONMENT_NAME = 'Car'  # Car or Quadruped

    FROM_RECORDING = True
    RECORDING_NAME = 'MPPI_DEBUGGING_SHORT.csv'
    RECORDING_FOLDER = './'
    RECORDING_PATH = os.path.join(RECORDING_FOLDER, RECORDING_NAME)

    # MAP_CONFIG_FILE =  "utilities/maps_files/Oschersleben.yaml"    
    MAP_CONFIG_FILE =  "utilities/maps/hangar9/config_map_gym.yaml"   
     
    ENV_CAR_PARAMETER_FILE = "utilities/car_files/gym_car_parameters.yml" # Car parameters for simulated car  
        
    
    DISABLE_AUTOMATIC_TIMEOUT = True
    PLACE_RANDOM_OBSTACLES = False # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    
    # Decide if to use PID as in the original F1TENTH implementation [angle, speed] Or bypass it [angular_vel, acceleration]
    WITH_PID = True # Warning: The planner classes that can not handle both (pp, ftg) will overwrite this setting
    
    KEYBOARD_INPUT_ENABLE = False  # Allows for keyboard input during experiment. Causes silent crash on some computers
    RENDER_MODE = "human_fast" # slow rendering (human) and fast rendering (human_fast) an no rendering (None)
    CAMERA_AUTO_FOLLOW = False  # Automatically follow the first car on the map
    
    DRAW_POSITION_HISTORY = True
    QUAD_VIZ = True  # Visualization, only for Quadruped

    
    
    
    ### Experiment Settings ###
    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENTS_IN_SEPARATE_PROGRAMS = False
    EXPERIMENT_LENGTH = 1000  # in timesteps, only valid if DISABLE_AUTOMATIC_TIMEOUT is True.
    
    SAVE_RECORDINGS = False
    SAVE_PLOTS = False # Only possible when SAVE_RECORDINGS is True
    
    ### State Estimation ###

    # Options for ODE_MODEL_OF_CAR_DYNAMICS: 'ODE:simple', 'ODE:ks', 'ODE:st' # TODO: Currently only st discerns correctly between scenario with and without PID
    ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:st'  # Its the model that the predictor uses. Only used for mpc predictions, if ODE predictor chosen
    
    # Car parameters for future state estimation (might derrive from the GYM_CAR_PARAMETER_FILE) for simulationg "wrong" model
    MPC_CAR_PARAMETER_FILE = "utilities/car_files/ini_car_parameters.yml" # Car parameters for MPC model prediction

    ONLY_ODOMETRY_AVAILABLE = False     # Decide if available state consists of full car state or only of odometry
    
    # Noise Level for the controller's state estimation
    # NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.5 # ftg: 0.5  # mppi: 2.0
    # NOISE_LEVEL_ANGULAR_CONTROL = 0.30  # ftg: 0.05  # mppi: 3.0
    NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0. # ftg: 0.5  # mppi: 2.0
    NOISE_LEVEL_ANGULAR_CONTROL = 0.  # ftg: 0.05  # mppi: 3.0
    NOISE_LEVEL_CAR_STATE = [ 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
    # NOISE_LEVEL_CAR_STATE = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
            
    # Nikita's Slip/Steer Predictor
    SLIP_STEER_PREDICTION = False
    NET_NAME_SLIP = 'GRU-15IN-64H1-64H2-1OUT-0'
    NET_NAME_STEER = 'GRU-14IN-64H1-64H2-1OUT-0'
    
    
    ### Controller Settings
    
    CONTROLLER = 'mpc'  # Options: 'manual' (requires connected joystick) ,'mpc', 'ftg' (follow the gap), neural (neural network),  'pp' (pure pursuit)

    TIMESTEP_CONTROL = 0.06    # Multiple of 0.01
    
    ACCELERATION_TIME = 1                   #nni 50, mpc 10 (necessary to overcome initial velocity of 0 m/s)
    ACCELERATION_AMPLITUDE = 10.0           #nni 2, mpc 10 [Float!]
    
    CONTROL_AVERAGE_WINDOW = (3, 1)     # Window for avg filter [angular, translational]
    
    FOLLOW_RANDOM_TARGETS = False

    LOOK_FORWARD_ONLY = False # If false the max range of LIDAR is considered, otherwise only forward cone

    ## Pure Pursuit Controller ##
    PP_WAYPOINT_VELOCITY_FACTOR = 0.5
    PP_LOOKAHEAD_DISTANCE = 1.82461887897713965 # lookahead distance [m]
    PP_BACKUP_LOOKAHEAD_POINT_INDEX = 1
    
    
    ## Neural Controller ##
    #Network to be used for Neural control in nni_planner   -> Path to model can be adapted in nni_planner (controller=neursl)
    PATH_TO_MODELS = 'SI_Toolkit_ASF/Experiments/Experiment-MPPI-Imitator/Models/'
    NET_NAME = 'Dense-24IN-64H1-64H2-2OUT-0'

  
    ## MPC Controller ##
    NUM_TRAJECTORIES_TO_PLOT = 20
    OPTIMIZE_EVERY_N_STEPS = 3
    
    ## overwriting config_controller.yaml
    mpc_calculate_optimal_trajectory= True
    mpc_optimizer = "mppi" # mppi or rpgd-tf
    
    ## overwriting config_optimizer.yaml
    mppi_mpc_horizon= 15                       # steps
    mppi_num_rollouts = 9000                    # Number of Monte Carlo samples
    mppi_LBD =0.01                              # Cost parameter lambda
    mppi_NU =2000.0                            # Exploration variance
    mppi_SQRTRHOINV =[ 0.05, 0.05 ]     
    mppi_period_interpolation_inducing_points = 1     
    
    rpgd_mpc_horizon= 15                       # steps
    
    
    ## overwriting config_cost_function.yaml
    cc_weight = 0.0                        #check that cc, ccrc and R are the same as in config_optimizers.yml
    ccrc_weight = 0.1

    R = 1.0                                # How much to punish Q, For MPPI YOU have to make sure that this is the same as in optimizer config, as it plays a special role in the optimization algorithm as well as is used in cost functions!
    steering_cost_weight = 0.0
    angular_velocity_cost_weight = 0.001
    slipping_cost_weight = 0.5
    terminal_speed_cost_weight = 0.0
    velocity_diff_to_waypoints_cost_weight = 0.1
    speed_control_diff_to_waypoints_cost_weight = 1.0  # Penalize difference between desired speed of control and the position's closest waypoint
    distance_to_waypoints_cost_weight = 10.0
    target_distance_cost_weight = 0.0            #only if you want to follow a target on an empty map
    
    acceleration_cost_weight = 0.0
    max_acceleration = 9.2
    desired_max_speed = 3.8                             # desired max speed for the car [m/s]
    waypoint_velocity_factor  = 0.45

    ### Other Settings ###
    
    GLOBALLY_DISABLE_COMPILATION = False # Disable TF Compilation 
    ROS_BRIDGE = None # Automatically determined on program start

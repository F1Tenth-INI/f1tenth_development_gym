class Settings:

    ENVIRONMENT_NAME = 'Car'  # Car or Quadruped
    FROM_RECORDING = False

    ### for slip steer estimatoin -> change path to net in nn_loader_race.py
    SLIP_STEER_PREDICTION = False
    NET_NAME_SLIP = 'GRU-15IN-64H1-64H2-1OUT-0'
    NET_NAME_STEER = 'GRU-14IN-64H1-64H2-1OUT-0'
    
    CONTROLLER = 'mpc'  # Options: 'mpc', 'ftg' (follow the gap), neural (neural network),  Out of order: 'pp' (pure pursuit)
    ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:st'  # Its the model that the predictor uses. Only used for mpc predictions, if ODE predictor chosen
    # Options for ODE_MODEL_OF_CAR_DYNAMICS: 'ODE:simple', 'ODE:ks', 'ODE:st' # TODO: Currently only st discerns correctly between scenario with and without PID


    #Network to be used for Neural control in nni_planner   -> Path to model can be adapted in nni_planner (controller=neursl)
    NET_NAME = 'LSTM-94IN-32H1-32H2-2OUT-0'


    # Decide if to use PID as in the original F1TENTH implementation
    # Or bypass it.
    # Warning: Even if set to True, the PID algorithm is modified
    # with respect to F1TENTH implementation! Check gym/f110_gym/envs/dynamics_models.py for more details
    WITH_PID = True

    DISABLE_AUTOMATIC_TIMEOUT = True

    QUAD_VIZ = True  # Visualization, only for Quadruped

    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENTS_IN_SEPARATE_PROGRAMS = False
    EXPERIMENT_LENGTH = 36000  # in timesteps, only valid if DISABLE_AUTOMATIC_TIMEOUT is True.

    TIMESTEP_CONTROL = 0.05    # Multiple of 0.01
    
    # The map config file contains all information about the map, including the map_path, starting positions, waypoint_file path
    # physical params etc.
    # If you want to create a new file, orientate on existing ones.
    # MAP_CONFIG_FILE =  "utilities/maps_files/config_Map.yaml"
    # MAP_CONFIG_FILE =  "utilities/maps_files/config_example_map.yaml"
    # MAP_CONFIG_FILE =  "utilities/maps_files/config_Oschersleben.yaml"
    MAP_CONFIG_FILE =  "utilities/maps_files/config_INI.yaml"
    # MAP_CONFIG_FILE =  "utilities/maps_files/config_Budapest.yaml"
    # MAP_CONFIG_FILE =  "utilities/maps_files/config_Sochi.yaml"
    
    # Car parameters
    ENV_CAR_PARAMETER_FILE = "utilities/car_files/gym_car_parameters.yml" # Car parameters for simulated car    
    MPC_CAR_PARAMETER_FILE = "utilities/car_files/ini_car_parameters.yml" # Car parameters for MPC model prediction
    # MPC_CAR_PARAMETER_FILE = "utilities/car_files/ini_car_parameters.yml"
    
    # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    PLACE_RANDOM_OBSTACLES = False
    
    FOLLOW_RANDOM_TARGETS = False
    SAVE_RECORDINGS = False

    #Set Noise Level
    NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0 # ftg: 0.5  # mppi: 2.0
    NOISE_LEVEL_ANGULAR_CONTROL = 0  # ftg: 0.05  # mppi: 3.0

    # Automatically follow the first car on the map
    CAMERA_AUTO_FOLLOW = True
    DRAW_POSITION_HISTORY = True

    # We can chose between slow rendering (human) and fast rendering (human_fast)
    # RENDER_MODE = None
    RENDER_MODE = "human_fast"
    # RENDER_MODE = "human"
    NUM_TRAJECTORIES_TO_PLOT = 10

    # If false the max range of LIDAR is considered, otherwise only forward cone
    LOOK_FORWARD_ONLY = False

    # Decide if available state consists of full car state or only of odometry
    ONLY_ODOMETRY_AVAILABLE = False

    KEYBOARD_INPUT_ENABLE = False  # Allows for keyboard input during experiment. Causes silent crash on some computers

    ROS_BRIDGE = None # Automatically determined on program start

class Settings:

    ENVIRONMENT_NAME = 'Car'  # Car or Quadruped

    CONTROLLER = 'neural'  # Options: 'mpc', 'ftg' (follow the gap), neural (neural network),  Out of order: 'pp' (pure pursuit)
    ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:st'  # Its the model that the predictor uses. Only used for mpc predictions, if ODE predictor chosen
        # Options for ODE_MODEL_OF_CAR_DYNAMICS: 'ODE:simple', 'ODE:ks', 'ODE:st' # TODO: Currently only st discerns correctly between scenario with and without PID


    #Network to be used for Neural control in nni_planner   -> Path to model can be adapted in nni_planner (controller=neursl)
    NET_NAME = 'Dense-44IN-64H1-64H2-2OUT-3_MPPI_wypts_oschers_speed7_noisydata'
    USE_WAYPOINTS = 'relative'                  #CHECK either: False (ftg), 'relative' or 'absolute'
    ACCELERATION_TIME = 30              #nni 50, mpc 10
    ACCELERATION_AMPLITUDE = 10         #nni 2, mpc 10


    # Decide if to use PID as in the original F1TENTH implementation
    # Or bypass it.
    # Warning: Even if set to True, the PID algorithm is modified
    # with respect to F1TENTH implementation! Check gym/f110_gym/envs/dynamics_models.py for more details
    WITH_PID = False                    #CHECK

    DISABLE_AUTOMATIC_TIMEOUT = True

    QUAD_VIZ = True  # Visualization, only for Quadruped

    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENTS_IN_SEPARATE_PROGRAMS = False
    EXPERIMENT_LENGTH = 3000 #36000  # in timesteps, only valid if DISABLE_AUTOMATIC_TIMEOUT is True.

    TIMESTEP_CONTROL = 0.03    # Multiple of 0.01
    
    # The map config file contains all information about the map, including the map_path, starting positions, waypoint_file path
    # physical params etc.
    # If you want to create a new file, orientate on existing ones.
    #MAP_CONFIG_FILE = "utilities/maps_files/config_Hockenheim.yaml"
    #MAP_CONFIG_FILE = "utilities/maps_files/config_Melbourne.yaml"
    #MAP_CONFIG_FILE = "utilities/maps_files/config_Oschersleben.yaml"
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_example_map.yaml"
    #MAP_CONFIG_FILE = "utilities/maps_files/config_empty_map.yaml"


    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Austin.yaml"               #Decrease_resolution_factor = 10 in config.yml
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_BrandsHatch.yaml"          #Decrease_resolution_factor = 8
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Budapest.yaml"             #Decrease_resolution_factor = 8
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Catalunya.yaml"            #Decrease_resolution_factor = 8
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Hockenheim.yaml"           #Decrease_resolution_factor = 8
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_IMS.yaml"                  #Decrease_resolution_factor = 12
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Melbourne.yaml"            #Decrease_resolution_factor = 6
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_MexicoCity.yaml"           #Decrease_resolution_factor = 9
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Montreal.yaml"              #Decrease_resolution_factor = 6   and  ignore_wypts = 2
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Monza.yaml"                #Decrease_resolution_factor = 6
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_MoscowRaceway.yaml"        #Decrease_resolution_factor = 10
    MAP_CONFIG_FILE =  "utilities/maps_files/config_Oschersleben.yaml"         #Decrease_resolution_factor = 1
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Sakhir.yaml"               #Decrease_resolution_factor = 11
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_SaoPaulo.yaml"             #Decrease_resolution_factor = 9 and ignore_wypts=2
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Sepang.yaml"               #Decrease_resolution_factor = 6
    #MAP_CONFIG_FILE =  "utilities/maps_files/config_Sochi.yaml"                #Decrease_resolution_factor = 10




    
    # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    PLACE_RANDOM_OBSTACLES = False
    
    FOLLOW_RANDOM_TARGETS = False
    SAVE_RECORDINGS = True

    #Set Noise Level
    NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.0 # ftg: 0.5  # mppi: 2.0
    NOISE_LEVEL_ANGULAR_CONTROL = 0.0  # ftg: 0.05  # mppi: 3.0

    # Automatically follow the first car on the map
    CAMERA_AUTO_FOLLOW = True
    DRAW_POSITION_HISTORY = True

    # We can chose between slow rendering (human) and fast rendering (human_fast)
    #RENDER_MODE = None
    #RENDER_MODE = "human_fast"
    RENDER_MODE = "human"
    NUM_TRAJECTORIES_TO_PLOT = 20

    # If false the max range of LIDAR is considered, otherwise only forward cone
    LOOK_FORWARD_ONLY = False

    # Decide if available state consists of full car state or only of odometry
    ONLY_ODOMETRY_AVAILABLE = False

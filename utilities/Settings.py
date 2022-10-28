# Configuration for main
# TODO: This all should end up in the main config
class Settings:

    CONTROLLER = 'mpc'

    SYSTEM = 'car'
    QUAD_VIZ = True

    NUMBER_OF_EXPERIMENTS = 1  # How many times to run the car racing experiment
    EXPERIMENTS_IN_SEPARATE_PROGRAMS = False
    EXPERIMENT_LENGTH = 36000  # in timesteps, only valid if default termination after two laps is off.

    TIMESTEP_CONTROL = 0.03    # Multiple of 0.01
    
    # The map config file contains all information about the map, including the map_path, starting positions
    # physical params etc.
    # If you want to create a new file, orientate on existing ones.
    # MAP_CONFIG_FILE =  "utilities/maps_files/config_example_map.yaml"
    MAP_CONFIG_FILE =  "utilities/maps_files/config_Oschersleben.yaml"
    MAP_WAYPOINT_FILE = 'utilities/maps_files/waypoints/Oschersleben_map_wpts_dense800_190'
    # MAP_CONFIG_FILE =  "utilities/maps_files/config_empty_map.yaml"
    # MAP_WAYPOINT_FILE =  None
    
    
    # You can place random obstacles on the map. Have a look at the obstacle settings in maps_files/random_obstacles.yaml
    PLACE_RANDOM_OBSTACLES = False
    
    FOLLOW_RANDOM_TARGETS = False
    SAVE_RECORDINGS = True


    # Automatically follow the first car on the map
    CAMERA_AUTO_FOLLOW = True
    DRAW_POSITION_HISTORY = True

    # We can chose between slow rendering (human) and fast rendering (human_fast)
    # RENDER_MODE = None
    RENDER_MODE = "human_fast"
    # RENDER_MODE = "human"
    NUM_TRAJECTORIES_TO_PLOT = 20

    # If false the max range of LIDAR is considered, otherwise only forward cone
    LOOK_FORWARD_ONLY = False

    # Decide if to use PID as in the original F1TENTH implementation
    # Or bypass it.
    # Warning: Even if set to True, the PID algorithm is modified
    # with respect to F1TENTH implementation! Check gym/f110_gym/envs/dynamics_models.py for more details
    WITH_PID = True

    # Decide if available state consists of full car state or only of odometry
    ONLY_ODOMETRY_AVAILABLE = False

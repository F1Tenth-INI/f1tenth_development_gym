# Configuration for main
class Settings:
    
    # The map config file contains all information about the map, including the map_path, starting positions
    # pysical params etc. 
    # If you want to create a new file, orientate on existing ones.
    # MAP_CONFIG_FILE =  "MultiAgents/config_example_map.yaml"
    MAP_CONFIG_FILE =  "maps_files/config_Oschersleben.yaml"
    
    
    # You can place random obstacles on the map. Have a look at the obstacle settings in MultiAgents/random_obstacles.yaml
    PLACE_RANDOM_OBSTACLES = False
    
    
    # Automatically follow the first car on the map
    CAMERA_AUTO_FOLLOW = True
    
    
    # We can chose between slow rendering (human) and fast rendering (human_fast)
    # RENDER_MODE = "human_fast"
    RENDER_MODE = "human"
    NUM_TRAJECTORIES_TO_PLOT = 10

    # If false the max range of LIDAR is considered, otherwise only forward cone
    LOOK_FORWARD_ONLY = False
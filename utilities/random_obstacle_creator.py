import numpy as np
import random
from PIL import Image
import math
import os
import shutil
import tempfile
import yaml
from argparse import Namespace
import atexit
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

obstacles_config_path = 'utilities/random_obstables.yaml'
path_where_to_save_the_map = 'utilities/maps_files/maps/WithRandomObstacles'

def create_obstacles(map_arr, map_resolution, obstacles_config_path=obstacles_config_path, start_point=None):

    with open(obstacles_config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    map_h_px = map_arr.shape[0]
    map_w_px = map_arr.shape[1]
    logger.info(f'adding {conf.obstacle_number} random obstacles each with dimension {conf.obstacle_dim_m} m^2')
    radius = math.ceil(conf.obstacle_dim_m / map_resolution / 2)
    obstacles = []
    # ensure there is at least odim space from any previous one
    min_obs_gap_px = int(conf.obstacle_min_space_m / map_resolution + (radius * 2))
    min_gap_to_start_px = int(conf.obstacle_min_space_to_start_m / map_resolution)
    its = 0
    while len(obstacles) < conf.obstacle_number:

        # Check weather there were already too many trials to place an obstacle
        its += 1
        if its >= conf.obstacle_number * 3:
            logger.warning(
                f'could not create {conf.obstacle_number} obstacles after {its} iterations, giving up making more')
            break


        random_point = (random.randrange(radius, map_w_px - radius), random.randrange(radius, map_h_px - radius))
        if math.dist(random_point, start_point) < min_gap_to_start_px:
            continue
        too_close = False
        for o in obstacles:
            if math.dist(o, random_point) < min_obs_gap_px:
                too_close = True
        if not too_close:
            obstacles.append(random_point)
            map_arr[random_point[0] - radius:random_point[0] + radius,
            random_point[1] - radius:random_point[1] + radius] = 0  # make them black

    map_arr_with_obstacles = map_arr

    return map_arr_with_obstacles


def get_starting_position_for_obstacle_creation(map_arr, solo_starting_positions, origin, map_resolution):
    map_h_px = map_arr.shape[0]
    orig_x = origin[0]
    orig_y = origin[1]
    # map starting position in px
    start_point = (int((solo_starting_positions[0][0] - orig_x) / map_resolution),
                   map_h_px - int((solo_starting_positions[0][1] - orig_y) / map_resolution))
    return start_point

def check_conf(conf):
    no_obstacles = False
    if not hasattr(conf, 'random_obstacle_seed'):
        logger.info('no random_obstacle_seed in conf, add one to get random obstacles')
        no_obstacles = True
    if not hasattr(conf, 'obstacle_dim_m'):
        conf.obstacle_dim_m = 0.3
        logger.warning(f'using default obstacle_dim_m={conf.obstacle_dim_m}')
    if not hasattr(conf, 'obstacle_number'):
        conf.obstacle_number = 1000
        logger.warning(f'using default obstacle_number={conf.obstacle_number}')
    if not hasattr(conf, 'obstacle_min_space_m'):
        conf.obstacle_min_space_m = 1
        logger.warning(f'using default obstacle_min_space_m={conf.obstacle_min_space_m}')
    if not hasattr(conf, 'obstacle_min_space_to_start_m'):
        conf.obstacle_min_space_to_start_m = 3
        logger.warning(f'using default obstacle_min_space_to_start_m={conf.obstacle_min_space_to_start_m}')
    conf.random_obstacle_map_filename = None
    if conf.random_obstacle_seed == 0:
        logger.info(f'random obstacles disabled by seed=0 in conf')
        no_obstacles = True
    return conf, no_obstacles


def save_map(map_img, map_template_path, path_where_to_save_the_map=path_where_to_save_the_map):
    dir = path_where_to_save_the_map
    os.makedirs(dir, exist_ok=True)
    trackname = os.path.split(map_template_path)[-1]
    temp_img_file = tempfile.NamedTemporaryFile(dir=path_where_to_save_the_map + '/', prefix=trackname + '-'+str(
                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))+'-',
                                                suffix='.png', delete=False)
    random_obstacle_map_filename = temp_img_file.name
    logger.info(f'making temporary map image file {random_obstacle_map_filename}')

    # save new image and associated yaml to temp folder
    map_img.save(random_obstacle_map_filename)
    map_img.close()
    new_map_path = random_obstacle_map_filename[:-4]
    # we need to copy yaml file too and have it deleted automatically
    temp_yaml_filename = new_map_path + '.yaml'
    shutil.copyfile(map_template_path + '.yaml', temp_yaml_filename)
    return new_map_path

def register_map(map_path, delete_random_obstacle_map):
    if Settings.DELETE_MAP_WITH_OBSTACLES_IF_CRASHED:
        atexit.register(delete_random_obstacle_map, map_path)


def add_random_obstacles(map_template_path: str = None, solo_starting_positions=[[0.0, 0.0, 0.0]],
                         path_where_to_save_the_new_map=path_where_to_save_the_map,
                         obstacles_config_path=obstacles_config_path):
    with open(obstacles_config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    conf, no_obstacles = check_conf(conf)  # Checks if config for obstacles contains all needed information
    if no_obstacles:
        return
    else:
        pass

    seed = conf.random_obstacle_seed
    # load map image
    if seed == -1:
        seed = random.randrange(10000)
    logger.info(f'using random obstacle seed {seed}')
    random.seed(seed)

    map_ext = '.png'
    map_img_path = os.path.splitext(map_template_path)[0] + map_ext
    map_img = Image.open(map_img_path)

    map_arr = np.array(map_img)  # don't transpose, will get transposed later

    # add random obstacles
    # load map yaml
    with open(map_template_path + '.yaml', 'r') as yaml_stream:
        try:
            map_metadata = yaml.safe_load(yaml_stream)
            map_resolution = map_metadata['resolution']  # m/px
            origin = map_metadata['origin']  # in meters
        except yaml.YAMLError as ex:
            raise Exception(f'cannot read map resolution or origin: {ex}')

    start_point = get_starting_position_for_obstacle_creation(map_arr, solo_starting_positions, origin, map_resolution)

    map_arr_with_obstacles = create_obstacles(map_arr, map_resolution, obstacles_config_path=obstacles_config_path, start_point=start_point)

    # save new image
    map_img_with_obstacles = Image.fromarray(map_arr_with_obstacles)  # 8 bit mono

    new_map_path = save_map(map_img_with_obstacles, map_template_path, path_where_to_save_the_map=path_where_to_save_the_new_map)

    return new_map_path

class RandomObstacleCreator:

    def __init__(self):
        self.map_name = None

    def add_random_obstacles(self, map_template_path: str, solo_starting_positions, path_where_to_save_the_new_map=path_where_to_save_the_map):
        new_map_path = add_random_obstacles(map_template_path, solo_starting_positions=solo_starting_positions, path_where_to_save_the_new_map=path_where_to_save_the_new_map)
        self.map_name = new_map_path
        register_map(new_map_path, self.delete_random_obstacle_map)
        return new_map_path

    def unregister_auto_delete(self):
        logger.info(f'will save {self.map_name}')
        atexit.unregister(self.delete_random_obstacle_map)

    def delete_random_obstacle_map(self, filename):
        try:
            logger.info(f'removing random obstacle {filename} .png and .yaml ')
            os.remove(filename + '.png')
            os.remove(filename + '.yaml')
        except:
            pass

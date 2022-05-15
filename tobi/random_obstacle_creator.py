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
logger = logging.getLogger(__name__)


class RandomObstacleCreator:

    def __init__(self):
        self.map_name=None

    def add_random_obstacles(self, racetrack:str, solo_starting_positions):
        with open('MultiAgents/random_obstables.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        if not hasattr(conf,'random_obstacle_seed'):
            logger.info('no random_obstacle_seed in conf, add one to get random obstacles')
            return
        seed=conf.random_obstacle_seed
        if not hasattr(conf,'obstacle_dim_m'):
            conf.obstacle_dim_m=0.3
            logger.warning(f'using default obstacle_dim_m={conf.obstacle_dim_m}')
        if not hasattr(conf,'obstacle_number'):
            conf.obstacle_number=1000
            logger.warning(f'using default obstacle_number={conf.obstacle_number}')
        if not hasattr(conf,'obstacle_min_space_m'):
            conf.obstacle_min_space_m=1
            logger.warning(f'using default obstacle_min_space_m={conf.obstacle_min_space_m}')
        if not hasattr(conf,'obstacle_min_space_to_start_m'):
            conf.obstacle_min_space_to_start_m=3
            logger.warning(f'using default obstacle_min_space_to_start_m={conf.obstacle_min_space_to_start_m}')
        conf.random_obstacle_map_filename=None
        if seed==0:
            logger.info(f'random obstacles disabled by seed=0 in conf')
            return
        # load map image
        if seed==-1:
            seed=random.randrange(10000)
        logger.info(f'using random obstacle seed {seed}')
        random.seed(seed)

        map_path=racetrack # we need to add the yaml for consistency with later code and usage of this conf field
        map_ext='.png'
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        map_arr = np.array(Image.open(map_img_path)) # don't transpose, will get transposed later
        map_h_px=map_arr.shape[0]
        map_w_px=map_arr.shape[1]

        # add random obstacles
        # load map yaml
        map_resolution=None
        with open(map_path+'.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata['resolution']  # m/px
                origin = map_metadata['origin']  # in meters
            except yaml.YAMLError as ex:
                raise Exception(f'cannot read map resolution or origin: {ex}')
        orig_x = origin[0]
        orig_y = origin[1]
        # map starting position in px
        trackname=os.path.split(racetrack)[-1]
        start_point= (int((solo_starting_positions[0][0] - orig_x) / map_resolution),
             map_h_px-int((solo_starting_positions[0][1] - orig_y) / map_resolution))
        logger.info(f'adding {conf.obstacle_number} random obstacles each with dimension {conf.obstacle_dim_m} m^2')
        radius=math.ceil(conf.obstacle_dim_m/map_resolution/2)
        obstacles=[]
        # ensure there is at least odim space from any previous one
        min_obs_gap_px=int(conf.obstacle_min_space_m/map_resolution+(radius*2))
        min_gap_to_start_px=int(conf.obstacle_min_space_to_start_m/map_resolution)
        its=0
        while len(obstacles)<conf.obstacle_number:
            its+=1
            if its>=conf.obstacle_number*3:
                logger.warning(f'could not create {conf.obstacle_number} obstacles after {its} iterations, giving up making more')
                break
            random_point=(random.randrange(radius,map_w_px-radius),random.randrange(radius,map_h_px-radius))
            if math.dist(random_point,start_point)<min_gap_to_start_px:
                continue
            too_close=False
            for o in obstacles:
                if math.dist(o,random_point)<min_obs_gap_px:
                    too_close=True
                    break
            if not too_close:
                obstacles.append(random_point)
                map_arr[random_point[0]-radius:random_point[0]+radius,random_point[1]-radius:random_point[1]+radius]=0 # make them black

        # save new image
        map_img2=Image.fromarray(map_arr) # 8 bit mono
        dir='MultiAgents/maps/WithRandomObstacles'
        os.makedirs(dir, exist_ok=True)
        temp_img_file=tempfile.NamedTemporaryFile(dir='MultiAgents/maps/WithRandomObstacles/', prefix=trackname + '-', suffix='.png', delete=False)
        random_obstacle_map_filename=temp_img_file.name
        logger.info(f'making temporary map image file {random_obstacle_map_filename}')

        # save new image and associated yaml to temp folder
        map_img2.save(random_obstacle_map_filename)
        map_img2.close()
        random_obstacle_map_filename_base=random_obstacle_map_filename[:-4]
        # we need to copy yaml file too and have it deleted automatically
        temp_yaml_filename=random_obstacle_map_filename_base+'.yaml'
        shutil.copyfile(map_path+'.yaml',temp_yaml_filename)
        racetrack=random_obstacle_map_filename_base # point to new yaml
        atexit.register(self.delete_random_obstacle_map, random_obstacle_map_filename_base)
        self.map_name=racetrack
        return racetrack

    def unregister_auto_delete(self):
        logger.info(f'will save {self.map_name}')
        atexit.unregister(self.delete_random_obstacle_map)

    def delete_random_obstacle_map(self, filename):
        try:
            logger.info(f'removing random obstacle {filename} .png and .yaml ')
            os.remove(filename+'.png')
            os.remove(filename+'.yaml')
        except:
            pass
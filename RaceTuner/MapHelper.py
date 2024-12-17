import os

import numpy as np
import yaml
from matplotlib.image import imread


class MapConfig:
    def __init__(self, map_name, path_to_maps):
        self.map_name = map_name
        self.path_to_maps = path_to_maps
        self.path_to_map_png = os.path.join(path_to_maps, map_name, map_name + ".png")
        self.path_to_map_config = os.path.join(path_to_maps, map_name, map_name + ".yaml")

    def load_map_image(self, grayscale=True):
        img = imread(self.path_to_map_png)
        if grayscale and img.ndim == 3:
            img = np.dot(img[..., :3], [0.2989, 0.587, 0.114])
        return img

    def load_map_config(self):
        with open(self.path_to_map_config, 'r') as file:
            return yaml.safe_load(file)

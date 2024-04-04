import os
import yaml
import copy
import math

import numpy as np

from PIL import Image

from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.map_utilities import MapUtilities

from scipy.signal import savgol_filter


class WaypointsGenerator:
    
    def __init__(self):
        # Settings
        self.reverse_direction = True
        self.default_speed = 2.0 #[m/s] default speed for generated waypoints
        self.every_nth_pixel = 2
        
        self.convolution_window_size = 4
        
        # Importted Settings
        self.map_name = Settings.MAP_NAME
        self.map_path = Settings.MAP_PATH
        
    
    def export_handdrawn_waypoints(self):
        
        # load map metadata
        config_file_path = os.path.join(self.map_path, self.map_name +".yaml")
        with open(config_file_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata['resolution']
                origin = map_metadata['origin']
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)
                
        # Load Map png images
        file_name = self.map_path + "/"+ self.map_name + "_wp_hand.png"
        img = Image.open(file_name)

        image_array = np.asarray(img)
        occupied = []
        print("Map image shape: ", image_array.shape)

        # Save red pixels separately
        for i, row in enumerate(image_array):
            for j, pixel in enumerate(row):
                value_red = np.sum(pixel[0])
                if(value_red >= 255 and pixel[1] == 0 and pixel[2] == 0): # fully red
                    occupied.append([j, -i])
                    
        # Generate track data
        track_points = []
        for point in occupied:
            track_point = [float(point[0]), float(point[1])]
            track_points.append(track_point)    
        
        # decrease density
        track_points = track_points[::self.every_nth_pixel]

        # Sort Trackpoints
        unsorted_track_points = copy.deepcopy(track_points)
        sorted_track_points = []

        current_point = track_points[0]
        while len(unsorted_track_points) > 0:
            index, current_point = MapUtilities.get_closest_point(current_point, unsorted_track_points)
            sorted_track_points.append(current_point)
            unsorted_track_points.pop(index)
        track_points = np.array(sorted_track_points, dtype=float)

        if(self.reverse_direction):
            track_points = np.flip(track_points, axis=0)

        # Transform waypoints so they fit onto map
        height = image_array.shape[0] * map_resolution
        track_points = track_points * map_resolution
        track_points[:,0] = track_points[:,0] + origin_x
        track_points[:,1] = track_points[:,1] + height + origin_y
        
        kernel = np.ones(self.convolution_window_size) / self.convolution_window_size
        track_points[:,0] = np.convolve(track_points[:,0], kernel, mode='same')
        track_points[:,1] = np.convolve(track_points[:,1], kernel, mode='same')

        # Convert into waypoint format
        waypoints = []
        for i, track_point in enumerate(track_points):
            current_point = track_points[i]
            next_point = track_points[(i+1)%len(track_points)]
            next_segment = next_point - current_point
            absolute_angle = math.atan(next_segment[1]/ next_segment[0])
            
            print(next_segment)
            print(absolute_angle)
    
            
            waypoint = [
                0., 
                track_point[0], 
                track_point[1], 
                absolute_angle ,
                0. ,
                self.default_speed,
                0.
            ]
            waypoints.append(waypoint)
        
        # Relative angle ( needs absolte angle )
        for i, track_point in enumerate(track_points):
            waypoints[i][4] = waypoints[(i+1)%len(waypoints)][3] -  waypoints[i][3]
            
        waypoints = np.array(waypoints).astype('float32')
        
        
        # Save to wp file
        wp_file_path = os.path.join(self.map_path,self.map_name + "_wp.csv")
        np.savetxt(wp_file_path, waypoints, delimiter=",", fmt='%.4f')


        # Save Speed scaling file
        data = {
            'global_limit': 1.0,
            'n_sectors': 1,
            'Sector0': {
                'start': 0,
                'end': len(waypoints),
                'scaling': 1.0,
                'only_FTG': False,
                'no_FTG': False,
                }
            }
        speed_scaling_file_path = os.path.join(self.map_path,"speed_scaling.yaml")
        with open(r''+speed_scaling_file_path, 'w') as file:
            documents = yaml.dump(data, file)
            
        print("Done. Waypoints generated from handdrawn line")



if __name__ == '__main__':
    waypoint_generator = WaypointsGenerator()
    waypoint_generator.export_handdrawn_waypoints()


import csv
import yaml
import pandas as pd
import numpy as np

    

def transform_track_data(map_name,
    map_scale_factor,
    width_scale_factor = 2.0,
    image_margin = 10 ):

    track_points = pd.read_csv("utilities/maps_files/tum_track_db/original/"+map_name+".csv").to_numpy()

    for track_point in track_points:
        track_point[0] =  map_scale_factor * track_point[0] 
        track_point[1] = map_scale_factor * track_point[1] 
        track_point[2] = map_scale_factor * width_scale_factor * track_point[2] 
        track_point[3] = map_scale_factor * width_scale_factor * track_point[3] 


    offset_x = (abs(np.min(track_points[:, 0])) + image_margin)
    offset_y = (abs(np.min(track_points[:, 1])) + image_margin)
    offset = [offset_x, offset_y]


    for track_point in track_points:
        track_point[0] =  track_point[0] + offset[0]
        track_point[1] = track_point[1] + offset[1]


    file = open("utilities/maps_files/tum_track_db/widened/"+map_name+".csv", 'w')
    writer = csv.writer(file)
    writer.writerows(track_points)
    file.close()
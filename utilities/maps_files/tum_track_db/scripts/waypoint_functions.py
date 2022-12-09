

import csv
import pandas as pd
import numpy as np


def transform_waypoints(file_name, scaling):
    
    offset_x =  0
    offset_y =   0
    
    waypoints = pd.read_csv("utilities/maps_files/tum_track_db/wp_original/"+file_name+".csv", header=2, sep=";").to_numpy()

    for waypoint in waypoints:
        
        waypoint[1] =  waypoint[1] + offset_x 
        waypoint[2] = - waypoint[2] + offset_y

        waypoint[0] =  waypoint[0] * scaling
        waypoint[1] =  waypoint[1] * scaling
        waypoint[2] = waypoint[2] * scaling
        waypoint[3] = waypoint[3] * scaling
        waypoint[4] = waypoint[4] * scaling
        
    file = open("utilities/maps_files/waypoints/"+file_name+".csv", 'w')
    writer = csv.writer(file)
    writer.writerows(waypoints)
    file.close()






import csv
import pandas as pd
import numpy as np

file_name = "Budapest_wp"
offset_x =  -104.72059020000002
offset_y = 36.96699079999996
waypoints = pd.read_csv("utilities/maps_files/tum_track_db/wp/"+file_name+".csv", header=2, sep=";").to_numpy()
offset = [offset_x, offset_y]


for waypoint in waypoints:
    
    waypoint[1] =  waypoint[1] + offset[0]
    waypoint[2] = - waypoint[2] + offset[1]

    
    


file = open("utilities/maps_files/waypoints/"+file_name+".csv", 'w')
writer = csv.writer(file)
writer.writerows(waypoints)
file.close()


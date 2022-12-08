

import csv
import pandas as pd
import numpy as np

file_name = "Hockenheim_wp"
offset_x =  -15.75749055 + 15
offset_y =   22.29677879999999 - 25
scaling = 0.05


waypoints = pd.read_csv("utilities/maps_files/tum_track_db/wp/"+file_name+".csv", header=2, sep=";").to_numpy()


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


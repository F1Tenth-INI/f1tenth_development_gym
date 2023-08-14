import math
import matplotlib.pyplot as plt
import csv
import copy

from utilities.waypoint_utils import *
from utilities.map_utilities import *
from PIL import Image
from numpy import asarray

''' INSTRUCTIONS
Creage a B/W .png file ( i recommend about 1000 x 1000 pixels) and draw the centerline of a track with one black continuous line. Save image to utilities/maps_files/handdrawn/[map_name].png (look @ example: double_s.png)

Then adjust the map_name variable below and run this file: python utilities/create_hand_drawn_map.py

Finally in config_Map.yaml (make sure you select in in Settings.py as map config) ), adjust the name of the maps and waypoint file to run it in the env
'''

# CHANGEABLE PARAMETERS:
map_name = "circle1"
scaling = 0.1
width = 5.0

# PATHS:
path_to_map_img = 'maps_files/handdrawn/'+map_name+'.png'
path_to_map_folder = "maps/"+map_name+"/"
img = Image.open(path_to_map_img)

# ---------------------------------------------------------------

# PIL images into NumPy arrays
image_array = asarray(img)
occupancy_grid = np.zeros(image_array.shape[:2])
occupied = []
#  shape
print(image_array.shape)

for i, row in enumerate(image_array):
    for j, pixel in enumerate(row):
        value = np.sum(pixel)
        if(value < 382): # Whiter than gray...
            occupancy_grid[i][j] = 1
            occupied.append([i, j])
            


# Generate track data
track_points = []
for point in occupied:
    track_point = [float(point[0]), float(point[1]), width, width]
    track_points.append(track_point)    


# Sort Trackpoints
unsorted_track_points = copy.deepcopy(track_points)
sorted_track_points = []

current_point = track_points[0]
while len(unsorted_track_points) > 0:
    index, current_point = MapUtilities.get_closest_point(current_point, unsorted_track_points)
    sorted_track_points.append(current_point)
    unsorted_track_points.pop(index)
    
track_points = np.array(sorted_track_points)


# decrease density
track_points = np.array(track_points[::10])
    
# Shift for min = 0
min_x = np.min(np.array(track_points)[:, 0])
min_y = np.min(np.array(track_points)[:, 1])
track_points = MapUtilities.transform_track_data(track_points, scaling, -min_x, -min_y, 1.0)

# Save trackpoints
np.savetxt('maps_files/handdrawn/'+map_name+'.csv',track_points,delimiter=",")
    
# Generate Waypoints from track centerline
waypoints = []
for track_point in track_points:
    waypoint = [0., track_point[0], -track_point[1], 0. ,0. ,0. ,0.]
    waypoints.append(waypoint)
np.savetxt('umaps_files/waypoints/'+map_name+'_wp.csv',waypoints,delimiter=",")


offset_x, offset_y = MapUtilities.draw_map_and_create_yaml(map_name, path_to_map, image_margin=10)
waypoints = MapUtilities.transform_waypoints(waypoints, scaling, offset_x= offset_x, offset_y=offset_y)
MapUtilities.save_waypoints(waypoints, map_name)
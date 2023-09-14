import math
import matplotlib.pyplot as plt
import csv
import copy

from utilities.waypoint_utils import *
from utilities.map_utilities import *
from utilities.waypoints_generator import *
from PIL import Image
from numpy import asarray

# For yaml file generation
import yaml

''' INSTRUCTIONS
NEW!------------------------------------
Simply upload the car trajectory png file under "utilities/maps_files/handdrawn/[map_name]" and run this file. 
Edit the "CHANGEABLE PARAMETERS" section as needed.
Working directory must be "f1tenth_development_gym" not "f1tenth_development_gym/utilities" when running this file!!!
-----------------------------------------

Creage a B/W .png file ( i recommend about 1000 x 1000 pixels) and draw the centerline of a track with one black continuous line. Save image to utilities/maps_files/handdrawn/[map_name].png (look @ example: double_s.png)

Then adjust the map_name variable below and run this file: python utilities/create_hand_drawn_map.py

Finally in config_Map.yaml (make sure you select in in Settings.py as map config) ), adjust the name of the maps and waypoint file to run it in the env
'''

# CHANGEABLE PARAMETERS:
map_name = "circle1"
scaling = 0.02
width = 40.0
default_speed = 5.0
default_acceleration = 1.0
reverse_direction = True

convolution_window_size = 4
map_resolution = 0.1

# PATHS:
path_to_map_img = 'utilities/maps_files/handdrawn/'+map_name+'.png'
path_to_map_folder = "utilities/maps/"+map_name+"/"
img = Image.open(path_to_map_img)

# ---------------------------------------------------------------

# Create folder if it doesn't exist
if not os.path.exists(path_to_map_folder):
    os.makedirs(path_to_map_folder)
    print("Folder created successfully.")
else:
    print("Folder already exists.")

# PIL images into NumPy arrays
image_array = asarray(img)
occupancy_grid = np.zeros(image_array.shape[:2])
occupied = []
#  shape
print("\nImage opened!\n")
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

print("\nTrack generated!\n")

# Sort Trackpoints
unsorted_track_points = copy.deepcopy(track_points)
sorted_track_points = []

current_point = track_points[0]
while len(unsorted_track_points) > 0:
    index, current_point = MapUtilities.get_closest_point(current_point, unsorted_track_points)
    sorted_track_points.append(current_point)
    unsorted_track_points.pop(index)
    
track_points = np.array(sorted_track_points)

print("\nTrack points sorted!\n")


# decrease density
track_points = np.array(track_points[::5])
    
# Shift for min = 0
min_x = np.min(np.array(track_points)[:, 0])
min_y = np.min(np.array(track_points)[:, 1])
track_points = MapUtilities.transform_track_data(track_points, scaling, -min_x, -min_y, 1.0)

# Save trackpoints
np.savetxt(path_to_map_folder+map_name+'.csv', track_points, delimiter=",")

print("\nTrackpoints saved!\n")
    
# Generate Waypoints from track centerline
"""waypoints = []
for track_point in track_points:
    waypoint = [0., track_point[0], -track_point[1], 0. ,0. ,0. ,0.]
    waypoints.append(waypoint)
np.savetxt(path_to_map_folder+map_name+'_wp.csv', waypoints, delimiter=",")"""

"""offset_x, offset_y = MapUtilities.draw_map_and_create_yaml(map_name, path_to_map_folder, image_margin=10)

# Transform waypoints so they fit onto map
height = image_array.shape[0] * map_resolution
track_points = track_points * map_resolution
track_points[:, 0] = track_points[:, 0] + offset_x
track_points[:, 1] = track_points[:, 1] + height + offset_y

kernel = np.ones(convolution_window_size) / convolution_window_size
track_points[:, 0] = np.convolve(track_points[:, 0], kernel, mode='same')
track_points[:, 1] = np.convolve(track_points[:, 1], kernel, mode='same')"""

kernel = np.ones(convolution_window_size) / convolution_window_size
track_points[:, 0] = np.convolve(track_points[:, 0], kernel, mode='same')
track_points[:, 1] = -np.convolve(track_points[:, 1], kernel, mode='same')

waypoints = []
traversed_distance = 0.0
for i, track_point in enumerate(track_points):
    current_point = track_points[i]
    next_point = track_points[(i + 1) % len(track_points)]
    prev_point = track_points[(i - 1) % len(track_points)]

    next_segment = next_point - current_point
    absolute_angle = math.atan2(next_segment[1], next_segment[0])

    prev_segment = current_point - prev_point
    prev_absolute_angle = math.atan2(prev_segment[1], prev_segment[0])

    relative_angle = absolute_angle - prev_absolute_angle

    print(next_segment)
    print(absolute_angle)

    waypoint = [
        traversed_distance,
        track_point[0],
        track_point[1],
        absolute_angle,
        relative_angle,
        default_speed,
        default_acceleration
    ]
    waypoints.append(waypoint)

    traversed_distance += np.sqrt(
                                    np.square(track_point[0] - next_point[0])
                                    + np.square(track_point[1] - next_point[1])
                                  )

# Relative angle ( needs absolte angle )
"""for i, track_point in enumerate(track_points):
    waypoints[i][4] = waypoints[(i + 1) % len(waypoints)][3] - waypoints[i][3]"""

waypoints = np.array(waypoints).astype('float32')

offset_x, offset_y = MapUtilities.draw_map_and_create_yaml(map_name, path_to_map_folder, image_margin=10)
waypoints = MapUtilities.transform_waypoints(waypoints, scaling, offset_x=offset_x, offset_y=offset_y)
MapUtilities.save_waypoints(waypoints, map_name, path_to_waypoints=path_to_map_folder)

print("\nWaypoints saved!\n")


# TEMPLATE YAML FILE GENERATION-----------------------

# Create data for speed_scaling.yaml
speed_scaling_data = {
    "global_limit": 0.5,
    "n_sectors": 1,
    "Sector0": {
        "start": 0,
        "end": len(track_points),
        "scaling": 0.5,
        "only_FTG": False,
        "no_FTG": False
    }
}

# Create data for config_map_gym.yaml
config_map_gym_data = {
    "map_ext": ".png",
    "map_path": path_to_map_folder+map_name,
    "waypoint_path": path_to_map_folder+map_name+"_wp",
    "starting_positions": [
        [0., 0., 0.0]
    ],
    "waypoints": {
        "decrease_resolution_factor": 1,
        "ignore_waypoints": 1
    }
}

# Write data to YAML files
with open(os.path.join(path_to_map_folder, "speed_scaling.yaml"), "w") as file1:
    yaml.dump(speed_scaling_data, file1, default_flow_style=False)

with open(os.path.join(path_to_map_folder, "config_map_gym.yaml"), "w") as file2:
    yaml.dump(config_map_gym_data, file2, default_flow_style=False)

print("\nYAML files created!\n")

from utilities.maps_files.tum_track_db.scripts.track_functions import *
from utilities.maps_files.tum_track_db.scripts.waypoint_functions import *


map_name = "BrandsHatch"
wp_file_name = map_name + "_wp"
scaling = 0.05

# Transform track such that all coordinates are positive and scale to an appropriate scale for f1tenth
transform_track_data(map_name, scaling, 1.)

# Draw track PNG and create YAML file with offset
draw_map_and_create_yaml(map_name=map_name, scaling=scaling)

# Transform waypoints such that the fit on the track again
transform_waypoints(wp_file_name, scaling)
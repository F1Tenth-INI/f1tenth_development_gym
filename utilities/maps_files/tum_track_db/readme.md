# TUM Track DB
The TUM racetrack database (https://github.com/TUMFTM/racetrack-database) delivers the centerlines and withs of famous racetracks.
We'd like to use them for f1tenth. For a realistic setup, they need to be scaled accordingly

Typically we scale them by a factor of 0.05 but double the width of the track.

We use https://github.com/TUMFTM/global_racetrajectory_optimization for generating the minimum curvage trajectory


# GeneratiWaypoints
1. Copy the original real sized racetrack file into the TUM-Global_racetrajectory_optimizer/inputs/tracks
2. Generate waypoints ( python main_globaltraj.py )
3. Rename the generated waypoint file to [map_name]_wp and copy Waypoints back to f1tenth development gym (/utilities/maps_files/tum_track_db/wp_original) Notice that its still the fully sized track's waypoints
4. Scale and Draw the track and transform the waypoints with this script:
    ```bash
    python utilities/maps_files/tum_track_db/scripts/convert_to_f1tenth.py
    ````


# Using new tracks/waypoints
1. Copy "config_Map.yaml" and rename it to "config_[map_name].yaml" as map config file.
2. In config_[map_name].yaml set the map_path to: 'utilities/maps_files/maps/[map_name]' You might want to change the initial yaw angle.
```yaml
    ...
   # map paths
    map_path: 'utilities/maps_files/maps/[map_name]'
    waypoint_path: 'utilities/maps_files/waypoints/[map_name]_wp'

    # starting pose for map
    starting_positions: 
    - [0., 0., {initial_yaw_angle}]
    ...
```
3. In Settings.py, select "config_[map_name].yaml" as map config file.
```python
    ...
    MAP_CONFIG_FILE =  "utilities/maps_files/config_[map_name].yaml"
    ...
```


3. run.py


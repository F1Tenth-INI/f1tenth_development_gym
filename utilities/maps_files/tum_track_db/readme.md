# TUM Track DB
The TUM racetrack database (https://github.com/TUMFTM/racetrack-database) delivers the centerlines and withs of famous racetracks.
We'd like to use them for f1tenth. For a realistic setup, they need to be scaled accordingly

Typically we scale them by a factor of 0.05 but double the width of the track.

We use https://github.com/TUMFTM/global_racetrajectory_optimization for generating the minimum curvage trajectory


# Generating Track pngs and Waypoints
1. Copy the original real sized racetrack file into the TUM-Global_racetrajectory_optimizer/inputs/tracks
2. Generate waypoints ( python main_globaltraj.py )
3. Copy Waypoints back to f1tenth development gym (/utilities/maps_files/tum_track_db/wp) Notice that its still the fully sized track's waypoints
4. Transform and draw the fully sized track 
    ```bash
    transform_and_draw.py
    ````
5. transform_and_dray.py will output the offset of the track that had to be induces for drawing ( no negative coordinates for points ). Insert the wp_offset into transform_wp.py
6. Run transform_wp.py to downscale the waypoints according to the new track size. Make sure, scale variable is the same.


# Using new tracks/waypoints
1. In Settings.py, select "config_custom.yml" as map config file. Also select the new waypoint file.
2. In config_custom.yaml set the map_path to: 'utilities/maps_files/maps/[map_name]'
3. run.py


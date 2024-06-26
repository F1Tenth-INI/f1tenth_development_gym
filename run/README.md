This folder contains routine, data-generation and helper scripts that can b e executed directly

## CI Test

This script is executed as soon as you push something on github. It will make sure that the program sill compiles and that the car is able to drive a lap.
To be sure it passes the test, you might want to run ci_test.py before pushing.
**Only code that passes the test is accepted on the main branch.**

```bash
python run/ci_test.py
```

## Minimum curvature waypoints

This is a script for calculating minimum curvature waypoints
A map (defined in the main README) should already exist.

- Set the map name in Settings.py
- Set the waypoints generation parameters (like MIN_CURV_SAFETY_WIDTH) in Settings.py
- Make sure the map image at utilities/maps/[MAP_NAME]/[MAP_NAME]\_wp_min_curve.png contains a clear closed contour ( where the waypoints are calculated around). If there is no contour, you can draw one with your favourite drawing software.
- run pythion script:

```bash
python run/create_min_curve_waypoints.py
```

- Enjoy driving on the minimum curvature raceline.

Warning: existing waypoint files will be overwritten. Make sure you have a backup.

## Data collection for race scenario

This script performs multiple experiments with different setings (defined in arrays at the beginning of the script). It can be used to collect data for a NeuralNetwork Immitator, that for example needs to be trained on different speed levels.

- Check the settings on the beginning of the file
- Set a unique DATASET_NAME
- WARNING: before runniong the script overnight, make sure to test it with a small amoount of data, otherwise the chances of a bad surprise in the morning are high...

```bash
python run/run_data_collection.py
```

# UNTESTED

## Experiment data distribution (by Nigalsan)

Distribute ExperimentRecordings into SI_Toolkit_ASF/Experiments/[EXPERIMENTNAME] folder. It will split test and training data.

## Perform Track Visualization (by Nigalsan)

Visualizes an Experiment Recording

## DataGen (MT by Gianni)

Generates a trajectory by applying random control to the car model. The data produced by DataGen can be used to train a neural network model predictor.

## Create Handdrawn Maps

Creage a B/W .png file ( i recommend about 1000 x 1000 pixels) and draw the centerline of a track with one black continuous line. Save image to utilities/maps_files/handdrawn/[map_name].png (look @ example: double_s.png)

Then adjust the map_name variable below and run this file: python utilities/create_hand_drawn_map.py

Finally in config_Map.yaml (make sure you select in in Settings.py as map config) ), adjust the name of the maps and waypoint file to run it in the env

## Create Handdrawn Waypoitns

Create waypoints that are drawn by hand onto the map.png

```bash
python run/generate_handdrawn_waypoints.py
```

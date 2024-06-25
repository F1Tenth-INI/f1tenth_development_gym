![Python 3.8 3.9](https://github.com/F1Tenth-INI/f1tenth_development_gym/actions/workflows/ci.yml/badge.svg)

## Setup

(Tested on Ubuntu20, MacOS at 2024-06-25)
Clone the repo including submodules

```bash
git clone --recurse-submodules git@github.com:F1Tenth-INI/f1tenth_development_gym.git
cd f1tenth_development_gym/
```

I highly recommend using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for virtual environments.
First create a conda environment.

```bash
conda create -n f1t python=3.9 # Working environment with Apple Silicon
conda activate f1t
```

Then install the gym inside the environment. Don't omit the trailing / on gym.

There is a chance that yout setuptools is too new for the gym version. If that is the case, you need to downgrade it.

```bash
python -m pip install "pip<24.1"
pip install setuptools==65.5.0 "wheel<0.40.0"
```

And now you can install the gym environment.

```bash
pip install --user -e gym/
```

Check if the submodules are present. The folder SI_Toolkit and Control_Toolkit should not be empty. If they are empty, run:

```bash
git submodule update --init --recursive
```

And then install the SI_Toolkit

```bash
python -m pip install --user -e ./SI_Toolkit
```

## Run

Run the simulation

```bash
python run.py
```

If you are running from terminal, please run all python scripts from the project's root folder. You might want to export the Python Path env variable:

```bash
export PYTHONPATH=./
```

### Settings

Have a look at the Settings file: [Settings.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/Settings.py) This file gives you an idea of what can be adjusted at the GYM.

Let's go through the most important ones:

- MAP_NAME: There are multiple maps available (from easy to quite tricky). Change the map name to face different challenges.
- CONTROLLER: We have implemented multiple controllers (again from easy to complicated). If you are new with F1TGym, checkout 'ftg' (Follow the Gap) first, then the 'pp' (Pure Pursuit) controller. There are a lot of Tutorials about how these controllers work online.
- SAVE_RECORDINGS: IF set to true, a Recording will be created (in the folder ExperimentRecordings), which contains all the information about the car state and sensory inputs during the simulation. Recordings can also be raplayed.

# Wording & Conventions

## Environment

Have a look at [run_simulations.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/run.py). This file represents the world. You can add one or multiple instances of car_system classes to the drivers array:

```python
drivers = [planner1,planner2]
```

## Car System

Have a look at [car_system.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/car_system.py).
This is a representation of the physical car. A car system fetches information and sensor data from the environment and will deliver it to the Planner.
The car system consists of everything that all cars (independent of the planner/controller) have in common. Features of the Car System are the following:

- Perceive the car's state
- Load global Waypoints
- Determine Local waypoints (dep. on position)
- Render data ( Lidar, Position history etc.)
- Record data from experiments
- An instance of a planner
- process_observation function that receives lidar data and returns a control command

Note that every feature of the Car System is also implemented on the Physical car in the f1tenth_gym_bridge.

## Planner

The next layer of abstraction is the planner. The planner is still system specific (resp. designed for the car/car environment) but it handles features that not all controllers have in common.

- process_observation function that receives lidar data and returns a control command

- (Optional) A controller
  - If we use a system agnostic controller from the [Control Toolkit](https://github.com/SensorsINI/Control_Toolkit/tree/7398fdf5c7c5a6d8615e68b9dc153b116d52564b), we use the planner to gather data to deliver it to the controller in the right format. See [mpc_planner.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/Control_Toolkit_ASF/Controllers/MPC/mpc_planner.py)
  - If we use car specific controllers, the controller might already be implemented in the planner instance -> See [pp_planner.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/Control_Toolkit_ASF/Controllers/PurePursuit/pp_planner.py)
- Everything else the controller needs ( fe. a cost function )

## Controller

A controller is system agnostic. That means it does not know (and care) about which system is controlled. It will only try to fullfil the objective delivere by the planner.
That's why [Control Toolkit](https://github.com/SensorsINI/Control_Toolkit/tree/7398fdf5c7c5a6d8615e68b9dc153b116d52564b) is a sub repository. In fact the same code will run to control the Car and also the CartPole f.e..
It you think of a PID controller, it only gets an objective (error) and will try to reach it (error -> 0) but has no information about the system.

## Car State

We have implemented different car models. But within the environment_gym, we basically stick to the following definition for a car state. It is an array of 9 variables:

- angular_vel_z: yaw rate
- linear_vel_x: velocity in x direction
- pose_theta: yaw angle
- pose_theta_cos- pose_theta_sin- pose_x: x position in global coordinates
- pose_y: y position in global coordinates
- slip_angle: slip angle at vehicle center
- steering_angle: steering angle of front wheels

Check the [TUM CommonRoad Vehiclemodels](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf?ref_type=heads) for further information. Attention: The state variable indices are not the same in our system, we sort the alphabetically!

Please access the state variables only by name, for example:
pos_x = s[POSE_X_IDX]
You can import the index names from utilities/[state_utilities.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/state_utilities.py)

## Waypoint

A waypoint is defined as an array of the following properties:

- Distance since start
- Position x
- Position y
- Absolute angle of vector connecting to next wp
- Relative angle
- Suggested velocity
- Suggested acceleration

Every waypoint describes a desired position, desired velocity and other features, that the car has to follow.
The waypoints are saved in the map folder under map_name_wp.csv
Please access the waypoint properties only by name, for example:
pos_x = wp[WP_S_IDX]

You can import the index names from utilities/[waypoint_utils.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/waypoint_utils.py)

For a new map, you can either calculate the waypoints with "minimum curvature optimization" (fast-driving) or with "draw by hand" (uncompliated)

## Map

All maps are loceted at utilities/maps/
A map called [ExampleMap] consist of a folder at utilities/maps/ExampleMap/` containing the following files:

- `ExampleMap.yaml` - Contains meta information about the resolution etc.
- `ExampleMap.pgm` - Original image of the map (from SLAM on physical car)
- `ExampleMap.png` - Original image in .png format (used by localiyation stack on phcsical car, should not be changed!)
- `ExampleMap_wp_min_curve.png` - Map that's used to calculate minimum curvature waypoints (can be editted to contain a closed contour)
- `ExampleMap_wp_hand.png` - Is used to create waypoints by hand

(only present after waypoint generation)

- `ExampleMap_wp.yaml` - List of waypoints in classic direction
- `ExampleMap_wp_reverse.yaml` - List of waypoints in reverfse direction
- `data/` - Containing image processing and other data of the map

# Develop

Please work on your own branches and do pull requests to the main branch.
If possible, seperate your code into your own folders.

Every driver class must have the function process_observation, with the following arguments:

```python
def process_observation(self, ranges=None, ego_odom=None):
      """
      gives actuation given observation
      @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
      @ ego_odom: A dict with following indices:
      {
          'pose_x': float,
          'pose_y': float,
          'pose_theta': float,
          'linear_vel_x': float,
          'linear_vel_y': float,
          'angular_vel_z': float,
      }
      """
      desired_speed = 0
      desired_angle = 0

      return desired_speed, desired_angle

```

The function should return the desired speed and the desired angle

## Control Toolkit

Control Toolkit is a system agnostic sub-repository, which provides the cores of the most important controllers.
It is used on multiple projects (f.e. [CartPole](https://github.com/SensorsINI/CartPoleSimulation)).

In the gym we have implemented controllers from the Control Toolkit. Every **Application Specific File** (which are specifically meant for controlling the car in the GYM environment) are in the folder Control_Toolkit_ASF

The Control Toolkit's config files are called

- [config_controllers.yml]([https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/Control_Toolkit_ASF/config_controllers.yml),
- [config_optimizers.yml]([https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/Control_Toolkit_ASF/config_optimizers.yml)
- [config_cost_function.yml]([https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/Control_Toolkit_ASF/config_cost_function.yml)

Have a look at them and see how the controllers can be tuned.

### MPC Controller

The MPC controller is implemented in the GYM with two optimizers:

- MPPI
- RPGD

#### Cost functions

The cost functions are in [Control_Toolkit_ASF/CostFunctions](Control_Toolkit_ASF/Cost_Functions).
The cost function properties are in cost function template: [f1t_cost_function.py](Control_Toolkit_ASF/Cost_Functions/f1t_cost_function.py), lines 63-72.

## SI Toolkit

Control Toolkit is a system agnostic sub-repository, which provides the cores for neural system identification and brunton plotting.
Like Control Toolkit, is used on multiple projects (f.e. [CartPole](https://github.com/SensorsINI/CartPoleSimulation)).
Every **Application Specific File** related to the SI Toolkit is in the folder SI_Toolkit_ASF.

On the controller side, these structure are at [SI_Toolkit_ASF/car_model.py](SI_Toolkit_ASF/car_model.py), lines 113-190.

<!-- For an arbitrary choise of controller, I removed the built-in PID controller from the base_class.py, such that the environment takes the actual motor inputs instead of the desired speed/angle. -->

# Neural Imitator

## Training

Collect experiment recordings with a controller of choise (fe. MPC - MPPI).

- For a higher variance data, set control noise up to 0.5

NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.5
NOISE_LEVEL_ANGULAR_CONTROL = 0.5

- Tune the controller for robustness ( it needs to be able to complete laps reliably )
- Delete all old experiment recordings in ExperimentRecordings/
- Set EXPERIMENT_LENGTH such that the car completes more than 2 laps
- set NUMBER_OF_EXPERIMENTS >= 10 depending on how much data you want to have
- Run experiments

- Create the following folders:
  - SI_Toolkit_ASF/Experiments/[Controller Name]/Recordings/Train
  - SI_Toolkit_ASF/Experiments/[Controller Name]/Recordings/Test
  - SI_Toolkit_ASF/Experiments/[Controller Name]/Recordings/Validate
- Distribute the experiment's CSV files into these 3 folders ( each 80%, 10%, 10% of the data points)
- in config_training.yml set path_to_experiment to [Controller Name]

- Create normalization file:

```bash
python SI_Toolkit_ASF/run/Create_normalization_file.py
```

- Check the histograms if training data makes sense
- in config_training.yml set NET_NAME, inputs and training settings
- Train Network:

```bash
 python SI_Toolkit_ASF/run/Train_Network.py
```

- create a file at SI_Toolkit_ASF/Experiments/[Controller Name]/Models/[Model Name]/notes.txt and write minimal documentation about the network (Maps, Controller, Settings, thoughts etc...)
  Congratulations, the Neural Controller is now ready to use.

## Run Neural Imitator

- in Settings.py, select the neural controller and the model name
- Deactivate the control noise
- Deactivate control averaging (NN does not like it )

```python
CONTROLLER = 'neural'
...
PATH_TO_MODELS = 'SI_Toolkit_ASF/Experiments/[Controller Name]/Models/'
NET_NAME = '[Model Name]'
...
NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.0
NOISE_LEVEL_ANGULAR_CONTROL = 0.0
...
CONTROL_AVERAGE_WINDOW = (1, 1)
...
```

- Make sure that the control_inputs in config_training.yml and nni_planner.py match. (Otherwise correct them in nni_planner)
- Run experiment
  Enjoy your realtime neural network MPPI imipator ( or how we call it: the INItator ).

# Brunton Test

Check config_testting.yml:

- Select a file in experiment recordings for reference
- Select the network you want to test

```bash
python SI_Toolkit_ASF/run/Run_Brunton_Test.py
```

# Neural Predictor

## Data Generation

To generate data, the best and fastest way is to use the data generator:

- Settings: `DataGen/config_data_gen.yml`
- Run: `python3 -m SI_Toolkit_ASF.run.run_data_generator_for_ML_Pipeline`

The provided settings are the ones that were found to work best for a time step of 0.04s. For other time steps, they might need to be adjusted. This creates a new folder in `SI_Toolkit_ASF/Experiments` with the data split into a train, test and validation folder.

## Data preprocessing

To remove outliers and add a derivative column, we need to preprocess the data.

- Settings: Everything need to be adjusted in the code itself (`SI_Toolkit_ASF/run/preprocess_data.py`)
- Run: `python3 -m SI_Toolkit_ASF.run.preprocess_data`

To only add derivative columns to the dataset:

- Run: `python3 -m SI_Toolkit_ASF.run.Add_derivative_to_csv`
- Settings: Adjust directly in file `SI_Toolkit_ASF/run/Add_derivative_to_csv.py`

## Training

To train, we first need to rename the columns of our training data. Use your IDE to rename all instances of `[translational]/[angular]_control_applied` in your dataset to `[translational]/[angular]_control`.

Then we need to create the normalization file:

- Settings:`SI_Toolkit_ASF/config_training.yml` and specify the desired `path_to_experiments`
- Run: `python3 -m SI_Toolkit_ASF.run`

Then set the desired model in `config_training.yml`. Settings that worked well are:

- Dense: `NET_NAME: 'Dense-128H1-128H2'`, 20 epochs, batch size of 32, wash out length 0, post wash out length 1, shift_labels 1 (important!).
- LSTM: `NET_NAME: 'Dense-128H1-128H2'`, 20 epochs, batch size of 32, wash out length 10, post wash out length 1, shift_labels 1.

Then run the training:

- Settings: `SI_Toolkit_ASF/config_training.yml`
- Run: `python3 -m SI_Toolkit_ASF.run.Train_Network`

## Evaluation

To check that your predictor works, run the Brunton test using:

- Settings: `SI_Toolkit_ASF/config_testing.py`
- Run: `python3 -m SI_Toolkit_ASF.run.Run_Brunton_Test``

# Generate miminum Curvature Waypoints

- Select the map yopu want to create the waypoints in Settings.py => MAP_NAME
- Make sure there is a valid MAP_NAME.yaml and MAP_NAME_wp_min_curve.png file in the map folder
- If there is no MAP_NAME_wp_min_curve, you can just copy and rename the original map PNG
- In the MAP_NAME_wp_min_curve.png, you can draw corrections, do deliver a nice closed contour.
- Set the MIN_CURV_SAFETY_WIDTH in Settings.py. Note that the car's width is included. It should not be < 0.8m.
- Run the script:

```bash
python run/create_min_curve_waypoints.py
```

The waypoints (and additional data) will be saved in the map folder.

# Info by the original authors

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Known issues

- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:

```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```

You can fix the error by installing a newer version of pyglet:

```bash
$ pip3 install pyglet==1.5.11
```

And you might see an error similar to

```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```

which could be ignored. The environment should still work without error.

## Citing

If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```

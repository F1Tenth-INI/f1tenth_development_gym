j![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)

# Notes by Florian

## Setup
I highly recommend using Conda for virtual environments.
First create a conda environment. 
```bash
conda create -n f1t python=3.8
conda activate f1t
```


Then install the gym inside the environment.
```bash
pip3 install --user -e gym/
```

To set up he SI_Toolkit, pull all sub modules:
```bash
git submodule update --init --recursive
git submodule update --recursive --remote
```
and then install the Toolkit packages: 
```bash
python -m pip install --user -e ./SI_Toolkit
python -m pip install --user -e ./Control_Toolkit
```

Finally copy Settings_Template.py.py to Settings.py, to have your own gitignored settings.
```bash
cp Settings_Template.py Settings.py
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
##Environment, CarModel and Controller

Have a look at [run_simulations.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/run.py). This file represents the world. You can add one or multiple instances of car_system classes to the planners array:
```python
##################### DEFINE DRIVERS HERE #####################    
drivers = [planner1,planner2]
###############################################################   
```

Have a look at [car_system.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/car_system.py). This file represents the car. Everyting that runs on this object can directly be applied on the physical car. Inside the carSystem, we can define a controller (planner).


### Settings
Have a look at the Settings file: [Settings.py](https://github.com/F1Tenth-INI/f1tenth_development_gym/blob/main/utilities/Settings.py) 

 
 ## Develop
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

## Controllers
### Cost functions
The cost functions are in [Control_Toolkit_ASF/CostFunctions](Control_Toolkit_ASF/Cost_Functions).
The cost function properties are in cost function template: [f1t_cost_function.py](Control_Toolkit_ASF/Cost_Functions/f1t_cost_function.py), lines 63-72.

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





# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
You can install the environment by running:

```bash
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip3 install --user -e gym/
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

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

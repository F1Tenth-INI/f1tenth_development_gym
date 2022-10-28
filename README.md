![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
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
and then install the packages: 
```bash
python -m pip install --user -e ./SI_Toolkit
python3 -m pip install --user -e ./SI_Toolkit_ASF/GlobalPackage
```
## Run

Please run all python scripts from the root folder

The  environment you should use is in the MultiAgents folder.
```bash
python MultiAgents/run.py
```

You can add one or multiple instances of driver classes to the drivers array:
```python
##################### DEFINE DRIVERS HERE #####################    
drivers = [planner1,planner2]
###############################################################   
```

### Settings
Have a look at the MultiAgent's settings file: [Settings.py](https://github.com/Florian-Bolli/f1tenth_development_gym/blob/main/MultiAgents/Settings.py) 

 
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

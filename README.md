![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)

# Changes by Florian
For an arbitrary choise of controller, I removed the built-in PID controller from the base_class.py, such that the environment takes the actual motor inputs instead of the desired speed/angle.

## Follow The Gap

The follow tha Gap controller can be tested in a single agent environment like this: 

```bash
cd FollowTheGap
python3 run.py
```

The Controller itself is implemented in ftg_planner. 
```process_observation(self, ranges=None, ego_odom=None) ```
This is the function that is required to take place in the ICRA gym race and is called by the ENV at every timestep with the according arguments.


## MPPI
```bash
cd MPPI
python3 run.py
```
An implementation of the MPPI Controller - Is much too slow to run real-time
Please don't touch yet - WORK IN PROGRESS


## Multi Agents
If you want to let two cars race against each other, there is a multi agent environment. You can plan the controls for the two different cars with different Planner classes / settings.

```bash
cd MultiAgent
python3 run.py
```


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

# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



from SI_Toolkit.computation_library import  NumpyLibrary


"""
Prototype of base classes
Replacement of the old RaceCar, Simulator classes in C++
Author: Hongrui Zheng
"""

import numpy as np

from f110_sim.envs.dynamic_model_pacejka_jit import car_dynamics_pacejka_jit, StateIndices
from sim.f110_sim.envs.car_model_jax import car_dynamics_pacejka_jax


from f110_sim.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_sim.envs.collision_models import get_vertices, collision_multiple


from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.car_files.vehicle_parameters import VehicleParameters

from math import fmod
# Wraps the angle into range [-π, π]
def wrap_angle_rad(angle: float) -> float:
    Modulo = fmod(angle, 2 * np.pi)  # positive modulo
    if Modulo < -np.pi:
        angle = Modulo + 2 * np.pi
    elif Modulo > np.pi:
        angle = Modulo - 2 * np.pi
    else:
        angle = Modulo
    return angle

class RaceCar(object):
    """
    Base level race car class, handles the physics and laser scan of a single vehicle

    Data Members:
        params (dict): vehicle parameters dictionary
        is_ego (bool): ego identifier
        time_step (float): physics timestep
        num_beams (int): number of beams in laser
        fov (float): field of view of laser
        state (np.ndarray (7, )): state vector [x, y, theta, vel, steer_angle, ang_vel, slip_angle]
        odom (np.ndarray(13, )): odometry vector [x, y, z, qx, qy, qz, qw, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        accel (float): current acceleration input
        steer_angle_vel (float): current steering velocity input
        in_collision (bool): collision indicator

    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(self, params, seed, is_ego=False, time_step=0.01, num_beams=1080, fov=4.7):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser

        Returns:
            None
        """

        # initialization
        self.params = params
        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov

        self.ode_implementation = Settings.SIM_ODE_IMPLEMENTATION

        # Handle the case where it's none of the specified options
        # For example, raise an error or set a default implementation
        if self.ode_implementation not in ['std', 'std_tf', 'pacejka', 'ODE_TF', 'jit_Pacejka', 'jax_pacejka']:
            raise ValueError(f"Unsupported ODE implementation: {self.ode_implementation}")

        self.state = np.zeros((StateIndices.number_of_states, ))
        if self.ode_implementation == 'ODE_TF':
            from SI_Toolkit_ASF.car_model import car_model
            self.car_model = car_model(
                model_of_car_dynamics = Settings.ODE_MODEL_OF_CAR_DYNAMICS,
                batch_size = 1, 
                car_parameter_file = Settings.ENV_CAR_PARAMETER_FILE, 
                dt = 0.01, 
                intermediate_steps=1,
                computation_lib=NumpyLibrary()
                )
            
            # In case you want to use other library than numpy
            # from SI_Toolkit.Compile import CompileAdaptive
            # self.step_dynamics = CompileAdaptive(self.car_model.lib)(self.car_model.step_dynamics)
            self.step_dynamics = self.car_model.step_dynamics
            self.step_dynamics_core = self.car_model.step_dynamics_core # step dynamics without constratins an PID

        if self.ode_implementation == 'jit_Pacejka':
            self.step_dynamics = car_dynamics_pacejka_jit
            u = np.array([0,0])
            
            car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
            self.car_params_array = car_params.to_np_array()
            
            self.state = self.step_dynamics(self.state, u, self.car_params_array, 0.01)   

        if self.ode_implementation == 'jax_pacejka':
            import jax.numpy as jnp
            self.step_dynamics = car_dynamics_pacejka_jax
            u = np.array([0,0])
            
            car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
            self.car_params_array = car_params.to_np_array()
            
            # Initialize with JAX conversion
            jax_state = jnp.array(self.state)
            jax_u = jnp.array(u)
            jax_params = jnp.array(self.car_params_array)
            new_state = self.step_dynamics(jax_state, jax_u, jax_params, 0.01)
            self.state = np.array(new_state)   

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # Effective control commands
        self.steering_speed_cmd = 0.0
        self.acceleration_x_cmd = 0.0
        self.u_pid_with_constrains = np.array([0.0, 0.0])

        # steering delay buffer
        self.steer_buffer = np.empty((0, ))
        self.steer_buffer_size = 1

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if RaceCar.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = RaceCar.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            RaceCar.cosines = np.zeros((num_beams, ))
            RaceCar.scan_angles = np.zeros((num_beams, ))
            RaceCar.side_distances = np.zeros((num_beams, ))

            dist_sides = params['width']/2.
            dist_fr = (params['lf']+params['lr'])/2.

            for i in range(num_beams):
                angle = -fov/2. + i*scan_ang_incr
                RaceCar.scan_angles[i] = angle
                RaceCar.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi/2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi/2.)
                        to_fr = dist_fr / np.sin(angle - np.pi/2.)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi/2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi/2)
                        to_fr = dist_fr / np.sin(-angle - np.pi/2)
                        RaceCar.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        self.params = params
    
    def set_map(self, map_path, map_ext):
        """
        Sets the map for scan simulator
        
        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """
        RaceCar.scan_simulator.set_map(map_path, map_ext)

    def reset(self, initial_state):
        """
        Resets the vehicle to a pose
        
        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to

        Returns:
            None
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state
        self.state = initial_state
        
        self.steer_buffer = np.empty((0, ))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def ray_cast_agents(self, scan):
        """
        Ray cast onto other agents in the env, modify original scan

        Args:
            scan (np.ndarray, (n, )): original scan range array

        Returns:
            new_scan (np.ndarray, (n, )): modified scan
        """

        # starting from original scan
        new_scan = scan
        
        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current oppoenent
            opp_vertices = get_vertices(opp_pose, self.params['length'], self.params['width'])

            pose = np.array([self.state[StateIndices.pose_x], self.state[StateIndices.pose_y], self.state[StateIndices.yaw_angle]])
            new_scan = ray_cast(pose, new_scan, self.scan_angles, opp_vertices)

        return new_scan

    def check_ttc(self, current_scan):
        """
        Check iTTC against the environment, sets vehicle states accordingly if collision occurs.
        Note that this does NOT check collision with other agents.

        state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        Args:
            current_scan

        Returns:
            None
        """
        
        in_collision = check_ttc_jit(current_scan, self.state[StateIndices.v_x], self.scan_angles, self.cosines, self.side_distances, self.ttc_thresh)

        # if in collision stop vehicle
        if in_collision:
            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, desired_steering_angle, desired_speed):
        """
        Steps the vehicle's physical simulation

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity

        Returns:
            current_scan
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        
        # Some models require PID control - but this is an old PID implementation!
        # acceleration, steering_angular_velocity = pid(desired_speed, desired_steering_angle, self.state[StateIndices.v_x], self.state[StateIndices.yaw_angle], self.params['sv_max'], self.params['a_max'], self.params['v_max'], self.params['v_min'])

        
        if(self.ode_implementation == 'pacejka'):
            s = self.state
            u = np.array([desired_steering_angle, desired_speed])
            self.state = self.dynamic_model.step(s, u)

        if self.ode_implementation == 'ODE_TF':
            s = np.expand_dims(self.state, 0).astype(np.float32)
            u = np.array([[desired_steering_angle, desired_speed]], dtype=np.float32)

            u_pid = self.car_model.pid(s, u)
            self.u_pid_with_constrains = self.car_model.apply_constrains(s, u_pid)

            self.steering_speed_cmd, self.acceleration_x_cmd = self.car_model.return_control_cmd_components(self.u_pid_with_constrains)

            s = self.car_model.step_dynamics_core(s, self.u_pid_with_constrains)[0]
            # wrap yaw angle
            s[POSE_THETA_IDX] = wrap_angle_rad(s[POSE_THETA_IDX])
            self.state = s
            

        
        elif self.ode_implementation == 'jit_Pacejka':
            u = np.array([desired_steering_angle, desired_speed])
            self.state = self.step_dynamics(self.state, u, self.car_params_array, 0.01)   
            
        elif self.ode_implementation == 'jax_pacejka':
            import jax.numpy as jnp
            u = np.array([desired_steering_angle, desired_speed])
            # Convert to JAX arrays and back to numpy
            jax_state = jnp.array(self.state)
            jax_u = jnp.array(u)
            jax_params = jnp.array(self.car_params_array)
            new_state = self.step_dynamics(jax_state, jax_u, jax_params, 0.01)
            self.state = np.array(new_state)   
            
        if self.ode_implementation == 'f1tenth_st':
            raise NotImplementedError("ODE implementation for 'f1tenth_st' is not yet implemented.")
        
        # update scan
        pose = np.array([self.state[StateIndices.pose_x], self.state[StateIndices.pose_y], self.state[StateIndices.yaw_angle]])
        current_scan = RaceCar.scan_simulator.scan(pose, self.scan_rng)

        return current_scan

    def update_opp_poses(self, opp_poses):
        """
        Updates the vehicle's information on other vehicles

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents

        Returns:
            None
        """
        self.opp_poses = opp_poses

    def update_scan(self, agent_scans, agent_index):
        """
        Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)

        Returns:
            None
        """

        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan

class Simulator(object):
    """
    Simulator class, handles the interaction and update of all vehicles in the environment

    Data Members:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agent
        collision_idx (np.ndarray(num_agents, )): which agent is each agent in collision with

    """

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step
            ego_idx (int, default=0): ego vehicle's index in list of agents

        Returns:
            None
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.agent_scans = []
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(params, self.seed, is_ego=True)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(params, self.seed)
                self.agents.append(agent)

    def set_map(self, map_path, map_ext):
        """
        Sets the map of the environment and sets the map for scan simulator of each agent

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)


    def update_params(self, params, agent_idx=-1):
        """
        Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def check_collision(self):
        """
        Checks for collision between agents using GJK and agents' body vertices

        Args:
            None

        Returns:
            None
        """
        # get vertices of all agents
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            pos_x = self.agents[i].state[StateIndices.pose_x]
            pos_y = self.agents[i].state[StateIndices.pose_y]
            yaw = self.agents[i].state[StateIndices.yaw_angle]
            all_vertices[i, :, :] = get_vertices(np.append([pos_x, pos_y], yaw), self.params['length'], self.params['width'])
        self.collisions, self.collision_idx = collision_multiple(all_vertices)

    def get_sim_observation(self):
        """
        Returns a dictionary containing car states, scans, and collisions for all agents.

        Returns:
            obs (dict): observation dictionary with keys:
                'car_states': np.ndarray of shape (num_agents, state_dim)
                'scans': list of np.ndarray, each scan for an agent
                'collisions': np.ndarray of shape (num_agents, )
        """
        car_states = np.array([agent.state for agent in self.agents])
        obs = {
            'car_states': car_states,
            'scans': self.agent_scans,
            'collisions': self.collisions.copy(),
            'ego_idx': self.ego_idx,
        }
        return obs
    
    def step(self, control_inputs):
        """
        Steps the simulation environment

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents, first column is desired steering angle, second column is desired velocity
        
        Returns:
            observations (dict): dictionary for observations: poses of agents, current laser scan of each agent, collision indicators, etc.
        """


        agent_scans = []

        # looping over agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)

            # update sim's information of agent poses
            pos_x = agent.state[StateIndices.pose_x]
            pos_y = agent.state[StateIndices.pose_y]
            yaw = agent.state[StateIndices.yaw_angle]
            self.agent_poses[i, :] = np.append([pos_x, pos_y], yaw)
        self.agent_scans = agent_scans

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            # update agent's information on other agents
            opp_poses = np.concatenate((self.agent_poses[0:i, :], self.agent_poses[i+1:, :]), axis=0)
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            agent.update_scan(agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.
       

        # fill in observations
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        observations = self.get_sim_observation()
        return observations

    def reset(self, initial_states):
        """
        Resets the simulation environment by given initial_states.

        Args:
            initial_states (np.ndarray (num_agents, 3)): initial_states to reset agents to

        Returns:
            obs (dict): initial observation dictionary
        """
        if initial_states.shape[0] != self.num_agents:
            raise ValueError('Number of initial_states for reset does not match number of agents.')

        # Reset all agents
        
        self.agent_scans = []
        for i in range(self.num_agents):
            agent = self.agents[i]
            agent.reset(initial_states[i, :])
            pose = np.array([agent.state[StateIndices.pose_x], agent.state[StateIndices.pose_y], agent.state[StateIndices.yaw_angle]])
            current_scan = RaceCar.scan_simulator.scan(pose, agent.scan_rng)
            self.agent_scans.append(current_scan)

        obs = self.get_sim_observation()
        return obs

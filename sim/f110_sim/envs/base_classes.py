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

from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax_from_settings


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


def normalize_state_yaw(state: np.ndarray) -> np.ndarray:
    """Wrap pose_theta to [-pi, pi] and keep cos/sin in sync."""
    yaw = wrap_angle_rad(float(state[POSE_THETA_IDX]))
    state[POSE_THETA_IDX] = yaw
    state[POSE_THETA_COS_IDX] = np.cos(yaw)
    state[POSE_THETA_SIN_IDX] = np.sin(yaw)
    return state

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

        # initialization - use VehicleParameters instead of params argument
        car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
        self.params = car_params.to_dict()
        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov

        self.ode_implementation = Settings.SIM_ODE_IMPLEMENTATION

        # Handle the case where it's none of the specified options
        # For example, raise an error or set a default implementation
        if self.ode_implementation not in ['std', 'std_tf', 'pacejka', 'ODE_TF', 'jax_pacejka', 'residual']:
            raise ValueError(f"Unsupported ODE implementation: {self.ode_implementation}")

        self.state = np.zeros((NUMBER_OF_STATES, ))
        if self.ode_implementation == 'ODE_TF':
            from SI_Toolkit_ASF.car_model import car_model
            self.car_model = car_model(
                model_of_car_dynamics = Settings.ODE_MODEL_OF_CAR_DYNAMICS,
                batch_size = 1, 
                car_parameter_file = Settings.ENV_CAR_PARAMETER_FILE, 
                dt = Settings.TIMESTEP_SIM,
                intermediate_steps=1,
                computation_lib=NumpyLibrary()
                )
            
            # In case you want to use other library than numpy
            # from SI_Toolkit.Compile import CompileAdaptive
            # self.step_dynamics = CompileAdaptive(self.car_model.lib)(self.car_model.step_dynamics)
            self.step_dynamics = self.car_model.step_dynamics
            self.step_dynamics_core = self.car_model.step_dynamics_core # step dynamics without constratins an PID

        if self.ode_implementation == 'jax_pacejka':
            import jax.numpy as jnp
            self.step_dynamics = car_dynamics_pacejka_jax_from_settings
            u = np.array([0, 0])

            car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
            self.car_params_array = car_params.to_np_array()

            jax_state = jnp.array(self.state)
            jax_u = jnp.array(u)
            jax_params = jnp.array(self.car_params_array)
            new_state = self.step_dynamics(jax_state, jax_u, jax_params, self.time_step, intermediate_steps=1)
            self.state = normalize_state_yaw(np.array(new_state, dtype=np.float64))
            
        if self.ode_implementation == 'residual':
            from TrainingLite.dynamic_residual_jax.dynamics_model_residual import DynamicsModelResidual
            self.dynamic_model = DynamicsModelResidual(dt=0.01)

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

        # state and control history (circular buffers of length 40)
        self.state_history = np.zeros((40, NUMBER_OF_STATES))
        self.control_history = np.zeros((40, 2))

    def update_params(self, params):
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        # Use VehicleParameters for consistency
        car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
        self.params = car_params.to_dict()
    
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
        if Settings.GLOBAL_SPEED_LIMIT is not None and len(self.state) > LINEAR_VEL_X_IDX:
            self.state[LINEAR_VEL_X_IDX] = np.clip(self.state[LINEAR_VEL_X_IDX], 0.0, float(Settings.GLOBAL_SPEED_LIMIT))
        
        self.steer_buffer = np.empty((0, ))
        
        # reset state and control history
        self.state_history = np.zeros((40, NUMBER_OF_STATES))
        self.control_history = np.zeros((40, 2))
        # Initialize with current state
        self.state_history[-1] = self.state.copy()

    def update_pose(self, desired_steering_angle, desired_speed):
        """
        Steps the vehicle's physical simulation

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        
        # Some models require PID control - but this is an old PID implementation!
        # acceleration, steering_angular_velocity = pid(desired_speed, desired_steering_angle, self.state[LINEAR_VEL_X_IDX], self.state[POSE_THETA_IDX], self.params['sv_max'], self.params['a_max'], self.params['v_max'], self.params['v_min'])

        # Track the applied control (will be set based on implementation)
        applied_control = np.array([desired_steering_angle, desired_speed])
        
        if(self.ode_implementation == 'pacejka'):
            s = self.state
            u = np.array([desired_steering_angle, desired_speed])
            applied_control = u
            self.state = self.dynamic_model.step(s, u)

        if self.ode_implementation == 'ODE_TF':
            s = np.expand_dims(self.state, 0).astype(np.float32)
            u = np.array([[desired_steering_angle, desired_speed]], dtype=np.float32)

            u_pid = self.car_model.pid(s, u)
            self.u_pid_with_constrains = self.car_model.apply_constrains(s, u_pid)

            self.steering_speed_cmd, self.acceleration_x_cmd = self.car_model.return_control_cmd_components(self.u_pid_with_constrains)

            # Track the actual applied control (after PID and constraints)
            applied_control = self.u_pid_with_constrains[0]

            s = self.car_model.step_dynamics_core(s, self.u_pid_with_constrains)[0]
            self.state = normalize_state_yaw(s)
            

        
        elif self.ode_implementation == 'jax_pacejka':
            import jax.numpy as jnp
            u = np.array([desired_steering_angle, desired_speed])
            applied_control = u
            jax_state = jnp.array(self.state)
            jax_u = jnp.array(u)
            jax_params = jnp.array(self.car_params_array)
            new_state = self.step_dynamics(
                jax_state, jax_u, jax_params, self.time_step, intermediate_steps=1)
            new_state_np = np.array(new_state, dtype=np.float64)
            if np.all(np.isfinite(new_state_np)):
                self.state = normalize_state_yaw(new_state_np)
            else:
                self.state[LINEAR_VEL_X_IDX] = 0.0
                self.state[LINEAR_VEL_Y_IDX] = 0.0
                self.state[ANGULAR_VEL_Z_IDX] = 0.0
            
        elif self.ode_implementation == 'residual':
            import jax.numpy as jnp
            u = np.array([desired_steering_angle, desired_speed])
            applied_control = u
            # Convert to JAX arrays and back to numpy
            jax_state = jnp.array(self.state)
            jax_u = jnp.array(u)

            # Residual network expects a control and state history of 10 timestepa at 0.04s intervals (every 4 timesteps at 0.01s)
            control_history = self.control_history[::4]
            state_history = self.state_history[::4]

            new_state_residual = self.dynamic_model.predict(jax_state, jax_u, dt=0.01, state_history=state_history, control_history=control_history)
            self.state = normalize_state_yaw(np.array(new_state_residual))
            
            
        if self.ode_implementation == 'f1tenth_st':
            raise NotImplementedError("ODE implementation for 'f1tenth_st' is not yet implemented.")

        # clip linear_vel_x to SpeedCap when set
        if Settings.GLOBAL_SPEED_LIMIT is not None:
            self.state[LINEAR_VEL_X_IDX] = np.clip(self.state[LINEAR_VEL_X_IDX], 0.0, float(Settings.GLOBAL_SPEED_LIMIT))

        # Update state and control history (circular buffer)
        self.state_history = np.roll(self.state_history, -1, axis=0)
        self.state_history[-1] = self.state.copy()

        self.control_history = np.roll(self.control_history, -1, axis=0)
        self.control_history[-1] = applied_control

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
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))
        self.sim_index = 0
        self._map_scan_simulator = None
        self.map_collision_margin = 0.005
        
        car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
        # self.params = car_params.to_dict()

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(params, self.seed, is_ego=True)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(params, self.seed)
                self.agents.append(agent)

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

    def set_map_collision_checker(self, scan_simulator, margin=0.005):
        """Attach the shared map distance-transform source used for wall collision."""
        self._map_scan_simulator = scan_simulator
        self.map_collision_margin = margin

    def check_map_collision(self):
        """Latch wall contact using body vertices vs the map distance transform."""
        if Settings.BLANK_MAP or self._map_scan_simulator is None:
            return

        scan_sim = self._map_scan_simulator
        if scan_sim.dt is None:
            return

        length = self.params['length']
        width = self.params['width']
        margin = self.map_collision_margin

        for agent in self.agents:
            pose = np.array(
                [
                    agent.state[POSE_X_IDX],
                    agent.state[POSE_Y_IDX],
                    agent.state[POSE_THETA_IDX],
                ],
                dtype=np.float64,
            )
            if scan_sim.check_body_collision(pose, length, width, margin=margin):
                agent.in_collision = True

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
            pos_x = self.agents[i].state[POSE_X_IDX]
            pos_y = self.agents[i].state[POSE_Y_IDX]
            yaw = self.agents[i].state[POSE_THETA_IDX]
            all_vertices[i, :, :] = get_vertices(np.append([pos_x, pos_y], yaw), self.params['length'], self.params['width'])
        self.collisions, self.collision_idx = collision_multiple(all_vertices)

    def get_sim_observation(self):
        """
        Returns world-sim outputs that are not already on agent objects.

        Car poses live on ``self.agents[i].state``; this dict carries
        collision flags and episode termination only.

        Returns:
            sim_obs (dict): keys ``collisions``, ``terminated``, ``ego_idx``
        """
        return {
            'collisions': self.collisions.copy(),
            'terminated': self.sim_index >= Settings.EXPERIMENT_MAX_LENGTH,
            'ego_idx': self.ego_idx,
        }
    
    def step(self, control_inputs):
        """
        Steps the simulation environment

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents, first column is desired steering angle, second column is desired velocity
        
        Returns:
            sim_obs (dict): collision flags and termination state.
        """

        for i, agent in enumerate(self.agents):
            agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])

            pos_x = agent.state[POSE_X_IDX]
            pos_y = agent.state[POSE_Y_IDX]
            yaw = agent.state[POSE_THETA_IDX]
            self.agent_poses[i, :] = np.append([pos_x, pos_y], yaw)

        self.check_map_collision()
        self.check_collision()

        for i, agent in enumerate(self.agents):
            if agent.in_collision:
                self.collisions[i] = 1.0

        sim_obs = self.get_sim_observation()
        self.sim_index += 1
        return sim_obs

    def reset(self, initial_states):
        """
        Resets the simulation environment by given initial_states.

        Args:
            initial_states (np.ndarray (num_agents, 3)): initial_states to reset agents to

        Returns:
            sim_obs (dict): initial world-sim observation dictionary
        """

        self.sim_index = 0
        if initial_states.shape[0] != self.num_agents:
            raise ValueError('Number of initial_states for reset does not match number of agents.')

        # Reset all agents
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))

        for i in range(self.num_agents):
            self.agents[i].reset(initial_states[i, :])

        return self.get_sim_observation()
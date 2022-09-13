import numpy as np
import tensorflow as tf
import torch

from typing import Optional, Tuple, Union
from gym import spaces

from utilities.Settings import Settings
from Control_Toolkit.others.environment import EnvironmentBatched, TensorFlowLibrary
from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    LINEAR_VEL_X_IDX,
    ANGULAR_VEL_Z_IDX,
    SLIP_ANGLE_IDX,
    STEERING_ANGLE_IDX,
    ANGULAR_CONTROL_IDX,
    TRANSLATIONAL_CONTROL_IDX,
)

from Control_Toolkit_ASF.CostFunctions.racing import racing

class f1t_model(EnvironmentBatched):

    """Accepts batches of data to environment

        :param Continuous_MountainCarEnv: _description_
        :type Continuous_MountainCarEnv: _type_
        """

    num_actions = 2
    num_states = ...  # FIXME: Car model can have a various number of states, depending on how precise you model it...

    def __init__(
            self,
            batch_size=1,
            computation_lib=TensorFlowLibrary,
            **kwargs
    ):

        self.config = {
            **kwargs,
        }

        self._state = None  # here just a placeholder, change line below
        self.state = None

        self.dt = kwargs["dt"]
        intermediate_steps = kwargs["intermediate_steps"]

        self._batch_size = batch_size

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = racing(self)

        # Added for F1TENTH

        self.model_of_car_dynamics = kwargs["model_of_car_dynamics"]

        clip_control_input = kwargs['CLIP_CONTROL_INPUT']
        if isinstance(clip_control_input[0], list):
            clip_control_input_low = self.lib.constant(clip_control_input[0], self.lib.float32)
            clip_control_input_high = self.lib.constant(clip_control_input[1], self.lib.float32)
        else:
            clip_control_input_high = self.lib.constant(clip_control_input, self.lib.float32)
            clip_control_input_low = -clip_control_input_high

        self.action_space = spaces.Box(
            np.array([clip_control_input_low]),
            np.array([clip_control_input_high]),
            dtype=np.float32,
        )  # Action space for [throttle, steer]

        # low = np.array([-1.0, -1.0, -4.0, -0.5])  # low range of observation space
        # high = np.array([1.0, 1.0, 4.0, 0.5])  # high range of observation space
        # self.observation_space = spaces.Box(
        #     low, high, dtype=np.float32
        # )  # Observation space for [x, y, theta]

        self._LIDAR = None  # here just a placeholder, change line below

        self._waypoints = None  # here just a placeholder, change line below

        self._target_position = None  # here just a placeholder, change line below

        if Settings.ONLY_ODOMETRY_AVAILABLE:
            self.next_step_output = self.next_state_output_odom
        else:
            self.next_step_output = self.next_state_output_full_state

        self.intermediate_steps = self.lib.to_tensor(intermediate_steps, self.lib.int32)
        self.intermediate_steps_float = self.lib.to_tensor(intermediate_steps, self.lib.float32)
        self.t_step = self.lib.to_tensor(self.dt / float(self.intermediate_steps), self.lib.float32)

        if self.model_of_car_dynamics == 'ODE:simple':
            self._step_dynamics = self._step_dynamics_simple
        elif self.model_of_car_dynamics == 'ODE:ks':
            self._step_dynamics = self._step_dynamics_ks
        elif self.model_of_car_dynamics == 'ODE:st':
            self.step_dynamics = self._step_dynamics_st
        elif self.model_of_car_dynamics == 'Neural Network':
            raise NotImplementedError('Neural network model for F1TENTH is not implemented yet')

    # region Updating variables

    # region Updating state

    @property
    def state(self):
        return self._state

    @property
    def state_tf(self):
        return self._state_tf

    @state.setter
    def state(self, state):
        self._state = state
        if state is None:
            pass
        elif not hasattr(self, "_state_tf"):
            self._state_tf = tf.Variable(state, dtype=tf.float32)
        else:
            self._state_tf.assign(state)

    # endregion

    # region Updating LIDAR
    
    @property
    def LIDAR(self):
        return self._LIDAR

    @property
    def LIDAR_tf(self):
        return self._LIDAR_tf

    @LIDAR.setter
    def LIDAR(self, LIDAR):
        self._LIDAR = LIDAR
        if not hasattr(self, "_LIDAR_tf"):
            self._LIDAR_tf = tf.Variable(LIDAR, dtype=tf.float32)
        else:
            self._LIDAR_tf.assign(LIDAR)
    
    # endregion

    # region Updating waypoints

    @property
    def waypoints(self):
        return self._waypoints

    @property
    def waypoints_tf(self):
        return self._waypoints_tf

    @waypoints.setter
    def waypoints(self, waypoints):
        self._waypoints = waypoints
        if not hasattr(self, "_waypoints_tf"):
            self._waypoints_tf = tf.Variable(waypoints, dtype=tf.float32)
        else:
            self._waypoints_tf.assign(waypoints)

    # endregion

    # region Updating target position
    @property
    def target_position(self):
        return self._target_position

    @property
    def target_position_tf(self):  # FIXME: What with torch?
        return self._target_position_tf

    @target_position.setter
    def target_position(self, target_position):
        self._target_position = target_position
        if not hasattr(self, "_target_position_tf"):
            self._target_position_tf = tf.Variable(target_position, dtype=tf.float32)  # FIXME: This kind of variable is not "declared" , neither in cartpole
        else:
            self._target_position_tf.assign(target_position)
            
    # endregion

    # endregion

    def step_dynamics(
            self,
            state: Union[np.ndarray, tf.Tensor, torch.Tensor],
            action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Union[np.ndarray, tf.Tensor, torch.Tensor]:

        state, action = self._expand_arrays(state, action)

        state = self._step_dynamics(state, action)

        return state

    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        # Perturb action if not in planning mode
        # TODO: Set explicitly whether in planning or simulation mode, not infer from batch size
        if self._batch_size == 1:
            action = self._apply_actuator_noise(action)

        self.state = self._step_dynamics(self.state, action)

        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        if self._batch_size == 1:
            self.renderer.render_step()
            return (
                self.lib.to_numpy(self.lib.squeeze(self.state)),
                float(reward),
                bool(done),
                {},
            )

        return self.state, reward, done, {}

    def reset(
            self,
            state: np.ndarray,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        if seed is not None:
            self._set_up_rng(seed)


        if self.lib.ndim(state) < 2:
            state = self.lib.unsqueeze(
                self.lib.to_tensor(state, self.lib.float32), 0
            )
        if self.lib.shape(state)[0] == 1:
            self.state = self.lib.tile(state, (self._batch_size, 1))
        else:
            self.state = state

        return self._get_reset_return_val()

    def render(self, mode="human"):
        pass

    def is_done(self, state):
        return False

    def get_reward(self, state, action):
        """
        This is a single step (stage cost) reward.
        Not used currently in F1TENTH.
        Instead, see the corresponding cost_functions_wrapper
        """
        reward = None
        return reward  # reward = - cost

    # region Various dynamical models for a car

    def _step_dynamics_simple(self, s, Q, params):
        '''
        Parallaley executes steps frim initial state s[i] with control input Q[i] for every i
        @param s: (batch_size, len(state)) all initial states for every step
        @param s: (batch_size, len(control_input)) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (batch_size, len(state)) all nexts states
        '''

        number_of_rollouts = self.lib.shape(s)[0]

        pose_x = s[:, POSE_X_IDX]
        pose_y = s[:, POSE_Y_IDX]
        pose_theta = s[:, POSE_THETA_IDX]

        speed = Q[:, TRANSLATIONAL_CONTROL_IDX]
        steering = Q[:, ANGULAR_CONTROL_IDX]

        for _ in range(self.intermediate_steps):
            pose_theta = pose_theta + 0.5 * (steering / self.intermediate_steps_float)
            pose_x = pose_x + self.t_step * speed * self.lib.cos(pose_theta)
            pose_y = pose_y + self.t_step * speed * self.lib.sin(pose_theta)

        angular_vel_z = self.lib.zeros([number_of_rollouts])
        linear_vel_x = self.lib.zeros([number_of_rollouts])
        pose_theta = pose_theta
        pose_theta_cos = self.lib.cos(pose_theta)
        pose_theta_sin = self.lib.sin(pose_theta)
        pose_x = pose_x
        pose_y = pose_y
        slip_angle = self.lib.zeros([number_of_rollouts])
        steering_angle = self.lib.zeros([number_of_rollouts])

        return self.next_step_output(angular_vel_z,
                                     linear_vel_x,
                                     pose_theta,
                                     pose_theta_cos,
                                     pose_theta_sin,
                                     pose_x,
                                     pose_y,
                                     slip_angle,
                                     steering_angle)

    def _step_dynamics_ks(self, s, Q, params):
        '''
        Parallaley executes steps frim initial state s[i] with control input Q[i] for every i
        @param s: (batch_size, len(state)) all initial states for every step
        @param s: (batch_size, len(control_input)) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (batch_size, len(state)) all nexts states
        '''
        lf = 0.15875
        lr = 0.17145
        lwb = lf + lr

        # Dimensions
        number_of_rollouts = self.lib.shape(s)[0]

        s_x = s[:, POSE_X_IDX]  # Pose X
        s_y = s[:, POSE_Y_IDX]  # Pose Y
        delta = s[:, STEERING_ANGLE_IDX]  # Fron Wheel steering angle
        theta = s[:, LINEAR_VEL_X_IDX]  # Speed
        psi = s[:, POSE_THETA_IDX]  # Yaw Angle

        delta_dot = Q[:, ANGULAR_CONTROL_IDX]  # steering angle velocity of front wheels
        theta_dot = Q[:, TRANSLATIONAL_CONTROL_IDX]  # longitudinal acceleration

        # Constaints
        theta_dot = self.accl_constraints(theta, theta_dot)
        delta_dot = self.steering_constraints(delta, delta_dot)

        # Euler stepping
        for _ in range(self.intermediate_steps):
            s_x_dot = theta * self.lib.cos(psi)
            s_y_dot = theta * self.lib.sin(psi)
            # delta_dot = delta_dot
            # theta_dot = theta_dot
            psi_dot = (theta / lwb) * self.lib.tan(delta)

            s_x = s_x + self.t_step * s_x_dot
            s_y = s_y + self.t_step * s_y_dot
            delta = delta + self.t_step * delta_dot
            theta = theta + self.t_step * theta_dot
            psi = psi + self.t_step * psi_dot

        angular_vel_z = self.lib.zeros([number_of_rollouts])
        linear_vel_x = theta
        pose_theta = psi
        pose_theta_cos = self.lib.cos(pose_theta)
        pose_theta_sin = self.lib.sin(pose_theta)
        pose_x = s_x
        pose_y = s_y
        slip_angle = self.lib.zeros([number_of_rollouts])
        steering_angle = delta

        return self.next_step_output(angular_vel_z,
                                     linear_vel_x,
                                     pose_theta,
                                     pose_theta_cos,
                                     pose_theta_sin,
                                     pose_x,
                                     pose_y,
                                     slip_angle,
                                     steering_angle)

    def _step_dynamics_st(self, s, Q, params):
        '''
        Parallaley executes steps frim initial state s[i] with control input Q[i] for every i
        @param s: (batch_size, len(state)) all initial states for every step
        @param s: (batch_size, len(control_input)) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (batch_size, len(state)) all nexts states
        '''

        # params
        mu = 1.0489  # friction coefficient  [-]
        C_Sf = 4.718  # cornering stiffness front [1/rad]
        C_Sr = 5.4562  # cornering stiffness rear [1/rad]
        lf = 0.15875  # distance from venter of gracity to front axle [m]
        lr = 0.17145  # distance from venter of gracity to rear axle [m]
        h = 0.074  # center of gravity height of toal mass [m]
        m = 3.74  # Total Mass of car [kg]
        I = 0.04712  # Moment of inertia for entire mass about z axis  [kgm^2]
        g = 9.81

        # State
        s_x = s[:, POSE_X_IDX]  # Pose X
        s_y = s[:, POSE_Y_IDX]  # Pose Y
        delta = s[:, STEERING_ANGLE_IDX]  # Fron Wheel steering angle
        theta = s[:, LINEAR_VEL_X_IDX]  # Speed
        psi = s[:, POSE_THETA_IDX]  # Yaw Angle
        psi_dot = s[:, ANGULAR_VEL_Z_IDX]  # Yaw Rate
        beta = s[:, SLIP_ANGLE_IDX]  # Slipping Angle

        # Control Input
        delta_dot = Q[:, 1]  # steering angle velocity of front wheels
        theta_dot = Q[:, 0]  # longitudinal acceleration

        # Constaints
        theta_dot = self.accl_constraints(theta, theta_dot)
        delta_dot = self.steering_constraints(delta, delta_dot)

        # switch to kinematic model for small velocities
        min_speed_st = 0.1
        speed_too_low_for_st_indices = self.lib.less(theta, min_speed_st)
        speed_not_too_low_for_st_indices = self.lib.logical_not(speed_too_low_for_st_indices)

        speed_too_low_for_st_indices = self.lib.cast(speed_too_low_for_st_indices, self.lib.float32)
        speed_not_too_low_for_st_indices = self.lib.cast(speed_not_too_low_for_st_indices, self.lib.float32)

        # TODO: Use ks model for slow speed

        for _ in range(self.intermediate_steps):
            s_x_dot = theta * self.lib.cos(psi + beta)
            s_y_dot = theta * self.lib.sin(psi + beta)

            # delta_dot = delta_dot
            # theta_dot = theta_dot
            # psi_dot = psi_dot

            psi_dot_dot = -mu * m / (theta * I * (lr + lf)) * (
                    lf ** 2 * C_Sf * (g * lr - theta_dot * h) + lr ** 2 * C_Sr * (g * lf + theta_dot * h)) * psi_dot \
                          + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + theta_dot * h) - lf * C_Sf * (
                    g * lr - theta_dot * h)) * beta \
                          + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - theta_dot * h) * delta

            beta_dot = (mu / (theta ** 2 * (lr + lf)) * (
                    C_Sr * (g * lf + theta_dot * h) * lr - C_Sf * (g * lr - theta_dot * h) * lf) - 1) * psi_dot \
                       - mu / (theta * (lr + lf)) * (
                               C_Sr * (g * lf + theta_dot * h) + C_Sf * (g * lr - theta_dot * h)) * beta \
                       + mu / (theta * (lr + lf)) * (C_Sf * (g * lr - theta_dot * h)) * delta

            s_x = s_x + self.t_step * s_x_dot
            s_y = s_y + self.t_step * s_y_dot
            delta = delta + self.t_step * delta_dot
            theta = theta + self.t_step * theta_dot
            psi = psi + self.t_step * psi_dot
            psi_dot = psi_dot + self.t_step * psi_dot_dot
            beta = beta + self.t_step * beta_dot

        angular_vel_z = psi_dot
        linear_vel_x = theta
        pose_theta = psi
        pose_theta_cos = self.lib.cos(pose_theta)
        pose_theta_sin = self.lib.sin(pose_theta)
        pose_x = s_x
        pose_y = s_y
        slip_angle = beta
        steering_angle = delta

        return self.next_step_output(angular_vel_z,
                                     linear_vel_x,
                                     pose_theta,
                                     pose_theta_cos,
                                     pose_theta_sin,
                                     pose_x,
                                     pose_y,
                                     slip_angle,
                                     steering_angle)

    def steering_constraints(self, steering_angle, steering_velocity):
        s_min = self.lib.constant([-0.4189], self.lib.float32)
        s_max = self.lib.constant([0.4189], self.lib.float32)
        sv_min = self.lib.constant([-3.2], self.lib.float32)
        sv_max = self.lib.constant([3.2], self.lib.float32)

        # Steering angle constraings
        steering_angle_not_too_low_indices = self.lib.greater(steering_angle, s_min)
        steering_angle_not_too_low_indices = self.lib.cast(steering_angle_not_too_low_indices, self.lib.float32)

        steering_angle_not_too_high_indices = self.lib.less(steering_angle, s_max)
        steering_angle_not_too_high_indices = self.lib.cast(steering_angle_not_too_high_indices, self.lib.float32)

        steering_velocity = steering_angle_not_too_low_indices * steering_velocity
        steering_velocity = steering_angle_not_too_high_indices * steering_velocity

        # Steering velocity is constrainted
        steering_velocity = self.lib.clip(steering_velocity, sv_min, sv_max)


        return steering_velocity

    def accl_constraints(self, vel, accl):
        v_switch = self.lib.constant([7.319], self.lib.float32)
        a_max = self.lib.constant([9.51], self.lib.float32)
        v_min = self.lib.constant([-5.0], self.lib.float32)
        v_max = self.lib.constant([20.0], self.lib.float32)

        # positive accl limit
        velocity_too_high_indices = self.lib.greater(vel, v_switch)
        velocity_not_too_high_indices = self.lib.logical_not(velocity_too_high_indices)
        velocity_too_high_indices = self.lib.cast(velocity_too_high_indices, self.lib.float32)
        velocity_not_too_high_indices = self.lib.cast(velocity_not_too_high_indices, self.lib.float32)

        pos_limit_velocity_too_high = (a_max * v_switch) / vel
        pos_limit_velocity_not_too_high = a_max

        pos_limit = (velocity_too_high_indices * pos_limit_velocity_too_high) + (
                velocity_not_too_high_indices * pos_limit_velocity_not_too_high)

        # accl limit reached?
        velocity_not_over_max_indices = self.lib.less(vel, v_max)
        velocity_not_over_max_indices = self.lib.cast(velocity_not_over_max_indices, self.lib.float32)

        velocity_not_under_min_indices = self.lib.greater(vel, v_min)
        velocity_not_under_min_indices = self.lib.cast(velocity_not_under_min_indices, self.lib.float32)

        accl = velocity_not_over_max_indices * accl
        accl = velocity_not_under_min_indices * accl

        accl = self.lib.clip(accl, a_max, pos_limit)

        return accl

    # endregion

    # region Formatting output of the step function

    def next_state_output_odom(self,
                               angular_vel_z,
                               linear_vel_x,
                               pose_theta,
                               pose_theta_cos,
                               pose_theta_sin,
                               pose_x,
                               pose_y,
                               slip_angle,
                               steering_angle):
        return self.lib.stack([
            angular_vel_z,
            linear_vel_x,
            pose_theta,
            pose_theta_cos,
            pose_theta_sin,
            pose_x,
            pose_y,
        ], axis=1)

    def next_state_output_full_state(self,
                                     angular_vel_z,
                                     linear_vel_x,
                                     pose_theta,
                                     pose_theta_cos,
                                     pose_theta_sin,
                                     pose_x,
                                     pose_y,
                                     slip_angle,
                                     steering_angle):
        return self.lib.stack([
            angular_vel_z,
            linear_vel_x,
            pose_theta,
            pose_theta_cos,
            pose_theta_sin,
            pose_x,
            pose_y,
            slip_angle,
            steering_angle
        ], axis=1)

    # endregion


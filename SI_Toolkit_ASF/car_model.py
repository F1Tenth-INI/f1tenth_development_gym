import tensorflow as tf

from SI_Toolkit.computation_library import TensorFlowLibrary
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

class car_model:

    num_actions = 2
    num_states = 9

    def __init__(
            self,
            model_of_car_dynamics: str,
            with_pid: bool,
            dt: float = 0.025,
            intermediate_steps=1,
            computation_lib=TensorFlowLibrary,
            **kwargs
    ):

        self.lib = computation_lib

        self.model_of_car_dynamics = model_of_car_dynamics
        self.with_pid = with_pid
        self.step_dynamics = None
        self.set_model_of_car_dynamics(model_of_car_dynamics)

        self.next_step_output = self.next_state_output_full_state

        self.dt = dt
        self.intermediate_steps = self.lib.to_tensor(intermediate_steps, self.lib.int32)
        self.intermediate_steps_float = self.lib.to_tensor(intermediate_steps, self.lib.float32)
        self.t_step = self.lib.to_tensor(self.dt / float(self.intermediate_steps), self.lib.float32)

    # region Various dynamical models for a car

    def set_model_of_car_dynamics(self, model_of_car_dynamics):
        if model_of_car_dynamics == 'ODE:simple':
            self.step_dynamics = self._step_dynamics_simple
        elif model_of_car_dynamics == 'ODE:ks':
            self.step_dynamics = self._step_dynamics_ks
        elif model_of_car_dynamics == 'ODE:st':
            if self.with_pid:
                self.step_dynamics = self._step_st_with_servo_and_motor_pid
            else:
                self.step_dynamics = self._step_dynamics_st
        else:
            raise NotImplementedError(
                '{} not recognized as a valid name for a model of car dynamics'.format(model_of_car_dynamics))

        self.model_of_car_dynamics = model_of_car_dynamics



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

        accl = self.lib.clip(accl, -a_max, pos_limit)

        return accl

    '''
    Extend the ST model with the simplified proportional Servo function (the physical car needs a desired angle as input)
    We need to consider the servo function as part of the car model

    The proportional servo function is adapted from the Gym simulator and defined as
    steering velocity = steer_diff / 0.1  if steer_diff > 0.0001 , otherwise 0

    @param desired_steering_angle: (batch_size) desired steering angle for the car
    @param current_steering_angle: (1) current steering angle for the car
    returns steering_velocity: steering velocity applied on the wheels
    '''

    def servo_proportional(self, desired_steering_angle, current_steering_angle):

        steering_angle_difference = desired_steering_angle - current_steering_angle

        steering_angle_difference_not_too_low_indices = tf.math.greater(tf.math.abs(steering_angle_difference), 0.01)
        steering_angle_difference_not_too_low_indices = tf.cast(steering_angle_difference_not_too_low_indices,
                                                                tf.float32)

        steering_velocity = steering_angle_difference_not_too_low_indices * (steering_angle_difference / 0.1)

        return steering_velocity

    def motor_controller_pid(self, desired_speed, current_speed):

        a_max = tf.constant([9.51])
        v_min = tf.constant([-5.0])
        v_max = tf.constant([20.0])

        speed_difference = desired_speed - current_speed

        forward_indices = tf.cast(tf.math.greater(tf.math.abs(current_speed), 0.0), tf.float32)
        backward_indices = tf.cast(tf.math.less(tf.math.abs(current_speed), 0.0), tf.float32)

        forward_accelerating_indices = forward_indices * tf.cast(tf.math.greater(tf.math.abs(speed_difference), 0.0),
                                                                 tf.float32)
        forward_breaking_indices = forward_indices * tf.cast(tf.math.less(tf.math.abs(speed_difference), 0.0),
                                                             tf.float32)

        backward_accelerating_indices = backward_indices * tf.cast(tf.math.less(tf.math.abs(speed_difference), 0.0),
                                                                   tf.float32)
        backward_breaking_indices = backward_indices * tf.cast(tf.math.greater(tf.math.abs(speed_difference), 0.0),
                                                               tf.float32)

        # fwd accl
        kp = 10.0 * a_max / v_max
        forward_acceleration = kp * forward_accelerating_indices * speed_difference

        # fwd break
        kp = 10.0 * a_max / (-v_min)
        forward_breaking = kp * forward_breaking_indices * speed_difference

        # bkw accl
        kp = 2.0 * a_max / (-v_min)
        backward_acceleration = kp * backward_accelerating_indices * speed_difference

        # bkw break
        kp = 2.0 * a_max / v_max
        backward_breaking = kp * backward_breaking_indices * speed_difference

        total_acceleration = forward_acceleration + forward_breaking + backward_acceleration + backward_breaking

        return total_acceleration

    def _step_st_with_servo_and_motor_pid(self, s, Q, params):
        # Control Input (desired speed, desired steering angle)
        desired_speed = Q[:, 0]  # longitudinal acceleration
        desired_angle = Q[:, 1]  # steering angle velocity of front wheels

        delta = s[:, STEERING_ANGLE_IDX]  # Fron Wheel steering angle
        vel_x = s[:, LINEAR_VEL_X_IDX]  # Speed

        delta_dot = self.servo_proportional(desired_angle, delta)
        vel_x_dot = self.motor_controller_pid(desired_speed, vel_x)

        Q_pid = tf.transpose(tf.stack([vel_x_dot, delta_dot]))

        return self._step_dynamics_st(s, Q_pid, params)

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

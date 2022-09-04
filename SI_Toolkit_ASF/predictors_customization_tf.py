import tensorflow as tf

from SI_Toolkit.TF.TF_Functions.Compile import Compile
import numpy as np

import yaml

from utilities.state_utilities import *

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

def next_state_output_odom(angular_vel_z,
                           linear_vel_x,
                           pose_theta,
                           pose_theta_cos,
                           pose_theta_sin,
                           pose_x,
                           pose_y,
                           slip_angle,
                           steering_angle):
    return tf.stack([
        angular_vel_z,
        linear_vel_x,
        pose_theta,
        pose_theta_cos,
        pose_theta_sin,
        pose_x,
        pose_y,
    ], axis=1)


def next_state_output_full_state(angular_vel_z,
                                 linear_vel_x,
                                 pose_theta,
                                 pose_theta_cos,
                                 pose_theta_sin,
                                 pose_x,
                                 pose_y,
                                 slip_angle,
                                 steering_angle):
    return tf.stack([
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


class next_state_predictor_ODE_tf():

    def __init__(self, dt, intermediate_steps, disable_individual_compilation=False):
        self.s = None

        self.intermediate_steps = tf.convert_to_tensor(intermediate_steps, dtype=tf.int32)
        self.intermediate_steps_float = tf.convert_to_tensor(intermediate_steps, dtype=tf.float32)
        self.t_step = tf.convert_to_tensor(dt / float(self.intermediate_steps), dtype=tf.float32)

        self.t_step = self.t_step

        if Settings.ONLY_ODOMETRY_AVAILABLE:
            self.next_step_output = next_state_output_odom
        else:
            self.next_step_output = next_state_output_full_state

        if disable_individual_compilation:
            # self.step = self._step
            self.step = self._step_st
        else:
            self.step = Compile(self._step_st)

    # @Compile
    def _step(self, s, Q, params):
        '''
        Parallaley executes steps frim initial state s[i] with control input Q[i] for every i
        @param s: (batch_size, len(state)) all initial states for every step
        @param s: (batch_size, len(control_input)) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (batch_size, len(state)) all nexts states
        '''

        number_of_rollouts = tf.shape(s)[0]

        pose_x = s[:, POSE_X_IDX]
        pose_y = s[:, POSE_Y_IDX]
        pose_theta = s[:, POSE_THETA_IDX]

        speed = Q[:, TRANSLATIONAL_CONTROL_IDX]
        steering = Q[:, ANGULAR_CONTROL_IDX]

        for _ in tf.range(self.intermediate_steps):
            pose_theta = pose_theta + 0.5 * (steering / self.intermediate_steps_float)
            pose_x = pose_x + self.t_step * speed * tf.math.cos(pose_theta)
            pose_y = pose_y + self.t_step * speed * tf.math.sin(pose_theta)

        angular_vel_z = tf.zeros([number_of_rollouts])
        linear_vel_x = tf.zeros([number_of_rollouts])
        pose_theta = pose_theta
        pose_theta_cos = tf.math.cos(pose_theta)
        pose_theta_sin = tf.math.sin(pose_theta)
        pose_x = pose_x
        pose_y = pose_y
        slip_angle = tf.zeros([number_of_rollouts])
        steering_angle = tf.zeros([number_of_rollouts])

        return self.next_step_output(angular_vel_z,
                                     linear_vel_x,
                                     pose_theta,
                                     pose_theta_cos,
                                     pose_theta_sin,
                                     pose_x,
                                     pose_y,
                                     slip_angle,
                                     steering_angle)

    def _step_ks(self, s, Q, params):
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
        number_of_rollouts = tf.shape(s)[0]

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
        for _ in tf.range(self.intermediate_steps):
            s_x_dot = tf.multiply(theta, tf.cos(psi))
            s_y_dot = tf.multiply(theta, tf.sin(psi))
            # delta_dot = delta_dot
            # theta_dot = theta_dot
            psi_dot = tf.divide(theta, lwb) * tf.tan(delta)

            s_x = s_x + self.t_step * s_x_dot
            s_y = s_y + self.t_step * s_y_dot
            delta = delta + self.t_step * delta_dot
            theta = theta + self.t_step * theta_dot
            psi = psi + self.t_step * psi_dot

        angular_vel_z = tf.zeros([number_of_rollouts])
        linear_vel_x = theta
        pose_theta = psi
        pose_theta_cos = tf.math.cos(pose_theta)
        pose_theta_sin = tf.math.sin(pose_theta)
        pose_x = s_x
        pose_y = s_y
        slip_angle = tf.zeros([number_of_rollouts])
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

    def _step_st(self, s, Q, params):
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
        speed_too_low_for_st_indices = tf.math.less(theta, min_speed_st)
        speed_not_too_low_for_st_indices = tf.math.logical_not(speed_too_low_for_st_indices)

        speed_too_low_for_st_indices = tf.cast(speed_too_low_for_st_indices, tf.float32)
        speed_not_too_low_for_st_indices = tf.cast(speed_not_too_low_for_st_indices, tf.float32)

        # TODO: Use ks model for slow speed

        for _ in tf.range(self.intermediate_steps):
            s_x_dot = tf.multiply(theta, tf.cos(tf.add(psi, beta)))
            s_y_dot = tf.multiply(theta, tf.sin(tf.add(psi, beta)))

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
        pose_theta_cos = tf.math.cos(pose_theta)
        pose_theta_sin = tf.math.sin(pose_theta)
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
        s_min = tf.constant([-0.4189])
        s_max = tf.constant([0.4189])
        sv_min = tf.constant([-3.2])
        sv_max = tf.constant([3.2])

        # Steering angle constraings
        steering_angle_not_too_low_indices = tf.math.greater(steering_angle, s_min)
        steering_angle_not_too_low_indices = tf.cast(steering_angle_not_too_low_indices, tf.float32)

        steering_angle_not_too_high_indices = tf.math.less(steering_angle, s_max)
        steering_angle_not_too_high_indices = tf.cast(steering_angle_not_too_high_indices, tf.float32)

        steering_velocity = tf.multiply(steering_angle_not_too_low_indices, steering_velocity)
        steering_velocity = tf.multiply(steering_angle_not_too_high_indices, steering_velocity)

        # Steering velocity is constrainted
        steering_velocity = tf.clip_by_value(steering_velocity, clip_value_min=sv_min, clip_value_max=sv_max)


        return steering_velocity

    def accl_constraints(self, vel, accl):
        v_switch = tf.constant([7.319])
        a_max = tf.constant([9.51])
        v_min = tf.constant([-5.0])
        v_max = tf.constant([20.0])

        # positive accl limit
        velocity_too_high_indices = tf.math.greater(vel, v_switch)
        velocity_not_too_high_indices = tf.math.logical_not(velocity_too_high_indices)
        velocity_too_high_indices = tf.cast(velocity_too_high_indices, tf.float32)
        velocity_not_too_high_indices = tf.cast(velocity_not_too_high_indices, tf.float32)

        pos_limit_velocity_too_high = tf.math.divide(a_max * v_switch, vel)
        pos_limit_velocity_not_too_high = a_max

        pos_limit = tf.multiply(velocity_too_high_indices, pos_limit_velocity_too_high) + tf.multiply(
            velocity_not_too_high_indices, pos_limit_velocity_not_too_high)

        # accl limit reached?
        velocity_not_over_max_indices = tf.math.less(vel, v_max)
        velocity_not_over_max_indices = tf.cast(velocity_not_over_max_indices, tf.float32)

        velocity_not_under_min_indices = tf.math.greater(vel, v_min)
        velocity_not_under_min_indices = tf.cast(velocity_not_under_min_indices, tf.float32)

        accl = tf.multiply(velocity_not_over_max_indices, accl)
        accl = tf.multiply(velocity_not_under_min_indices, accl)

        accl = tf.clip_by_value(accl, clip_value_min=-a_max, clip_value_max=10000)
        accl = tf.clip_by_value(accl, clip_value_min=-100000, clip_value_max=pos_limit)

        return accl






class predictor_output_augmentation_tf:
    def __init__(self, net_info, disable_individual_compilation=False):
        self.net_output_indices = {key: value for value, key in enumerate(net_info.outputs)}
        indices_augmentation = []
        features_augmentation = []
        if 'angular_vel_z' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['angular_vel_z'])
            features_augmentation.append('angular_vel_z')
        if 'linear_vel_x' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['linear_vel_x'])
            features_augmentation.append('linear_vel_x')
        if 'linear_vel_y' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['linear_vel_y'])
            features_augmentation.append('linear_vel_y')

        if 'pose_theta' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['pose_theta'])
            features_augmentation.append('pose_theta')
        if 'pose_theta_cos' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['pose_theta_cos'])
            features_augmentation.append('pose_theta_cos')
        if 'pose_theta_sin' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['pose_theta_sin'])
            features_augmentation.append('pose_theta_sin')

        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

        if 'pose_theta' in net_info.outputs:
            self.index_pose_theta = tf.convert_to_tensor(self.net_output_indices['pose_theta'])
        if 'pose_theta_sin' in net_info.outputs:
            self.index_pose_theta_sin = tf.convert_to_tensor(self.net_output_indices['pose_theta_sin'])
        if 'pose_theta_cos' in net_info.outputs:
            self.index_pose_theta_cos = tf.convert_to_tensor(self.net_output_indices['pose_theta_cos'])

        if disable_individual_compilation:
            self.augment = self._augment
        else:
            self.augment = Compile(self._augment)

    def get_indices_augmentation(self):
        return self.indices_augmentation

    def get_features_augmentation(self):
        return self.features_augmentation

    def _augment(self, net_output):

        output = net_output
        if 'angular_vel_z' in self.features_augmentation:
            angular_vel_z = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, angular_vel_z], axis=-1)
        if 'linear_vel_x' in self.features_augmentation:
            linear_vel_x = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, linear_vel_x], axis=-1)
        if 'linear_vel_y' in self.features_augmentation:
            linear_vel_y = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, linear_vel_y], axis=-1)

        if 'pose_theta' in self.features_augmentation:
            pose_theta = tf.math.atan2(
                net_output[..., self.index_pose_theta_sin],
                net_output[..., self.index_pose_theta_cos])[:, :,
                         tf.newaxis]  # tf.math.atan2 removes the features (last) dimension, so it is added back with [:, :, tf.newaxis]
            output = tf.concat([output, pose_theta], axis=-1)

        if 'angle_sin' in self.features_augmentation:
            pose_theta_sin = \
                tf.sin(net_output[..., self.index_pose_theta])[:, :, tf.newaxis]
            output = tf.concat([output, pose_theta_sin], axis=-1)

        if 'angle_cos' in self.features_augmentation:
            pose_theta_cos = \
                tf.cos(net_output[..., self.index_pose_theta])[:, :, tf.newaxis]
            output = tf.concat([output, pose_theta_cos], axis=-1)

        return output

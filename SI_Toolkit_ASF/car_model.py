import yaml
import math
import tensorflow as tf
from utilities.path_helper_ros import *
from utilities.car_files.vehicle_parameters import VehicleParameters
from SI_Toolkit.computation_library import TensorFlowLibrary,NumpyLibrary

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
    num_states  = 9

    def __init__(
            self,
            model_of_car_dynamics: str,
            batch_size: int,
            car_parameter_file: str, # file containing the car parameters
            dt: float = 0.03,
            intermediate_steps=1,
            computation_lib=TensorFlowLibrary,
            wrap_angle=True,
            **kwargs
    ):
        self.lib = computation_lib
        self.wrap_angle = wrap_angle

        # config = yaml.load(open(os.path.join(gym_path, "config.yml"), "r"), Loader=yaml.FullLoader)
        # self.car_parameters = yaml.load(open(os.path.join(gym_path,car_parameter_file), "r"), Loader=yaml.FullLoader)
        self.car_parameters =  VehicleParameters(car_parameter_file)
        
        self.s_min = self.lib.constant(self.car_parameters.s_min, self.lib.float32)
        self.s_max = self.lib.constant(self.car_parameters.s_max, self.lib.float32)
        self.sv_min = self.lib.constant(self.car_parameters.sv_min, self.lib.float32)
        self.sv_max = self.lib.constant(self.car_parameters.sv_max, self.lib.float32)
        
        self.v_switch = self.lib.constant(self.car_parameters.v_switch, self.lib.float32)
        self.a_max = self.lib.constant(self.car_parameters.a_max, self.lib.float32)
        self.a_min = self.lib.constant(self.car_parameters.a_min, self.lib.float32)
        self.v_min = self.lib.constant(self.car_parameters.v_min, self.lib.float32)
        self.v_max = self.lib.constant(self.car_parameters.v_max, self.lib.float32)
        
        lf = self.car_parameters.lf  # distance from venter of gracity to front axle [m]
        lr = self.car_parameters.lr  # distance from venter of gracity to rear axle [m]
        self.lwb = self.lib.constant(lf + lr, self.lib.float32)
      
        self.model_of_car_dynamics = model_of_car_dynamics
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
            self.step_dynamics = self._step_model_with_servo_and_motor_pid_(self._step_dynamics_simple)
           
        elif model_of_car_dynamics == 'ODE:ks':
            self.step_dynamics = self._step_model_with_servo_and_motor_pid_(self._step_dynamics_ks)
          
        elif model_of_car_dynamics == 'ODE:st':
            self.step_dynamics = self._step_model_with_servo_and_motor_pid_(self._step_dynamics_st)
            
        elif model_of_car_dynamics == 'ODE:pacejka':
            self.step_dynamics = self._step_model_with_servo_and_motor_pid_(self._step_dynamics_pacejka)
            
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

        pose_x = s[:, POSE_X_IDX]
        pose_y = s[:, POSE_Y_IDX]
        pose_theta = s[:, POSE_THETA_IDX]

        steering = Q[:, ANGULAR_CONTROL_IDX]
        speed = Q[:, TRANSLATIONAL_CONTROL_IDX]

        for _ in range(self.intermediate_steps):
            pose_theta = pose_theta + 0.5 * (steering / self.intermediate_steps_float)
            pose_x = pose_x + self.t_step * speed * self.lib.cos(pose_theta)
            pose_y = pose_y + self.t_step * speed * self.lib.sin(pose_theta)

        pose_theta_cos = self.lib.cos(pose_theta)
        pose_theta_sin = self.lib.sin(pose_theta)
        if self.wrap_angle:
            pose_theta = self.lib.atan2(pose_theta_sin, pose_theta_cos)
        pose_x = pose_x
        pose_y = pose_y
        angular_vel_z = self.lib.zeros_like(pose_x)
        linear_vel_x = self.lib.zeros_like(pose_x)
        slip_angle = self.lib.zeros_like(pose_x)
        steering_angle = self.lib.zeros_like(pose_x)

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
        Parallely executes steps from initial state s[i] with control input Q[i] for every i
        @param s: (batch_size, len(state)) all initial states for every step
        @param s: (batch_size, len(control_input)) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (batch_size, len(state)) all nexts states
        '''
        
        angular_vel_z = s[:, ANGULAR_VEL_Z_IDX]
        v_x = s[:, LINEAR_VEL_X_IDX]
        psi = s[:, POSE_THETA_IDX]
      
        s_x = s[:, POSE_X_IDX]
        s_y = s[:, POSE_Y_IDX]
        beta = s[:, SLIP_ANGLE_IDX]
        delta = s[:, STEERING_ANGLE_IDX]

        delta_dot, v_x_dot = self.lib.unstack(Q, 2, 1)
        
        # Constraints
        v_x_dot = self.accl_constraints(v_x, v_x_dot)
        delta_dot = self.steering_constraints(delta, delta_dot)

        # Euler stepping
        for _ in range(self.intermediate_steps):
            s_x_dot = v_x * self.lib.cos(psi)
            s_y_dot = v_x * self.lib.sin(psi)
            # delta_dot = delta_dot
            # v_x_dot = v_x_dot
            psi_dot = (v_x / self.lwb) * self.lib.tan(delta)
            psi_dot_dot = (v_x_dot * self.lib.tan(delta) / self.lwb) \
                + v_x * delta_dot / (self.lwb * self.lib.cos(delta) ** 2)

            s_x = s_x + self.t_step * s_x_dot
            s_y = s_y + self.t_step * s_y_dot
            delta = self.lib.clip(delta + self.t_step * delta_dot, self.s_min, self.s_max)
            v_x = v_x + self.t_step * v_x_dot
            psi = psi + self.t_step * psi_dot
            angular_vel_z = angular_vel_z + self.t_step * psi_dot_dot

        angular_vel_z = angular_vel_z
        linear_vel_x = v_x
        pose_theta_cos = self.lib.cos(psi)
        pose_theta_sin = self.lib.sin(psi)
        if self.wrap_angle:
            pose_theta = self.lib.atan2(pose_theta_sin, pose_theta_cos)
        else:
            pose_theta = psi
        pose_x = s_x
        pose_y = s_y
        slip_angle = self.lib.zeros_like(beta)
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
        @param Q: (batch_size, len(control_input)) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (batch_size, len(state)) all nexts states
        '''

        # params
        mu = self.car_parameters.mu  # friction coefficient  [-]
        C_Sf = self.car_parameters.C_Sf  # cornering stiffness front [1/rad]
        C_Sr = self.car_parameters.C_Sr  # cornering stiffness rear [1/rad]
        lf = self.car_parameters.lf  # distance from venter of gracity to front axle [m]
        lr = self.car_parameters.lr  # distance from venter of gracity to rear axle [m]
        h = self.car_parameters.h  # center of gravity height of toal mass [m]
        m = self.car_parameters.m  # Total Mass of car [kg]
        I = self.car_parameters.I_z  # Moment of inertia for entire mass about z axis  [kgm^2]
        g = self.car_parameters.g

        # State
        s_x = s[:, POSE_X_IDX]  # Pose X
        s_y = s[:, POSE_Y_IDX]  # Pose Y
        delta = s[:, STEERING_ANGLE_IDX]  # Fron Wheel steering angle
        v_x = s[:, LINEAR_VEL_X_IDX]  # Speed
        psi = s[:, POSE_THETA_IDX]  # Yaw Angle
        psi_dot = s[:, ANGULAR_VEL_Z_IDX]  # Yaw Rate
        beta = s[:, SLIP_ANGLE_IDX]  # Slipping Angle

        # Variable utils, mbakka
        # Control Input
        delta_dot = Q[:, ANGULAR_CONTROL_IDX]  # steering angle velocity of front wheels
        v_x_dot = Q[:, TRANSLATIONAL_CONTROL_IDX]  # longitudinal acceleration

        # v_x = tf.clip_by_value(v_x, 0.11, 1000)
        # min_vel_x = tf.reduce_min(v_x)
        # if(tf.less(min_vel_x, 0.5)):
        #     return self._step_dynamics_ks(s,Q, params)

        for _ in range(self.intermediate_steps):
            # Constaints
            v_x_dot = self.accl_constraints(v_x, v_x_dot)
            delta_dot = self.steering_constraints(delta, delta_dot)

            # In case speed dropy to < 0.2 during rollout
            # ST model needs a lin_vel_x of > 0.1 to work
            #v_x = tf.clip_by_value(v_x, 0.2, 1000)

            s_x_dot = v_x * self.lib.cos(psi + beta)
            s_y_dot = v_x * self.lib.sin(psi + beta)
            # delta_dot = delta_dot
            # v_x_dot = v_x_dot

            v_x = self.lib.where(v_x == 0.0, 1e-10, v_x)  # Add small value zero values to avoid division by zero
            psi_dot_dot = -mu * m / (v_x * I * (lr + lf)) * (
                    lf ** 2 * C_Sf * (g * lr - v_x_dot * h) + lr ** 2 * C_Sr * (g * lf + v_x_dot * h)) * psi_dot \
                          + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + v_x_dot * h) - lf * C_Sf * (
                    g * lr - v_x_dot * h)) * beta \
                          + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - v_x_dot * h) * delta

            beta_dot = (mu / (v_x ** 2 * (lr + lf)) * (
                    C_Sr * (g * lf + v_x_dot * h) * lr - C_Sf * (g * lr - v_x_dot * h) * lf) - 1) * psi_dot \
                       - mu / (v_x * (lr + lf)) * (
                               C_Sr * (g * lf + v_x_dot * h) + C_Sf * (g * lr - v_x_dot * h)) * beta \
                       + mu / (v_x * (lr + lf)) * (C_Sf * (g * lr - v_x_dot * h)) * delta

            s_x = s_x + self.t_step * s_x_dot
            s_y = s_y + self.t_step * s_y_dot
            delta = self.lib.clip(delta + self.t_step * delta_dot, self.s_min, self.s_max)
            v_x = v_x + self.t_step * v_x_dot
            psi = psi + self.t_step * psi_dot
            psi_dot = psi_dot + self.t_step * psi_dot_dot
            beta = beta + self.t_step * beta_dot

        angular_vel_z = psi_dot
        linear_vel_x = v_x
        pose_theta_cos = self.lib.cos(psi)
        pose_theta_sin = self.lib.sin(psi)
        if self.wrap_angle:
            pose_theta = self.lib.atan2(pose_theta_sin, pose_theta_cos)
        else:
            pose_theta = psi
        pose_x = s_x
        pose_y = s_y
        slip_angle = beta
        steering_angle = delta

        s_next_ks = self._step_dynamics_ks(s, Q, None)
        s_next_ts = self.next_step_output(angular_vel_z,
                                          linear_vel_x,
                                          pose_theta,
                                          pose_theta_cos,
                                          pose_theta_sin,
                                          pose_x,
                                          pose_y,
                                          slip_angle,
                                          steering_angle)

        # switch to kinematic model for small velocities
        min_speed_st = self.car_parameters.min_speed_st
        speed_too_low_for_st_indices = self.lib.less(v_x, min_speed_st)
        speed_too_low_for_st_indices = self.lib.reshape(speed_too_low_for_st_indices, (-1, 1))
        state_len = self.lib.shape(s)[1]
        ks_or_ts = self.lib.repeat(speed_too_low_for_st_indices, state_len, 1)
        next_step = self.lib.where(ks_or_ts, s_next_ks, s_next_ts)

        return next_step

    def _step_dynamics_pacejka(self, s, Q, params):
        
        # Q: Control after PID (steering velocity: delta_dot, acceleration_x: v_x_dot)
                
        # params
        mu = self.car_parameters.mu  # friction coefficient  [-]
        lf = self.car_parameters.lf  # distance from center of gravity to front axle [m]
        lr = self.car_parameters.lr  # distance from center of gravity to rear axle [m]
        h_cg = self.car_parameters.h  # center of gravity height of total mass [m]
        m = self.car_parameters.m  # Total Mass of car [kg]
        I_z = self.car_parameters.I_z  # Moment of inertia for entire mass about z axis  [kgm^2]
        g_ = self.car_parameters.g # gravity [m/s^2]

        # pacejka tire model parameters
        B_f = self.car_parameters.C_Pf[0]
        C_f = self.car_parameters.C_Pf[1]
        D_f = self.car_parameters.C_Pf[2]
        E_f = self.car_parameters.C_Pf[3]
        B_r = self.car_parameters.C_Pr[0]
        C_r = self.car_parameters.C_Pr[1]
        D_r = self.car_parameters.C_Pr[2]
        E_r = self.car_parameters.C_Pr[3]

        # State
        s_x = s[:, POSE_X_IDX]  # Pose X
        s_y = s[:, POSE_Y_IDX]  # Pose Y
        delta = s[:, STEERING_ANGLE_IDX]  # Front Wheel steering angle
        velocity = s[:, LINEAR_VEL_X_IDX] # Speed
        psi = s[:, POSE_THETA_IDX]  # Yaw Angle
        psi_dot = s[:, ANGULAR_VEL_Z_IDX]  # Yaw Rate
        delta_dot = Q[:, ANGULAR_CONTROL_IDX]  # steering angle velocity of front wheels
        v_x_dot = Q[:, TRANSLATIONAL_CONTROL_IDX]  # longitudinal acceleration
        slip_angle = s[:, SLIP_ANGLE_IDX]  # Slip Angle
        
        v_x = velocity * self.lib.cos(slip_angle)  # Longitudinal Velocity
        v_y = velocity * self.lib.sin(slip_angle)  # Lateral Velocity;
        
        # Constraints
        v_x_dot = self.accl_constraints(v_x, v_x_dot)
        delta_dot = self.steering_constraints(delta, delta_dot)

        for _ in range(self.intermediate_steps):
            v_x = tf.where(v_x == 0, tf.constant(1e-8), v_x)
            alpha_f = -self.lib.atan((v_y + psi_dot * lf) / (v_x)) + delta
            alpha_r = -self.lib.atan((v_y - psi_dot * lr) / v_x )

            # compute vertical tire forces
            F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
            F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

            F_yf = mu * F_zf * D_f * self.lib.sin(C_f * self.lib.atan(B_f * alpha_f - E_f*(B_f * alpha_f - self.lib.atan(B_f * alpha_f))))
            F_yr = mu * F_zr * D_r * self.lib.sin(C_r * self.lib.atan(B_r * alpha_r - E_r*(B_r * alpha_r - self.lib.atan(B_r * alpha_r))))

            d_pos_x = v_x * self.lib.cos(psi) - v_y * self.lib.sin(psi)
            d_pos_y = v_x * self.lib.sin(psi) + v_y * self.lib.cos(psi)
            d_psi = psi_dot
            d_v_x = v_x_dot
            d_v_y = 1/m * (F_yr + F_yf) - v_x * psi_dot
            d_psi_dot = 1/I_z * (-lr * F_yr + lf * F_yf)

            s_x = s_x + self.t_step * d_pos_x
            s_y = s_y + self.t_step * d_pos_y
            delta = self.lib.clip(delta + self.t_step * delta_dot, self.s_min, self.s_max)
            # v_x = v_x + self.lib.multiply(self.t_step, d_v_x)
            v_x = v_x + self.t_step *d_v_x
            v_y = v_y + self.t_step * d_v_y
            psi = psi + self.t_step * d_psi
            psi_dot = psi_dot + self.t_step * d_psi_dot

        angular_vel_z = psi_dot
        linear_vel_x = v_x
        pose_theta_cos = self.lib.cos(psi)
        pose_theta_sin = self.lib.sin(psi)
        if self.wrap_angle:
            pose_theta = self.lib.atan2(pose_theta_sin, pose_theta_cos)
        else:
            pose_theta = psi
        pose_x = s_x
        pose_y = s_y
        slip_angle = self.lib.atan(v_y / v_x)  # Calculate slip angle for consistency
        steering_angle = delta

        s_next_ks = self._step_dynamics_ks(s, Q, None)
        s_next_ts = self.next_step_output(angular_vel_z,
                                        linear_vel_x,
                                        pose_theta,
                                        pose_theta_cos,
                                        pose_theta_sin,
                                        pose_x,
                                        pose_y,
                                        slip_angle,
                                        steering_angle)

        # switch to kinematic model for small velocities
        # Define speed thresholds
        low_speed_threshold = self.lib.constant(0.5, self.lib.float32)
        high_speed_threshold = self.lib.constant(3, self.lib.float32)

        # Calculate weights for each element in v_x
        weights = (v_x - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
        weights = self.lib.clip(weights, 0, 1)  # Ensure weights are in [0, 1]
        weights = self.lib.reshape(weights, [-1, 1]) 

        counter_weights = self.lib.ones(self.lib.shape(weights)) - weights
        counter_weights = self.lib.reshape(counter_weights, [-1, 1]) 
        # Interpolate between the simple and complex models
        next_step = counter_weights * s_next_ks + weights * s_next_ts

        return next_step


    def steering_constraints(self, steering_angle, steering_velocity):

        cond1 = self.lib.logical_and(steering_angle <= self.s_min, steering_velocity <= 0.0)
        cond2 = self.lib.logical_and(steering_angle >= self.s_max, steering_velocity >= 0.0)

        good_steering_velocity_indices = self.lib.cast(self.lib.logical_not(self.lib.logical_or(cond1, cond2)), self.lib.float32)

        # Steering velocity is constrainted
        steering_velocity = self.lib.clip(steering_velocity * good_steering_velocity_indices, self.sv_min, self.sv_max)

        return steering_velocity

    def accl_constraints(self, vel, accl):

        # positive accl limit
        velocity_too_high_indices = self.lib.greater(vel, self.v_switch)
        pos_limit = self.lib.where(velocity_too_high_indices, (self.a_max * self.v_switch) / vel, self.a_max)

        # mbakka commented the following paragraph
        """
        pos_limit_velocity_too_high = (a_max * v_switch) / vel
        pos_limit_velocity_not_too_high = a_max
        
        pos_limit = (velocity_too_high_indices * pos_limit_velocity_too_high) + (
               velocity_not_too_high_indices * pos_limit_velocity_not_too_high)
        """

        # accl limit reached?

        v_less_vmin = self.lib.less(vel, self.v_min)
        accl_negative = self.lib.less(accl, 0)
        v_less_vmin_and_accl_negative = self.lib.logical_and(v_less_vmin, accl_negative)

        v_greater_vmax = self.lib.greater(vel, self.v_max)
        accl_positive = self.lib.greater(accl, 0)
        v_greater_vmax_and_accl_positive = self.lib.logical_and(v_greater_vmax, accl_positive)

        condition = self.lib.logical_or(v_less_vmin_and_accl_negative, v_greater_vmax_and_accl_positive)

        accl = self.lib.where(condition, 0., accl)

        accl = self.lib.clip(accl, self.a_min, pos_limit)

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
        
        steering_diff_low = self.car_parameters.steering_diff_low
        servo_p = self.car_parameters.servo_p
        
        steering_angle_difference = desired_steering_angle - current_steering_angle

        steering_angle_difference_not_too_low_indices = self.lib.greater(self.lib.abs(steering_angle_difference), steering_diff_low)
        steering_angle_difference_not_too_low_indices = self.lib.cast(steering_angle_difference_not_too_low_indices,
                                                                self.lib.float32)

        steering_velocity = steering_angle_difference_not_too_low_indices * (steering_angle_difference * servo_p)
        
        steering_velocity = self.lib.clip(steering_velocity, self.car_parameters.sv_min, self.car_parameters.sv_max)

        return steering_velocity

    def motor_controller_pid(self, desired_speed, current_speed):
        
        speed_difference = desired_speed - current_speed

        forward_indices = current_speed > 0.0
        backward_indices = ~forward_indices

        positive_speed_difference = speed_difference > 0.0
        negative_speed_difference = ~positive_speed_difference
        
        forward_accelerating_indices = forward_indices & positive_speed_difference
        forward_breaking_indices = forward_indices & negative_speed_difference
        backward_accelerating_indices = backward_indices & negative_speed_difference
        backward_breaking_indices = backward_indices & positive_speed_difference
        
        max_a_v = self.a_max / self.v_max * speed_difference
        min_a_v = self.a_max / (-self.v_min) * speed_difference
       
        zeros = self.lib.zeros_like(speed_difference)
        # fwd accl
        forward_acceleration = self.lib.where(forward_accelerating_indices, 10.0 * max_a_v, zeros)

        # fwd break
        forward_breaking = self.lib.where(forward_breaking_indices, 10.0 * min_a_v, zeros)

        # bkw accl
        backward_acceleration = self.lib.where(backward_accelerating_indices, 2.0 * min_a_v, zeros)

        # bkw break
        backward_breaking = self.lib.where(backward_breaking_indices, 2.0 * max_a_v, zeros)

        total_acceleration = forward_acceleration + forward_breaking + backward_acceleration + backward_breaking

        return total_acceleration

    def _step_model_with_servo_and_motor_pid_(self, model):

        def _step_model_with_servo_and_motor_pid(s, Q, params):

            # Control Input (desired speed, desired steering angle)
            desired_angle, desired_speed = self.lib.unstack(Q, 2, 1)

            delta = s[:, STEERING_ANGLE_IDX]  # Fron Wheel steering angle
            vel_x = s[:, LINEAR_VEL_X_IDX]  # Speed

            delta_dot = self.servo_proportional(desired_angle, delta)
            vel_x_dot = self.motor_controller_pid(desired_speed, vel_x)

            Q_pid = self.lib.permute(self.lib.stack([delta_dot, vel_x_dot]))

            return model(s, Q_pid, params)

        return _step_model_with_servo_and_motor_pid

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

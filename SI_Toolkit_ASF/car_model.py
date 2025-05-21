import yaml
import math
from utilities.path_helper_ros import *
from utilities.car_files.vehicle_parameters import VehicleParameters
from SI_Toolkit.computation_library import NumpyLibrary

from utilities.state_utilities import *


def _ks_step_factory(lib, car_parameters, t_step):

    def _ks_step(s_x, s_y, delta, v_x, psi, angular_vel_z, delta_dot, v_x_dot):
        s_x_dot = v_x * lib.cos(psi)
        s_y_dot = v_x * lib.sin(psi)
        psi_dot = (v_x / car_parameters.l_wb) * lib.tan(delta)
        psi_dot_dot = (v_x_dot * lib.tan(delta) / car_parameters.l_wb) \
                        + v_x * delta_dot / (car_parameters.l_wb * lib.cos(delta) ** 2)

        s_x = s_x + t_step * s_x_dot
        s_y = s_y + t_step * s_y_dot
        delta = lib.clip(delta + t_step * delta_dot, car_parameters.s_min, car_parameters.s_max)
        v_x = v_x + t_step * v_x_dot
        psi = psi + t_step * psi_dot
        angular_vel_z = angular_vel_z + t_step * psi_dot_dot

        return s_x, s_y, delta, v_x, psi, angular_vel_z

    return _ks_step


def _pacejka_step_factory(lib, car_parameters, t_step):

    # params
    lf = car_parameters.lf  # distance from center of gravity to front axle [m]
    lr = car_parameters.lr  # distance from center of gravity to rear axle [m]
    h_cg = car_parameters.h  # center of gravity height of total mass [m]
    m = car_parameters.m  # Total Mass of car [kg]
    I_z = car_parameters.I_z  # Moment of inertia for entire mass about z axis  [kgm^2]
    g_ = car_parameters.g  # gravity [m/s^2]

    # pacejka tire model parameters
    B_f = car_parameters.C_Pf[0]
    C_f = car_parameters.C_Pf[1]
    D_f = car_parameters.C_Pf[2]
    E_f = car_parameters.C_Pf[3]
    B_r = car_parameters.C_Pr[0]
    C_r = car_parameters.C_Pr[1]
    D_r = car_parameters.C_Pr[2]
    E_r = car_parameters.C_Pr[3]

    def _pacejka_step(s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot, mu):
        v_x_safe = lib.where(v_x < 1.0e-3, lib.constant(1.0e-3, lib.float32), v_x)
        alpha_f = -lib.atan((v_y + psi_dot * lf) / v_x_safe) + delta
        alpha_r = -lib.atan((v_y - psi_dot * lr) / v_x_safe)

        # compute vertical tire forces
        F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
        F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

        F_yf = mu * F_zf * D_f * lib.sin(
            C_f * lib.atan(B_f * alpha_f - E_f * (B_f * alpha_f - lib.atan(B_f * alpha_f))))
        F_yr = mu * F_zr * D_r * lib.sin(
            C_r * lib.atan(B_r * alpha_r - E_r * (B_r * alpha_r - lib.atan(B_r * alpha_r))))

        d_pos_x = v_x * lib.cos(psi) - v_y * lib.sin(psi)
        d_pos_y = v_x * lib.sin(psi) + v_y * lib.cos(psi)
        d_psi = psi_dot
        d_v_x = v_x_dot
        d_v_y = 1 / m * (F_yr + F_yf) - v_x * psi_dot
        # print d_v_y + v_x * psi_dot
        # Should be equal to IMUs a_y

        d_psi_dot = 1 / I_z * (-lr * F_yr + lf * F_yf)

        s_x = s_x + t_step * d_pos_x
        s_y = s_y + t_step * d_pos_y
        delta = lib.clip(delta + t_step * delta_dot, car_parameters.s_min, car_parameters.s_max)
        v_x = v_x + t_step * d_v_x
        v_y = v_y + t_step * d_v_y
        psi = psi + t_step * d_psi
        psi_dot = psi_dot + t_step * d_psi_dot
        return s_x, s_y, delta, v_x, v_y, psi, psi_dot

    return _pacejka_step


class car_model:

    num_actions = len(CONTROL_INPUTS)
    num_states  = len(STATE_VARIABLES)

    def __init__(
            self,
            model_of_car_dynamics: str,
            batch_size: int,
            car_parameter_file: str, # file containing the car parameters
            dt: float = 0.03,
            intermediate_steps=1,
            computation_lib = NumpyLibrary(),
            variable_parameters=None,
            **kwargs
    ):
        self.lib = computation_lib

        self.car_parameters = VehicleParameters(car_parameter_file)

        self.num_actions = self.lib.to_tensor(self.num_actions, self.lib.int32)
        self.num_states = self.lib.to_tensor(self.num_states, self.lib.int32)

        # self.POSE_THETA_IDX = self.lib.constant(POSE_THETA_IDX, self.lib.int32)
        # self.POSE_X_IDX = self.lib.to_tensor(POSE_X_IDX, self.lib.int32)
        # self.POSE_Y_IDX = self.lib.to_tensor(POSE_Y_IDX, self.lib.int32)
        # self.LINEAR_VEL_X_IDX = self.lib.to_tensor(LINEAR_VEL_X_IDX, self.lib.int32)
        # self.LINEAR_VEL_Y_IDX = self.lib.to_tensor(LINEAR_VEL_Y_IDX, self.lib.int32)
        # self.ANGULAR_VEL_Z_IDX = self.lib.to_tensor(ANGULAR_VEL_Z_IDX, self.lib.int32)
        # self.SLIP_ANGLE_IDX = self.lib.to_tensor(SLIP_ANGLE_IDX, self.lib.int32)
        # self.STEERING_ANGLE_IDX = self.lib.to_tensor(STEERING_ANGLE_IDX, self.lib.int32)
        #
        # self.ANGULAR_CONTROL_IDX = self.lib.to_tensor(ANGULAR_CONTROL_IDX, self.lib.int32)
        # self.TRANSLATIONAL_CONTROL_IDX = self.lib.to_tensor(TRANSLATIONAL_CONTROL_IDX, self.lib.int32)


        self.POSE_THETA_IDX = int(POSE_THETA_IDX)
        self.POSE_X_IDX = int(POSE_X_IDX)
        self.POSE_Y_IDX = int(POSE_Y_IDX)
        self.LINEAR_VEL_X_IDX = int(LINEAR_VEL_X_IDX)
        self.LINEAR_VEL_Y_IDX = int(LINEAR_VEL_Y_IDX)
        self.ANGULAR_VEL_Z_IDX = int(ANGULAR_VEL_Z_IDX)
        self.SLIP_ANGLE_IDX = int(SLIP_ANGLE_IDX)
        self.STEERING_ANGLE_IDX = int(STEERING_ANGLE_IDX)

        self.ANGULAR_CONTROL_IDX = int(ANGULAR_CONTROL_IDX)
        self.TRANSLATIONAL_CONTROL_IDX = int(TRANSLATIONAL_CONTROL_IDX)
            
              
        self.model_of_car_dynamics = model_of_car_dynamics
        self.step_dynamics = None
        self.step_dynamics_core = None
        self.set_model_of_car_dynamics(model_of_car_dynamics)


        self.dt = dt
        self.intermediate_steps = int(intermediate_steps)
        self.intermediate_steps_float = float(intermediate_steps)
        self.t_step = float(self.dt / float(self.intermediate_steps))

        self.variable_parameters = variable_parameters
        if self.variable_parameters is not None and hasattr(self.variable_parameters, 'mu'):
            self.mu = self.variable_parameters.mu
        else:
            self.mu = self.lib.to_tensor(self.car_parameters.mu, self.lib.float32)

        self._ks_step = _ks_step_factory(self.lib, self.car_parameters, self.t_step)
        self._pacejka_step = _pacejka_step_factory(self.lib, self.car_parameters, self.t_step)


    # region Various dynamical models for a car

    def set_model_of_car_dynamics(self, model_of_car_dynamics):

        if model_of_car_dynamics == 'ODE:ks':
            self.step_dynamics = self._step_model_with_servo_pid_with_constrains_(self._step_dynamics_ks)
        elif model_of_car_dynamics == 'ODE:pacejka':
            self.step_dynamics = self._step_model_with_servo_pid_with_constrains_(self._step_dynamics_pacejka)
        elif model_of_car_dynamics == 'ODE:ks_pacejka':
            self.step_dynamics = self._step_model_with_servo_pid_with_constrains_(self._step_dynamics_ks_pacejka)

        if model_of_car_dynamics == 'ODE:ks':
            self.step_dynamics_core = self._step_dynamics_ks
        elif model_of_car_dynamics == 'ODE:pacejka':
            self.step_dynamics_core = self._step_dynamics_pacejka
        elif model_of_car_dynamics == 'ODE:ks_pacejka':
            self.step_dynamics_core = self._step_dynamics_ks_pacejka
        else:
            raise NotImplementedError(
                '{} not recognized as a valid name for a model of car dynamics'.format(model_of_car_dynamics))

        self.model_of_car_dynamics = model_of_car_dynamics

    def change_friction_coefficient(self, friction_coefficient):
        self.car_parameters.mu = friction_coefficient
        self.set_model_of_car_dynamics(self.model_of_car_dynamics)

    # Kinematic model: applicable for low speeds
    def _step_dynamics_ks(self, s, Q):
        '''
        Parallely executes steps from initial state s[i] with control input Q[i] for every i
        @param s: (batch_size, len(state)) all initial states for every step
        @param s: (batch_size, len(control_input)) all control inputs for every step
        returns s_next: (batch_size, len(state)) all nexts states
        '''
        
        angular_vel_z = s[:, self.ANGULAR_VEL_Z_IDX]
        v_x = s[:, self.LINEAR_VEL_X_IDX]
        psi = s[:,self.POSE_THETA_IDX]
      
        s_x = s[:, self.POSE_X_IDX]
        s_y = s[:, self.POSE_Y_IDX]
        beta = s[:, self.SLIP_ANGLE_IDX]
        delta = s[:, self.STEERING_ANGLE_IDX]

        delta_dot, v_x_dot = self.lib.unstack(Q, 2, 1)

        i = 0
        while i < self.intermediate_steps:
            s_x, s_y, delta, v_x, psi, angular_vel_z = \
            self._ks_step(s_x, s_y, delta, v_x, psi, angular_vel_z, delta_dot, v_x_dot)
            i += 1

        linear_vel_x = v_x
        pose_theta_cos = self.lib.cos(psi)
        pose_theta_sin = self.lib.sin(psi)

        # Angle is not wraped in ks model. Check Model Pacejka: The models are fused there and this doesnt work if the angle is warapped already
        pose_theta = psi
        
        
        slip_angle = self.lib.zeros_like(beta)
        v_y = self.lib.zeros_like(beta)
        

        return self.next_step_output(angular_vel_z,
                                     v_x,
                                     v_y,
                                     pose_theta,
                                     pose_theta_cos,
                                     pose_theta_sin,
                                     s_x,
                                     s_y,
                                     slip_angle,
                                     delta)

   

    def _step_dynamics_pacejka(self, s, Q):

        mu = self.mu
        # Q: Control after PID (steering velocity: delta_dot, acceleration_x: v_x_dot)

        # State
        s_x = s[:, self.POSE_X_IDX]  # Pose X
        s_y = s[:, self.POSE_Y_IDX]  # Pose Y
        delta = s[:, self.STEERING_ANGLE_IDX]  # Front Wheel steering angle
        v_x = s[:, self.LINEAR_VEL_X_IDX]  # Longitudinal velocity
        v_y = s[:, self.LINEAR_VEL_Y_IDX]  # Lateral velocity    
        psi = s[:,self.POSE_THETA_IDX]  # Yaw Angle
        psi_dot = s[:, self.ANGULAR_VEL_Z_IDX]  # Yaw Rate
        delta_dot = Q[:, self.ANGULAR_CONTROL_IDX]  # steering angle velocity of front wheels
        v_x_dot = Q[:, self.TRANSLATIONAL_CONTROL_IDX]  # longitudinal acceleration

        i = 0
        while i < self.intermediate_steps:
            s_x, s_y, delta, v_x, v_y, psi, psi_dot = \
                self._pacejka_step(s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot, mu)
            i += 1

        pose_theta = psi
        pose_theta_cos = self.lib.cos(psi)
        pose_theta_sin = self.lib.sin(psi)
        
        pose_x = s_x
        pose_y = s_y
        v_x_safe = self.lib.where(v_x < 1.0e-3, self.lib.constant(1.0e-3, self.lib.float32), v_x)
        slip_angle = self.lib.atan(v_y / v_x_safe)  # Calculate slip angle for consistency

        s_next_ts = self.next_step_output(psi_dot,
                                        v_x,
                                        v_y,
                                        pose_theta,
                                        pose_theta_cos,
                                        pose_theta_sin,
                                        pose_x,
                                        pose_y,
                                        slip_angle,
                                        delta)
     
        next_step = s_next_ts
        return next_step



    def _step_dynamics_ks_pacejka(self, s, Q):

        v_x = s[:, self.LINEAR_VEL_X_IDX]  # Longitudinal velocity

        # switch to kinematic model for small velocities
        # Define speed thresholds

        s_next_ks = self._step_dynamics_ks(s, Q)
        s_next_ts = self._step_dynamics_pacejka(s, Q)

        low_speed_threshold = self.lib.constant(0.5, self.lib.float32)
        high_speed_threshold = self.lib.constant(3, self.lib.float32)

        # Calculate weights for each element in v_x (smooth transission from pacejka to KS model)
        # weights = (v_x - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
        # weights = self.lib.clip(weights, 0, 1)  # Ensure weights are in [0, 1]

        midpoint = (low_speed_threshold + high_speed_threshold) / self.lib.constant(2, self.lib.float32)
        k = self.lib.constant(4.8, self.lib.float32)
        weights = self.lib.constant(1, self.lib.float32) / (self.lib.constant(1, self.lib.float32) + self.lib.exp(
            -k * (v_x - midpoint)))

        weights = self.lib.reshape(weights, (-1, 1))
        weights_shape = self.lib.shape(weights)
        counter_weights = self.lib.ones(weights_shape) - weights
        counter_weights = self.lib.reshape(counter_weights, (-1, 1)) 
        
        # Interpolate between the simple and complex models
        next_step = counter_weights * s_next_ks + weights * s_next_ts

        return next_step


    def steering_constraints(self, steering_angle, steering_velocity):

        cond1 = self.lib.logical_and(steering_angle <= self.car_parameters.s_min, steering_velocity <= 0.0)
        cond2 = self.lib.logical_and(steering_angle >= self.car_parameters.s_max, steering_velocity >= 0.0)

        good_steering_velocity_indices = self.lib.cast(self.lib.logical_not(self.lib.logical_or(cond1, cond2)), self.lib.float32)

        # Steering velocity is constrainted
        steering_velocity = self.lib.clip(steering_velocity * good_steering_velocity_indices, self.car_parameters.sv_min, self.car_parameters.sv_max)

        return steering_velocity

    def accl_constraints(self, vel, accl):

        # velocity too low
        vel = self.lib.where(vel == 0, self.lib.constant(0.0001, self.lib.float32), vel)

        # positive accl limit
        velocity_too_high_indices = self.lib.greater(vel, self.car_parameters.v_switch)
        pos_limit = self.lib.where(velocity_too_high_indices, (self.car_parameters.a_max * self.car_parameters.v_switch) / vel, self.car_parameters.a_max)

        # accl limit reached?
        v_less_vmin = self.lib.less(vel, self.car_parameters.v_min)
        accl_negative = self.lib.less(accl, 0)
        v_less_vmin_and_accl_negative = self.lib.logical_and(v_less_vmin, accl_negative)

        v_greater_vmax = self.lib.greater(vel, self.car_parameters.v_max)
        accl_positive = self.lib.greater(accl, 0)
        v_greater_vmax_and_accl_positive = self.lib.logical_and(v_greater_vmax, accl_positive)

        condition = self.lib.logical_or(v_less_vmin_and_accl_negative, v_greater_vmax_and_accl_positive)

        accl = self.lib.where(condition, 0., accl)

        # Constraint longitudinal acceleration by motor power
        # accl = self.lib.clip(accl, self.car_parameters.a_min, pos_limit)
        a_min = self.lib.constant(self.car_parameters.a_min, t=self.lib.float32)
        accl = self.lib.clip(accl, a_min, pos_limit)

        # Constraint longitudinal acceleration by slipping
        max_acceleration = self.car_parameters.g * self.car_parameters.mu        
        accl = self.lib.clip(accl, -max_acceleration, max_acceleration)
        
        
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

    # Deprecated. Motor PID is no longer a part of the car model
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
        
        max_a_v = self.car_parameters.a_max / self.car_parameters.v_max * speed_difference
        min_a_v = self.car_parameters.a_max / (-self.car_parameters.v_min) * speed_difference
       
        zeros = self.lib.zeros_like(speed_difference)
        # fwd accl
        forward_acceleration = self.lib.where(forward_accelerating_indices, 10.0 * max_a_v, zeros)

        # fwd break
        forward_breaking = self.lib.where(forward_breaking_indices, 0.5 * min_a_v, zeros)

        # bkw accl
        backward_acceleration = self.lib.where(backward_accelerating_indices, 2.0 * min_a_v, zeros)

        # bkw break
        backward_breaking = self.lib.where(backward_breaking_indices, 2.0 * max_a_v, zeros)

        total_acceleration = forward_acceleration + forward_breaking + backward_acceleration + backward_breaking

        return total_acceleration

    def _step_model_with_servo_pid_with_constrains_(self, model):

        def _step_model_with_servo_and_motor_pid_with_constrains(s, Q):

            Q_pid = self.pid(s, Q)
            Q_pid_with_constrains = self.apply_constrains(s, Q_pid)

            return model(s, Q_pid_with_constrains)

        return _step_model_with_servo_and_motor_pid_with_constrains

# endregion

    def pid(self, s, Q):

        # Control Input (desired speed, desired steering angle)
        desired_angle, translational_control = self.lib.unstack(Q, 2, 1)

        delta = s[:, self.STEERING_ANGLE_IDX]  # Front Wheel steering angle
        vel_x = s[:, self.LINEAR_VEL_X_IDX]  # Longitudinal velocity

        delta_dot = self.servo_proportional(desired_angle, delta)
        
        vel_x_dot = translational_control   

        # Q_pid = self.lib.permute(self.lib.stack([delta_dot, vel_x_dot]))
        Q_pid = self.lib.stack([delta_dot, vel_x_dot], axis=1)

        return Q_pid


    def apply_constrains(self, s, Q_pid):

        delta = s[:, self.STEERING_ANGLE_IDX]  # Front wheels steering angle
        v_x = s[:, self.LINEAR_VEL_X_IDX]  # Longitudinal velocity

        delta_dot = Q_pid[:, self.ANGULAR_CONTROL_IDX]  # Front wheels steering angle velocity
        v_x_dot = Q_pid[:, self.TRANSLATIONAL_CONTROL_IDX]  # Longitudinal acceleration

        delta_dot = self.steering_constraints(delta, delta_dot)
        v_x_dot = self.accl_constraints(v_x, v_x_dot)

        Q_pid_with_constrains = self.lib.stack([delta_dot, v_x_dot], axis=1)

        return Q_pid_with_constrains

    def return_control_cmd_components(self, control_cmd):
        steering_speed = control_cmd[:, self.ANGULAR_CONTROL_IDX]
        acceleration_x = control_cmd[:, self.TRANSLATIONAL_CONTROL_IDX]
        return steering_speed, acceleration_x



# region Formatting output of the step function

    def next_step_output(self,
                                     angular_vel_z,
                                     linear_vel_x,
                                     linear_vel_y,
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
            linear_vel_y,
            pose_theta,
            pose_theta_cos,
            pose_theta_sin,
            pose_x,
            pose_y,
            slip_angle,
            steering_angle
        ], axis=1)

# endregion

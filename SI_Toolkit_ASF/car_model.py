import yaml
import math
from utilities.path_helper_ros import *
from utilities.car_files.vehicle_parameters import VehicleParameters
from SI_Toolkit.computation_library import NumpyLibrary

from utilities.state_utilities import *

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
            **kwargs
    ):
        self.lib = computation_lib

        self.car_parameters = VehicleParameters(car_parameter_file)
              
        self.model_of_car_dynamics = model_of_car_dynamics
        self.step_dynamics = None
        self.step_dynamics_core = None
        self.set_model_of_car_dynamics(model_of_car_dynamics)


        self.dt = dt
        self.intermediate_steps = self.lib.to_tensor(intermediate_steps, self.lib.int32)
        self.intermediate_steps_float = self.lib.to_tensor(intermediate_steps, self.lib.float32)
        self.t_step = self.lib.to_tensor(self.dt / float(self.intermediate_steps), self.lib.float32)

    # region Various dynamical models for a car

    def set_model_of_car_dynamics(self, model_of_car_dynamics):

        if model_of_car_dynamics == 'ODE:ks':
            self.step_dynamics = self._step_model_with_servo_and_motor_pid_with_constrains_(self._step_dynamics_ks)
        elif model_of_car_dynamics == 'ODE:pacejka':
            self.step_dynamics = self._step_model_with_servo_and_motor_pid_with_constrains_(self._step_dynamics_pacejka)
        elif model_of_car_dynamics == 'ODE:ks_pacejka':
            self.step_dynamics = self._step_model_with_servo_and_motor_pid_with_constrains_(self._step_dynamics_ks_pacejka)

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


    # Kinematic model: applicable for low speeds
    def _step_dynamics_ks(self, s, Q):
        '''
        Parallely executes steps from initial state s[i] with control input Q[i] for every i
        @param s: (batch_size, len(state)) all initial states for every step
        @param s: (batch_size, len(control_input)) all control inputs for every step
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

        # Euler stepping
        for _ in range(self.intermediate_steps):
            s_x_dot = v_x * self.lib.cos(psi)
            s_y_dot = v_x * self.lib.sin(psi)
            # delta_dot = delta_dot
            # v_x_dot = v_x_dot
            psi_dot = (v_x / self.car_parameters.l_wb) * self.lib.tan(delta)
            psi_dot_dot = (v_x_dot * self.lib.tan(delta) / self.car_parameters.l_wb) \
                + v_x * delta_dot / (self.car_parameters.l_wb * self.lib.cos(delta) ** 2)

            s_x = s_x + self.t_step * s_x_dot
            s_y = s_y + self.t_step * s_y_dot
            delta = self.lib.clip(delta + self.t_step * delta_dot, self.car_parameters.s_min, self.car_parameters.s_max)
            v_x = v_x + self.t_step * v_x_dot
            psi = psi + self.t_step * psi_dot
            angular_vel_z = angular_vel_z + self.t_step * psi_dot_dot

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
        v_x = s[:, LINEAR_VEL_X_IDX]  # Longitudinal velocity
        v_y = s[:, LINEAR_VEL_Y_IDX]  # Lateral velocity    
        psi = s[:, POSE_THETA_IDX]  # Yaw Angle
        psi_dot = s[:, ANGULAR_VEL_Z_IDX]  # Yaw Rate
        delta_dot = Q[:, ANGULAR_CONTROL_IDX]  # steering angle velocity of front wheels
        v_x_dot = Q[:, TRANSLATIONAL_CONTROL_IDX]  # longitudinal acceleration

        for _ in range(self.intermediate_steps):
            v_x = self.lib.where(v_x == 0, self.lib.constant(1e-5, self.lib.float32), v_x)
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
            # print d_v_y + v_x * psi_dot
            # Should be equal to IMUs a_y
            
            d_psi_dot = 1/I_z * (-lr * F_yr + lf * F_yf)

            s_x = s_x + self.t_step * d_pos_x
            s_y = s_y + self.t_step * d_pos_y
            delta = self.lib.clip(delta + self.t_step * delta_dot, self.car_parameters.s_min, self.car_parameters.s_max)
            v_x = v_x + self.t_step * d_v_x
            v_y = v_y + self.t_step * d_v_y
            psi = psi + self.t_step * d_psi
            psi_dot = psi_dot + self.t_step * d_psi_dot

        pose_theta = psi
        pose_theta_cos = self.lib.cos(psi)
        pose_theta_sin = self.lib.sin(psi)
        
        pose_x = s_x
        pose_y = s_y
        slip_angle = self.lib.atan(v_y / v_x)  # Calculate slip angle for consistency

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

        v_x = s[:, LINEAR_VEL_X_IDX]  # Longitudinal velocity

        # switch to kinematic model for small velocities
        # Define speed thresholds

        s_next_ks = self._step_dynamics_ks(s, Q)
        s_next_ts = self._step_dynamics_pacejka(s, Q)

        low_speed_threshold = self.lib.constant(0.5, self.lib.float32)
        high_speed_threshold = self.lib.constant(3, self.lib.float32)

        # Calculate weights for each element in v_x (smooth transission from pacejka to KS model)
        weights = (v_x - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
        weights = self.lib.clip(weights, 0, 1)  # Ensure weights are in [0, 1]
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

        accl = self.lib.clip(accl, self.car_parameters.a_min, pos_limit)

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
        
        max_a_v = self.car_parameters.a_max / self.car_parameters.v_max * speed_difference
        min_a_v = self.car_parameters.a_max / (-self.car_parameters.v_min) * speed_difference
       
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

    def _step_model_with_servo_and_motor_pid_with_constrains_(self, model):

        def _step_model_with_servo_and_motor_pid_with_constrains(s, Q):

            Q_pid = self.pid(s, Q)
            Q_pid_with_constrains = self.apply_constrains(s, Q_pid)

            return model(s, Q_pid_with_constrains)

        return _step_model_with_servo_and_motor_pid_with_constrains

# endregion

    def pid(self, s, Q):

        # Control Input (desired speed, desired steering angle)
        desired_angle, desired_speed = self.lib.unstack(Q, 2, 1)

        delta = s[:, STEERING_ANGLE_IDX]  # Front Wheel steering angle
        vel_x = s[:, LINEAR_VEL_X_IDX]  # Longitudinal velocity

        delta_dot = self.servo_proportional(desired_angle, delta)
        vel_x_dot = self.motor_controller_pid(desired_speed, vel_x)

        Q_pid = self.lib.permute(self.lib.stack([delta_dot, vel_x_dot]))

        return Q_pid


    def apply_constrains(self, s, Q_pid):

        delta = s[:, STEERING_ANGLE_IDX]  # Front wheels steering angle
        v_x = s[:, LINEAR_VEL_X_IDX]  # Longitudinal velocity

        delta_dot = Q_pid[:, ANGULAR_CONTROL_IDX]  # Front wheels steering angle velocity
        v_x_dot = Q_pid[:, TRANSLATIONAL_CONTROL_IDX]  # Longitudinal acceleration

        delta_dot = self.steering_constraints(delta, delta_dot)
        v_x_dot = self.accl_constraints(v_x, v_x_dot)

        Q_pid_with_constrains = self.lib.permute(self.lib.stack([delta_dot, v_x_dot]))

        return Q_pid_with_constrains

    def return_control_cmd_components(self, control_cmd):
        steering_speed = control_cmd[:, ANGULAR_CONTROL_IDX]
        acceleration_x = control_cmd[:, TRANSLATIONAL_CONTROL_IDX]
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

import math
import numpy as np  
from utilities.car_files.vehicle_parameters import VehicleParameters
import inspect

class StateIndices:
    pose_x = 0
    pose_y = 1
    yaw_angle = 2
    v_x = 3
    v_y = 4
    yaw_rate = 5
    steering_angle = 6
    
    
class ControlIndices:
    desired_steering_angle = 0
    acceleration = 1


class DynamicModelPacejka:
    
    # Dynamically count number of states
    number_of_states = len([name for name, value in inspect.getmembers(StateIndices) if not (name.startswith('__') and name.endswith('__')) and not inspect.isroutine(value)])

    # Create a reverse mapping dictionary as a class attribute
    name_dict = {value: name for name, value in vars(StateIndices).items() if not name.startswith('__')}

    @classmethod
    def getStateName(cls, index):
        # Use the reverse mapping to get the state name from the index
        return cls.name_dict.get(index, "Index not found")
    
    def __init__(self, dt=0.01 , intermediate_steps=1):
        self.dt = dt
        self.vehicle_parameters = VehicleParameters()
        self.intermediate_steps = intermediate_steps
        
    @staticmethod
    def speed_pid(current_speed: float, desired_speed: float, p: VehicleParameters) -> float:
        # acceleration - Exactly as original F1TENTH
        vel_diff = desired_speed - current_speed
        # currently forward
        if current_speed > 0.:
            if (vel_diff > 0):
                # accelerate
                kp = 10.0 * p.a_max / p.v_max
                accl = kp * vel_diff
            else:
                # braking
                kp = 10.0 * p.a_max / (-p.v_min)
                accl = kp * vel_diff
        # currently backwards
        else:
            if (vel_diff > 0):
                # braking
                kp = 2.0 * p.a_max / p.v_max
                accl = kp * vel_diff
            else:
                # accelerating
                kp = 2.0 * p.a_max / (-p.v_min)
                accl = kp * vel_diff
                
        accl = max(p.a_min, min(accl, p.a_max))
        return accl

    @staticmethod
    def servo_pid(current_steering: float, desired_steering: float, p: VehicleParameters) -> float:
        
        error = desired_steering - current_steering
        d_steering_angle = p.servo_p * error
    
        d_steering_angle = max(p.sv_min, min(d_steering_angle, p.sv_max))
        return d_steering_angle
    
    # @njit(cache=True)
    def accl_constraints(self, vel, accl):
        """
        Acceleration constraints, adjusts the acceleration based on constraints

            Args:
                vel (float): current velocity of the vehicle
                accl (float): unconstraint desired acceleration

            Returns:
                accl (float): adjusted acceleration
        """
        p = self.vehicle_parameters

        # positive accl limit
        if vel > p.v_switch:
            pos_limit = p.a_max*p.v_switch/vel
        else:
            pos_limit = p.a_max

        # accl limit reached?
        if (vel <= p.v_min and accl <= 0) or (vel >= p.v_max and accl >= 0):
            accl = 0.
        elif accl <= -p.a_max:
            accl = -p.a_max
        elif accl >= pos_limit:
            accl = pos_limit

        return accl

    # @njit(cache=True)
    def steering_constraint(self, steering_angle, steering_velocity):
        """
        Steering constraints, adjusts the steering velocity based on constraints

            Args:
                steering_angle (float): current steering_angle of the vehicle
                steering_velocity (float): unconstraint desired steering_velocity

            Returns:
                steering_velocity (float): adjusted steering velocity
        """
        p = self.vehicle_parameters
        # constraint steering velocity
        if (steering_angle <= p.s_min and steering_velocity <= 0) or (steering_angle >= p.s_max and steering_velocity >= 0):
            steering_velocity = 0.
        elif steering_velocity <= p.sv_min:
            steering_velocity = p.sv_min
        elif steering_velocity >= p.sv_max:
            steering_velocity = p.sv_max

        return steering_velocity


    def vehicle_dynamics(self, x, u) -> np.ndarray:
        """
        single-track vehicle dynamics with linear and pacejka tire models
        Inputs:
            :param x: vehicle state vector (x, y, yaw, v_x, v_y, omega)
                x0: x position
                x1: y position
                x2: yaw angle
                x3: longitudinal velocity
                x4: lateral velocity
                x5: yaw rate
            :param u: vehicle input vector (steering angle, longitudinal acceleration)
                u0: desired steering angle
                u1: desired speed
            :param p: vehicle parameter vector 
            :param type: tire model type (linear or pacejka)
        Outputs:
            :return f: derivative of vehicle state vector
        """
        p = self.vehicle_parameters
        g_ = p.g
        mu = p.mu
        
        # pacejka tire model parameters
        B_f = p.C_Pf[0]
        C_f = p.C_Pf[1]
        D_f = p.C_Pf[2]
        E_f = p.C_Pf[3]
        B_r = p.C_Pr[0]
        C_r = p.C_Pr[1]
        D_r = p.C_Pr[2]
        E_r = p.C_Pr[3]

        # vehicle parameters
        lf = p.lf
        lr = p.lr
        h = p.h 
        m = p.m 
        I_z = p.I_z
        
        # Pid controller
        acceleration_x = DynamicModelPacejka.speed_pid(x[StateIndices.v_x], u[ControlIndices.acceleration], p)
        steering_velocity = DynamicModelPacejka.servo_pid(x[StateIndices.steering_angle], u[ControlIndices.desired_steering_angle], p)
        
        # constraints
        acceleration_x = self.accl_constraints(x[StateIndices.v_x], acceleration_x)
        steering_velocity = self.steering_constraint(x[StateIndices.steering_angle], steering_velocity)

        # compute lateral tire slip angles
        if x[StateIndices.v_x] == 0 :
            x[StateIndices.v_x] = 1e-8 
        alpha_f = -math.atan((x[StateIndices.v_y] + x[StateIndices.yaw_rate] * lf) / (x[StateIndices.v_x])) + x[StateIndices.steering_angle]
        alpha_r = -math.atan((x[StateIndices.v_y] - x[StateIndices.yaw_rate] * lr) / x[StateIndices.v_x] )

        # compute vertical tire forces
        F_zf = m * (-acceleration_x * h + g_ * lr) / (lr + lf)
        F_zr = m * (acceleration_x * h + g_ * lf) / (lr + lf)

        F_yf = F_yr = 0
        
        # calculate combined slip lateral forces
        F_yf = mu * F_zf * D_f * math.sin(C_f * math.atan(B_f * alpha_f - E_f*(B_f * alpha_f - math.atan(B_f * alpha_f))))
        F_yr = mu * F_zr * D_r * math.sin(C_r * math.atan(B_r * alpha_r - E_r*(B_r * alpha_r - math.atan(B_r * alpha_r))))

        d_pos_x = x[StateIndices.v_x]*math.cos(x[StateIndices.yaw_angle]) - x[StateIndices.v_y]*math.sin(x[StateIndices.yaw_angle])
        d_pos_y = x[StateIndices.v_x]*math.sin(x[StateIndices.yaw_angle]) + x[StateIndices.v_y]*math.cos(x[StateIndices.yaw_angle])
        d_yaw_angle = x[StateIndices.yaw_rate]
        d_v_x = acceleration_x
        d_v_y = 1/m * (F_yr + F_yf) - x[StateIndices.v_x] * x[StateIndices.yaw_rate]
        d_yaw_rate = 1/I_z * (-lr * F_yr + lf * F_yf)
        d_steering_angle = steering_velocity
        
        d_s_simple = np.zeros(7)
        d_s_pacejka = np.zeros(7)
    
        d_s_simple[StateIndices.pose_x] = x[StateIndices.v_x]*np.cos(x[StateIndices.yaw_angle])
        d_s_simple[StateIndices.pose_y] = x[StateIndices.v_x]*np.sin(x[StateIndices.yaw_angle])
        d_s_simple[StateIndices.steering_angle] = (u[ControlIndices.desired_steering_angle] - x[StateIndices.steering_angle])
        d_s_simple[StateIndices.v_x] = d_v_x 
        d_s_simple[StateIndices.yaw_angle] = x[StateIndices.v_x]/p.l_wb*np.tan(x[StateIndices.steering_angle])
        
    
        d_s_pacejka[StateIndices.pose_x] = d_pos_x
        d_s_pacejka[StateIndices.pose_y] = d_pos_y
        d_s_pacejka[StateIndices.yaw_angle] = d_yaw_angle
        d_s_pacejka[StateIndices.v_x] = d_v_x
        d_s_pacejka[StateIndices.v_y] = d_v_y
        d_s_pacejka[StateIndices.yaw_rate] = d_yaw_rate
        d_s_pacejka[StateIndices.steering_angle] = d_steering_angle
        
        # Fuse the two models 
        low_speed_threshold = 0.5  # Below this speed, use the simple model
        high_speed_threshold = 3  # Above this speed, use the complex model
        if x[StateIndices.v_x] <= low_speed_threshold:
            weight = 0
        elif x[StateIndices.v_x] >= high_speed_threshold:
            weight = 1
        else:
            weight = (x[StateIndices.v_x] - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
        d_s = (1 - weight) * d_s_simple + weight * d_s_pacejka


        return d_s


    def step(self, state, control):
        
        p = self.vehicle_parameters
        
        for i in range(self.intermediate_steps):
            
            delta = self.dt / self.intermediate_steps * self.vehicle_dynamics(state, control)
            state = state + delta
            
            # Constraints
            state[StateIndices.yaw_angle] = (state[StateIndices.yaw_angle] + np.pi) % (2 * np.pi) - np.pi
            state[StateIndices.yaw_rate] = max(-15, min(state[StateIndices.yaw_rate], 15))
            state[StateIndices.steering_angle] = max(p.s_min, min(state[StateIndices.steering_angle], p.s_max))
        
        return state

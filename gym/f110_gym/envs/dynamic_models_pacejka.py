import math
import numpy as np  
from utilities.car_files.gym_car_parameters import GymCarParameters
class StateIndices:
    pose_x = 0
    pose_y = 1
    yaw_angle = 2
    v_x = 3
    v_y = 4
    yaw_rate = 5
    steering_angle = 6
    
    # Create a reverse mapping dictionary as a class attribute
    name_dict = {value: name for name, value in vars().items() if not name.startswith('__')}

    @classmethod
    def getStateName(cls, index):
        # Use the reverse mapping to get the state name from the index
        return cls.name_dict.get(index, "Index not found")
    
    
class ControlIndices:
    desired_steering_angle = 0
    acceleration = 1
    

class VehicleParameters:
    def __init__(self):
        self.C_0d = 0.41117415569890003
        self.C_Pf = [
            4.47161357602916,
            0.1320293068694414,
            12.267008918241816,
            1.5562751013900538
        ]
        self.C_Pr = [
            9.999999999999812,
            1.4999999999992566,
            1.3200250015860229,
            1.0999999999999999
        ]
        self.C_R = 3.693303119695026
        self.C_acc = 7.135521073243542
        self.C_d = -0.8390993957160475
        self.C_dec = 5.935972263881579
        self.I_z = 0.05797
        self.a_max = 3
        self.a_min = -3
        self.h_cg = 0.014
        self.lf = 0.162
        self.lr = 0.145
        self.l_wb = 0.307
        self.m = 3.54
        self.mu = 0.8
        
        self.v_max = 20
        self.v_min = -5



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
            
        accl = max(-5, min(accl, 10))
    
    return accl

    

def vehicle_dynamics_pacejka(x, u) -> np.ndarray:
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
            u1: longitudinal acceleration
        :param p: vehicle parameter vector 
        :param type: tire model type (linear or pacejka)
    Outputs:
        :return f: derivative of vehicle state vector
    """
    p = VehicleParameters()
    g_ = 9.81
    mu = p.mu
    B_f = p.C_Pf[0]
    C_f = p.C_Pf[1]
    D_f = p.C_Pf[2]
    E_f = p.C_Pf[3]
    B_r = p.C_Pr[0]
    C_r = p.C_Pr[1]
    D_r = p.C_Pr[2]
    E_r = p.C_Pr[3]

    lf = p.lf
    lr = p.lr
    h = p.h_cg 
    m = p.m 
    I_z = p.I_z
    
    
    acceleration_x = speed_pid(x[StateIndices.v_x], u[ControlIndices.acceleration], p)
    # acceleration_x = u[ControlIndices.acceleration]    

    # compute lateral tire slip angles
    if x[StateIndices.v_x] == 0 :
        x[StateIndices.v_x] = 1e-8 
    alpha_f = -math.atan((x[StateIndices.v_y] + x[StateIndices.yaw_rate] * lf) / (x[StateIndices.v_x])) + x[StateIndices.steering_angle]
    alpha_r = -math.atan((x[StateIndices.v_y] - x[StateIndices.yaw_rate] * lr) / x[StateIndices.v_x] )

    # compute vertical tire forces
    F_zf = m * (-u[1] * h + g_ * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g_ * lf) / (lr + lf)

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
    
    
    threshold = 0.1  # Define what "close" means, adjust as necessary
    error = u[ControlIndices.desired_steering_angle] - x[StateIndices.steering_angle]

    if error < -threshold:
        d_steering_angle = -4.5
    elif error > threshold:
        d_steering_angle = 4.5
    else:
        d_steering_angle = 1.0 * error
    
        
    
    d_s_simple = np.zeros(7)
    d_s_pacejka = np.zeros(7)
    
    low_speed_threshold = 0.5  # Below this speed, use the simple model
    high_speed_threshold = 4  # Above this speed, use the complex model
    if x[StateIndices.v_x] <= low_speed_threshold:
        weight = 0
    elif x[StateIndices.v_x] >= high_speed_threshold:
        weight = 1
    else:
        weight = (x[StateIndices.v_x] - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
        
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
        
        
    d_s = (1 - weight) * d_s_simple + weight * d_s_pacejka

    

    return d_s


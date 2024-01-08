import numpy as np

from utilities.state_utilities import *




        
class ImuSimulator:
    
    def __init__(self):
        
        self.car_state = np.zeros(9)
        
        self.d_x_car = 0
        self.d_y_car = 0    
        self.d_theta_car = 0
        
        self.imu_data = {
            "a_x": 0,
            "a_y": 0,
            "a_z": -9.81,  # Gravity
            "ang_x": 0,
            "ang_y": 0,
            "ang_z": 0,
        }
        
        print("IMU Sim initialized")
        
    
    
    
    def set_state(self, car_state):
            
        s1 = self.car_state # old car statr
        s2 = car_state # new car state
        
        
        d_x = s2[POSE_X_IDX] - s1[POSE_X_IDX]
        d_y = s2[POSE_Y_IDX] - s1[POSE_Y_IDX]
        d_theta = s2[POSE_THETA_IDX] - s1[POSE_THETA_IDX]
        # Normalize the difference in theta to be within the range -pi to pi
        d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))
                
        
        d_x_car = d_x * s1[POSE_THETA_COS_IDX] + d_y * s1[POSE_THETA_SIN_IDX]
        d_y_car = -d_x * s1[POSE_THETA_SIN_IDX] + d_y * s1[POSE_THETA_COS_IDX]
        d_theta_car = d_theta
        
        a_x_car = d_x_car - self.d_x_car
        a_y_car = d_y_car - self.d_y_car
        
        # print("v_x", car_state[LINEAR_VEL_X_IDX])
        
        self.imu_data["a_x"] = a_x_car  / (Settings.TIMESTEP_CONTROL ** 2)
        self.imu_data["a_y"] = a_y_car  / (Settings.TIMESTEP_CONTROL ** 2)
        self.imu_data["ang_z"] = d_theta_car  / Settings.TIMESTEP_CONTROL
                
        self.car_state = car_state
        self.d_x_car = d_x_car
        self.d_y_car = d_y_car
        self.d_theta_car = d_theta_car

        return self.imu_data
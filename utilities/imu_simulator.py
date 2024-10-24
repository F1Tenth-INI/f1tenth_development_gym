import numpy as np
from utilities.Settings import Settings
from utilities.state_utilities import *


class IMUSimulator:
    
    # Indices
    dx_idx = 0
    dy_idx = 1
    dtheta_idx = 2

    imu_dict_keys = ['imu_dd_x', 'imu_dd_y', 'imu_dd_yaw']
 
    def __init__(self):
        
        # Last car state variables
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_theta = 0.0
        
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_dtheta = 0.0
        
        

        

    def update_car_state(self, car_state, delta_time=Settings.TIMESTEP_CONTROL):
        """
        Update the car state and calculate the acceleration in the car's coordinate frame.
        
        @car_state: A dict with the following keys:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
        }
        @delta_time: Time difference between the current and last state (default is 1 for simplicity)
        """

        
   
        # Extract current and last state variables
        current_pose_x = car_state[POSE_X_IDX]
        current_pose_y = car_state[POSE_Y_IDX]
        current_yaw = car_state[POSE_THETA_IDX]
        
        # Calculate the change in position
        delta_pose_x = current_pose_x - self.last_x
        delta_pose_y = current_pose_y - self.last_y
        
        # Calculate the linear velocity based on the change in position
        current_linear_vel_x = delta_pose_x / delta_time
        current_linear_vel_y = delta_pose_y / delta_time
        
        # Calculate the change in orientation
        delta_yaw = current_yaw - self.last_theta
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
        
        # Calculate the angular velocity based on the change in orientation
        current_angular_vel_z = delta_yaw / delta_time
        
        delta_linear_vel_x = current_linear_vel_x - self.last_dx
        delta_linear_vel_y = current_linear_vel_y -  self.last_dy
        delta_angular_vel_z = current_angular_vel_z - self.last_dtheta
        
        
        # Calculate the acceleration in the car's coordinate frame
        accel_x = delta_linear_vel_x / delta_time
        accel_y = delta_linear_vel_y / delta_time
        accel_angular_z = delta_angular_vel_z / delta_time
        
        # Rotate the acceleration to the car's coordinate frame
        accel_x_car = accel_x * np.cos(current_yaw) + accel_y * np.sin(current_yaw)
        accel_y_car = -accel_x * np.sin(current_yaw) + accel_y * np.cos(current_yaw)
        
        # Print or store the calculated acceleration
        # print(f"Acceleration in car's frame: accel_x_car={accel_x_car}, accel_y_car={accel_y_car}, accel_angular_z={accel_angular_z}")
        
        # Update the last car state
        self.last_x = current_pose_x
        self.last_y = current_pose_y
        self.last_theta = current_yaw
        
        self.last_dx = current_linear_vel_x
        self.last_dy = current_linear_vel_y
        self.last_dtheta = current_angular_vel_z
        
        imu_array = np.array([accel_x_car, accel_y_car, accel_angular_z])
        return imu_array
    
    
    
    @staticmethod
    def array_to_dict(imu_array):
        imu_dict = {
            'imu_dd_x': imu_array[IMUSimulator.dx_idx],
            'imu_dd_y': imu_array[IMUSimulator.dy_idx],
            'imu_dd_yaw': imu_array[IMUSimulator.dtheta_idx],
        }
        
        return imu_dict

# Example usage
if __name__ == "__main__":
    imu_simulator = IMUSimulator()
    
    # Example car states
    car_state_1 = {
        'pose_x': 0.0,
        'pose_y': 0.0,
        'pose_theta': 0.0,
        'linear_vel_x': 1.0,
        'linear_vel_y': 0.0,
        'angular_vel_z': 0.1,
    }
    
    car_state_2 = {
        'pose_x': 1.0,
        'pose_y': 0.0,
        'pose_theta': 0.1,
        'linear_vel_x': 1.5,
        'linear_vel_y': 0.2,
        'angular_vel_z': 0.15,
    }
    
    imu_simulator.update_car_state(car_state_1)
    imu_simulator.update_car_state(car_state_2)
import numpy as np
from utilities.Settings import Settings
from utilities.state_utilities import *


class IMUSimulator:
    
    # Indices
    ddx_idx = 0     # Acceleration in x direction
    ddy_idx = 1     # Acceleration in y direction
    dyaw_idx = 2    # Angular velocity in z direction
 
    def __init__(self):
        
        # Last car state variables
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_avz = 0.0
        
    
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
        pose_x = car_state[POSE_X_IDX]
        pose_y = car_state[POSE_Y_IDX]
        yaw = car_state[POSE_THETA_IDX]
        
        # Calculate velocity based on the change in position
        v_x = (pose_x - self.last_x) / delta_time
        v_y = (pose_y - self.last_y) / delta_time
        
        # Calculate the change in orientation
        delta_yaw = yaw - self.last_yaw
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
        av_z = delta_yaw / delta_time
        
        # Calculate the acceleration 
        a_x = (v_x - self.last_vx) / delta_time
        a_y = (v_y -  self.last_vy) / delta_time        
        
        # Rotate the acceleration to the car's coordinate frame
        a_x_car = a_x * np.cos(yaw) + a_y * np.sin(yaw)
        a_y_car = -a_x * np.sin(yaw) + a_y * np.cos(yaw)
        
        # Print or store the calculated acceleration
        # print(f"Acceleration in car's frame: accel_x_car={a_x_car}, accel_y_car={a_y_car}, angular_vel_z={av_z}")
        
        # Update the last car state
        self.last_x = pose_x
        self.last_y = pose_y
        self.last_yaw = yaw
        
        self.last_vx = v_x
        self.last_vy = v_y
        self.last_avz = av_z
        
        imu_array = np.array([a_x_car, a_y_car, av_z])
        return imu_array
    
    
    
    @staticmethod
    def array_to_dict(imu_array):
        imu_dict = {
            'imu_a_x': imu_array[IMUSimulator.ddx_idx], 
            'imu_a_y': imu_array[IMUSimulator.ddy_idx],
            'imu_av_z': imu_array[IMUSimulator.dyaw_idx],
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
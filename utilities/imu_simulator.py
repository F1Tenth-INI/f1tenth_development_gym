import numpy as np
import math
from utilities.Settings import Settings
from utilities.state_utilities import *


class IMUSimulator:
    
    # Indices for 6-DOF IMU data (accelerometer + gyroscope + orientation)
    # Linear acceleration (3-axis)
    accel_x_idx = 0
    accel_y_idx = 1
    accel_z_idx = 2
    
    # Angular velocity (3-axis) - gyroscope
    gyro_x_idx = 3  # Roll rate
    gyro_y_idx = 4  # Pitch rate
    gyro_z_idx = 5  # Yaw rate
    
    # Euler angles (3-axis) - orientation
    euler_roll_idx = 6   # Roll angle
    euler_pitch_idx = 7  # Pitch angle
    euler_yaw_idx = 8    # Yaw angle
    
    # Quaternion (4-axis) - precise orientation
    quat_w_idx = 9
    quat_x_idx = 10
    quat_y_idx = 11
    quat_z_idx = 12
    
    # Total IMU data dimension
    IMU_DATA_DIM = 13
 
    def __init__(self, noise_level=0.01, bias_std=0.001, temperature_coeff=0.0001):
        """
        Initialize IMU simulator with realistic sensor characteristics.
        
        Args:
            noise_level (float): Standard deviation of Gaussian noise for accelerometer and gyroscope
            bias_std (float): Standard deviation of bias drift
            temperature_coeff (float): Temperature coefficient for bias drift
        """
        # Last car state variables
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_avz = 0.0
        
        # Sensor characteristics
        self.noise_level = noise_level
        self.bias_std = bias_std
        self.temperature_coeff = temperature_coeff
        
        # Initialize sensor biases (random walk)
        # Accelerometer biases
        self.accel_bias_x = np.random.normal(0, bias_std)
        self.accel_bias_y = np.random.normal(0, bias_std)
        self.accel_bias_z = np.random.normal(0, bias_std)
        
        # Gyroscope biases
        self.gyro_bias_x = np.random.normal(0, bias_std)
        self.gyro_bias_y = np.random.normal(0, bias_std)
        self.gyro_bias_z = np.random.normal(0, bias_std)
        
        # Temperature simulation
        self.temperature = 25.0  # Room temperature in Celsius
        self.temperature_drift = 0.0
        
        # First update flag
        self.first_update = True
        
    
    def update_car_state(self, car_state, delta_time=Settings.TIMESTEP_CONTROL):
        """
        Update the car state and calculate full 6-DOF IMU data with orientation.
        
        Args:
            car_state: numpy array containing car state [x, y, theta, vel, steer_angle, ang_vel, slip_angle]
            delta_time: Time difference between the current and last state
        
        Returns:
            numpy array: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, 
                         roll, pitch, yaw, quat_w, quat_x, quat_y, quat_z] with sensor noise and bias
        """
        # Handle first update
        if self.first_update:
            self.last_x = car_state[POSE_X_IDX]
            self.last_y = car_state[POSE_Y_IDX]
            self.last_yaw = car_state[POSE_THETA_IDX]
            self.last_vx = car_state[LINEAR_VEL_X_IDX] if len(car_state) > LINEAR_VEL_X_IDX else 0.0
            self.last_vy = car_state[LINEAR_VEL_Y_IDX] if len(car_state) > LINEAR_VEL_Y_IDX else 0.0
            self.last_avz = car_state[ANGULAR_VEL_Z_IDX] if len(car_state) > ANGULAR_VEL_Z_IDX else 0.0
            self.first_update = False
            return np.zeros(self.IMU_DATA_DIM)
        
        # Extract current state variables
        pose_x = car_state[POSE_X_IDX]
        pose_y = car_state[POSE_Y_IDX]
        yaw = car_state[POSE_THETA_IDX]
        
        # Get velocities from state if available, otherwise calculate from position
        if len(car_state) > LINEAR_VEL_X_IDX:
            v_x = car_state[LINEAR_VEL_X_IDX]
            v_y = car_state[LINEAR_VEL_Y_IDX]
        else:
            v_x = (pose_x - self.last_x) / delta_time
            v_y = (pose_y - self.last_y) / delta_time
        
        # Get angular velocity from state if available, otherwise calculate from orientation
        if len(car_state) > ANGULAR_VEL_Z_IDX:
            av_z = car_state[ANGULAR_VEL_Z_IDX]
        else:
            delta_yaw = yaw - self.last_yaw
            delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
            av_z = delta_yaw / delta_time
        
        # Calculate acceleration in car's local frame
        # v_x and v_y are already in the car's local frame (body frame)
        # No coordinate transformation needed - they are already in the correct frame
        a_x_car = (v_x - self.last_vx) / delta_time
        a_y_car = (v_y - self.last_vy) / delta_time
        
        # For 3D simulation, we need to simulate pitch and roll
        # In a 2D simulation, we'll add small random variations
        pitch = np.random.normal(0, 0.01)  # Small pitch variation
        roll = np.random.normal(0, 0.01)   # Small roll variation
        
        # Calculate 3D acceleration including gravity
        # Gravity affects the Z-axis based on pitch and roll
        gravity_x = 9.81 * np.sin(pitch)
        gravity_y = 9.81 * np.sin(roll) * np.cos(pitch)
        gravity_z = 9.81 * np.cos(roll) * np.cos(pitch)
        
        a_x_3d = a_x_car + gravity_x
        a_y_3d = a_y_car + gravity_y
        a_z_3d = gravity_z  # Vertical acceleration due to gravity
        
        # Calculate 3D angular velocities
        # In 2D simulation, we only have yaw rate, but add small pitch/roll rates
        gyro_x = np.random.normal(0, 0.01)  # Small roll rate
        gyro_y = np.random.normal(0, 0.01)  # Small pitch rate
        gyro_z = av_z  # Actual yaw rate from simulation
        
        # Calculate Euler angles (roll, pitch, yaw)
        euler_roll = roll
        euler_pitch = pitch
        euler_yaw = yaw
        
        # Calculate quaternion from Euler angles
        quat_w, quat_x, quat_y, quat_z = self._euler_to_quaternion(euler_roll, euler_pitch, euler_yaw)
        
        # Update temperature drift (simulate temperature changes)
        self.temperature_drift += np.random.normal(0, 0.1)
        self.temperature = 25.0 + self.temperature_drift
        
        # Update sensor biases (random walk)
        self.accel_bias_x += np.random.normal(0, self.bias_std * delta_time)
        self.accel_bias_y += np.random.normal(0, self.bias_std * delta_time)
        self.accel_bias_z += np.random.normal(0, self.bias_std * delta_time)
        self.gyro_bias_x += np.random.normal(0, self.bias_std * delta_time)
        self.gyro_bias_y += np.random.normal(0, self.bias_std * delta_time)
        self.gyro_bias_z += np.random.normal(0, self.bias_std * delta_time)
        
        # Add temperature-dependent bias
        temp_factor = 1.0 + self.temperature_coeff * (self.temperature - 25.0)
        
        # Apply noise and bias to measurements
        imu_data = np.zeros(self.IMU_DATA_DIM)
        
        # Accelerometer data
        imu_data[self.accel_x_idx] = a_x_3d + self.accel_bias_x * temp_factor + np.random.normal(0, self.noise_level)
        imu_data[self.accel_y_idx] = a_y_3d + self.accel_bias_y * temp_factor + np.random.normal(0, self.noise_level)
        imu_data[self.accel_z_idx] = a_z_3d + self.accel_bias_z * temp_factor + np.random.normal(0, self.noise_level)
        
        # Gyroscope data
        imu_data[self.gyro_x_idx] = gyro_x + self.gyro_bias_x * temp_factor + np.random.normal(0, self.noise_level)
        imu_data[self.gyro_y_idx] = gyro_y + self.gyro_bias_y * temp_factor + np.random.normal(0, self.noise_level)
        imu_data[self.gyro_z_idx] = gyro_z + self.gyro_bias_z * temp_factor + np.random.normal(0, self.noise_level)
        
        # Euler angles
        imu_data[self.euler_roll_idx] = euler_roll
        imu_data[self.euler_pitch_idx] = euler_pitch
        imu_data[self.euler_yaw_idx] = euler_yaw
        
        # Quaternion
        imu_data[self.quat_w_idx] = quat_w
        imu_data[self.quat_x_idx] = quat_x
        imu_data[self.quat_y_idx] = quat_y
        imu_data[self.quat_z_idx] = quat_z
        
        # Update the last car state
        self.last_x = pose_x
        self.last_y = pose_y
        self.last_yaw = yaw
        self.last_vx = v_x
        self.last_vy = v_y
        self.last_avz = av_z
        
        return imu_data
    
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion.
        
        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians  
            yaw: Yaw angle in radians
            
        Returns:
            tuple: (w, x, y, z) quaternion components
        """
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return w, x, y, z
    
    def reset(self):
        """Reset the IMU simulator to initial state."""
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_avz = 0.0
        self.first_update = True
        
        # Reset biases
        self.accel_bias_x = np.random.normal(0, self.bias_std)
        self.accel_bias_y = np.random.normal(0, self.bias_std)
        self.accel_bias_z = np.random.normal(0, self.bias_std)
        self.gyro_bias_x = np.random.normal(0, self.bias_std)
        self.gyro_bias_y = np.random.normal(0, self.bias_std)
        self.gyro_bias_z = np.random.normal(0, self.bias_std)
        
        # Reset temperature
        self.temperature = 25.0
        self.temperature_drift = 0.0
    
    def get_sensor_info(self):
        """Get current sensor information for debugging."""
        return {
            'temperature': self.temperature,
            'accel_bias_x': self.accel_bias_x,
            'accel_bias_y': self.accel_bias_y,
            'accel_bias_z': self.accel_bias_z,
            'gyro_bias_x': self.gyro_bias_x,
            'gyro_bias_y': self.gyro_bias_y,
            'gyro_bias_z': self.gyro_bias_z,
            'noise_level': self.noise_level
        }
    
    def calibrate(self, num_samples=100):
        """
        Perform basic calibration by averaging measurements over time.
        This simulates a simple bias estimation.
        """
        print(f"Calibrating IMU with {num_samples} samples...")
        # In a real implementation, this would collect samples and estimate bias
        # For simulation, we'll just reduce the bias slightly
        self.accel_bias_x *= 0.5
        self.accel_bias_y *= 0.5
        self.accel_bias_z *= 0.5
        self.gyro_bias_x *= 0.5
        self.gyro_bias_y *= 0.5
        self.gyro_bias_z *= 0.5
        print("IMU calibration complete.")
    
    @staticmethod
    def array_to_dict(imu_array):
        """Convert IMU array to dictionary format."""
        imu_dict = {
            # Accelerometer data
            'imu_a_x': imu_array[IMUSimulator.accel_x_idx],
            'imu_a_y': imu_array[IMUSimulator.accel_y_idx],
            'imu_a_z': imu_array[IMUSimulator.accel_z_idx],
            
            # Gyroscope data
            'imu_gyro_x': imu_array[IMUSimulator.gyro_x_idx],
            'imu_gyro_y': imu_array[IMUSimulator.gyro_y_idx],
            'imu_gyro_z': imu_array[IMUSimulator.gyro_z_idx],
            
            # Euler angles
            'imu_roll': imu_array[IMUSimulator.euler_roll_idx],
            'imu_pitch': imu_array[IMUSimulator.euler_pitch_idx],
            'imu_yaw': imu_array[IMUSimulator.euler_yaw_idx],
            
            # Quaternion
            'imu_quat_w': imu_array[IMUSimulator.quat_w_idx],
            'imu_quat_x': imu_array[IMUSimulator.quat_x_idx],
            'imu_quat_y': imu_array[IMUSimulator.quat_y_idx],
            'imu_quat_z': imu_array[IMUSimulator.quat_z_idx],
        }
        return imu_dict
    
    @staticmethod
    def dict_to_array(imu_dict):
        """Convert IMU dictionary to array format."""
        return np.array([
            imu_dict['imu_a_x'],
            imu_dict['imu_a_y'],
            imu_dict['imu_a_z'],
            imu_dict['imu_gyro_x'],
            imu_dict['imu_gyro_y'],
            imu_dict['imu_gyro_z'],
            imu_dict['imu_roll'],
            imu_dict['imu_pitch'],
            imu_dict['imu_yaw'],
            imu_dict['imu_quat_w'],
            imu_dict['imu_quat_x'],
            imu_dict['imu_quat_y'],
            imu_dict['imu_quat_z']
        ])

# Example usage
if __name__ == "__main__":
    # Create IMU simulator with custom noise levels
    imu_simulator = IMUSimulator(noise_level=0.02, bias_std=0.005)
    
    # Example car states (using proper state array format)
    car_state_1 = create_car_state({
        'pose_x': 0.0, 'pose_y': 0.0, 'pose_theta': 0.0,
        'linear_vel_x': 1.0, 'linear_vel_y': 0.0, 'angular_vel_z': 0.1,
        'slip_angle': 0.0, 'steering_angle': 0.0
    })
    car_state_2 = create_car_state({
        'pose_x': 1.0, 'pose_y': 0.0, 'pose_theta': 0.1,
        'linear_vel_x': 1.5, 'linear_vel_y': 0.2, 'angular_vel_z': 0.15,
        'slip_angle': 0.0, 'steering_angle': 0.1
    })
    car_state_3 = create_car_state({
        'pose_x': 2.0, 'pose_y': 0.5, 'pose_theta': 0.2,
        'linear_vel_x': 2.0, 'linear_vel_y': 0.3, 'angular_vel_z': 0.1,
        'slip_angle': 0.0, 'steering_angle': 0.05
    })
    
    print("Enhanced IMU Sensor Simulation Demo (6-DOF + Orientation)")
    print("=" * 60)
    
    # First update (should return zeros)
    imu_data_1 = imu_simulator.update_car_state(car_state_1)
    print(f"First update (zeros): {imu_data_1[:6]}...")  # Show first 6 values
    
    # Second update
    imu_data_2 = imu_simulator.update_car_state(car_state_2)
    print(f"Second update: {imu_data_2[:6]}...")
    
    # Third update
    imu_data_3 = imu_simulator.update_car_state(car_state_3)
    print(f"Third update: {imu_data_3[:6]}...")
    
    # Convert to dictionary format
    imu_dict = imu_simulator.array_to_dict(imu_data_3)
    print(f"\nIMU Dictionary (sample):")
    print(f"  Accel: X={imu_dict['imu_a_x']:.3f}, Y={imu_dict['imu_a_y']:.3f}, Z={imu_dict['imu_a_z']:.3f}")
    print(f"  Gyro:  X={imu_dict['imu_gyro_x']:.3f}, Y={imu_dict['imu_gyro_y']:.3f}, Z={imu_dict['imu_gyro_z']:.3f}")
    print(f"  Euler: Roll={imu_dict['imu_roll']:.3f}, Pitch={imu_dict['imu_pitch']:.3f}, Yaw={imu_dict['imu_yaw']:.3f}")
    
    # Show sensor info
    sensor_info = imu_simulator.get_sensor_info()
    print(f"\nSensor Info:")
    print(f"  Temperature: {sensor_info['temperature']:.1f}°C")
    print(f"  Noise level: {sensor_info['noise_level']}")
    
    # Demonstrate calibration
    imu_simulator.calibrate()
    
    # Test reset
    imu_simulator.reset()
    print("\nIMU reset complete.")
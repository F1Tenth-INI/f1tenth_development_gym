#!/usr/bin/env python3
"""
IMU Utilities for data conversion and processing.
This module provides utility functions for working with IMU data.
"""

import numpy as np


class IMUUtilities:
    """Utility class for IMU data processing and conversion."""
    
    # Indices for 6-DOF IMU data (accelerometer + gyroscope + orientation)
    # Linear acceleration (3-axis)
    ACCEL_X_IDX = 0
    ACCEL_Y_IDX = 1
    ACCEL_Z_IDX = 2
    
    # Angular velocity (3-axis) - gyroscope
    GYRO_X_IDX = 3  # Roll rate
    GYRO_Y_IDX = 4  # Pitch rate
    GYRO_Z_IDX = 5  # Yaw rate
    
    # Euler angles (3-axis) - orientation
    EULER_ROLL_IDX = 6   # Roll angle
    EULER_PITCH_IDX = 7  # Pitch angle
    EULER_YAW_IDX = 8    # Yaw angle
    
    # Quaternion (4-axis) - precise orientation
    QUAT_W_IDX = 9
    QUAT_X_IDX = 10
    QUAT_Y_IDX = 11
    QUAT_Z_IDX = 12
    
    # Total IMU data dimension
    IMU_DATA_DIM = 13
    
    @staticmethod
    def imu_array_to_dict(imu_array):
        """
        Convert IMU array to dictionary format for recording.
        
        Args:
            imu_array: numpy array containing IMU data [accel_x, accel_y, accel_z, 
                      gyro_x, gyro_y, gyro_z, roll, pitch, yaw, quat_w, quat_x, quat_y, quat_z]
        
        Returns:
            dict: Dictionary with IMU data keys and values
        """
        if len(imu_array) < IMUUtilities.IMU_DATA_DIM:
            # Handle old format or incomplete data
            return IMUUtilities._handle_legacy_format(imu_array)
        
        imu_dict = {
            # Accelerometer data
            'imu_a_x': imu_array[IMUUtilities.ACCEL_X_IDX],
            'imu_a_y': imu_array[IMUUtilities.ACCEL_Y_IDX],
            'imu_a_z': imu_array[IMUUtilities.ACCEL_Z_IDX],
            
            # Gyroscope data
            'imu_gyro_x': imu_array[IMUUtilities.GYRO_X_IDX],
            'imu_gyro_y': imu_array[IMUUtilities.GYRO_Y_IDX],
            'imu_gyro_z': imu_array[IMUUtilities.GYRO_Z_IDX],
            
            # Euler angles
            'imu_roll': imu_array[IMUUtilities.EULER_ROLL_IDX],
            'imu_pitch': imu_array[IMUUtilities.EULER_PITCH_IDX],
            'imu_yaw': imu_array[IMUUtilities.EULER_YAW_IDX],
            
            # Quaternion
            'imu_quat_w': imu_array[IMUUtilities.QUAT_W_IDX],
            'imu_quat_x': imu_array[IMUUtilities.QUAT_X_IDX],
            'imu_quat_y': imu_array[IMUUtilities.QUAT_Y_IDX],
            'imu_quat_z': imu_array[IMUUtilities.QUAT_Z_IDX],
        }
        return imu_dict
    
    @staticmethod
    def _handle_legacy_format(imu_array):
        """
        Handle legacy 3-DOF IMU format for backward compatibility.
        
        Args:
            imu_array: numpy array with 3 elements [accel_x, accel_y, angular_vel_z]
        
        Returns:
            dict: Dictionary with legacy format converted to new format
        """
        if len(imu_array) >= 3:
            # Legacy format: [accel_x, accel_y, angular_vel_z]
            imu_dict = {
                'imu_a_x': imu_array[0],
                'imu_a_y': imu_array[1], 
                'imu_a_z': 0.0,
                'imu_gyro_x': 0.0,
                'imu_gyro_y': 0.0,
                'imu_gyro_z': imu_array[2],  # Use angular_vel_z as gyro_z
                'imu_roll': 0.0,
                'imu_pitch': 0.0,
                'imu_yaw': 0.0,
                'imu_quat_w': 1.0,
                'imu_quat_x': 0.0,
                'imu_quat_y': 0.0,
                'imu_quat_z': 0.0,
            }
        else:
            # Fallback to zeros
            imu_dict = {
                'imu_a_x': 0.0,
                'imu_a_y': 0.0,
                'imu_a_z': 0.0,
                'imu_gyro_x': 0.0,
                'imu_gyro_y': 0.0,
                'imu_gyro_z': 0.0,
                'imu_roll': 0.0,
                'imu_pitch': 0.0,
                'imu_yaw': 0.0,
                'imu_quat_w': 1.0,
                'imu_quat_x': 0.0,
                'imu_quat_y': 0.0,
                'imu_quat_z': 0.0,
            }
        return imu_dict
    
    @staticmethod
    def dict_to_imu_array(imu_dict):
        """
        Convert IMU dictionary to array format.
        
        Args:
            imu_dict: Dictionary with IMU data keys
        
        Returns:
            numpy array: IMU data array
        """
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
    
    @staticmethod
    def get_imu_data_summary(imu_array):
        """
        Get a summary of IMU data for debugging/logging.
        
        Args:
            imu_array: numpy array containing IMU data
        
        Returns:
            dict: Summary with key statistics
        """
        if len(imu_array) < IMUUtilities.IMU_DATA_DIM:
            return {"error": "Incomplete IMU data", "length": len(imu_array)}
        
        return {
            "accel_magnitude": np.sqrt(imu_array[0]**2 + imu_array[1]**2 + imu_array[2]**2),
            "gyro_magnitude": np.sqrt(imu_array[3]**2 + imu_array[4]**2 + imu_array[5]**2),
            "yaw_rate": imu_array[IMUUtilities.GYRO_Z_IDX],
            "yaw_angle": imu_array[IMUUtilities.EULER_YAW_IDX],
            "quat_magnitude": np.sqrt(imu_array[9]**2 + imu_array[10]**2 + imu_array[11]**2 + imu_array[12]**2)
        }
    
    @staticmethod
    def validate_imu_data(imu_array):
        """
        Validate IMU data for reasonable values.
        
        Args:
            imu_array: numpy array containing IMU data
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if len(imu_array) != IMUUtilities.IMU_DATA_DIM:
            return False, f"Expected {IMUUtilities.IMU_DATA_DIM} elements, got {len(imu_array)}"
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(imu_array)):
            return False, "IMU data contains NaN or infinite values"
        
        # Check for reasonable acceleration values (should be around 9.81 m/s² due to gravity)
        accel_magnitude = np.sqrt(imu_array[0]**2 + imu_array[1]**2 + imu_array[2]**2)
        if accel_magnitude < 5.0 or accel_magnitude > 50.0:
            return False, f"Unreasonable acceleration magnitude: {accel_magnitude:.2f} m/s²"
        
        # Check quaternion normalization (should be close to 1.0)
        quat_magnitude = np.sqrt(imu_array[9]**2 + imu_array[10]**2 + imu_array[11]**2 + imu_array[12]**2)
        if abs(quat_magnitude - 1.0) > 0.1:
            return False, f"Quaternion not normalized: magnitude = {quat_magnitude:.3f}"
        
        return True, "IMU data is valid"


# Example usage
if __name__ == "__main__":
    # Test the IMU utilities
    print("IMU Utilities Test")
    print("=" * 30)
    
    # Create sample IMU data
    sample_imu = np.array([
        1.0, 2.0, 9.81,  # accel x, y, z
        0.1, 0.2, 0.3,   # gyro x, y, z
        0.05, 0.1, 0.2,  # roll, pitch, yaw
        0.995, 0.05, 0.1, 0.0  # quat w, x, y, z
    ])
    
    # Test conversion
    imu_dict = IMUUtilities.imu_array_to_dict(sample_imu)
    print("IMU Dictionary:")
    for key, value in imu_dict.items():
        print(f"  {key}: {value:.3f}")
    
    # Test validation
    is_valid, message = IMUUtilities.validate_imu_data(sample_imu)
    print(f"\nValidation: {is_valid} - {message}")
    
    # Test summary
    summary = IMUUtilities.get_imu_data_summary(sample_imu)
    print(f"\nSummary: {summary}")
    
    print("\n✅ IMU Utilities test completed!")

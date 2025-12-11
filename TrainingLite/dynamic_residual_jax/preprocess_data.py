import os
import numpy as np
import pandas as pd
from utilities.imu_simulator import *
from utilities.state_utilities import STATE_VARIABLES
from utilities.ekf import EKF, alpha_beta_filter


imu_simulator = IMUSimulator()




def car_state_from_row(row: pd.Series) -> dict:
    # Select all the cols defined in STATE_VARIABLES from the dataframe row
    state = []
    for var in STATE_VARIABLES:
        if var in row.index:
            state.append(row[var])
    return state

def cleanup(df : pd.DataFrame) -> pd.DataFrame:
    # delete all columns with cs_a_, cs_t_, Lidar in the name
    cols_to_delete = [ 'cs_a_', 'cs_t_', 'LIDAR' , 'WYP']
    for col in df.columns:
        if any(substring in col for substring in cols_to_delete):
            df = df.drop(columns=[col])
    return df

def add_simulated_imu_data(df: pd.DataFrame) -> pd.DataFrame:
    imu_simulator.reset()
    simulated_imu_data = {
        'imu_accel_x': [],
        'imu_accel_y': [],
        'imu_gyro_z': []
    }
    for index, row in df.iterrows():
        # Extract necessary data from the dataframe row
        state = car_state_from_row(row)
        
        # Simulate IMU data
        imu_data = imu_simulator.update_car_state(state, 0.04)  # Assuming a timestep of 0.04s
        
        
        # Append simulated data to lists
        simulated_imu_data['imu_accel_x'].append(imu_data[imu_simulator.accel_x_idx])
        simulated_imu_data['imu_accel_y'].append(imu_data[imu_simulator.accel_y_idx])
        simulated_imu_data['imu_gyro_z'].append(imu_data[imu_simulator.gyro_z_idx])
        
    # Add simulated IMU data to the dataframe
    df['imu_accel_x'] = simulated_imu_data['imu_accel_x']
    df['imu_accel_y'] = simulated_imu_data['imu_accel_y']
    df['imu_gyro_z'] = simulated_imu_data['imu_gyro_z']
    
    return df
    
def filter_imu_data(df: pd.DataFrame) -> pd.DataFrame:
    imu1_a_x_data = df['imu_accel_x'].tolist()
    imu1_a_y_data = df['imu_accel_y'].tolist()
    imu1_gyro_z_data = df['imu_gyro_z'].tolist()
    
    imu1_a_x_filtered = alpha_beta_filter(imu1_a_x_data, alpha=0.2)
    imu1_a_y_filtered = alpha_beta_filter(imu1_a_y_data, alpha=0.2)
    imu1_gyro_z_filtered = alpha_beta_filter(imu1_gyro_z_data, alpha=0.2)
    
    df['imu_accel_x'] = imu1_a_x_filtered
    df['imu_accel_y'] = imu1_a_y_filtered
    df['imu_gyro_z'] = imu1_gyro_z_filtered
    
    return df
       
def filter_v_x(df: pd.DataFrame) -> pd.DataFrame:
    pose_x_data = df['pose_x'].tolist()
    linear_vel_x_data = df['linear_vel_x'].tolist()
    imu1_a_x_data = df['imu_accel_x'].tolist()
    time_data = df['time'].tolist() 
    
    imu1_a_x_filtered = df['imu_accel_x'].tolist()
    
    
    ekf = EKF(
        initial_x=pose_x_data[0], 
        initial_vx=linear_vel_x_data[0],
        process_noise_pos=0.01,      # Lower = more trust in IMU prediction
        process_noise_vel=0.01,      # Lower = more trust in IMU prediction
        measurement_noise_pos=0.1,    # Higher = less weight to position
        measurement_noise_vel=0.1     # Higher = less weight to velocity
    )
    linear_vel_x_ekf = []

    for i in range(len(df)):
        if i == 0:
            # Initialize with first measurement
            linear_vel_x_ekf.append(linear_vel_x_data[0])
        else:
            # Calculate time step
            dt = time_data[i] - time_data[i-1]
            
            # Prediction step using filtered IMU acceleration
            ekf.predict(imu1_a_x_filtered[i-1], dt)
            
            # Update step using position and velocity measurements
            ekf.update(pose_x_data[i], linear_vel_x_data[i])
            
            # Store the estimated velocity
            linear_vel_x_ekf.append(ekf.get_velocity())
            
    df['linear_vel_x_ekf'] = linear_vel_x_ekf
    return df


if __name__ == "__main__":
    # Input CSV path
    input_csv = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/AnalyseData/PhysicalData/2025_11_28/2025-11-28_07-58-30_Recording1_0_IPZ10_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None__filtered.csv"
    
    # Output directory
    output_dir = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/TrainingLite/dynamic_residual_jax/training_data"
    
    # Process the data
    df = pd.read_csv(input_csv, comment='#')
    df = cleanup(df)
    df = add_simulated_imu_data(df)
    df = filter_imu_data(df)
    df = filter_v_x(df)
    
    # Save dataframe
    output_path = os.path.join(output_dir, "processed_data.csv")
   
    df.to_csv(output_path, index=False)
    print(f"\nDone! Training data saved to: {output_path}")


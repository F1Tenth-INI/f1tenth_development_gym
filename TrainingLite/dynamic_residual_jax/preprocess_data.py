import os
import numpy as np
import pandas as pd
from utilities.imu_simulator import *
from utilities.state_utilities import STATE_VARIABLES
from utilities.ekf import EKF, alpha_beta_filter, moving_average_zero_phase
from sim.f110_sim.envs.car_model_jax import (car_steps_sequential_jax)
from utilities.car_files.vehicle_parameters import VehicleParameters

imu_simulator = IMUSimulator()




def car_state_from_row(row: pd.Series) -> dict:
    # Select all the cols defined in STATE_VARIABLES from the dataframe row
    state = []
    for var in STATE_VARIABLES:
        if var in row.index:
            state.append(row[var])
    return state

def control_from_row(row: pd.Series) -> dict:
    control = []
    control.append(row['angular_control_executed'])
    control.append(row['translational_control_executed'])
  
        
    return control

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
    
    imu1_a_x_filtered = moving_average_zero_phase(imu1_a_x_data, window_size=5)
    imu1_a_y_filtered = moving_average_zero_phase(imu1_a_y_data, window_size=5)
    imu1_gyro_z_filtered = moving_average_zero_phase(imu1_gyro_z_data, window_size=5)
    
    df['imu_accel_x'] = imu1_a_x_filtered
    df['imu_accel_y'] = imu1_a_y_filtered
    df['imu_gyro_z'] = imu1_gyro_z_filtered
    
    return df
       
def filter_states(df: pd.DataFrame) -> pd.DataFrame:
    
    # angular_vel_z
    angular_vel_z_data = df['angular_vel_z'].tolist()
    angular_vel_z_filtered = moving_average_zero_phase(angular_vel_z_data, window_size=5)
    df['angular_vel_z'] = angular_vel_z_filtered
    
    #linear_vel_x
    linear_vel_x_data = df['linear_vel_x'].tolist()
    linear_vel_x_filtered = moving_average_zero_phase(linear_vel_x_data, window_size=7)
    df['linear_vel_x'] = linear_vel_x_filtered
    
    # linear_vel_y
    linear_vel_y_data = df['linear_vel_y'].tolist()
    linear_vel_y_filtered = moving_average_zero_phase(linear_vel_y_data, window_size=7)
    df['linear_vel_y'] = linear_vel_y_filtered
    
    return df

       
def efk_filter_v_x(df: pd.DataFrame) -> pd.DataFrame:
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
            
    df['linear_vel_x'] = linear_vel_x_ekf
    return df

def add_state_deltas(df: pd.DataFrame) -> pd.DataFrame:
    for index, row in df.iterrows():
        if index == len(df) - 1:
            for var in STATE_VARIABLES:
                df.at[index, f'delta_{var}'] = 0.0
        else:
            for var in STATE_VARIABLES:
                delta =  df.at[index + 1 , var] - df.at[index , var]
                df.at[index, f'delta_{var}'] = delta / 0.04
    return df


def filter_deltas(df: pd.DataFrame) -> pd.DataFrame:
    for var in ['linear_vel_x', 'linear_vel_y', 'angular_vel_z']:
        delta_var_data = df[f'delta_{var}'].tolist()
        delta_var_filtered = moving_average_zero_phase(delta_var_data, window_size=7)
        df[f'delta_{var}'] = delta_var_filtered
    return df

def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
            
    car_params = VehicleParameters()
        
    for index, row in df.iterrows():
        # Extract necessary data from the dataframe row
        state = car_state_from_row(row)
        control = control_from_row(row)
        
        # Simulate next state
        next_states = car_steps_sequential_jax(
            s0=np.array(state),
            Q_sequence=np.array([control]),
            car_params=car_params.to_np_array(),
            dt=0.04,
            horizon=1,
            model_type='pacejka',
            intermediate_steps=1
        )
        
        next_state = next_states[-1]
        # Append simulated data to lists
        for i, var in enumerate(STATE_VARIABLES):
            df.at[index, f'predicted_{var}_0'] = next_state[i]
            df.at[index, f'predicted_delta_{var}_0'] = (next_state[i] - state[i]) / 0.04  # Placeholder, will be computed later
            
        # Now the predicted next_state and plredicted_delta is still on the same line as the actual state and control. Needs to be on the next line.
        df[f'predicted_{var}_0'] = df[f'predicted_{var}_0'].shift(1)
        df[f'predicted_delta_{var}_0'] = df[f'predicted_delta_{var}_0'].shift(1)
            
    return df


def calculate_residuals(df: pd.DataFrame) -> pd.DataFrame:
    for var in ['delta_angular_vel_z', 'delta_linear_vel_x', 'delta_linear_vel_y', 'delta_pose_theta_sin', 'delta_pose_theta_cos']:
        for j in range(1):  # horizon
            predicted_col = f'predicted_{var}_{j}'
            if predicted_col in df.columns:
                # Residual is the difference between ctual delta and the predicted delta
                df[f'residual_{var}_{j}'] = df[var] - df[predicted_col]
    return df


if __name__ == "__main__":
    # Input CSV path
    input_csv = "AnalyseData/PhysicalData/2025_11_28/2025-11-28_07-58-30_Recording1_0_IPZ10_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None_.csv"
    # Output directory
    output_dir = "TrainingLite/dynamic_residual_jax/training_data"
    
    # Process the data
    df = pd.read_csv(input_csv, comment='#')
    df = cleanup(df)
    df = add_simulated_imu_data(df)
    df = filter_imu_data(df)
    df = filter_states(df)
    df = add_state_deltas(df)
    df = filter_deltas(df)
    df = add_predictions(df)
    df = calculate_residuals(df)
    
    # Save dataframe
    output_path = os.path.join(output_dir, "processed_data.csv")
   
    df.to_csv(output_path, index=False)
    print(f"\nDone! Training data saved to: {output_path}")


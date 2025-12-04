"""
Script to prepare training data for dynamic residual model.

Reads physical recording CSV, extracts states and controls, uses dynamic model
to compute expected states from previous rows' state and control.
Each row contains: current state, control, and expected state (from previous row).
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
import os
import sys

from scipy.signal import lfilter, butter


# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Verify the path is correct
if not os.path.exists(os.path.join(project_root, 'sim')):
    raise RuntimeError(f"Project root not found correctly. Expected 'sim' directory at {project_root}")

from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax
from utilities.car_files.vehicle_parameters import VehicleParameters


def read_csv_with_comments(filepath):
    """Read CSV file that may have comment lines starting with '#'."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the first non-comment line (header)
    header_idx = None
    for i, line in enumerate(lines):
        if not line.strip().startswith('#'):
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("No header found in CSV file")
    
    # Read CSV starting from header
    df = pd.read_csv(filepath, skiprows=header_idx)
    return df


def extract_state_from_row(row):
    """Extract state array from a DataFrame row in the correct order."""
    state = np.array([
        row['angular_vel_z'],
        row['linear_vel_x'],
        row['linear_vel_y'],
        row['pose_theta'],
        row['pose_theta_cos'],
        row['pose_theta_sin'],
        row['pose_x'],
        row['pose_y'],
        row['slip_angle'],
        row['steering_angle']
    ], dtype=np.float32)
    return state


def extract_control_from_row(row):
    """Extract control array from a DataFrame row."""
    control = np.array([
        row['angular_control_executed'],
        row['translational_control_executed']
    ], dtype=np.float32)
    return control


def process_data(input_csv_path, output_dir, car_param_file='gym_car_parameters.yml', dt=0.04):
    """
    Process CSV data to create training dataset.
    
    For each row i (starting from row 1):
    - Current state: from row i
    - Control: from row i
    - Expected state: computed from row i-1's state and control using dynamic model
    
    Args:
        input_csv_path: Path to input CSV file
        output_dir: Directory to save output CSV
        car_param_file: Car parameter YAML file name
        dt: Time step (default 0.04s)
    """
    print(f"Reading CSV from: {input_csv_path}")
    df = read_csv_with_comments(input_csv_path)
    print(f"Loaded {len(df)} rows")

    # Compensate or delay by shifting control signals by 1 row
    # df['angular_control_executed'] = df['angular_control_executed'].shift(4)
    # df['translational_control_executed'] = df['translational_control_executed'].shift(4)
    
    
    # Load car parameters
    print(f"Loading car parameters from: {car_param_file}")
    car_params_obj = VehicleParameters(car_param_file)
    car_params = jnp.array(car_params_obj.to_np_array(), dtype=jnp.float32)
    
    # Extract state and control columns
    state_cols = [
        'angular_vel_z', 'linear_vel_x', 'linear_vel_y', 'pose_theta',
        'pose_theta_cos', 'pose_theta_sin', 'pose_x', 'pose_y',
        'slip_angle', 'steering_angle'
    ]
    control_cols = ['angular_control_executed', 'translational_control_executed']
    
    # Verify all required columns exist
    missing_state = [col for col in state_cols if col not in df.columns]
    missing_control = [col for col in control_cols if col not in df.columns]
    
    if missing_state:
        raise ValueError(f"Missing state columns: {missing_state}")
    if missing_control:
        raise ValueError(f"Missing control columns: {missing_control}")
    
    # Apply delay-free lowpass filter to specified signals before processing
    filter_signals = ['linear_vel_x', 'linear_vel_y']
    # filter_signals = []
    filter_cutoff_hz = 1.0
    filter_order = 2

    sampling_rate = 1.0 / dt
    nyquist = sampling_rate / 2.0
    normalized_cutoff = filter_cutoff_hz / nyquist
    
    # df['linear_vel_y'] = -df['linear_vel_y'] * 0.1
    
    # Design Butterworth lowpass filter (causal, minimal delay)
    b, a = butter(filter_order, normalized_cutoff, btype='low')
    
    # Apply forward-only filter to each specified signal (causal, delay-free for practical purposes)
    for signal in filter_signals:
        df[signal] = lfilter(b, a, df[signal])
       
    # Initialize expected state columns in DataFrame
    for col in state_cols:
        df[f'expected_state_{col}'] = np.nan
    
    print("Processing rows...")
    # Start from row 1 (skip row 0 since we need previous row to compute expected state)
    valid_indices = []
    for idx in range(1, len(df)):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(df) - 1} rows")
        
        current_row = df.iloc[idx]
        previous_row = df.iloc[idx - 1]
        
        # Extract previous state and control to compute expected state
        previous_state = extract_state_from_row(previous_row)
        previous_control = extract_control_from_row(previous_row)
        
        # Check for NaN values
        if (np.any(np.isnan(previous_state)) or np.any(np.isnan(previous_control))):
            print(f"  Warning: NaN values found in row {idx}, skipping...")
            continue
        
        # Convert previous state and control to JAX arrays
        jax_prev_state = jnp.array(previous_state, dtype=jnp.float32)
        jax_prev_control = jnp.array(previous_control, dtype=jnp.float32)
        
        # Compute expected state from previous row's state and control
        expected_state = car_dynamics_pacejka_jax(jax_prev_state, jax_prev_control, car_params, dt)
        expected_state_np = np.array(expected_state)
        
        # Store expected state directly in DataFrame
        for i, col in enumerate(state_cols):
            df.loc[idx, f'expected_state_{col}'] = expected_state_np[i]
        
        valid_indices.append(idx)

    print(f"Processed {len(valid_indices)} valid rows out of {len(df) - 1} total rows (excluding first row)")
    
    # Use pure pandas operations to compute deltas and residuals in full DataFrame
    for col in state_cols:
        if col == 'pose_theta':
            # Handle angular wrap-around for pose_theta (radians)
            # Delta state: current - previous (using shift), normalized to [-pi, pi]
            delta = df[col] - df[col].shift(1)
            df[f'delta_state_{col}'] = np.arctan2(np.sin(delta), np.cos(delta))
            df[f'change_rate_{col}'] = df[f'delta_state_{col}'] / dt
            
            # Expected state delta: expected - previous state, normalized to [-pi, pi]
            delta_expected = df[f'expected_state_{col}'] - df[col].shift(1)
            df[f'expected_state_delta_{col}'] = np.arctan2(np.sin(delta_expected), np.cos(delta_expected))
            df[f'expected_state_change_rate_{col}'] = df[f'expected_state_delta_{col}'] / dt
            
            # Residual: current - expected, normalized to [-pi, pi]
            residual = df[col] - df[f'expected_state_{col}']
            df[f'residual_state_{col}'] = np.arctan2(np.sin(residual), np.cos(residual))
        else:
            # Delta state: current - previous (using shift)
            df[f'delta_state_{col}'] = df[col] - df[col].shift(1)
            df[f'change_rate_{col}'] = df[f'delta_state_{col}'] / dt
            
            # Expected state delta: expected - previous state
            df[f'expected_state_delta_{col}'] = df[f'expected_state_{col}'] - df[col].shift(1)
            df[f'expected_state_change_rate_{col}'] = df[f'expected_state_delta_{col}'] / dt
            
            # Residual: current - expected
            df[f'residual_state_{col}'] = df[col] - df[f'expected_state_{col}']
    
    # Create output DataFrame with only valid rows
    output_df = df.iloc[valid_indices].copy()
    
    # Rename control columns
    output_df = output_df.rename(columns={
        'angular_control_executed': 'angular_control_executed',
        'translational_control_executed': 'translational_control_executed'
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    input_basename = os.path.basename(input_csv_path)
    output_filename = input_basename.replace('.csv', '_training_data.csv')
    output_path = os.path.join(output_dir, output_filename)
    
    # Save to CSV
    print(f"Saving training data to: {output_path}")
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} rows to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Input CSV path
    input_csv = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/AnalyseData/PhysicalData/2025_10_10/IPZ8_rpgd/2025-10-10_02-08-52_Recording1_0_IPZ8_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None_.csv"
    
    # Output directory
    output_dir = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/TrainingLite/dynamic_residual_jax/training_data"
    
    # Process the data
    output_path = process_data(input_csv, output_dir, dt=0.04)
    print(f"\nDone! Training data saved to: {output_path}")


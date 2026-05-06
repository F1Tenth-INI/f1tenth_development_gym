import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from utilities.state_utilities import STATE_VARIABLES

def moving_average_zero_phase(data: list, window_size: int) -> np.ndarray:
    """
    Apply a zero-phase moving average filter to the data.
    
    This function applies a moving average filter forward and backward to eliminate
    phase shift, which is important for preserving the timing of features in the signal.
    
    Parameters:
    - data: Input signal as a list or array
    - window_size: Size of the moving average window (must be odd)
    
    Returns:
    - Filtered signal as numpy array
    """
    data = np.array(data)
    
    # Ensure window_size is odd for symmetric filtering
    if window_size % 2 == 0:
        window_size += 1
    
    # Create moving average filter coefficients
    b = np.ones(window_size) / window_size
    
    # Apply zero-phase filtering using filtfilt (forward-backward filtering)
    filtered = signal.filtfilt(b, 1, data)
    
    return filtered

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the data using the same filtering approach as in preprocess_data.py
    Returns a new dataframe with filtered columns.
    """
    df_filtered = df.copy()
    
    # Filter IMU data
    if 'imu_accel_x' in df.columns:
        imu1_a_x_data = df['imu_accel_x'].tolist()
        imu1_a_x_filtered = moving_average_zero_phase(imu1_a_x_data, window_size=5)
        df_filtered['imu_accel_x'] = imu1_a_x_filtered
    
    if 'imu_accel_y' in df.columns:
        imu1_a_y_data = df['imu_accel_y'].tolist()
        imu1_a_y_filtered = moving_average_zero_phase(imu1_a_y_data, window_size=5)
        df_filtered['imu_accel_y'] = imu1_a_y_filtered
    
    if 'imu_gyro_z' in df.columns:
        imu1_gyro_z_data = df['imu_gyro_z'].tolist()
        imu1_gyro_z_filtered = moving_average_zero_phase(imu1_gyro_z_data, window_size=5)
        df_filtered['imu_gyro_z'] = imu1_gyro_z_filtered
    
    # Filter all state variables from STATE_VARIABLES
    # Use same window sizes as in preprocess_data.py
    # Special handling for pose_theta: filter cos/sin and reconstruct to avoid wrapping issues
    
    for state_var in STATE_VARIABLES:
        if state_var in df.columns:
            # Skip pose_theta - we'll reconstruct it from filtered cos/sin
            if state_var == 'pose_theta':
                continue
            
            state_data = df[state_var].tolist()
            # Use window_size=5 for angular_vel_z, window_size=7 for linear velocities
            # For other states, use a default window_size=5
            if state_var == 'angular_vel_z':
                window_size = 5
            elif state_var in ['linear_vel_x', 'linear_vel_y']:
                window_size = 7
            else:
                # For pose variables and others, use window_size=5
                window_size = 5
            
            state_filtered = moving_average_zero_phase(state_data, window_size=window_size)
            df_filtered[state_var] = state_filtered
    
    # Reconstruct pose_theta from filtered cos and sin to avoid wrapping issues
    if 'pose_theta' in df.columns and 'pose_theta_cos' in df_filtered.columns and 'pose_theta_sin' in df_filtered.columns:
        # Use atan2 to reconstruct the angle from filtered cos and sin
        df_filtered['pose_theta'] = np.arctan2(df_filtered['pose_theta_sin'], df_filtered['pose_theta_cos'])
    
    return df_filtered

def calculate_noise(df_original: pd.DataFrame, df_filtered: pd.DataFrame, columns: list) -> dict:
    """
    Calculate noise (difference between original and filtered data) for specified columns.
    Returns a dictionary with noise data and statistics.
    """
    noise_data = {}
    
    for col in columns:
        if col in df_original.columns and col in df_filtered.columns:
            original = df_original[col].values
            filtered = df_filtered[col].values
            
            # Special handling for pose_theta: compute angular difference to handle wrapping
            if col == 'pose_theta':
                # Compute angular difference using atan2 to properly handle wrapping
                # This gives the shortest angular distance between two angles
                noise = np.arctan2(np.sin(original - filtered), np.cos(original - filtered))
            else:
                noise = original - filtered
            
            mu = np.mean(noise)
            sigma = np.std(noise)
            
            noise_data[col] = {
                'noise': noise,
                'mu': mu,
                'sigma': sigma
            }
    
    return noise_data

def plot_histograms(noise_data: dict, output_dir: str, filename_prefix: str = ""):
    """
    Plot histograms of noise for each column and save them.
    """
    n_cols = len(noise_data)
    if n_cols == 0:
        print("No noise data to plot")
        return
    
    # Calculate grid size
    n_rows = int(np.ceil(np.sqrt(n_cols)))
    n_cols_grid = int(np.ceil(n_cols / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(5*n_cols_grid, 4*n_rows))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (col, data) in enumerate(noise_data.items()):
        ax = axes[idx]
        noise = data['noise']
        mu = data['mu']
        sigma = data['sigma']

        # Calculate 3-sigma range
        if sigma > 1e-10:  # Avoid divide by zero
            x_min = mu - 3 * sigma
            x_max = mu + 3 * sigma
        else:
            # If sigma is very small, use a small range around mu
            x_min = mu - 0.1
            x_max = mu + 0.1

        # Filter noise data to only include values within 3-sigma range
        # This prevents outliers from affecting the density calculation
        noise_filtered = noise[(noise >= x_min) & (noise <= x_max)]
        
        # If no data in range, use original data but still limit the plot
        if len(noise_filtered) == 0:
            noise_filtered = noise
            # Use actual data range if 3-sigma is too restrictive
            x_min = noise.min()
            x_max = noise.max()

        # Use a reasonable number of bins (80 bins for good resolution)
        n_bins = 80
        bins = np.linspace(x_min, x_max, n_bins + 1)

        # Plot histogram with filtered data
        ax.hist(noise_filtered, bins=bins, density=True, alpha=0.7,
                edgecolor='black', linewidth=0.3)

        # Overlay normal distribution with calculated mu and sigma (only if sigma > 0)
        if sigma > 1e-10:  # Avoid divide by zero
            x = np.linspace(x_min, x_max, 200)
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, y, 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        else:
            ax.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'μ={mu:.4f}, σ≈0')

        # Set x-axis limits to the 3-sigma range
        ax.set_xlim(x_min, x_max)
        
        ax.set_xlabel('Noise')
        ax.set_ylabel('Density')
        ax.set_title(f'{col}\nμ={mu:.6f}, σ={sigma:.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{filename_prefix}noise_histograms.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram plot to: {output_path}")
    plt.close()

def plot_time_series(df_original: pd.DataFrame, df_filtered: pd.DataFrame, columns: list, 
                     output_dir: str, filename_prefix: str = "", start_idx: int = 200, end_idx: int = 700):
    """
    Plot time series comparing original and filtered data for each column.
    
    Parameters:
    - df_original: Original unfiltered dataframe
    - df_filtered: Filtered dataframe
    - columns: List of columns to plot
    - output_dir: Directory to save the plot
    - filename_prefix: Prefix for the output filename
    - start_idx: Starting timestep index (default: 200)
    - end_idx: Ending timestep index (default: 700)
    """
    # Filter columns to only those that exist in both dataframes
    valid_columns = [col for col in columns if col in df_original.columns and col in df_filtered.columns]
    
    if len(valid_columns) == 0:
        print("No valid columns to plot for time series")
        return
    
    # Ensure indices are within bounds
    n_rows = len(df_original)
    start_idx = max(0, min(start_idx, n_rows - 1))
    end_idx = max(start_idx + 1, min(end_idx, n_rows))
    
    # Create timestep array for x-axis
    timesteps = np.arange(start_idx, end_idx)
    
    # Create subplots: all stacked vertically (single column, full width)
    n_cols_plot = len(valid_columns)
    fig, axes = plt.subplots(n_cols_plot, 1, figsize=(12, 3*n_cols_plot))
    
    # Ensure axes is always a list/array
    if n_cols_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(valid_columns):
        ax = axes[idx]
        
        # Extract data for the specified time range
        original_data = df_original[col].iloc[start_idx:end_idx].values
        filtered_data = df_filtered[col].iloc[start_idx:end_idx].values
        
        # Plot both original and filtered data
        ax.plot(timesteps, original_data, 'b-', alpha=0.6, linewidth=1, label='Original', zorder=1)
        ax.plot(timesteps, filtered_data, 'r-', alpha=0.8, linewidth=1.5, label='Filtered', zorder=2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_title(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{filename_prefix}time_series_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved time series comparison plot to: {output_path}")
    plt.close()

def print_statistics(noise_data: dict):
    """
    Print statistics (mu and sigma) for each column.
    """
    print("\n" + "="*80)
    print("NOISE STATISTICS")
    print("="*80)
    print(f"{'Column':<30} {'μ (mean)':<20} {'σ (std)':<20}")
    print("-"*80)
    
    for col, data in noise_data.items():
        mu = data['mu']
        sigma = data['sigma']
        print(f"{col:<30} {mu:<20.6f} {sigma:<20.6f}")
    
    print("="*80 + "\n")

def analyze_noise(input_csv: str, output_dir: str = None, columns: list = None):
    """
    Main function to analyze noise in physical data.
    
    Parameters:
    - input_csv: Path to input CSV file
    - output_dir: Directory to save output plots (default: same as input file directory)
    - columns: List of columns to analyze (default: None, analyzes all filterable columns)
    """
    # Read the CSV file
    print(f"Reading data from: {input_csv}")
    df_original = pd.read_csv(input_csv, comment='#')
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine columns to analyze
    if columns is None:
        columns = []
        # Include only state variables from STATE_VARIABLES (no IMU data)
        for col in STATE_VARIABLES:
            if col in df_original.columns:
                columns.append(col)
    
    if len(columns) == 0:
        print("No filterable columns found in the data")
        return
    
    print(f"Analyzing noise for columns: {columns}")
    
    # Filter the data
    print("Filtering data...")
    df_filtered = filter_data(df_original)
    
    # Calculate noise
    print("Calculating noise (difference between original and filtered)...")
    noise_data = calculate_noise(df_original, df_filtered, columns)
    
    # Print statistics
    print_statistics(noise_data)
    
    # Plot histograms
    print("Generating histograms...")
    filename_prefix = os.path.splitext(os.path.basename(input_csv))[0] + "_"
    plot_histograms(noise_data, output_dir, filename_prefix)
    
    # Plot time series comparison
    print("Generating time series comparison plots...")
    plot_time_series(df_original, df_filtered, columns, output_dir, filename_prefix, start_idx=200, end_idx=700)
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'column': list(noise_data.keys()),
        'mu': [noise_data[col]['mu'] for col in noise_data.keys()],
        'sigma': [noise_data[col]['sigma'] for col in noise_data.keys()]
    })
    stats_path = os.path.join(output_dir, f'{filename_prefix}noise_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to: {stats_path}")
    
    return noise_data

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        # Default: use a file from the physical data directory
        input_csv = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/AnalyseData/NoiseAnalysis/2026-01-16_01-22-28_Recording1_0_IPZ20_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None_.csv"
    
    # Optional: specify output directory
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Optional: specify columns to analyze
    columns = None
    if len(sys.argv) > 3:
        columns = sys.argv[3].split(',')
    
    analyze_noise(input_csv, output_dir, columns)

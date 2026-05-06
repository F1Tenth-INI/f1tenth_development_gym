import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "AnalyseData/PhysicalData/2026_01_16/2026-01-16_01-50-44_Recording1_0_IPZ20_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None_.csv"
print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path, comment='#')

# Get IMU1 columns only
imu_accel_cols = ['imu1_a_x', 'imu1_a_y']
imu_gyro_cols = ['imu1_av_z']

# Filter to only include columns that exist in the dataframe
imu_accel_cols = [col for col in imu_accel_cols if col in df.columns]
imu_gyro_cols = [col for col in imu_gyro_cols if col in df.columns]

print(f"Found accelerometer columns: {imu_accel_cols}")
print(f"Found gyroscope columns: {imu_gyro_cols}")

# Limit to 500 timesteps
max_timesteps = 500
df = df.iloc[:max_timesteps]

# Create timestep array
timesteps = np.arange(len(df))

# Create figure with subplots
n_plots = 0
plot_configs = []

if imu_accel_cols:
    n_plots += 1
    plot_configs.append(('Accelerometer', imu_accel_cols, 'm/s²'))

if imu_gyro_cols:
    n_plots += 1
    plot_configs.append(('Gyroscope', imu_gyro_cols, 'rad/s'))

if n_plots == 0:
    print("No IMU data found!")
    exit(1)

fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots))
if n_plots == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (title, cols, ylabel) in enumerate(plot_configs):
    ax = axes[idx]
    
    for col in cols:
        if col in df.columns:
            ax.plot(timesteps, df[col], label=col, alpha=0.8, linewidth=2.5)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
output_dir = "AnalyseData/IMU"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.basename(csv_path).replace('.csv', '_imu_plot.png')
output_path = os.path.join(output_dir, filename)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved IMU plot to: {output_path}")
plt.close()

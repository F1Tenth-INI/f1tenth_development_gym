import csv

def alpha_beta_filter(data, alpha=0.1):
    """
    Apply an alpha-beta filter to smooth the data.
    
    Parameters:
    - data: list of measurements
    - alpha: smoothing factor (0 < alpha < 1), lower = more smoothing
    
    Returns:
    - filtered_data: list of filtered values
    """
    filtered = [0.0] * len(data)
    filtered[0] = float(data[0])  # Initialize with first measurement
    
    for i in range(1, len(data)):
        # Prediction step
        prediction = filtered[i-1]
        
        # Update step
        residual = float(data[i]) - prediction
        filtered[i] = prediction + alpha * residual
    
    return filtered


class EKF:
    """
    Minimal Extended Kalman Filter for fusing position and IMU acceleration.
    State: [x, vx] (position and velocity in x direction)
    
    Noise parameters control the filter's trust in different sources:
    - process_noise: Lower values = more trust in IMU/model predictions
    - measurement_noise: Higher values = less trust in measurements (less weight)
    """
    def __init__(self, initial_x=0.0, initial_vx=0.0, 
                 process_noise_pos=0.1, process_noise_vel=0.1,
                 measurement_noise_pos=0.1, measurement_noise_vel=0.1):
        # Initial state
        self.x = initial_x  # position
        self.vx = initial_vx  # velocity
        
        # Process noise covariance (Q)
        self.Q = [[process_noise_pos**2, 0],
                  [0, process_noise_vel**2]]
        
        # Measurement noise covariance (R)
        self.R = [[measurement_noise_pos**2, 0],
                  [0, measurement_noise_vel**2]]
        
        # State covariance (P)
        self.P = [[1.0, 0],
                  [0, 1.0]]
    
    def predict(self, accel, dt):
        """
        Prediction step using IMU acceleration.
        
        Parameters:
        - accel: acceleration from IMU (imu1_a_x)
        - dt: time step
        """
        # State transition: x_k+1 = x_k + vx_k * dt + 0.5 * a_k * dt^2
        #                   vx_k+1 = vx_k + a_k * dt
        self.x = self.x + self.vx * dt + 0.5 * accel * dt * dt
        self.vx = self.vx + accel * dt
        
        # State transition matrix (F)
        F = [[1.0, dt],
             [0.0, 1.0]]
        
        # Update covariance: P = F * P * F^T + Q
        # Simplified computation
        P00 = self.P[0][0] + dt * (self.P[0][1] + self.P[1][0]) + dt * dt * self.P[1][1] + self.Q[0][0]
        P01 = self.P[0][1] + dt * self.P[1][1]
        P10 = self.P[1][0] + dt * self.P[1][1]
        P11 = self.P[1][1] + self.Q[1][1]
        
        self.P = [[P00, P01],
                  [P10, P11]]
    
    def update(self, meas_x, meas_vx):
        """
        Update step using position and velocity measurements.
        
        Parameters:
        - meas_x: measured position (pose_x)
        - meas_vx: measured velocity (linear_vel_x)
        """
        # Measurement vector
        z = [meas_x, meas_vx]
        
        # Predicted measurement (H is identity for this case)
        h = [self.x, self.vx]
        
        # Innovation (residual)
        y = [z[0] - h[0], z[1] - h[1]]
        
        # Innovation covariance: S = H * P * H^T + R
        # Since H is identity, S = P + R
        S00 = self.P[0][0] + self.R[0][0]
        S01 = self.P[0][1]
        S10 = self.P[1][0]
        S11 = self.P[1][1] + self.R[1][1]
        
        # Determinant of S
        det_S = S00 * S11 - S01 * S10
        
        if abs(det_S) < 1e-10:
            return  # Skip update if singular
        
        # Kalman gain: K = P * H^T * S^-1
        # Since H is identity, K = P * S^-1
        inv_S00 = S11 / det_S
        inv_S01 = -S01 / det_S
        inv_S10 = -S10 / det_S
        inv_S11 = S00 / det_S
        
        K00 = self.P[0][0] * inv_S00 + self.P[0][1] * inv_S10
        K01 = self.P[0][0] * inv_S01 + self.P[0][1] * inv_S11
        K10 = self.P[1][0] * inv_S00 + self.P[1][1] * inv_S10
        K11 = self.P[1][0] * inv_S01 + self.P[1][1] * inv_S11
        
        # Update state: x = x + K * y
        self.x = self.x + K00 * y[0] + K01 * y[1]
        self.vx = self.vx + K10 * y[0] + K11 * y[1]
        
        # Update covariance: P = (I - K * H) * P
        # Since H is identity, P = (I - K) * P
        I_minus_K00 = 1.0 - K00
        I_minus_K11 = 1.0 - K11
        
        P00_new = I_minus_K00 * self.P[0][0] - K01 * self.P[1][0]
        P01_new = I_minus_K00 * self.P[0][1] - K01 * self.P[1][1]
        P10_new = -K10 * self.P[0][0] + I_minus_K11 * self.P[1][0]
        P11_new = -K10 * self.P[0][1] + I_minus_K11 * self.P[1][1]
        
        self.P = [[P00_new, P01_new],
                  [P10_new, P11_new]]
    
    def get_velocity(self):
        """Get the current velocity estimate."""
        return self.vx

# Read the CSV file, skipping comment lines
input_file = '2025-11-28_07-58-30_Recording1_0_IPZ10_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None_.csv'

# Read header comments and find data start
header_lines = []
data_start_line = 0
with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith('#'):
            header_lines.append(line)
            data_start_line = i + 1
        else:
            break

# Read the CSV data
rows = []
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    # Skip comment lines
    for _ in range(data_start_line):
        next(reader, None)
    
    # Read header row
    header = next(reader)
    
    # Read all data rows
    for row in reader:
        rows.append(row)

# Find column indices
linear_vel_x_idx = header.index('linear_vel_x')
linear_vel_y_idx = header.index('linear_vel_y')
time_idx = header.index('time')
pose_x_idx = header.index('pose_x')
imu1_a_x_idx = header.index('imu1_a_x')

# Extract data
linear_vel_x_data = [float(row[linear_vel_x_idx]) for row in rows]
linear_vel_y_data = [float(row[linear_vel_y_idx]) for row in rows]
time_data = [float(row[time_idx]) for row in rows]
pose_x_data = [float(row[pose_x_idx]) for row in rows]
imu1_a_x_data = [float(row[imu1_a_x_idx]) for row in rows]

# Apply delay-free filter to IMU acceleration
# Using exponential moving average for minimal delay
imu_filter_alpha = 0.2  # Smoothing factor (0 < alpha < 1), lower = more smoothing
imu1_a_x_filtered = alpha_beta_filter(imu1_a_x_data, alpha=imu_filter_alpha)

# Apply alpha-beta filter
alpha = 0.1  # Smoothing factor (adjust as needed, lower = more smoothing)
linear_vel_x_filtered = alpha_beta_filter(linear_vel_x_data, alpha=alpha)
linear_vel_y_filtered = alpha_beta_filter(linear_vel_y_data, alpha=alpha)

# Apply EKF
# EKF tuning parameters:
# - Higher measurement_noise_pos = less weight to position measurements
# - Higher measurement_noise_vel = less weight to velocity measurements  
# - Lower process_noise_pos/vel = more weight to IMU (process model)
# - Higher process_noise_pos/vel = less weight to IMU (process model)
ekf = EKF(
    initial_x=pose_x_data[0], 
    initial_vx=linear_vel_x_data[0],
    process_noise_pos=0.01,      # Lower = more trust in IMU prediction
    process_noise_vel=0.01,      # Lower = more trust in IMU prediction
    measurement_noise_pos=0.1,    # Higher = less weight to position
    measurement_noise_vel=0.1     # Higher = less weight to velocity
)
linear_vel_x_ekf = []

for i in range(len(rows)):
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

# Calculate delta (change over one timestep)
linear_vel_x_ekf_delta = []
for i in range(len(linear_vel_x_ekf)):
    if i == 0:
        # First timestep: delta is 0 (no previous value)
        linear_vel_x_ekf_delta.append(0.0)
    else:
        # Delta is the change from previous timestep
        delta = linear_vel_x_ekf[i] - linear_vel_x_ekf[i-1]
        linear_vel_x_ekf_delta.append(delta)

# Create output filename
output_file = input_file.replace('.csv', '_filtered.csv')

# Write the filtered data to a new CSV file
with open(output_file, 'w', newline='') as f:
    # Write header comments
    for line in header_lines:
        f.write(line)
    
    # Add new columns to header
    new_header = header + ['linear_vel_x_filtered', 'linear_vel_y_filtered', 'imu1_a_x_filtered', 
                          'linear_vel_x_ekf', 'linear_vel_x_ekf_delta']
    
    # Write header row
    writer = csv.writer(f)
    writer.writerow(new_header)
    
    # Write data rows with filtered values
    for i, row in enumerate(rows):
        new_row = row + [str(linear_vel_x_filtered[i]), str(linear_vel_y_filtered[i]), 
                        str(imu1_a_x_filtered[i]), str(linear_vel_x_ekf[i]), str(linear_vel_x_ekf_delta[i])]
        writer.writerow(new_row)

print(f"Filtered data saved to: {output_file}")
print(f"Applied alpha-beta filter with alpha={alpha} to velocity data")
print(f"Applied delay-free filter with alpha={imu_filter_alpha} to IMU acceleration (imu1_a_x)")
print(f"Applied EKF fusing position (pose_x) with filtered IMU acceleration")
print(f"Original columns: linear_vel_x, linear_vel_y, imu1_a_x")
print(f"New columns: linear_vel_x_filtered, linear_vel_y_filtered, imu1_a_x_filtered, linear_vel_x_ekf, linear_vel_x_ekf_delta")
print(f"Total rows processed: {len(rows)}")

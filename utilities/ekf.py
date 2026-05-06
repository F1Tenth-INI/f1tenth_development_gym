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


def moving_average_zero_phase(data, window_size=5):
    """
    Zero-phase moving average filter (no delay in the middle of the signal).
    
    window_size should be odd: 3,5,7,...
    """
    assert window_size % 2 == 1, "window_size must be odd"
    n = len(data)
    half = window_size // 2
    out = [0.0] * n

    # Simple edge handling: extend the signal with edge values
    extended = ([data[0]] * half) + list(data) + ([data[-1]] * half)

    # Precompute cumulative sum for speed
    cumsum = [0.0]
    for x in extended:
        cumsum.append(cumsum[-1] + float(x))

    for i in range(n):
        start = i
        end = i + window_size
        window_sum = cumsum[end] - cumsum[start]
        out[i] = window_sum / window_size

    return out

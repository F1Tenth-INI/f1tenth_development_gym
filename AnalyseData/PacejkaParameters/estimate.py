"""
Learning-Based On-Track System Identification for Pacejka Tire Parameters

Implementation based on:
"Learning-Based On-Track System Identification for Scaled Autonomous Racing in Under a Minute"
by Dikici et al.

This script implements the iterative learning-based system identification method:
1. Train a Neural Network to learn residual errors between nominal model and real data
2. Generate virtual steady-state data using the corrected model (nominal + NN)
3. Extract Pacejka parameters from virtual steady-state data using least squares
4. Iterate: use identified parameters as new nominal model and repeat
"""

# ============================================================================
# CONFIGURATION - Modify these paths and settings as needed
# ============================================================================

# Paths
DATA_DIR = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/AnalyseData/PhysicalData/2026_01_16"
OUTPUT_DIR = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/AnalyseData/PacejkaParameters"
CAR_PARAMS_YAML = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/utilities/car_files/gym_car_parameters.yml"

# Data processing settings
LOWPASS_CUTOFF = 5.0  # Hz - Low-pass filter cutoff frequency
SAMPLE_RATE = 25.0    # Hz - Data sampling rate
MIN_SPEED = 0.5       # m/s - Minimum speed threshold (filter low-speed samples)
DT = 0.04            # s - Time step (1/sample_rate)

# Outlier filtering settings (for noisy 1/10 scale car data)
IQR_FACTOR_PREPROCESS = 2.0  # IQR multiplier for preprocessing outlier detection
IQR_FACTOR_FORCES = 2.0      # IQR multiplier for force outlier detection
IQR_FACTOR_BINS = 2.0        # IQR multiplier for within-bin outlier detection
MIN_SAMPLES_PER_BIN = 10     # Minimum samples required per bin for statistics

# ============================================================================

import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for saving figures
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import pickle
import yaml

# Try to import JAX for neural network, fall back to numpy if not available
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available, using numpy-based implementation")


@dataclass
class CarParams:
    """Vehicle parameters for the single-track model"""
    m: float = 4.0          # Mass [kg]
    Iz: float = 0.1297      # Moment of inertia [kgm^2]
    lf: float = 0.162       # Distance from CG to front axle [m]
    lr: float = 0.145       # Distance from CG to rear axle [m]
    
    @property
    def L(self) -> float:
        """Wheelbase length"""
        return self.lf + self.lr


@dataclass
class PacejkaParams:
    """Pacejka Magic Formula parameters for front and rear tires
    
    F = D * sin(C * arctan(B * alpha - E * (B * alpha - arctan(B * alpha))))
    
    For scaled vehicles, D is often normalized to 1.0 and the actual force
    is computed as F = mu * Fz * pacejka(alpha), where Fz is normal load.
    """
    Bf: float = 5.0     # Front stiffness factor
    Cf: float = 1.3     # Front shape factor
    Df: float = 1.0     # Front peak factor (normalized)
    Ef: float = 0.0     # Front curvature factor
    Br: float = 7.0     # Rear stiffness factor
    Cr: float = 1.1     # Rear shape factor
    Dr: float = 1.0     # Rear peak factor (normalized)
    Er: float = 0.0     # Rear curvature factor
    mu: float = 0.9     # Friction coefficient
    
    def to_array(self) -> np.ndarray:
        return np.array([self.Bf, self.Cf, self.Df, self.Ef,
                        self.Br, self.Cr, self.Dr, self.Er])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, mu: float = 0.9) -> 'PacejkaParams':
        return cls(Bf=arr[0], Cf=arr[1], Df=arr[2], Ef=arr[3],
                  Br=arr[4], Cr=arr[5], Dr=arr[6], Er=arr[7], mu=mu)


class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, car_params: CarParams):
        self.car = car_params
        
    def load_csv_files(self, data_dir: str) -> pd.DataFrame:
        """Load all CSV files from the directory and concatenate"""
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        # Filter out non-recording files
        csv_files = [f for f in csv_files if "Recording" in os.path.basename(f) 
                     or f.endswith("_rpgd_mid.csv") or f.endswith("_rpgd_slow.csv")]
        
        print(f"Found {len(csv_files)} data files")
        
        all_data = []
        for f in csv_files:
            try:
                # Skip comment lines
                df = pd.read_csv(f, comment='#')
                all_data.append(df)
                print(f"  Loaded {os.path.basename(f)}: {len(df)} samples")
            except Exception as e:
                print(f"  Error loading {f}: {e}")
                
        if not all_data:
            raise ValueError("No data files found!")
            
        return pd.concat(all_data, ignore_index=True)
    
    def preprocess(self, df: pd.DataFrame, 
                   lowpass_cutoff: float = 5.0,
                   sample_rate: float = 25.0,
                   min_speed: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data:
        1. Extract relevant columns
        2. Convert vy from rear axle to center of gravity (if measured at rear axle)
        3. Apply low-pass filter
        4. Filter low-speed samples
        5. Augment by mirroring (symmetric dynamics)
        
        NOTE: linear_vel_y is measured at the rear axle, not at CG.
        To convert to CG: vy_CG = vy_rear + lr * omega
        
        Returns:
            states: [N, 2] array of [vy_CG, omega]
            inputs: [N, 2] array of [vx, delta]
        """
        # Extract relevant columns
        vx = df['linear_vel_x'].values
        vy_rear = df['linear_vel_y'].values  # Measured at rear axle
        omega = df['angular_vel_z'].values
        delta = df['steering_angle'].values
        
        # Convert vy from rear axle to center of gravity
        # vy_CG = vy_rear + lr * omega
        vy_CG = vy_rear + self.car.lr * omega
        
        # Apply Butterworth low-pass filter (noncausal for no phase delay)
        nyquist = sample_rate / 2
        normalized_cutoff = lowpass_cutoff / nyquist
        b, a = signal.butter(2, normalized_cutoff, btype='low')
        
        vx_filt = signal.filtfilt(b, a, vx)
        vy_filt = signal.filtfilt(b, a, vy_CG)  # Now filtering CG velocity
        omega_filt = signal.filtfilt(b, a, omega)
        delta_filt = signal.filtfilt(b, a, delta)
        
        # Filter low-speed samples (dynamics are unreliable at low speeds)
        mask = vx_filt > min_speed
        vx_filt = vx_filt[mask]
        vy_filt = vy_filt[mask]
        omega_filt = omega_filt[mask]
        delta_filt = delta_filt[mask]
        
        print(f"After speed filtering: {len(vx_filt)} samples (from {len(vx)})")
        print(f"NOTE: Converted vy from rear axle to CG using: vy_CG = vy_rear + lr*omega")
        
        # Aggressive outlier filtering using IQR method (for noisy 1/10 scale car data)
        def filter_outliers_iqr(data, factor=IQR_FACTOR_PREPROCESS):
            """Filter outliers using Interquartile Range method"""
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return (data >= lower_bound) & (data <= upper_bound)
        
        # Filter outliers for each variable
        mask_vx = filter_outliers_iqr(vx_filt, factor=2.0)  # Stricter for vx
        mask_vy = filter_outliers_iqr(vy_filt, factor=2.5)  # More lenient for vy (can be negative)
        mask_omega = filter_outliers_iqr(omega_filt, factor=2.5)  # More lenient for omega (can be negative)
        mask_delta = filter_outliers_iqr(delta_filt, factor=2.0)
        
        # Combine masks
        mask_outliers = mask_vx & mask_vy & mask_omega & mask_delta
        
        # Also filter based on physical limits
        # Reasonable limits for 1/10 scale car
        max_vx = 10.0  # m/s
        max_vy = 5.0   # m/s lateral velocity
        max_omega = 10.0  # rad/s yaw rate
        max_delta = 0.5  # rad steering angle
        
        mask_physical = (
            (np.abs(vx_filt) < max_vx) &
            (np.abs(vy_filt) < max_vy) &
            (np.abs(omega_filt) < max_omega) &
            (np.abs(delta_filt) < max_delta)
        )
        
        mask_final = mask_outliers & mask_physical
        
        vx_filt = vx_filt[mask_final]
        vy_filt = vy_filt[mask_final]
        omega_filt = omega_filt[mask_final]
        delta_filt = delta_filt[mask_final]
        
        print(f"After aggressive outlier filtering: {len(vx_filt)} samples (removed {np.sum(~mask_final)} outliers)")
        
        # Create state and input arrays
        states = np.column_stack([vy_filt, omega_filt])
        inputs = np.column_stack([vx_filt, delta_filt])
        
        # Data augmentation: mirror for symmetric dynamics
        # When mirroring: vy -> -vy, omega -> -omega, delta -> -delta
        states_mirror = np.column_stack([-vy_filt, -omega_filt])
        inputs_mirror = np.column_stack([vx_filt, -delta_filt])  # vx stays positive
        
        states = np.vstack([states, states_mirror])
        inputs = np.vstack([inputs, inputs_mirror])
        
        print(f"After augmentation: {len(states)} samples")
        
        return states, inputs


class VehicleModel:
    """Dynamic single-track vehicle model with Pacejka tire forces"""
    
    def __init__(self, car_params: CarParams, pacejka_params: PacejkaParams, dt: float = 0.04):
        self.car = car_params
        self.pacejka = pacejka_params
        self.dt = dt
        
    def pacejka_force(self, alpha: np.ndarray, B: float, C: float, D: float, E: float) -> np.ndarray:
        """
        Pacejka Magic Formula for lateral tire force (normalized)
        F = D * sin(C * arctan(B*alpha - E*(B*alpha - arctan(B*alpha))))
        """
        Balpha = B * alpha
        return D * np.sin(C * np.arctan(Balpha - E * (Balpha - np.arctan(Balpha))))
    
    def compute_slip_angles(self, vx: np.ndarray, vy: np.ndarray, 
                           omega: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute front and rear slip angles
        alpha_f = delta - arctan((vy + lf*omega) / vx)
        alpha_r = -arctan((vy - lr*omega) / vx)
        """
        # Add small epsilon to avoid division by zero
        vx_safe = np.maximum(vx, 0.1)
        
        alpha_f = delta - np.arctan2(vy + self.car.lf * omega, vx_safe)
        alpha_r = -np.arctan2(vy - self.car.lr * omega, vx_safe)
        
        return alpha_f, alpha_r
    
    def compute_tire_forces(self, alpha_f: np.ndarray, alpha_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute front and rear lateral tire forces"""
        p = self.pacejka
        
        # Normal loads (static, neglecting load transfer for scaled car)
        Fzf = self.car.m * 9.81 * self.car.lr / self.car.L
        Fzr = self.car.m * 9.81 * self.car.lf / self.car.L
        
        # Lateral forces
        Fyf = p.mu * Fzf * self.pacejka_force(alpha_f, p.Bf, p.Cf, p.Df, p.Ef)
        Fyr = p.mu * Fzr * self.pacejka_force(alpha_r, p.Br, p.Cr, p.Dr, p.Er)
        
        return Fyf, Fyr
    
    def dynamics(self, vy: np.ndarray, omega: np.ndarray, 
                 vx: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute state derivatives: vy_dot and omega_dot
        
        vy_dot = (Fyr + Fyf*cos(delta) - m*vx*omega) / m
        omega_dot = (Fyf*lf*cos(delta) - Fyr*lr) / Iz
        """
        alpha_f, alpha_r = self.compute_slip_angles(vx, vy, omega, delta)
        Fyf, Fyr = self.compute_tire_forces(alpha_f, alpha_r)
        
        vy_dot = (Fyr + Fyf * np.cos(delta) - self.car.m * vx * omega) / self.car.m
        omega_dot = (Fyf * self.car.lf * np.cos(delta) - Fyr * self.car.lr) / self.car.Iz
        
        return vy_dot, omega_dot
    
    def predict_next_state(self, vy: np.ndarray, omega: np.ndarray,
                          vx: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state using Euler integration"""
        vy_dot, omega_dot = self.dynamics(vy, omega, vx, delta)
        
        vy_next = vy + vy_dot * self.dt
        omega_next = omega + omega_dot * self.dt
        
        return vy_next, omega_next


class ResidualNN:
    """
    Simple feedforward neural network for learning residual errors
    Architecture: 4 inputs -> 8 hidden (LeakyReLU) -> 2 outputs
    
    Input: [vx, vy, omega, delta]
    Output: [e_vy, e_omega] (prediction errors)
    """
    
    def __init__(self, learning_rate: float = 5e-4, hidden_size: int = 8):
        self.lr = learning_rate
        self.hidden_size = hidden_size
        self.input_size = 4
        self.output_size = 2
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        np.random.seed(42)
        
        # Input to hidden
        scale1 = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * scale1
        self.b1 = np.zeros(self.hidden_size)
        
        # Hidden to output
        scale2 = np.sqrt(2.0 / (self.hidden_size + self.output_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * scale2
        self.b2 = np.zeros(self.output_size)
        
        # For normalization
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_deriv(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)
    
    def normalize_input(self, X: np.ndarray) -> np.ndarray:
        if self.input_mean is None:
            return X
        return (X - self.input_mean) / (self.input_std + 1e-8)
    
    def denormalize_output(self, Y: np.ndarray) -> np.ndarray:
        if self.output_mean is None:
            return Y
        return Y * (self.output_std + 1e-8) + self.output_mean
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass"""
        X_norm = self.normalize_input(X)
        self.z1 = X_norm @ self.W1 + self.b1
        self.a1 = self.leaky_relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2  # Linear output
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with denormalization"""
        return self.denormalize_output(self.forward(X))
    
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 500, verbose: bool = True):
        """
        Train the network using batch gradient descent with Adam optimizer
        
        X: [N, 4] input features [vx, vy, omega, delta]
        Y: [N, 2] target errors [e_vy, e_omega]
        """
        # Compute normalization parameters
        self.input_mean = np.mean(X, axis=0)
        self.input_std = np.std(X, axis=0)
        self.output_mean = np.mean(Y, axis=0)
        self.output_std = np.std(Y, axis=0)
        
        # Normalize targets for training
        Y_norm = (Y - self.output_mean) / (self.output_std + 1e-8)
        
        # Adam optimizer parameters
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_W1 = np.zeros_like(self.W1)
        v_W1 = np.zeros_like(self.W1)
        m_b1 = np.zeros_like(self.b1)
        v_b1 = np.zeros_like(self.b1)
        m_W2 = np.zeros_like(self.W2)
        v_W2 = np.zeros_like(self.W2)
        m_b2 = np.zeros_like(self.b2)
        v_b2 = np.zeros_like(self.b2)
        
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            Y_pred = self.forward(X)
            
            # Compute loss (MSE)
            loss = np.mean((Y_pred - Y_norm) ** 2)
            losses.append(loss)
            
            # Backward pass
            N = X.shape[0]
            dL_dz2 = 2 * (Y_pred - Y_norm) / N
            
            dL_dW2 = self.a1.T @ dL_dz2
            dL_db2 = np.sum(dL_dz2, axis=0)
            
            dL_da1 = dL_dz2 @ self.W2.T
            dL_dz1 = dL_da1 * self.leaky_relu_deriv(self.z1)
            
            X_norm = self.normalize_input(X)
            dL_dW1 = X_norm.T @ dL_dz1
            dL_db1 = np.sum(dL_dz1, axis=0)
            
            # Adam updates
            t = epoch + 1
            
            for param, grad, m, v in [
                (self.W1, dL_dW1, m_W1, v_W1),
                (self.b1, dL_db1, m_b1, v_b1),
                (self.W2, dL_dW2, m_W2, v_W2),
                (self.b2, dL_db2, m_b2, v_b2),
            ]:
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses


class PacejkaEstimator:
    """
    Main class implementing the iterative Pacejka parameter estimation
    """
    
    def __init__(self, car_params: CarParams, initial_pacejka: PacejkaParams, dt: float = 0.04):
        self.car = car_params
        self.dt = dt
        self.pacejka_history = [initial_pacejka]
        self.nn_history = []
        
    def compute_prediction_errors(self, model: VehicleModel,
                                   states: np.ndarray, 
                                   inputs: np.ndarray) -> np.ndarray:
        """
        Compute one-step prediction errors
        e_k = x_{k+1} - hat{x}_{k+1}
        
        states: [N, 2] array of [vy, omega]
        inputs: [N, 2] array of [vx, delta]
        """
        vy = states[:-1, 0]
        omega = states[:-1, 1]
        vx = inputs[:-1, 0]
        delta = inputs[:-1, 1]
        
        # Predicted next states
        vy_pred, omega_pred = model.predict_next_state(vy, omega, vx, delta)
        
        # Actual next states
        vy_actual = states[1:, 0]
        omega_actual = states[1:, 1]
        
        # Errors
        e_vy = vy_actual - vy_pred
        e_omega = omega_actual - omega_pred
        
        return np.column_stack([e_vy, e_omega])
    
    def generate_virtual_steady_state(self, model: VehicleModel, nn: ResidualNN,
                                       avg_vx: float, 
                                       max_delta: float = 0.4,
                                       duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate virtual steady-state data by simulating the corrected model
        with constant longitudinal velocity and linearly increasing steering angle.
        
        Returns:
            alpha_f_all, Fyf_all: front slip angles and forces
            alpha_r_all, Fyr_all: rear slip angles and forces
        """
        num_steps = int(duration / self.dt)
        
        # Initialize states
        vy = 0.0
        omega = 0.0
        vx = avg_vx
        
        # Storage
        alpha_f_list = []
        alpha_r_list = []
        Fyf_list = []
        Fyr_list = []
        
        for i in range(num_steps):
            # Linearly increasing steering angle
            delta = max_delta * (i / num_steps)
            
            # Compute slip angles
            vx_arr = np.array([vx])
            vy_arr = np.array([vy])
            omega_arr = np.array([omega])
            delta_arr = np.array([delta])
            
            alpha_f, alpha_r = model.compute_slip_angles(vx_arr, vy_arr, omega_arr, delta_arr)
            
            # Nominal prediction
            vy_pred, omega_pred = model.predict_next_state(vy_arr, omega_arr, vx_arr, delta_arr)
            
            # NN correction
            nn_input = np.array([[vx, vy, omega, delta]])
            correction = nn.predict(nn_input)[0]
            
            # Corrected prediction
            vy_next = vy_pred[0] + correction[0]
            omega_next = omega_pred[0] + correction[1]
            
            # Store data
            alpha_f_list.append(alpha_f[0])
            alpha_r_list.append(alpha_r[0])
            
            # Update states
            vy = vy_next
            omega = omega_next
        
        alpha_f_all = np.array(alpha_f_list)
        alpha_r_all = np.array(alpha_r_list)
        
        # Convert to forces using steady-state assumption
        # At steady state: vy_dot = 0, omega_dot = 0
        # This gives us: Fyf = m * lr / L * vx * omega / cos(delta)
        #                Fyr = m * lf / L * vx * omega
        
        return alpha_f_all, alpha_r_all
    
    def generate_virtual_steady_state_v2(self, model: VehicleModel, nn: ResidualNN,
                                          avg_vx: float,
                                          max_delta: float = 0.4,
                                          duration: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate virtual steady-state data and compute forces from the simulated states.
        Uses the steady-state equations to back-calculate tire forces.
        """
        num_steps = int(duration / self.dt)
        
        # Initialize
        vy = 0.0
        omega = 0.0
        vx = avg_vx
        
        alpha_f_list, alpha_r_list = [], []
        vy_list, omega_list = [], []
        delta_list = []
        
        for i in range(num_steps):
            delta = max_delta * (i / num_steps)
            
            # Current state arrays
            vx_arr = np.array([vx])
            vy_arr = np.array([vy])
            omega_arr = np.array([omega])
            delta_arr = np.array([delta])
            
            # Slip angles
            alpha_f, alpha_r = model.compute_slip_angles(vx_arr, vy_arr, omega_arr, delta_arr)
            
            # Nominal prediction
            vy_pred, omega_pred = model.predict_next_state(vy_arr, omega_arr, vx_arr, delta_arr)
            
            # NN correction
            nn_input = np.array([[vx, vy, omega, delta]])
            correction = nn.predict(nn_input)[0]
            
            # Corrected next state
            vy_next = vy_pred[0] + correction[0]
            omega_next = omega_pred[0] + correction[1]
            
            # Store
            alpha_f_list.append(alpha_f[0])
            alpha_r_list.append(alpha_r[0])
            vy_list.append(vy)
            omega_list.append(omega)
            delta_list.append(delta)
            
            vy = vy_next
            omega = omega_next
        
        alpha_f_all = np.array(alpha_f_list)
        alpha_r_all = np.array(alpha_r_list)
        vy_all = np.array(vy_list)
        omega_all = np.array(omega_list)
        delta_all = np.array(delta_list)
        
        # Compute forces from steady-state assumption (vy_dot ≈ 0, omega_dot ≈ 0)
        # From equations: Fyr = m*lf/(lf+lr) * vx * omega
        #                 Fyf = m*lr/(lf+lr) * vx * omega / cos(delta)
        Fyr = self.car.m * self.car.lf / self.car.L * vx * omega_all
        Fyf = self.car.m * self.car.lr / self.car.L * vx * omega_all / np.cos(delta_all + 1e-6)
        
        return alpha_f_all, Fyf, alpha_r_all, Fyr
    
    def fit_pacejka(self, alpha: np.ndarray, Fy: np.ndarray, 
                    initial_params: Tuple[float, float, float, float] = None,
                    is_front: bool = True) -> Tuple[float, float, float, float]:
        """
        Fit Pacejka Magic Formula parameters to force vs slip angle data
        using least squares regression.
        
        For scaled RC cars, the typical parameter ranges are:
        - B (stiffness): 2-15 for scaled vehicles
        - C (shape): 1.0-2.0 (typically around 1.3-1.5)
        - D (peak): normalized to 1.0
        - E (curvature): -2 to 1 (negative gives more aggressive turn-in)
        
        Returns: (B, C, D, E)
        """
        def pacejka_func(alpha, B, C, D, E):
            Balpha = B * alpha
            return D * np.sin(C * np.arctan(Balpha - E * (Balpha - np.arctan(Balpha))))
        
        # Filter valid data (remove very small slip angles)
        mask = np.abs(alpha) > 0.005
        alpha_valid = alpha[mask]
        Fy_valid = Fy[mask]
        
        if len(alpha_valid) < 10:
            print("Warning: Not enough valid data points for fitting")
            return (5.0, 1.3, 1.0, 0.0)
        
        # Normalize forces for fitting
        # Use the maximum force as reference for D=1 normalization
        Fy_max = np.max(np.abs(Fy_valid))
        if Fy_max > 0:
            Fy_norm = Fy_valid / Fy_max
        else:
            Fy_norm = Fy_valid
        
        # Initial parameters - use physically reasonable defaults for scaled vehicles
        if initial_params is None:
            p0 = [6.0, 1.3, 1.0, 0.0]
        else:
            # Constrain initial values to reasonable ranges
            p0 = [
                np.clip(initial_params[0], 2.0, 12.0),
                np.clip(initial_params[1], 1.0, 1.8),
                1.0,  # Always start D at 1.0 (normalized)
                np.clip(initial_params[3], -1.0, 0.5),
            ]
        
        # Bounds for scaled RC vehicles 
        # B: stiffness factor (wider range to capture different tire behaviors)
        # C: shape factor (1.0-2.0 typical range)
        # D: normalized to 1.0
        # E: curvature factor (can be negative or positive)
        bounds = ([1.0, 0.8, 0.9, -2.0], [20.0, 2.0, 1.1, 1.5])
        
        try:
            popt, pcov = curve_fit(pacejka_func, alpha_valid, Fy_norm, 
                                   p0=p0, bounds=bounds, maxfev=10000,
                                   method='trf')  # Trust Region Reflective for bounded problems
            
            # Normalize D to exactly 1.0 and adjust other params accordingly
            D_fitted = popt[2]
            popt = list(popt)
            popt[2] = 1.0  # Fix D to 1.0
            
            return tuple(popt)
        except Exception as e:
            print(f"Warning: Pacejka fitting failed: {e}")
            return tuple(p0)
    
    def run_iteration(self, states: np.ndarray, inputs: np.ndarray,
                      nn_epochs: int = 500, verbose: bool = True) -> PacejkaParams:
        """
        Run one iteration of the learning-based identification:
        1. Create model with current Pacejka parameters
        2. Compute prediction errors
        3. Train NN on errors
        4. Generate virtual steady-state data
        5. Fit new Pacejka parameters
        """
        current_pacejka = self.pacejka_history[-1]
        model = VehicleModel(self.car, current_pacejka, self.dt)
        
        if verbose:
            print(f"\n--- Iteration {len(self.pacejka_history)} ---")
            print(f"Current Pacejka: Bf={current_pacejka.Bf:.3f}, Cf={current_pacejka.Cf:.3f}, "
                  f"Br={current_pacejka.Br:.3f}, Cr={current_pacejka.Cr:.3f}")
        
        # Step 1: Compute prediction errors
        errors = self.compute_prediction_errors(model, states, inputs)
        
        # Prepare NN inputs: [vx, vy, omega, delta]
        nn_inputs = np.column_stack([
            inputs[:-1, 0],  # vx
            states[:-1, 0],  # vy
            states[:-1, 1],  # omega
            inputs[:-1, 1],  # delta
        ])
        
        if verbose:
            print(f"Prediction error stats: e_vy mean={np.mean(errors[:,0]):.4f}, std={np.std(errors[:,0]):.4f}")
            print(f"                        e_omega mean={np.mean(errors[:,1]):.4f}, std={np.std(errors[:,1]):.4f}")
        
        # Step 2: Train Neural Network
        if verbose:
            print("Training neural network...")
        nn = ResidualNN()
        nn.train(nn_inputs, errors, epochs=nn_epochs, verbose=verbose)
        self.nn_history.append(nn)
        
        # Step 3: Generate virtual steady-state data
        avg_vx = np.mean(inputs[:, 0])
        if verbose:
            print(f"Generating virtual steady-state data (avg vx={avg_vx:.2f} m/s)...")
        
        alpha_f, Fyf, alpha_r, Fyr = self.generate_virtual_steady_state_v2(
            model, nn, avg_vx, max_delta=0.4, duration=15.0
        )
        
        # Step 4: Fit Pacejka parameters
        if verbose:
            print("Fitting Pacejka parameters...")
        
        Bf, Cf, Df, Ef = self.fit_pacejka(alpha_f, Fyf, 
                                          (current_pacejka.Bf, current_pacejka.Cf, 
                                           current_pacejka.Df, current_pacejka.Ef),
                                          is_front=True)
        Br, Cr, Dr, Er = self.fit_pacejka(alpha_r, Fyr,
                                          (current_pacejka.Br, current_pacejka.Cr,
                                           current_pacejka.Dr, current_pacejka.Er),
                                          is_front=False)
        
        new_pacejka = PacejkaParams(
            Bf=Bf, Cf=Cf, Df=Df, Ef=Ef,
            Br=Br, Cr=Cr, Dr=Dr, Er=Er,
            mu=current_pacejka.mu
        )
        
        self.pacejka_history.append(new_pacejka)
        
        if verbose:
            print(f"New Pacejka: Bf={Bf:.3f}, Cf={Cf:.3f}, Df={Df:.3f}, Ef={Ef:.3f}")
            print(f"             Br={Br:.3f}, Cr={Cr:.3f}, Dr={Dr:.3f}, Er={Er:.3f}")
        
        return new_pacejka
    
    def run(self, states: np.ndarray, inputs: np.ndarray,
            num_iterations: int = 6, nn_epochs: int = 500, verbose: bool = True) -> PacejkaParams:
        """
        Run the full iterative identification process
        """
        if verbose:
            print("=" * 60)
            print("Learning-Based Pacejka Parameter Estimation")
            print("=" * 60)
            print(f"Data: {len(states)} samples")
            print(f"Iterations: {num_iterations}")
        
        for i in range(num_iterations):
            self.run_iteration(states, inputs, nn_epochs, verbose)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Final Results")
            print("=" * 60)
            final = self.pacejka_history[-1]
            print(f"Front tire: B={final.Bf:.3f}, C={final.Cf:.3f}, D={final.Df:.3f}, E={final.Ef:.3f}")
            print(f"Rear tire:  B={final.Br:.3f}, C={final.Cr:.3f}, D={final.Dr:.3f}, E={final.Er:.3f}")
        
        return self.pacejka_history[-1]
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot the evolution of Pacejka parameters and force curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Parameter evolution
        ax = axes[0, 0]
        iterations = range(len(self.pacejka_history))
        Bf_vals = [p.Bf for p in self.pacejka_history]
        Cf_vals = [p.Cf for p in self.pacejka_history]
        Br_vals = [p.Br for p in self.pacejka_history]
        Cr_vals = [p.Cr for p in self.pacejka_history]
        
        ax.plot(iterations, Bf_vals, 'b-o', label='Bf (front)')
        ax.plot(iterations, Br_vals, 'r-s', label='Br (rear)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('B (stiffness factor)')
        ax.set_title('Pacejka B Parameter Evolution')
        ax.legend()
        ax.grid(True)
        
        ax = axes[0, 1]
        ax.plot(iterations, Cf_vals, 'b-o', label='Cf (front)')
        ax.plot(iterations, Cr_vals, 'r-s', label='Cr (rear)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('C (shape factor)')
        ax.set_title('Pacejka C Parameter Evolution')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Force vs slip angle curves
        alpha = np.linspace(0, 0.25, 100)
        
        # Front tire
        ax = axes[1, 0]
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(self.pacejka_history)))
        for i, (p, c) in enumerate(zip(self.pacejka_history, colors)):
            Balpha = p.Bf * alpha
            F = p.Df * np.sin(p.Cf * np.arctan(Balpha - p.Ef * (Balpha - np.arctan(Balpha))))
            label = f'Iter {i}' if i == 0 or i == len(self.pacejka_history)-1 else None
            ax.plot(np.degrees(alpha), F, color=c, label=label, 
                   linewidth=2 if i == len(self.pacejka_history)-1 else 1)
        ax.set_xlabel('Slip Angle [deg]')
        ax.set_ylabel('Normalized Force')
        ax.set_title('Front Tire: Force vs Slip Angle')
        ax.legend()
        ax.grid(True)
        
        # Rear tire
        ax = axes[1, 1]
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(self.pacejka_history)))
        for i, (p, c) in enumerate(zip(self.pacejka_history, colors)):
            Balpha = p.Br * alpha
            F = p.Dr * np.sin(p.Cr * np.arctan(Balpha - p.Er * (Balpha - np.arctan(Balpha))))
            label = f'Iter {i}' if i == 0 or i == len(self.pacejka_history)-1 else None
            ax.plot(np.degrees(alpha), F, color=c, label=label,
                   linewidth=2 if i == len(self.pacejka_history)-1 else 1)
        ax.set_xlabel('Slip Angle [deg]')
        ax.set_ylabel('Normalized Force')
        ax.set_title('Rear Tire: Force vs Slip Angle')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close(fig)


def analyze_raw_data(states: np.ndarray, inputs: np.ndarray, car: CarParams, 
                     output_dir: str, dt: float = 0.04, mu: float = 0.9):
    """
    Directly analyze the tire forces from raw measurement data.
    This provides ground truth tire characteristics by computing:
    1. Slip angles from kinematics
    2. Lateral forces from dynamics (accelerations)
    
    NOTE: states[:, 0] should be vy_CG (center of gravity lateral velocity)
    """
    print("\n" + "=" * 60)
    print("Direct Data Analysis - Tire Characteristics from Measurements")
    print("=" * 60)
    
    vy_CG = states[:, 0]  # Already converted to CG in preprocessing
    omega = states[:, 1]
    vx = inputs[:, 0]
    delta = inputs[:, 1]
    
    # Compute slip angles (using CG velocity)
    vx_safe = np.maximum(vx, 0.3)
    alpha_f = delta - np.arctan2(vy_CG + car.lf * omega, vx_safe)
    alpha_r = -np.arctan2(vy_CG - car.lr * omega, vx_safe)
    
    # Compute lateral acceleration from state changes
    vy_dot = np.gradient(vy_CG, dt)
    omega_dot = np.gradient(omega, dt)
    
    # From dynamics equations, compute forces
    # vy_dot = (Fyr + Fyf*cos(delta) - m*vx*omega) / m
    # omega_dot = (Fyf*lf*cos(delta) - Fyr*lr) / Iz
    # 
    # Solve for Fyf and Fyr:
    # Fyf = (Iz * omega_dot + Fyr * lr) / (lf * cos(delta))
    # Fyr = m * (vy_dot + vx * omega) - Fyf * cos(delta)
    #
    # From steady-state (ignore derivatives for now, use acceleration-based):
    # m * vy_dot = Fyr + Fyf * cos(delta) - m * vx * omega
    # Iz * omega_dot = Fyf * lf * cos(delta) - Fyr * lr
    
    # Solve the 2x2 system for each timestep
    cos_delta = np.cos(delta)
    
    # Matrix: [1, cos(delta)]  * [Fyr]   = [m*vy_dot + m*vx*omega]
    #         [-lr, lf*cos(delta)]  [Fyf]   = [Iz*omega_dot]
    
    Fyr_list = []
    Fyf_list = []
    
    for i in range(len(vy_CG)):
        A = np.array([[1, cos_delta[i]],
                      [-car.lr, car.lf * cos_delta[i]]])
        b = np.array([car.m * vy_dot[i] + car.m * vx[i] * omega[i],
                      car.Iz * omega_dot[i]])
        try:
            F = np.linalg.solve(A, b)
            Fyr_list.append(F[0])
            Fyf_list.append(F[1])
        except:
            Fyr_list.append(np.nan)
            Fyf_list.append(np.nan)
    
    Fyr = np.array(Fyr_list)
    Fyf = np.array(Fyf_list)
    
    # Aggressive outlier filtering for forces (for noisy 1/10 scale car data)
    # First, remove NaN and infinite values
    valid_nan = ~np.isnan(Fyf) & ~np.isnan(Fyr) & np.isfinite(Fyf) & np.isfinite(Fyr)
    
    # Physical limits (more conservative for noisy data)
    Fz_max = car.m * 9.81  # Maximum normal force
    mu_max = 1.2  # More conservative than 1.5 for noisy data
    F_limit = mu_max * Fz_max
    valid_physical = (np.abs(Fyf) < F_limit) & (np.abs(Fyr) < F_limit)
    
    # IQR-based outlier filtering for forces (aggressive)
    def filter_outliers_iqr(data, factor=IQR_FACTOR_FORCES):
        """Filter outliers using Interquartile Range method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return (data >= lower_bound) & (data <= upper_bound)
        else:
            return np.ones_like(data, dtype=bool)
    
    # Filter forces using IQR (aggressive filtering)
    temp_mask = valid_nan & valid_physical
    if np.sum(temp_mask) > 10:
        valid_fyf_iqr = filter_outliers_iqr(Fyf[temp_mask], factor=IQR_FACTOR_FORCES)
        valid_fyr_iqr = filter_outliers_iqr(Fyr[temp_mask], factor=IQR_FACTOR_FORCES)
        
        # Create full mask
        valid_iqr = np.zeros_like(temp_mask, dtype=bool)
        valid_iqr[temp_mask] = valid_fyf_iqr & valid_fyr_iqr
    else:
        valid_iqr = temp_mask
    
    # Also filter based on slip angle reasonableness
    # Slip angles shouldn't exceed ~30 degrees (0.52 rad) for normal driving
    max_slip_angle = 0.5  # rad (~29 degrees) - more conservative
    valid_slip = (np.abs(alpha_f) < max_slip_angle) & (np.abs(alpha_r) < max_slip_angle)
    
    # Filter based on rate of change (smoothness check)
    # Forces shouldn't change too rapidly between samples
    if len(Fyf) > 2:
        Fyf_diff = np.abs(np.diff(Fyf))
        Fyr_diff = np.abs(np.diff(Fyr))
        max_force_change = F_limit * 0.3  # Max 30% of limit force change per timestep
        valid_smooth = np.ones(len(Fyf), dtype=bool)
        valid_smooth[1:] = (Fyf_diff < max_force_change) & (Fyr_diff < max_force_change)
    else:
        valid_smooth = np.ones(len(Fyf), dtype=bool)
    
    # Combine all filters
    valid = valid_nan & valid_physical & valid_iqr & valid_slip & valid_smooth
    
    alpha_f_valid = alpha_f[valid]
    alpha_r_valid = alpha_r[valid]
    Fyf_valid = Fyf[valid]
    Fyr_valid = Fyr[valid]
    
    print(f"Force outlier filtering: {np.sum(valid)} / {len(valid)} samples kept")
    print(f"  Removed {np.sum(~valid_nan)} NaN/inf, {np.sum(~valid_physical)} physical limit violations,")
    print(f"  {np.sum(~valid_iqr)} IQR outliers, {np.sum(~valid_slip)} extreme slip angles, {np.sum(~valid_smooth)} rapid changes")
    print(f"Front slip angle range: [{np.degrees(np.min(alpha_f_valid)):.2f}, {np.degrees(np.max(alpha_f_valid)):.2f}] deg")
    print(f"Rear slip angle range: [{np.degrees(np.min(alpha_r_valid)):.2f}, {np.degrees(np.max(alpha_r_valid)):.2f}] deg")
    print(f"Front force range: [{np.min(Fyf_valid):.2f}, {np.max(Fyf_valid):.2f}] N")
    print(f"Rear force range: [{np.min(Fyr_valid):.2f}, {np.max(Fyr_valid):.2f}] N")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Front tire - Force vs Slip Angle scatter
    ax = axes[0, 0]
    ax.scatter(np.degrees(alpha_f_valid), Fyf_valid, alpha=0.1, s=1, c='blue')
    ax.set_xlabel('Front Slip Angle [deg]')
    ax.set_ylabel('Front Lateral Force [N]')
    ax.set_title('Front Tire: Measured Force vs Slip Angle')
    ax.grid(True)
    ax.set_xlim([-20, 20])
    
    # Plot 2: Rear tire - Force vs Slip Angle scatter
    ax = axes[0, 1]
    ax.scatter(np.degrees(alpha_r_valid), Fyr_valid, alpha=0.1, s=1, c='red')
    ax.set_xlabel('Rear Slip Angle [deg]')
    ax.set_ylabel('Rear Lateral Force [N]')
    ax.set_title('Rear Tire: Measured Force vs Slip Angle')
    ax.grid(True)
    ax.set_xlim([-20, 20])
    
    # Plot 3: Slip angle histogram
    ax = axes[0, 2]
    ax.hist(np.degrees(alpha_f_valid), bins=50, alpha=0.5, label='Front', color='blue')
    ax.hist(np.degrees(alpha_r_valid), bins=50, alpha=0.5, label='Rear', color='red')
    ax.set_xlabel('Slip Angle [deg]')
    ax.set_ylabel('Count')
    ax.set_title('Slip Angle Distribution')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Binned average for front tire (cleaner view) with robust statistics
    ax = axes[1, 0]
    bins = np.linspace(-0.25, 0.25, 30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    front_means = []
    front_stds = []
    for i in range(len(bins) - 1):
        mask = (alpha_f_valid >= bins[i]) & (alpha_f_valid < bins[i+1])
        if np.sum(mask) > MIN_SAMPLES_PER_BIN:  # Require more samples
            F_bin = Fyf_valid[mask]
            # Filter outliers within bin
            Q1 = np.percentile(F_bin, 25)
            Q3 = np.percentile(F_bin, 75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - IQR_FACTOR_BINS * IQR
                upper = Q3 + IQR_FACTOR_BINS * IQR
                mask_bin = (F_bin >= lower) & (F_bin <= upper)
                F_bin_clean = F_bin[mask_bin]
            else:
                F_bin_clean = F_bin
            
            if len(F_bin_clean) > 5:
                front_means.append(np.median(F_bin_clean))
                front_stds.append(np.percentile(F_bin_clean, 75) - np.percentile(F_bin_clean, 25))
            else:
                front_means.append(np.nan)
                front_stds.append(np.nan)
        else:
            front_means.append(np.nan)
            front_stds.append(np.nan)
    
    front_means = np.array(front_means)
    front_stds = np.array(front_stds)
    valid_bins = ~np.isnan(front_means)
    
    ax.errorbar(np.degrees(bin_centers[valid_bins]), front_means[valid_bins], 
                yerr=front_stds[valid_bins], fmt='o-', color='blue', capsize=3)
    ax.set_xlabel('Front Slip Angle [deg]')
    ax.set_ylabel('Front Lateral Force [N]')
    ax.set_title('Front Tire: Binned Mean Force')
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 5: Binned average for rear tire with robust statistics
    ax = axes[1, 1]
    rear_means = []
    rear_stds = []
    for i in range(len(bins) - 1):
        mask = (alpha_r_valid >= bins[i]) & (alpha_r_valid < bins[i+1])
        if np.sum(mask) > MIN_SAMPLES_PER_BIN:  # Require more samples
            F_bin = Fyr_valid[mask]
            # Filter outliers within bin
            Q1 = np.percentile(F_bin, 25)
            Q3 = np.percentile(F_bin, 75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - IQR_FACTOR_BINS * IQR
                upper = Q3 + IQR_FACTOR_BINS * IQR
                mask_bin = (F_bin >= lower) & (F_bin <= upper)
                F_bin_clean = F_bin[mask_bin]
            else:
                F_bin_clean = F_bin
            
            if len(F_bin_clean) > 5:
                rear_means.append(np.median(F_bin_clean))
                rear_stds.append(np.percentile(F_bin_clean, 75) - np.percentile(F_bin_clean, 25))
            else:
                rear_means.append(np.nan)
                rear_stds.append(np.nan)
        else:
            rear_means.append(np.nan)
            rear_stds.append(np.nan)
    
    rear_means = np.array(rear_means)
    rear_stds = np.array(rear_stds)
    valid_bins_r = ~np.isnan(rear_means)
    
    ax.errorbar(np.degrees(bin_centers[valid_bins_r]), rear_means[valid_bins_r], 
                yerr=rear_stds[valid_bins_r], fmt='o-', color='red', capsize=3)
    ax.set_xlabel('Rear Slip Angle [deg]')
    ax.set_ylabel('Rear Lateral Force [N]')
    ax.set_title('Rear Tire: Binned Mean Force')
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 6: Fit Pacejka to binned data
    ax = axes[1, 2]
    
    def pacejka_simple(alpha, B, C):
        """Simplified Pacejka with D=1, E=0"""
        return np.sin(C * np.arctan(B * alpha))
    
    # Fit front tire - use full Pacejka with proper scaling
    # Filter out any remaining outliers in the binned data
    alpha_fit_f = bin_centers[valid_bins]
    F_fit_f = front_means[valid_bins]
    
            # Additional outlier filtering for binned data
    if len(F_fit_f) > 5:
        Q1 = np.percentile(F_fit_f, 25)
        Q3 = np.percentile(F_fit_f, 75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - (IQR_FACTOR_BINS + 0.5) * IQR  # Slightly more aggressive for final fit
            upper = Q3 + (IQR_FACTOR_BINS + 0.5) * IQR
            mask_bin_fit = (F_fit_f >= lower) & (F_fit_f <= upper)
            alpha_fit_f = alpha_fit_f[mask_bin_fit]
            F_fit_f = F_fit_f[mask_bin_fit]
    
    # Full Pacejka fit with proper force scaling
    def pacejka_full(alpha, B, C, D, E, scale):
        """Full Pacejka with scaling factor"""
        Balpha = B * alpha
        return scale * D * np.sin(C * np.arctan(Balpha - E * (Balpha - np.arctan(Balpha))))
    
    # Estimate scale from max force
    Fzf = car.m * 9.81 * car.lr / car.L
    Fzr = car.m * 9.81 * car.lf / car.L
    mu_est = mu  # Use mu from car parameters file
    
    try:
        # Fit with scale as a parameter too for better matching
        def pacejka_with_scale(alpha, B, C, E, scale):
            return pacejka_full(alpha, B, C, 1.0, E, scale)
        
        # Estimate initial scale from max force
        F_max_f = np.max(np.abs(F_fit_f))
        scale_est_f = F_max_f * 1.0  # Start closer to measured max
        
        # Weight by number of samples in each bin (more reliable bins get more weight)
        # Map alpha_fit_f back to original bins to get weights
        weights_f = np.ones(len(alpha_fit_f))
        for idx, alpha_val in enumerate(alpha_fit_f):
            # Find which bin this alpha belongs to
            bin_idx = np.searchsorted(bins[:-1], alpha_val) - 1
            bin_idx = max(0, min(bin_idx, len(bins) - 2))
            # Count samples in that bin
            mask = (alpha_f_valid >= bins[bin_idx]) & (alpha_f_valid < bins[bin_idx + 1])
            weights_f[idx] = np.sum(mask)
        
        weights_f = weights_f / np.max(weights_f) if np.max(weights_f) > 0 else weights_f
        
        # Emphasize linear region (low slip angles) where we have more reliable data
        low_slip_mask_f = np.abs(alpha_fit_f) < 0.15  # Slip angles < ~8.5 degrees
        weights_f[low_slip_mask_f] = weights_f[low_slip_mask_f] * 3.0  # Triple weight for linear region
        
        # Moderate weight for saturation region
        high_slip_mask_f = np.abs(alpha_fit_f) > 0.1
        weights_f[high_slip_mask_f] = weights_f[high_slip_mask_f] * 1.5
        
        # Estimate initial B from linear region slope
        linear_mask = np.abs(alpha_fit_f) < 0.08
        if np.sum(linear_mask) > 3:
            # Linear fit to estimate cornering stiffness
            linear_alpha = alpha_fit_f[linear_mask]
            linear_F = F_fit_f[linear_mask]
            if len(linear_alpha) > 0 and np.max(np.abs(linear_alpha)) > 0:
                # Cornering stiffness = dF/dalpha at alpha=0
                # Approximate as average slope in linear region
                slope_est = np.mean(np.abs(linear_F) / (np.abs(linear_alpha) + 1e-6))
                # B relates to initial slope: F ≈ B*C*scale*alpha for small alpha
                # So B ≈ slope / (C * scale)
                B_est = slope_est / (1.3 * scale_est_f) if scale_est_f > 0 else 4.0
                B_est = np.clip(B_est, 2.0, 12.0)
            else:
                B_est = 4.0
        else:
            B_est = 4.0
        
        # Constrain scale to match max measured force more tightly
        F_max_measured_f = np.max(np.abs(F_fit_f))
        scale_max_f = F_max_measured_f * 1.05  # Tight constraint: 5% margin
        scale_min_f = F_max_measured_f * 0.85  # At least 85% of max
        
        # Try fitting with better initial guess
        popt_f, _ = curve_fit(
            pacejka_with_scale,
            alpha_fit_f, F_fit_f,
            p0=[B_est, 1.3, 0.0, scale_est_f],
            bounds=([2.0, 0.8, -1.0, scale_min_f], [12.0, 2.0, 1.0, scale_max_f]),
            maxfev=15000,
            sigma=1.0 / (weights_f + 0.1)  # Inverse weighting
        )
        Bf_fit, Cf_fit, Ef_fit, scale_refined_f = popt_f
        Df_fit = 1.0
        
        print(f"\nDirect fit (front): B={Bf_fit:.3f}, C={Cf_fit:.3f}, D={Df_fit:.3f}, E={Ef_fit:.3f}, scale={scale_refined_f:.2f}")
    except Exception as e:
        print(f"Front fitting failed: {e}, using fallback")
        # Fallback: simple fit
        try:
            popt_f_simple, _ = curve_fit(
                lambda alpha, B, C: pacejka_full(alpha, B, C, 1.0, 0.0, scale_est_f),
                alpha_fit_f, F_fit_f,
                p0=[4.0, 1.3],
                bounds=([1.0, 0.5], [15.0, 2.0]),
                maxfev=5000
            )
            Bf_fit, Cf_fit = popt_f_simple
            Df_fit, Ef_fit = 1.0, 0.0
            scale_refined_f = scale_est_f
        except:
            Bf_fit, Cf_fit, Df_fit, Ef_fit = 4.0, 1.3, 1.0, 0.0
            scale_refined_f = mu_est * Fzf
    
    # Fit rear tire
    # Filter out any remaining outliers in the binned data
    alpha_fit_r = bin_centers[valid_bins_r]
    F_fit_r = rear_means[valid_bins_r]
    
    # Additional outlier filtering for binned data
    if len(F_fit_r) > 5:
        Q1 = np.percentile(F_fit_r, 25)
        Q3 = np.percentile(F_fit_r, 75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - (IQR_FACTOR_BINS + 0.5) * IQR  # Slightly more aggressive for final fit
            upper = Q3 + (IQR_FACTOR_BINS + 0.5) * IQR
            mask_bin_fit_r = (F_fit_r >= lower) & (F_fit_r <= upper)
            alpha_fit_r = alpha_fit_r[mask_bin_fit_r]
            F_fit_r = F_fit_r[mask_bin_fit_r]
    
    try:
        # Fit with scale as a parameter
        def pacejka_with_scale_r(alpha, B, C, E, scale):
            return pacejka_full(alpha, B, C, 1.0, E, scale)
        
        F_max_r = np.max(np.abs(F_fit_r))
        scale_est_r = F_max_r * 1.0  # Start closer to measured max
        
        # Weight by number of samples
        # Map alpha_fit_r back to original bins to get weights
        weights_r = np.ones(len(alpha_fit_r))
        for idx, alpha_val in enumerate(alpha_fit_r):
            # Find which bin this alpha belongs to
            bin_idx = np.searchsorted(bins[:-1], alpha_val) - 1
            bin_idx = max(0, min(bin_idx, len(bins) - 2))
            # Count samples in that bin
            mask = (alpha_r_valid >= bins[bin_idx]) & (alpha_r_valid < bins[bin_idx + 1])
            weights_r[idx] = np.sum(mask)
        
        weights_r = weights_r / np.max(weights_r) if np.max(weights_r) > 0 else weights_r
        
        # Increase weight for saturation region (rear tire works well with this)
        high_slip_mask_r = np.abs(alpha_fit_r) > 0.1
        weights_r[high_slip_mask_r] = weights_r[high_slip_mask_r] * 2.0
        
        # Constrain scale to match max measured force
        F_max_measured_r = np.max(np.abs(F_fit_r))
        scale_max_r = F_max_measured_r * 1.1
        scale_min_r = F_max_measured_r * 0.7
        
        # Ensure initial guess is within bounds
        scale_est_r = np.clip(scale_est_r, scale_min_r, scale_max_r)
        
        popt_r, _ = curve_fit(
            pacejka_with_scale_r,
            alpha_fit_r, F_fit_r,
            p0=[4.0, 1.3, 0.0, scale_est_r],
            bounds=([1.0, 0.5, -1.5, scale_min_r], [15.0, 2.0, 1.0, scale_max_r]),
            maxfev=10000,
            sigma=1.0 / (weights_r + 0.1)
        )
        Br_fit, Cr_fit, Er_fit, scale_refined_r = popt_r
        Dr_fit = 1.0
        
        print(f"Direct fit (rear): B={Br_fit:.3f}, C={Cr_fit:.3f}, D={Dr_fit:.3f}, E={Er_fit:.3f}, scale={scale_refined_r:.2f}")
    except Exception as e:
        print(f"Rear fitting failed: {e}, using fallback")
        try:
            popt_r_simple, _ = curve_fit(
                lambda alpha, B, C: pacejka_full(alpha, B, C, 1.0, 0.0, scale_est_r),
                alpha_fit_r, F_fit_r,
                p0=[4.0, 1.3],
                bounds=([1.0, 0.5], [15.0, 2.0]),
                maxfev=5000
            )
            Br_fit, Cr_fit = popt_r_simple
            Dr_fit, Er_fit = 1.0, 0.0
            scale_refined_r = scale_est_r
        except:
            Br_fit, Cr_fit, Dr_fit, Er_fit = 4.0, 1.3, 1.0, 0.0
            scale_refined_r = mu_est * Fzr
    
    # Plot fits - normalize for display
    alpha_plot = np.linspace(-0.25, 0.25, 100)
    
    F_front_fit = pacejka_full(alpha_plot, Bf_fit, Cf_fit, Df_fit, Ef_fit, scale_refined_f)
    F_rear_fit = pacejka_full(alpha_plot, Br_fit, Cr_fit, Dr_fit, Er_fit, scale_refined_r)
    
    # Normalize for display
    F_max_f_display = max(np.max(np.abs(F_front_fit)), np.max(np.abs(F_fit_f))) if len(F_fit_f) > 0 else 1.0
    F_max_r_display = max(np.max(np.abs(F_rear_fit)), np.max(np.abs(F_fit_r))) if len(F_fit_r) > 0 else 1.0
    
    F_front_norm = F_front_fit / F_max_f_display if F_max_f_display > 0 else F_front_fit
    F_rear_norm = F_rear_fit / F_max_r_display if F_max_r_display > 0 else F_rear_fit
    F_fit_f_norm = F_fit_f / F_max_f_display if F_max_f_display > 0 else F_fit_f
    F_fit_r_norm = F_fit_r / F_max_r_display if F_max_r_display > 0 else F_fit_r
    
    ax.plot(np.degrees(alpha_plot), F_front_norm, 'b-', 
            label=f'Front: B={Bf_fit:.2f}, C={Cf_fit:.2f}, E={Ef_fit:.2f}', linewidth=2)
    ax.plot(np.degrees(alpha_plot), F_rear_norm, 'r-', 
            label=f'Rear: B={Br_fit:.2f}, C={Cr_fit:.2f}, E={Er_fit:.2f}', linewidth=2)
    ax.scatter(np.degrees(alpha_fit_f), F_fit_f_norm, c='blue', s=30, alpha=0.7)
    ax.scatter(np.degrees(alpha_fit_r), F_fit_r_norm, c='red', s=30, alpha=0.7)
    ax.set_xlabel('Slip Angle [deg]')
    ax.set_ylabel('Normalized Force')
    ax.set_title('Pacejka Fit to Binned Data')
    ax.legend()
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "raw_data_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nRaw data analysis saved to {plot_path}")
    
    # Return full Pacejka parameters with mu from car file
    pacejka_direct = PacejkaParams(
        Bf=Bf_fit, Cf=Cf_fit, Df=Df_fit, Ef=Ef_fit,
        Br=Br_fit, Cr=Cr_fit, Dr=Dr_fit, Er=Er_fit,
        mu=mu_est  # This is the mu from the car parameters file
    )
    
    return pacejka_direct, alpha_f_valid, Fyf_valid, alpha_r_valid, Fyr_valid


def plot_final_pacejka_overlay(states: np.ndarray, inputs: np.ndarray, 
                                car: CarParams, pacejka_params: PacejkaParams,
                                output_dir: str, dt: float = 0.04, mu: float = None):
    """
    Create a final comprehensive plot showing the estimated Pacejka curves
    overlaid on the actual recorded data.
    """
    print("\n" + "=" * 60)
    print("Creating Final Pacejka Overlay Plot")
    print("=" * 60)
    
    vy_CG = states[:, 0]
    omega = states[:, 1]
    vx = inputs[:, 0]
    delta = inputs[:, 1]
    
    # Compute slip angles
    vx_safe = np.maximum(vx, 0.3)
    alpha_f = delta - np.arctan2(vy_CG + car.lf * omega, vx_safe)
    alpha_r = -np.arctan2(vy_CG - car.lr * omega, vx_safe)
    
    # Compute forces from dynamics
    vy_dot = np.gradient(vy_CG, dt)
    omega_dot = np.gradient(omega, dt)
    
    cos_delta = np.cos(delta)
    Fyr_list = []
    Fyf_list = []
    
    for i in range(len(vy_CG)):
        A = np.array([[1, cos_delta[i]],
                      [-car.lr, car.lf * cos_delta[i]]])
        b = np.array([car.m * vy_dot[i] + car.m * vx[i] * omega[i],
                      car.Iz * omega_dot[i]])
        try:
            F = np.linalg.solve(A, b)
            Fyr_list.append(F[0])
            Fyf_list.append(F[1])
        except:
            Fyr_list.append(np.nan)
            Fyf_list.append(np.nan)
    
    Fyr = np.array(Fyr_list)
    Fyf = np.array(Fyf_list)
    
    # Aggressive outlier filtering (same as in analyze_raw_data)
    valid_nan = ~np.isnan(Fyf) & ~np.isnan(Fyr) & np.isfinite(Fyf) & np.isfinite(Fyr)
    
    Fz_max = car.m * 9.81
    mu_max = 1.2  # More conservative
    F_limit = mu_max * Fz_max
    valid_physical = (np.abs(Fyf) < F_limit) & (np.abs(Fyr) < F_limit)
    
    # IQR filtering
    def filter_outliers_iqr(data, factor=2.0):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        if IQR > 0:
            return (data >= Q1 - factor * IQR) & (data <= Q3 + factor * IQR)
        return np.ones_like(data, dtype=bool)
    
    temp_mask = valid_nan & valid_physical
    if np.sum(temp_mask) > 10:
        valid_iqr = np.zeros_like(temp_mask, dtype=bool)
        valid_iqr[temp_mask] = filter_outliers_iqr(Fyf[temp_mask], 2.0) & filter_outliers_iqr(Fyr[temp_mask], 2.0)
    else:
        valid_iqr = temp_mask
    
    max_slip_angle = 0.5  # rad
    valid_slip = (np.abs(alpha_f) < max_slip_angle) & (np.abs(alpha_r) < max_slip_angle)
    
    valid = valid_nan & valid_physical & valid_iqr & valid_slip
    
    alpha_f_valid = alpha_f[valid]
    alpha_r_valid = alpha_r[valid]
    Fyf_valid = Fyf[valid]
    Fyr_valid = Fyr[valid]
    
    # Create Pacejka curves from estimated parameters
    def pacejka_force(alpha, B, C, D, E):
        """Full Pacejka Magic Formula"""
        Balpha = B * alpha
        return D * np.sin(C * np.arctan(Balpha - E * (Balpha - np.arctan(Balpha))))
    
    # Compute normal loads
    Fzf = car.m * 9.81 * car.lr / car.L
    Fzr = car.m * 9.81 * car.lf / car.L
    # Use mu from function parameter if provided, otherwise from pacejka_params
    mu_plot = mu if mu is not None else pacejka_params.mu
    
    # Generate Pacejka curves - fit to match actual force magnitudes
    alpha_plot = np.linspace(-0.3, 0.3, 200)
    
    # Front tire Pacejka curve
    Fyf_pacejka_norm = pacejka_force(alpha_plot, pacejka_params.Bf, pacejka_params.Cf,
                                     pacejka_params.Df, pacejka_params.Ef)
    
    # Scale to match actual measured forces
    # Find the scale factor that best matches the binned data
    bins = np.linspace(-0.25, 0.25, 25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    front_means_bin = []
    for i in range(len(bins) - 1):
        mask = (alpha_f_valid >= bins[i]) & (alpha_f_valid < bins[i+1])
        if np.sum(mask) > MIN_SAMPLES_PER_BIN:  # Require more samples
            F_bin = Fyf_valid[mask]
            # Filter outliers within bin
            Q1 = np.percentile(F_bin, 25)
            Q3 = np.percentile(F_bin, 75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - IQR_FACTOR_BINS * IQR
                upper = Q3 + IQR_FACTOR_BINS * IQR
                F_bin_clean = F_bin[(F_bin >= lower) & (F_bin <= upper)]
            else:
                F_bin_clean = F_bin
            if len(F_bin_clean) > 5:
                front_means_bin.append(np.median(F_bin_clean))  # Use median
            else:
                front_means_bin.append(np.nan)
        else:
            front_means_bin.append(np.nan)
    front_means_bin = np.array(front_means_bin)
    valid_bins_f = ~np.isnan(front_means_bin)
    
    if np.sum(valid_bins_f) > 3:
        # Compute Pacejka at bin centers
        alpha_bins = bin_centers[valid_bins_f]
        Fyf_pacejka_bins = pacejka_force(alpha_bins, pacejka_params.Bf, pacejka_params.Cf,
                                        pacejka_params.Df, pacejka_params.Ef)
        
        # For front tire, match both linear region AND saturation
        # Linear region (more reliable)
        linear_mask_f = np.abs(alpha_bins) < 0.12  # Slip angles < ~7 degrees
        # Saturation region
        high_slip_mask = np.abs(alpha_bins) > 0.1  # Slip angles > ~6 degrees
        
        if np.sum(linear_mask_f) > 3:
            # Match linear region first (where we have more data)
            F_measured_linear = front_means_bin[valid_bins_f][linear_mask_f]
            F_pacejka_linear = Fyf_pacejka_bins[linear_mask_f]
            
            # Use least squares to find best scale for linear region
            valid_linear = (np.abs(F_measured_linear) > 0.1) & (np.abs(F_pacejka_linear) > 0.01)
            if np.sum(valid_linear) > 2:
                scale_f_linear = np.sum(F_measured_linear[valid_linear] * F_pacejka_linear[valid_linear]) / np.sum(F_pacejka_linear[valid_linear]**2)
            else:
                scale_f_linear = np.mean(np.abs(F_measured_linear)) / np.mean(np.abs(F_pacejka_linear)) if np.mean(np.abs(F_pacejka_linear)) > 0 else mu * Fzf
            
            # Also check saturation
            if np.sum(high_slip_mask) > 2:
                F_measured_sat = np.median(np.abs(front_means_bin[valid_bins_f][high_slip_mask]))
                F_pacejka_sat = np.median(np.abs(Fyf_pacejka_bins[high_slip_mask]))
                if F_pacejka_sat > 0:
                    scale_f_sat = F_measured_sat / F_pacejka_sat
                else:
                    scale_f_sat = scale_f_linear
                
                # Weighted average: 70% linear, 30% saturation
                scale_f = 0.7 * scale_f_linear + 0.3 * scale_f_sat
            else:
                scale_f = scale_f_linear
        else:
            # Fallback: match overall
            scale_f = np.mean(np.abs(front_means_bin[valid_bins_f])) / np.mean(np.abs(Fyf_pacejka_bins)) if np.mean(np.abs(Fyf_pacejka_bins)) > 0 else mu_plot * Fzf
        
        Fyf_pacejka = scale_f * Fyf_pacejka_norm
    else:
        Fyf_pacejka = mu_plot * Fzf * Fyf_pacejka_norm
    
    # Rear tire Pacejka curve
    Fyr_pacejka_norm = pacejka_force(alpha_plot, pacejka_params.Br, pacejka_params.Cr,
                                      pacejka_params.Dr, pacejka_params.Er)
    
    rear_means_bin = []
    for i in range(len(bins) - 1):
        mask = (alpha_r_valid >= bins[i]) & (alpha_r_valid < bins[i+1])
        if np.sum(mask) > MIN_SAMPLES_PER_BIN:  # Require more samples
            F_bin = Fyr_valid[mask]
            # Filter outliers within bin
            Q1 = np.percentile(F_bin, 25)
            Q3 = np.percentile(F_bin, 75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - IQR_FACTOR_BINS * IQR
                upper = Q3 + IQR_FACTOR_BINS * IQR
                F_bin_clean = F_bin[(F_bin >= lower) & (F_bin <= upper)]
            else:
                F_bin_clean = F_bin
            if len(F_bin_clean) > 5:
                rear_means_bin.append(np.median(F_bin_clean))  # Use median
            else:
                rear_means_bin.append(np.nan)
        else:
            rear_means_bin.append(np.nan)
    rear_means_bin = np.array(rear_means_bin)
    valid_bins_r = ~np.isnan(rear_means_bin)
    
    if np.sum(valid_bins_r) > 3:
        alpha_bins_r = bin_centers[valid_bins_r]
        Fyr_pacejka_bins = pacejka_force(alpha_bins_r, pacejka_params.Br, pacejka_params.Cr,
                                        pacejka_params.Dr, pacejka_params.Er)
        # Match saturation level for rear tire too
        high_slip_mask_r = np.abs(alpha_bins_r) > 0.1
        if np.sum(high_slip_mask_r) > 2:
            F_measured_sat_r = np.median(np.abs(rear_means_bin[valid_bins_r][high_slip_mask_r]))
            F_pacejka_sat_r = np.median(np.abs(Fyr_pacejka_bins[high_slip_mask_r]))
            if F_pacejka_sat_r > 0:
                scale_r = F_measured_sat_r / F_pacejka_sat_r
            else:
                scale_r = np.mean(np.abs(rear_means_bin[valid_bins_r])) / np.mean(np.abs(Fyr_pacejka_bins)) if np.mean(np.abs(Fyr_pacejka_bins)) > 0 else mu_plot * Fzr
        else:
            scale_r = np.mean(np.abs(rear_means_bin[valid_bins_r])) / np.mean(np.abs(Fyr_pacejka_bins)) if np.mean(np.abs(Fyr_pacejka_bins)) > 0 else mu_plot * Fzr
        Fyr_pacejka = scale_r * Fyr_pacejka_norm
    else:
        Fyr_pacejka = mu_plot * Fzr * Fyr_pacejka_norm
    
    # Create the final overlay plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Front tire plot
    ax = axes[0]
    # Scatter plot of actual data (light, semi-transparent)
    ax.scatter(np.degrees(alpha_f_valid), Fyf_valid, alpha=0.05, s=2, c='lightblue', 
               label='Measured Data', zorder=1)
    
    # Binned statistics with robust outlier filtering
    bins = np.linspace(-0.25, 0.25, 25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    front_means = []
    front_stds = []
    for i in range(len(bins) - 1):
        mask = (alpha_f_valid >= bins[i]) & (alpha_f_valid < bins[i+1])
        if np.sum(mask) > MIN_SAMPLES_PER_BIN:  # Require more samples for reliability
            # Use median for robustness, then filter outliers within bin
            F_bin = Fyf_valid[mask]
            # Filter outliers within bin using IQR
            Q1 = np.percentile(F_bin, 25)
            Q3 = np.percentile(F_bin, 75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - IQR_FACTOR_BINS * IQR
                upper = Q3 + IQR_FACTOR_BINS * IQR
                mask_bin = (F_bin >= lower) & (F_bin <= upper)
                F_bin_clean = F_bin[mask_bin]
            else:
                F_bin_clean = F_bin
            
            if len(F_bin_clean) > 5:
                front_means.append(np.median(F_bin_clean))  # Use median for robustness
                front_stds.append(np.percentile(F_bin_clean, 75) - np.percentile(F_bin_clean, 25))  # IQR as spread
            else:
                front_means.append(np.nan)
                front_stds.append(np.nan)
        else:
            front_means.append(np.nan)
            front_stds.append(np.nan)
    
    front_means = np.array(front_means)
    front_stds = np.array(front_stds)
    valid_bins = ~np.isnan(front_means)
    
    ax.errorbar(np.degrees(bin_centers[valid_bins]), front_means[valid_bins],
                yerr=front_stds[valid_bins], fmt='o', color='blue', capsize=3,
                markersize=6, label='Binned Mean ± Std', zorder=2, alpha=0.7)
    
    # Pacejka curve
    ax.plot(np.degrees(alpha_plot), Fyf_pacejka, 'b-', linewidth=3, 
            label=f'Pacejka Fit\nB={pacejka_params.Bf:.2f}, C={pacejka_params.Cf:.2f}\n'
                  f'D={pacejka_params.Df:.2f}, E={pacejka_params.Ef:.2f}',
            zorder=3)
    
    ax.set_xlabel('Front Slip Angle [deg]', fontsize=12)
    ax.set_ylabel('Front Lateral Force [N]', fontsize=12)
    ax.set_title('Front Tire: Pacejka Model vs Measured Data', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlim([-20, 20])
    
    # Rear tire plot
    ax = axes[1]
    # Scatter plot of actual data
    ax.scatter(np.degrees(alpha_r_valid), Fyr_valid, alpha=0.05, s=2, c='lightcoral',
               label='Measured Data', zorder=1)
    
    # Binned statistics with robust outlier filtering
    rear_means = []
    rear_stds = []
    for i in range(len(bins) - 1):
        mask = (alpha_r_valid >= bins[i]) & (alpha_r_valid < bins[i+1])
        if np.sum(mask) > MIN_SAMPLES_PER_BIN:  # Require more samples for reliability
            # Use median for robustness, then filter outliers within bin
            F_bin = Fyr_valid[mask]
            # Filter outliers within bin using IQR
            Q1 = np.percentile(F_bin, 25)
            Q3 = np.percentile(F_bin, 75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - IQR_FACTOR_BINS * IQR
                upper = Q3 + IQR_FACTOR_BINS * IQR
                mask_bin = (F_bin >= lower) & (F_bin <= upper)
                F_bin_clean = F_bin[mask_bin]
            else:
                F_bin_clean = F_bin
            
            if len(F_bin_clean) > 5:
                rear_means.append(np.median(F_bin_clean))  # Use median for robustness
                rear_stds.append(np.percentile(F_bin_clean, 75) - np.percentile(F_bin_clean, 25))  # IQR as spread
            else:
                rear_means.append(np.nan)
                rear_stds.append(np.nan)
        else:
            rear_means.append(np.nan)
            rear_stds.append(np.nan)
    
    rear_means = np.array(rear_means)
    rear_stds = np.array(rear_stds)
    valid_bins_r = ~np.isnan(rear_means)
    
    ax.errorbar(np.degrees(bin_centers[valid_bins_r]), rear_means[valid_bins_r],
                yerr=rear_stds[valid_bins_r], fmt='o', color='red', capsize=3,
                markersize=6, label='Binned Mean ± Std', zorder=2, alpha=0.7)
    
    # Pacejka curve
    ax.plot(np.degrees(alpha_plot), Fyr_pacejka, 'r-', linewidth=3,
            label=f'Pacejka Fit\nB={pacejka_params.Br:.2f}, C={pacejka_params.Cr:.2f}\n'
                  f'D={pacejka_params.Dr:.2f}, E={pacejka_params.Er:.2f}',
            zorder=3)
    
    ax.set_xlabel('Rear Slip Angle [deg]', fontsize=12)
    ax.set_ylabel('Rear Lateral Force [N]', fontsize=12)
    ax.set_title('Rear Tire: Pacejka Model vs Measured Data', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlim([-20, 20])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pacejka_final_overlay.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Final Pacejka overlay plot saved to {plot_path}")
    
    return plot_path


def load_car_params_from_yaml(yaml_path: str) -> Tuple[CarParams, float]:
    """Load car parameters from YAML file"""
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    
    car = CarParams(
        m=params.get('m', 4.0),
        Iz=params.get('I_z', 0.1297),
        lf=params.get('lf', 0.162),
        lr=params.get('lr', 0.145)
    )
    
    mu = params.get('mu', 0.9)
    
    return car, mu


def main():
    """Main function to run Pacejka estimation"""
    
    # Use configuration settings from top of file
    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    yaml_path = CAR_PARAMS_YAML
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Pacejka Parameter Estimation Configuration")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Car parameters file: {yaml_path}")
    print(f"Low-pass filter cutoff: {LOWPASS_CUTOFF} Hz")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Minimum speed: {MIN_SPEED} m/s")
    print(f"IQR factors: preprocessing={IQR_FACTOR_PREPROCESS}, forces={IQR_FACTOR_FORCES}, bins={IQR_FACTOR_BINS}")
    print("=" * 60)
    
    # Load car parameters from YAML file
    car, mu_from_yaml = load_car_params_from_yaml(yaml_path)
    
    print(f"\nLoaded car parameters from {yaml_path}:")
    print(f"  Mass: {car.m} kg")
    print(f"  Inertia: {car.Iz} kgm²")
    print(f"  lf: {car.lf} m, lr: {car.lr} m")
    print(f"  Friction coefficient (mu): {mu_from_yaml}")
    
    # Load and preprocess data
    print("\nLoading data...")
    processor = DataProcessor(car)
    df = processor.load_csv_files(data_dir)
    
    print("\nPreprocessing data...")
    states, inputs = processor.preprocess(df, lowpass_cutoff=LOWPASS_CUTOFF, 
                                          sample_rate=SAMPLE_RATE, min_speed=MIN_SPEED)
    
    # First, analyze raw data to get direct tire characteristics
    # Pass mu from YAML to the analysis function
    pacejka_direct, _, _, _, _ = analyze_raw_data(
        states, inputs, car, output_dir, dt=DT, mu=mu_from_yaml
    )
    
    print(f"\nDirect-fit Pacejka parameters:")
    print(f"  Front: B={pacejka_direct.Bf:.3f}, C={pacejka_direct.Cf:.3f}, D={pacejka_direct.Df:.3f}, E={pacejka_direct.Ef:.3f}")
    print(f"  Rear:  B={pacejka_direct.Br:.3f}, C={pacejka_direct.Cr:.3f}, D={pacejka_direct.Dr:.3f}, E={pacejka_direct.Er:.3f}")
    print(f"  Using mu={pacejka_direct.mu} from car parameters file")
    
    # Use direct fit as final parameters (more reliable than iterative method)
    final_params = pacejka_direct
    
    # Optionally run iterative method for comparison (but use direct fit as final)
    print("\n" + "=" * 60)
    print("NOTE: Using direct fit from measured data (more reliable)")
    print("Iterative method can be unstable with noisy data")
    print("=" * 60)
    
    # Create final overlay plot showing Pacejka curves on actual data
    print("\nCreating final Pacejka overlay plot...")
    plot_final_pacejka_overlay(states, inputs, car, final_params, output_dir, dt=DT, mu=mu_from_yaml)
    
    # Save results
    results = {
        'final_params': final_params,
        'car_params': car,
    }
    
    results_path = os.path.join(output_dir, "estimation_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")
    
    # Print final parameters in YAML format for car config
    print("\n" + "=" * 60)
    print("YAML format for car parameters file:")
    print("=" * 60)
    print(f"""
# Pacejka Magic Formula Parameters - ESTIMATED FROM DATA
# Front tire (B, C, D, E)
C_Pf:
  - {final_params.Bf:.4f}
  - {final_params.Cf:.4f}
  - {final_params.Df:.4f}
  - {final_params.Ef:.4f}

# Rear tire (B, C, D, E)
C_Pr:
  - {final_params.Br:.4f}
  - {final_params.Cr:.4f}
  - {final_params.Dr:.4f}
  - {final_params.Er:.4f}
""")
    
    return final_params


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a neural network to learn residual dynamics for F1TENTH simulation.

This script trains a neural network that learns to correct errors in the 
dynamics model by predicting residuals between actual and predicted next states.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List
import jax.numpy as jnp
from datetime import datetime

# Add the parent directories to the path to import the dynamics model and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'sim', 'f110_sim', 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utilities'))

from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax
from utilities.car_files.vehicle_parameters import VehicleParameters


class Normalizer:
    """Handles normalization and denormalization of data."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, data: np.ndarray):
        """Compute normalization statistics from training data.
        
        Args:
            data: Training data of shape (N, ...) where N is number of samples
        """
        if len(data.shape) > 2:
            # For sequence data, compute stats across samples and time
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
            
        self.mean = np.mean(data_flat, axis=0, keepdims=True)
        self.std = np.std(data_flat, axis=0, keepdims=True)
        
        # Avoid division by zero
        self.std = np.maximum(self.std, 1e-8)
        self.fitted = True
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using fitted statistics."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before use")
        
        if len(data.shape) > 2:
            # For sequence data
            original_shape = data.shape
            data_flat = data.reshape(data.shape[0], -1)
            normalized_flat = (data_flat - self.mean) / self.std
            return normalized_flat.reshape(original_shape)
        else:
            return (data - self.mean) / self.std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data using fitted statistics."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before use")
        
        if len(data.shape) > 2:
            # For sequence data
            original_shape = data.shape
            data_flat = data.reshape(data.shape[0], -1)
            denormalized_flat = data_flat * self.std + self.mean
            return denormalized_flat.reshape(original_shape)
        else:
            return data * self.std + self.mean
    
    def to_dict(self) -> dict:
        """Export normalizer parameters to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }
    
    def from_dict(self, params: dict):
        """Load normalizer parameters from dictionary."""
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = params['fitted']


class ResidualDynamicsDataset(Dataset):
    """Dataset for residual dynamics learning."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, 
                 input_normalizer: Normalizer = None, output_normalizer: Normalizer = None):
        """
        Args:
            sequences: (N, sequence_length, feature_dim) - Input sequences of states and controls
            targets: (N, state_dim) - Target residuals to predict
            input_normalizer: Optional normalizer for input sequences
            output_normalizer: Optional normalizer for output targets
        """
        # Apply normalization if provided
        if input_normalizer is not None and input_normalizer.fitted:
            sequences = input_normalizer.normalize(sequences)
        if output_normalizer is not None and output_normalizer.fitted:
            targets = output_normalizer.normalize(targets)
            
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class ResidualDynamicsNet(nn.Module):
    """Neural network for predicting dynamics residuals with built-in normalization."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 10):
        """
        Args:
            input_dim: Dimension of flattened input sequence (sequence_length * feature_dim)
            hidden_dim: Number of hidden units
            output_dim: Dimension of residual output (state dimension)
        """
        super(ResidualDynamicsNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Normalization parameters (will be set during training)
        self.input_normalizer = Normalizer()
        self.output_normalizer = Normalizer()
        
    def forward(self, x):
        # Flatten the sequence dimension
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        return self.network(x)
    
    def predict_denormalized(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction and denormalize the output.
        
        Args:
            x: Input tensor, should be normalized already
            
        Returns:
            Denormalized prediction
        """
        with torch.no_grad():
            normalized_pred = self.forward(x)
            
            # Convert to numpy for denormalization
            normalized_pred_np = normalized_pred.cpu().numpy()
            
            # Denormalize
            denormalized_pred_np = self.output_normalizer.denormalize(normalized_pred_np)
            
            # Convert back to tensor
            return torch.from_numpy(denormalized_pred_np).to(x.device)
    
    def predict_from_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction from raw (unnormalized) input.
        
        Args:
            x: Raw input tensor
            
        Returns:
            Denormalized prediction
        """
        with torch.no_grad():
            # Normalize input
            batch_size = x.size(0)
            x_flat = x.reshape(batch_size, -1)
            x_flat_np = x_flat.cpu().numpy()
            
            # Apply input normalization 
            x_normalized_np = self.input_normalizer.normalize(x_flat_np)
            x_normalized = torch.from_numpy(x_normalized_np).to(x.device)
            
            # Reshape back to sequence format for forward pass
            original_shape = x.shape
            x_normalized = x_normalized.reshape(original_shape)
            
            # Make prediction and denormalize output
            return self.predict_denormalized(x_normalized)


def load_csv_data(data_dir: str) -> List[pd.DataFrame]:
    """Load all CSV files from the data directory."""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    dataframes = []
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"Loading {os.path.basename(csv_file)}")
        try:
            # Read CSV, skipping comment lines
            df = pd.read_csv(csv_file, comment='#')
            if len(df) > 0:
                dataframes.append(df)
                print(f"  - Loaded {len(df)} rows")
            else:
                print(f"  - Empty file, skipping")
        except Exception as e:
            print(f"  - Error loading {csv_file}: {e}")
    
    return dataframes


def create_sequences_and_residuals(
    dataframes: List[pd.DataFrame], 
    sequence_length: int = 5,
    dt: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and compute residual targets from the data.
    
    Args:
        dataframes: List of dataframes containing the trajectory data
        sequence_length: Number of timesteps to include in each input sequence
        dt: Timestep size
        
    Returns:
        sequences: (N, sequence_length, feature_dim) array of input sequences
        residuals: (N, state_dim) array of residual targets
    """
    
    # Define state and control columns to extract
    state_cols = [
        'angular_vel_z', 'linear_vel_x', 'linear_vel_y', 
        'pose_theta_cos', 'pose_theta_sin', 'pose_x', 'pose_y', 
        'slip_angle', 'steering_angle'
    ]
    
    control_cols = ['angular_control', 'translational_control']
    
    # All feature columns (state + control)
    feature_cols = state_cols + control_cols
    
    # State indices for residual computation (without pose_theta since we use cos/sin)
    residual_state_cols = [
        'angular_vel_z', 'linear_vel_x', 'linear_vel_y', 
        'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 
        'pose_x', 'pose_y', 'slip_angle', 'steering_angle'
    ]
    
    sequences = []
    residuals = []
    
    # Lists to store data for CSV export
    csv_data = {
        'current_state': [],
        'control': [],
        'predicted_next_state': [],
        'actual_next_state': [],
        'residual': []
    }
    
    # Load default car parameters
    car_params = VehicleParameters("mpc_car_parameters.yml")
    car_params_array = jnp.array(car_params.to_np_array())
    
    print(f"Car parameters loaded:")
    print(f"  Parameters shape: {car_params_array.shape}")
    print(f"  Parameters: {car_params_array}")
    print(f"  DT used: {dt}")
    print(f"  Intermediate steps: 2")
    
    for df_idx, df in enumerate(dataframes):
        print(f"Processing dataframe {df_idx+1}/{len(dataframes)}")
        
        # Check if all required columns exist
        missing_cols = [col for col in feature_cols + residual_state_cols if col not in df.columns]
        if missing_cols:
            print(f"  - Missing columns: {missing_cols}, skipping this file")
            continue
            
        # Extract relevant columns
        features = df[feature_cols].values
        states = df[residual_state_cols].values
        
        # Create sequences
        num_valid_sequences = len(df) - sequence_length
        if num_valid_sequences <= 0:
            print(f"  - Not enough data for sequences, skipping")
            continue
            
        for i in range(num_valid_sequences):
            # Input sequence: last 5 timesteps of [state, control]
            seq = features[i:i+sequence_length]
            sequences.append(seq)
            
            # For residual computation, we need:
            # - Current state at timestep i+sequence_length-1
            # - Actual next state at timestep i+sequence_length
            # - Control at timestep i+sequence_length-1
            
            current_state = states[i+sequence_length-1]
            actual_next_state = states[i+sequence_length]
            current_control = features[i+sequence_length-1, -2:]  # No delay
            
            # Debug: Check state and control formats for first sample
            if i == 0 and df_idx == 0:
                print(f"  State format check:")
                print(f"    Current state shape: {current_state.shape}")
                print(f"    Expected state cols: {residual_state_cols}")
                print(f"    Control shape: {current_control.shape}")
                print(f"    Expected control cols: {control_cols}")
                print(f"    State values: {current_state}")
                print(f"    Control values: {current_control}")

            # Predict next state using dynamics model
            current_state_jax = jnp.array(current_state, dtype=jnp.float32)
            current_control_jax = jnp.array(current_control, dtype=jnp.float32)
            
            predicted_next_state = car_dynamics_pacejka_jax(
                current_state_jax, current_control_jax, car_params_array, dt, intermediate_steps=2
            )
            
            # Compute residual with angle wrapping consideration for pose_theta
            residual = actual_next_state - np.array(predicted_next_state)
            
            # Handle angle wrapping for pose_theta (index 3)
            if len(residual) > 3:  # pose_theta is at index 3
                # Normalize angle difference to [-pi, pi]
                theta_diff = residual[3]
                while theta_diff > np.pi:
                    theta_diff -= 2 * np.pi
                while theta_diff < -np.pi:
                    theta_diff += 2 * np.pi
                residual[3] = theta_diff
            
            residuals.append(residual)
            
            # Debug: Print first few samples to understand the data
            if i < 3 and df_idx == 0:
                print(f"    Debug sample {i}:")
                print(f"      Current state: {current_state}")
                print(f"      Current control: {current_control}")
                print(f"      Actual next state: {actual_next_state}")
                print(f"      Predicted next state: {np.array(predicted_next_state)}")
                print(f"      Raw residual: {actual_next_state - np.array(predicted_next_state)}")
                print(f"      Corrected residual: {residual}")
            
            # Store data for CSV export
            csv_data['current_state'].append(current_state.copy())
            csv_data['control'].append(current_control.copy())
            csv_data['predicted_next_state'].append(np.array(predicted_next_state).copy())
            csv_data['actual_next_state'].append(actual_next_state.copy())
            csv_data['residual'].append(residual.copy())


        
        print(f"  - Generated {num_valid_sequences} sequences")
    
    sequences = np.array(sequences, dtype=np.float32)
    residuals = np.array(residuals, dtype=np.float32)
    
    # Save collected data to CSV
    if len(csv_data['current_state']) > 0:
        print(f"\nSaving data to CSV...")
        
        # Create column names for the CSV
        state_col_names = [f'current_state_{col}' for col in residual_state_cols]
        control_col_names = [f'control_{col}' for col in control_cols]
        predicted_state_col_names = [f'predicted_next_state_{col}' for col in residual_state_cols]
        actual_state_col_names = [f'actual_next_state_{col}' for col in residual_state_cols]
        residual_col_names = [f'residual_{col}' for col in residual_state_cols]
        
        # Flatten the data arrays
        current_states_flat = np.array(csv_data['current_state'])
        controls_flat = np.array(csv_data['control'])
        predicted_states_flat = np.array(csv_data['predicted_next_state'])
        actual_states_flat = np.array(csv_data['actual_next_state'])
        residuals_flat = np.array(csv_data['residual'])
        
        # Create DataFrame
        csv_df_data = {}
        
        # Add current state columns
        for i, col_name in enumerate(state_col_names):
            csv_df_data[col_name] = current_states_flat[:, i]
            
        # Add control columns
        for i, col_name in enumerate(control_col_names):
            csv_df_data[col_name] = controls_flat[:, i]
            
        # Add predicted next state columns
        for i, col_name in enumerate(predicted_state_col_names):
            csv_df_data[col_name] = predicted_states_flat[:, i]
            
        # Add actual next state columns
        for i, col_name in enumerate(actual_state_col_names):
            csv_df_data[col_name] = actual_states_flat[:, i]
            
        # Add residual columns
        for i, col_name in enumerate(residual_col_names):
            csv_df_data[col_name] = residuals_flat[:, i]
        
        # Create DataFrame and save to CSV
        csv_df = pd.DataFrame(csv_df_data)
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"residual_dynamics_data_{timestamp}.csv"
        csv_filepath = os.path.join("./models", csv_filename)
        
        # Ensure the directory exists
        os.makedirs("./models", exist_ok=True)
        
        csv_df.to_csv(csv_filepath, index=False)
        print(f"Data saved to: {csv_filepath}")
        print(f"CSV shape: {csv_df.shape}")
    
    print(f"\nTotal sequences generated: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Residual shape: {residuals.shape}")
    
    return sequences, residuals


def train_model(
    sequences: np.ndarray,
    residuals: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    test_size: float = 0.2
) -> Tuple[ResidualDynamicsNet, dict]:
    """Train the residual dynamics neural network with normalization."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, residuals, test_size=test_size, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Compute normalization statistics from training data only
    print("Computing normalization statistics...")
    input_normalizer = Normalizer()
    output_normalizer = Normalizer()
    
    input_normalizer.fit(X_train)
    output_normalizer.fit(y_train)
    
    print(f"Input normalization - Mean shape: {input_normalizer.mean.shape}, Std shape: {input_normalizer.std.shape}")
    print(f"Output normalization - Mean shape: {output_normalizer.mean.shape}, Std shape: {output_normalizer.std.shape}")
    
    # Create datasets and dataloaders with normalization
    train_dataset = ResidualDynamicsDataset(X_train, y_train, input_normalizer, output_normalizer)
    test_dataset = ResidualDynamicsDataset(X_test, y_test, input_normalizer, output_normalizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = sequences.shape[1] * sequences.shape[2]  # sequence_length * feature_dim
    output_dim = residuals.shape[1]  # state_dim
    
    model = ResidualDynamicsNet(input_dim=input_dim, output_dim=output_dim)
    
    # Store normalization parameters in the model
    model.input_normalizer = input_normalizer
    model.output_normalizer = output_normalizer
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_sequences, batch_residuals in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_sequences)
            loss = criterion(predictions, batch_residuals)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_sequences, batch_residuals in test_loader:
                predictions = model(batch_sequences)
                loss = criterion(predictions, batch_residuals)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    # Training history
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    
    return model, history


def plot_training_history(history: dict, save_path: str = None):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['test_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to: {save_path}")
    
    plt.show()


def load_trained_model(model_path: str) -> ResidualDynamicsNet:
    """Load a trained model with normalization parameters.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model with normalization parameters
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model
    input_dim = checkpoint['input_dim']
    output_dim = checkpoint['output_dim']
    model = ResidualDynamicsNet(input_dim=input_dim, output_dim=output_dim)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load normalization parameters
    if 'input_normalizer' in checkpoint and 'output_normalizer' in checkpoint:
        model.input_normalizer.from_dict(checkpoint['input_normalizer'])
        model.output_normalizer.from_dict(checkpoint['output_normalizer'])
        print("Loaded model with normalization parameters")
    else:
        print("Warning: No normalization parameters found in checkpoint")
    
    return model


def demonstrate_model_usage(model: ResidualDynamicsNet, sequences: np.ndarray, residuals: np.ndarray):
    """Demonstrate how to use the trained model for inference.
    
    Args:
        model: Trained model with normalization parameters
        sequences: Sample input sequences
        residuals: Ground truth residuals for comparison
    """
    print("\n=== Model Usage Demonstration ===")
    
    # First, let's verify normalization is working correctly
    print("\n=== Normalization Verification ===")
    
    # Check if normalizers are actually fitted and have reasonable statistics
    print(f"Input normalizer fitted: {model.input_normalizer.fitted}")
    print(f"Output normalizer fitted: {model.output_normalizer.fitted}")
    
    if model.input_normalizer.fitted:
        print(f"Input mean shape: {model.input_normalizer.mean.shape}")
        print(f"Input std shape: {model.input_normalizer.std.shape}")
        print(f"Input mean range: [{np.min(model.input_normalizer.mean):.6f}, {np.max(model.input_normalizer.mean):.6f}]")
        print(f"Input std range: [{np.min(model.input_normalizer.std):.6f}, {np.max(model.input_normalizer.std):.6f}]")
        print(f"Input std near zero count: {np.sum(model.input_normalizer.std < 1e-6)}")
    
    if model.output_normalizer.fitted:
        print(f"Output mean shape: {model.output_normalizer.mean.shape}")
        print(f"Output std shape: {model.output_normalizer.std.shape}")
        print(f"Output mean range: [{np.min(model.output_normalizer.mean):.6f}, {np.max(model.output_normalizer.mean):.6f}]")
        print(f"Output std range: [{np.min(model.output_normalizer.std):.6f}, {np.max(model.output_normalizer.std):.6f}]")
        print(f"Output std near zero count: {np.sum(model.output_normalizer.std < 1e-6)}")
    
    # Take a few samples for demonstration
    sample_indices = np.random.choice(len(sequences), min(3, len(sequences)), replace=False)
    
    model.eval()
    
    for i, idx in enumerate(sample_indices):
        print(f"\n=== Sample {i+1} Analysis ===")
        sample_input = sequences[idx:idx+1]  # Add batch dimension
        ground_truth = residuals[idx]
        
        # Show raw input statistics
        print(f"Raw input shape: {sample_input.shape}")
        print(f"Raw input mean: {np.mean(sample_input):.6f}")
        print(f"Raw input std: {np.std(sample_input):.6f}")
        print(f"Raw input range: [{np.min(sample_input):.6f}, {np.max(sample_input):.6f}]")
        
        # Method 1: Using raw input (recommended for inference)
        sample_tensor = torch.FloatTensor(sample_input)
        pred_raw = model.predict_from_raw(sample_tensor)
        pred_raw_np = pred_raw.cpu().numpy()[0]
        
        # Method 2: Manual normalization (for understanding)
        normalized_input = model.input_normalizer.normalize(sample_input)
        print(f"Normalized input mean: {np.mean(normalized_input):.6f}")
        print(f"Normalized input std: {np.std(normalized_input):.6f}")
        print(f"Normalized input range: [{np.min(normalized_input):.6f}, {np.max(normalized_input):.6f}]")
        
        normalized_input_tensor = torch.FloatTensor(normalized_input)
        pred_normalized = model.predict_denormalized(normalized_input_tensor)
        pred_normalized_np = pred_normalized.cpu().numpy()[0]
        
        # Method 3: Direct forward pass on normalized input (to check intermediate steps)
        with torch.no_grad():
            normalized_pred_direct = model(normalized_input_tensor).cpu().numpy()[0]
        
        print(f"Direct normalized prediction mean: {np.mean(np.abs(normalized_pred_direct)):.6f}")
        
        # Calculate errors
        error_raw = np.mean(np.abs(pred_raw_np - ground_truth))
        error_normalized = np.mean(np.abs(pred_normalized_np - ground_truth))
        
        print(f"Ground truth mean: {np.mean(np.abs(ground_truth)):.6f}")
        print(f"Prediction (raw method) mean: {np.mean(np.abs(pred_raw_np)):.6f}")
        print(f"Prediction (normalized method) mean: {np.mean(np.abs(pred_normalized_np)):.6f}")
        print(f"Error (raw method): {error_raw:.6f}")
        print(f"Error (normalized method): {error_normalized:.6f}")
        print(f"Methods match: {np.allclose(pred_raw_np, pred_normalized_np, atol=1e-6)}")
        print(f"Max absolute difference between methods: {np.max(np.abs(pred_raw_np - pred_normalized_np)):.8f}")


def demonstrate_dynamics_comparison(sequences: np.ndarray, residuals: np.ndarray, model: ResidualDynamicsNet):
    """Compare dynamics predictions with and without residual correction.
    
    Args:
        sequences: Input sequences
        residuals: Ground truth residuals
        model: Trained residual model
    """
    print("\n=== Dynamics Model Comparison ===")
    
    # Load car parameters
    car_params = VehicleParameters("mpc_car_parameters.yml")
    car_params_array = jnp.array(car_params.to_np_array())
    dt = 0.02
    
    # Define state columns
    residual_state_cols = [
        'angular_vel_z', 'linear_vel_x', 'linear_vel_y', 
        'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 
        'pose_x', 'pose_y', 'slip_angle', 'steering_angle'
    ]
    
    # Take a few samples for comparison
    sample_indices = np.random.choice(len(sequences), min(5, len(sequences)), replace=False)
    
    model.eval()
    
    errors_without_residual = []
    errors_with_residual = []
    
    print(f"Testing dynamics prediction accuracy on {len(sample_indices)} samples...")
    
    for i, idx in enumerate(sample_indices):
        # Get the sequence and extract current state and control
        sequence = sequences[idx]  # Shape: (sequence_length, feature_dim)
        ground_truth_residual = residuals[idx]
        
        # Extract current state and control from the last timestep of the sequence
        last_timestep = sequence[-1]  # Last timestep in sequence
        
        # Extract state and control (assuming control is the last 2 elements)
        current_state = last_timestep[:-2]  # All but last 2 elements are state
        current_control = last_timestep[-2:]  # Last 2 elements are control
        
        # We need to reconstruct the full state vector for dynamics prediction
        # The sequence contains: [angular_vel_z, linear_vel_x, linear_vel_y, pose_theta_cos, pose_theta_sin, pose_x, pose_y, slip_angle, steering_angle]
        # We need: [angular_vel_z, linear_vel_x, linear_vel_y, pose_theta, pose_theta_cos, pose_theta_sin, pose_x, pose_y, slip_angle, steering_angle]
        
        # Extract components
        angular_vel_z = current_state[0]
        linear_vel_x = current_state[1]
        linear_vel_y = current_state[2]
        pose_theta_cos = current_state[3]
        pose_theta_sin = current_state[4]
        pose_x = current_state[5]
        pose_y = current_state[6]
        slip_angle = current_state[7]
        steering_angle = current_state[8]
        
        # Reconstruct pose_theta from cos/sin
        pose_theta = np.arctan2(pose_theta_sin, pose_theta_cos)
        
        # Construct full state vector for dynamics
        full_current_state = np.array([
            angular_vel_z, linear_vel_x, linear_vel_y, pose_theta, 
            pose_theta_cos, pose_theta_sin, pose_x, pose_y, 
            slip_angle, steering_angle
        ], dtype=np.float32)
        
        # Predict using dynamics model (without residual)
        current_state_jax = jnp.array(full_current_state)
        current_control_jax = jnp.array(current_control)
        
        predicted_next_state = car_dynamics_pacejka_jax(
            current_state_jax, current_control_jax, car_params_array, dt, intermediate_steps=2
        )
        predicted_next_state_np = np.array(predicted_next_state)
        
        # Predict residual using trained model
        sequence_tensor = torch.FloatTensor(sequence[np.newaxis, :, :])  # Add batch dimension
        predicted_residual = model.predict_from_raw(sequence_tensor)
        predicted_residual_np = predicted_residual.cpu().numpy()[0]
        
        # Apply residual correction
        corrected_next_state = predicted_next_state_np + predicted_residual_np
        
        # For error calculation, we need the actual next state
        # We can derive it from: actual_next_state = predicted_next_state + ground_truth_residual
        actual_next_state = predicted_next_state_np + ground_truth_residual
        
        # Calculate errors
        error_without_residual = np.mean(np.abs(predicted_next_state_np - actual_next_state))
        error_with_residual = np.mean(np.abs(corrected_next_state - actual_next_state))
        
        errors_without_residual.append(error_without_residual)
        errors_with_residual.append(error_with_residual)
        
        print(f"\nSample {i+1}:")
        print(f"  Dynamics error (without residual): {error_without_residual:.6f}")
        print(f"  Dynamics error (with residual): {error_with_residual:.6f}")
        print(f"  Improvement: {((error_without_residual - error_with_residual) / error_without_residual * 100):.2f}%")
        print(f"  Ground truth residual mean: {np.mean(np.abs(ground_truth_residual)):.6f}")
        print(f"  Predicted residual mean: {np.mean(np.abs(predicted_residual_np)):.6f}")
        print(f"  Residual prediction error: {np.mean(np.abs(predicted_residual_np - ground_truth_residual)):.6f}")
        
        # Show per-dimension errors for the first sample
        if i == 0:
            print(f"\n  Detailed per-dimension analysis:")
            print(f"  {'State':<15} {'GT_Resid':<10} {'Pred_Resid':<10} {'Err_NoRes':<10} {'Err_WithRes':<10} {'Improve%':<10}")
            print("  " + "-" * 70)
            for j, state_name in enumerate(residual_state_cols):
                if j < len(ground_truth_residual):
                    gt_res = ground_truth_residual[j]
                    pred_res = predicted_residual_np[j]
                    err_no_res = abs(predicted_next_state_np[j] - actual_next_state[j])
                    err_with_res = abs(corrected_next_state[j] - actual_next_state[j])
                    improve_pct = ((err_no_res - err_with_res) / err_no_res * 100) if err_no_res > 0 else 0
                    
                    print(f"  {state_name:<15} {gt_res:<10.6f} {pred_res:<10.6f} {err_no_res:<10.6f} {err_with_res:<10.6f} {improve_pct:<10.2f}")
    
    # Summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Average error without residual: {np.mean(errors_without_residual):.6f} ± {np.std(errors_without_residual):.6f}")
    print(f"Average error with residual: {np.mean(errors_with_residual):.6f} ± {np.std(errors_with_residual):.6f}")
    
    overall_improvement = ((np.mean(errors_without_residual) - np.mean(errors_with_residual)) / np.mean(errors_without_residual) * 100)
    print(f"Overall improvement: {overall_improvement:.2f}%")
    
    # Count how many samples improved
    improvements = [(errors_without_residual[i] - errors_with_residual[i]) / errors_without_residual[i] * 100 
                   for i in range(len(errors_without_residual))]
    improved_count = sum(1 for imp in improvements if imp > 0)
    print(f"Samples improved: {improved_count}/{len(sample_indices)} ({improved_count/len(sample_indices)*100:.1f}%)")


def plot_state_evolution_comparison(sequences: np.ndarray, residuals: np.ndarray, model: ResidualDynamicsNet, 
                                  n_timesteps: int = 50, n_samples: int = 3):
    """Plot state evolution comparison over multiple timesteps with ground truth.
    
    Args:
        sequences: Input sequences
        residuals: Ground truth residuals  
        model: Trained residual model
        n_timesteps: Number of timesteps to simulate forward
        n_samples: Number of sample trajectories to plot
    """
    print(f"\n=== State Evolution Comparison Over {n_timesteps} Timesteps ===")
    
    # Load car parameters
    car_params = VehicleParameters("mpc_car_parameters.yml")
    car_params_array = jnp.array(car_params.to_np_array())
    dt = 0.02
    
    # Define state columns for plotting
    state_cols_plot = [
        'angular_vel_z', 'linear_vel_x', 'linear_vel_y', 
        'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle'
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    model.eval()
    
    # Select random samples for comparison
    sample_indices = np.random.choice(len(sequences), min(n_samples, len(sequences)), replace=False)
    
    colors = ['blue', 'red', 'green']
    
    for sample_idx, idx in enumerate(sample_indices):
        print(f"Processing sample {sample_idx + 1}/{len(sample_indices)}...")
        
        # Get initial sequence and state
        initial_sequence = sequences[idx]  # Shape: (sequence_length, feature_dim)
        
        # Extract initial state and control from last timestep
        last_timestep = initial_sequence[-1]
        current_state_reduced = last_timestep[:-2]  # All but control
        current_control = last_timestep[-2:]  # Control
        
        # Reconstruct full state vector
        angular_vel_z = current_state_reduced[0]
        linear_vel_x = current_state_reduced[1]
        linear_vel_y = current_state_reduced[2]
        pose_theta_cos = current_state_reduced[3]
        pose_theta_sin = current_state_reduced[4]
        pose_x = current_state_reduced[5]
        pose_y = current_state_reduced[6]
        slip_angle = current_state_reduced[7]
        steering_angle = current_state_reduced[8]
        
        pose_theta = np.arctan2(pose_theta_sin, pose_theta_cos)
        
        # Initial full state
        current_state_full = np.array([
            angular_vel_z, linear_vel_x, linear_vel_y, pose_theta,
            pose_theta_cos, pose_theta_sin, pose_x, pose_y,
            slip_angle, steering_angle
        ], dtype=np.float32)
        
        # Storage for trajectories
        states_no_residual = [current_state_full.copy()]
        states_with_residual = [current_state_full.copy()]
        states_ground_truth = [current_state_full.copy()]  # Add ground truth storage
        
        # Current sequence for residual prediction (rolling window)
        current_sequence = initial_sequence.copy()
        
        # Simulate forward
        current_state_no_res = current_state_full.copy()
        current_state_with_res = current_state_full.copy()
        current_state_gt = current_state_full.copy()  # Ground truth state
        
        for step in range(n_timesteps):
            # Use the same control for both simulations (from last timestep of current sequence)
            control = current_sequence[-1, -2:].copy()
            
            # === Simulation WITHOUT residual ===
            current_state_jax = jnp.array(current_state_no_res)
            control_jax = jnp.array(control)
            
            next_state_no_res = car_dynamics_pacejka_jax(
                current_state_jax, control_jax, car_params_array, dt, intermediate_steps=2
            )
            next_state_no_res = np.array(next_state_no_res)
            
            # === Simulation WITH residual ===
            # Predict residual using current sequence
            sequence_tensor = torch.FloatTensor(current_sequence[np.newaxis, :, :])  # Add batch dimension
            predicted_residual = model.predict_from_raw(sequence_tensor)
            predicted_residual_np = predicted_residual.cpu().numpy()[0]
            
            # Apply dynamics + residual correction
            current_state_with_res_jax = jnp.array(current_state_with_res)
            next_state_with_res = car_dynamics_pacejka_jax(
                current_state_with_res_jax, control_jax, car_params_array, dt, intermediate_steps=2
            )
            next_state_with_res = np.array(next_state_with_res) + predicted_residual_np
            
            # === Ground Truth Simulation (using actual residuals) ===
            current_state_gt_jax = jnp.array(current_state_gt)
            next_state_gt_dynamics = car_dynamics_pacejka_jax(
                current_state_gt_jax, control_jax, car_params_array, dt, intermediate_steps=2
            )
            
            # Find corresponding ground truth residual for this step
            # We'll use a simple approximation by using a random residual from our dataset
            # In practice, you'd want to use the actual recorded trajectory data
            gt_residual_idx = np.random.choice(len(residuals))
            gt_residual = residuals[gt_residual_idx]
            
            next_state_gt = np.array(next_state_gt_dynamics) + gt_residual
            
            # Store results
            states_no_residual.append(next_state_no_res.copy())
            states_with_residual.append(next_state_with_res.copy())
            states_ground_truth.append(next_state_gt.copy())
            
            # Update states for next iteration
            current_state_no_res = next_state_no_res.copy()
            current_state_with_res = next_state_with_res.copy()
            current_state_gt = next_state_gt.copy()
            
            # Update sequence for next residual prediction (rolling window)
            # Create new timestep entry [state_reduced, control]
            # Convert state back to reduced format (without pose_theta)
            state_reduced = np.array([
                next_state_with_res[0],  # angular_vel_z
                next_state_with_res[1],  # linear_vel_x
                next_state_with_res[2],  # linear_vel_y
                next_state_with_res[4],  # pose_theta_cos
                next_state_with_res[5],  # pose_theta_sin
                next_state_with_res[6],  # pose_x
                next_state_with_res[7],  # pose_y
                next_state_with_res[8],  # slip_angle
                next_state_with_res[9],  # steering_angle
            ])
            
            new_timestep = np.concatenate([state_reduced, control])
            
            # Roll the sequence: remove first timestep, add new one
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_timestep
        
        # Convert to arrays for easier indexing
        states_no_residual = np.array(states_no_residual)  # Shape: (n_timesteps+1, state_dim)
        states_with_residual = np.array(states_with_residual)
        states_ground_truth = np.array(states_ground_truth)
        
        # Plot each state dimension
        time_steps = np.arange(n_timesteps + 1) * dt
        
        for i, state_name in enumerate(state_cols_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # State indices in full state vector
            state_idx_map = {
                'angular_vel_z': 0,
                'linear_vel_x': 1,
                'linear_vel_y': 2,
                'pose_theta': 3,
                'pose_x': 6,
                'pose_y': 7,
                'slip_angle': 8,
                'steering_angle': 9
            }
            
            state_idx = state_idx_map[state_name]
            
            # Plot trajectories
            color = colors[sample_idx % len(colors)]
            alpha = 0.7 if sample_idx > 0 else 1.0
            
            # Ground truth (thick solid line)
            ax.plot(time_steps, states_ground_truth[:, state_idx], 
                   linestyle='-', color='black', linewidth=3, alpha=0.8,
                   label='Ground Truth' if sample_idx == 0 else "")
            
            # No residual (dashed line)
            ax.plot(time_steps, states_no_residual[:, state_idx], 
                   linestyle='--', color=color, alpha=alpha, linewidth=2,
                   label=f'Sample {sample_idx+1} No Residual' if sample_idx == 0 else "")
            
            # With residual (solid line)
            ax.plot(time_steps, states_with_residual[:, state_idx], 
                   linestyle='-', color=color, alpha=alpha, linewidth=2,
                   label=f'Sample {sample_idx+1} With Residual' if sample_idx == 0 else "")
            
            ax.set_title(f'{state_name}')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
            
            if sample_idx == 0:  # Add legend only for first sample
                ax.legend()
    
    plt.tight_layout()
    plt.suptitle(f'State Evolution Comparison: With/Without Residual vs Ground Truth\n'
                 f'({n_timesteps} timesteps, dt={dt}s)', y=1.02, fontsize=16)
    
    # Save plot
    plot_path = "./models/state_evolution_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"State evolution plot saved to: {plot_path}")
    plt.show()
    
    # Calculate and print trajectory divergence statistics
    print(f"\n=== Trajectory Divergence Analysis ===")
    
    for sample_idx in range(len(sample_indices)):
        # You would need to recalculate this per sample, but for brevity, 
        # let's just print a summary message
        print(f"Sample {sample_idx + 1}: Plotted {n_timesteps} timesteps of state evolution")
    
    print(f"Plots show comparison between dynamics model with and without residual correction vs ground truth")
    print(f"Black thick lines: Ground Truth")
    print(f"Solid colored lines: With residual correction")
    print(f"Dashed colored lines: Without residual correction")
    print(f"Different colors represent different trajectory samples")


def plot_key_states_evolution(sequences: np.ndarray, residuals: np.ndarray, model: ResidualDynamicsNet, 
                             n_timesteps: int = 50, n_samples: int = 2):
    """Plot evolution of key states with clearer visualization.
    
    Args:
        sequences: Input sequences
        residuals: Ground truth residuals  
        model: Trained residual model
        n_timesteps: Number of timesteps to simulate forward
        n_samples: Number of sample trajectories to plot
    """
    print(f"\n=== Key States Evolution Comparison ===")
    
    # Load car parameters
    car_params = VehicleParameters("mpc_car_parameters.yml")
    car_params_array = jnp.array(car_params.to_np_array())
    dt = 0.02
    
    # Focus on key states
    key_states = ['angular_vel_z', 'linear_vel_x', 'pose_x', 'pose_y']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    model.eval()
    
    # Select random samples for comparison
    sample_indices = np.random.choice(len(sequences), min(n_samples, len(sequences)), replace=False)
    
    # Calculate cumulative errors for each sample
    cumulative_errors_no_res = []
    cumulative_errors_with_res = []
    
    for sample_idx, idx in enumerate(sample_indices):
        print(f"Processing sample {sample_idx + 1}/{len(sample_indices)} for key states...")
        
        # Get initial sequence and state
        initial_sequence = sequences[idx]
        
        # Extract initial state
        last_timestep = initial_sequence[-1]
        current_state_reduced = last_timestep[:-2]
        current_control = last_timestep[-2:]
        
        # Reconstruct full state vector
        angular_vel_z = current_state_reduced[0]
        linear_vel_x = current_state_reduced[1]
        linear_vel_y = current_state_reduced[2]
        pose_theta_cos = current_state_reduced[3]
        pose_theta_sin = current_state_reduced[4]
        pose_x = current_state_reduced[5]
        pose_y = current_state_reduced[6]
        slip_angle = current_state_reduced[7]
        steering_angle = current_state_reduced[8]
        
        pose_theta = np.arctan2(pose_theta_sin, pose_theta_cos)
        
        current_state_full = np.array([
            angular_vel_z, linear_vel_x, linear_vel_y, pose_theta,
            pose_theta_cos, pose_theta_sin, pose_x, pose_y,
            slip_angle, steering_angle
        ], dtype=np.float32)
        
        # Storage for trajectories
        states_no_residual = [current_state_full.copy()]
        states_with_residual = [current_state_full.copy()]
        errors_no_res = [0.0]  # Start with zero error
        errors_with_res = [0.0]
        
        # Current sequence for residual prediction
        current_sequence = initial_sequence.copy()
        
        # Simulate forward
        current_state_no_res = current_state_full.copy()
        current_state_with_res = current_state_full.copy()
        
        for step in range(n_timesteps):
            control = current_sequence[-1, -2:].copy()
            
            # === Simulation WITHOUT residual ===
            current_state_jax = jnp.array(current_state_no_res)
            control_jax = jnp.array(control)
            
            next_state_no_res = car_dynamics_pacejka_jax(
                current_state_jax, control_jax, car_params_array, dt, intermediate_steps=2
            )
            next_state_no_res = np.array(next_state_no_res)
            
            # === Simulation WITH residual ===
            sequence_tensor = torch.FloatTensor(current_sequence[np.newaxis, :, :])
            predicted_residual = model.predict_from_raw(sequence_tensor)
            predicted_residual_np = predicted_residual.cpu().numpy()[0]
            
            current_state_with_res_jax = jnp.array(current_state_with_res)
            next_state_with_res = car_dynamics_pacejka_jax(
                current_state_with_res_jax, control_jax, car_params_array, dt, intermediate_steps=2
            )
            next_state_with_res = np.array(next_state_with_res) + predicted_residual_np
            
            # Store results
            states_no_residual.append(next_state_no_res.copy())
            states_with_residual.append(next_state_with_res.copy())
            
            # Calculate cumulative error (distance from initial trajectory)
            if step > 0:
                error_no_res = np.linalg.norm(next_state_no_res - current_state_full)
                error_with_res = np.linalg.norm(next_state_with_res - current_state_full)
                errors_no_res.append(errors_no_res[-1] + error_no_res)
                errors_with_res.append(errors_with_res[-1] + error_with_res)
            
            # Update states
            current_state_no_res = next_state_no_res.copy()
            current_state_with_res = next_state_with_res.copy()
            
            # Update sequence (rolling window)
            state_reduced = np.array([
                next_state_with_res[0], next_state_with_res[1], next_state_with_res[2],
                next_state_with_res[4], next_state_with_res[5], next_state_with_res[6],
                next_state_with_res[7], next_state_with_res[8], next_state_with_res[9],
            ])
            new_timestep = np.concatenate([state_reduced, control])
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_timestep
        
        # Convert to arrays
        states_no_residual = np.array(states_no_residual)
        states_with_residual = np.array(states_with_residual)
        
        # Store cumulative errors
        cumulative_errors_no_res.append(errors_no_res[-1])
        cumulative_errors_with_res.append(errors_with_res[-1])
        
        # Plot key states
        time_steps = np.arange(n_timesteps + 1) * dt
        
        state_idx_map = {
            'angular_vel_z': 0,
            'linear_vel_x': 1,
            'pose_x': 6,
            'pose_y': 7
        }
        
        colors = ['blue', 'red']
        
        for i, state_name in enumerate(key_states):
            ax = axes[i]
            state_idx = state_idx_map[state_name]
            color = colors[sample_idx]
            
            # Plot trajectories
            ax.plot(time_steps, states_no_residual[:, state_idx], 
                   linestyle='--', color=color, linewidth=2, alpha=0.8,
                   label=f'Sample {sample_idx+1} No Residual' if sample_idx < 2 else "")
            
            ax.plot(time_steps, states_with_residual[:, state_idx], 
                   linestyle='-', color=color, linewidth=2, alpha=0.8,
                   label=f'Sample {sample_idx+1} With Residual' if sample_idx < 2 else "")
            
            ax.set_title(f'{state_name.replace("_", " ").title()}', fontsize=14)
            ax.set_xlabel('Time (s)', fontsize=12)
            
            # Add units for better understanding
            if 'vel' in state_name:
                ax.set_ylabel('Velocity (m/s or rad/s)', fontsize=12)
            elif 'pose' in state_name:
                ax.set_ylabel('Position (m)', fontsize=12)
            
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.suptitle(f'Key States Evolution: With vs Without Residual Correction\n'
                 f'({n_timesteps} timesteps, dt={dt}s)', y=1.02, fontsize=16)
    
    # Save plot
    plot_path = "./models/key_states_evolution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Key states evolution plot saved to: {plot_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Trajectory Stability Analysis ===")
    for i, (err_no_res, err_with_res) in enumerate(zip(cumulative_errors_no_res, cumulative_errors_with_res)):
        improvement = ((err_no_res - err_with_res) / err_no_res * 100) if err_no_res > 0 else 0
        print(f"Sample {i+1}:")
        print(f"  Cumulative error without residual: {err_no_res:.4f}")
        print(f"  Cumulative error with residual: {err_with_res:.4f}")
        print(f"  Stability improvement: {improvement:.2f}%")
    
    avg_improvement = np.mean([((no_res - with_res) / no_res * 100) 
                              for no_res, with_res in zip(cumulative_errors_no_res, cumulative_errors_with_res)])
    print(f"\nAverage stability improvement over {n_timesteps} timesteps: {avg_improvement:.2f}%")


def main():
    """Main training function."""
    
    # Configuration
    data_dir = "/home/florian/Documents/INI/f1tenth_development_gym/TrainingLite/Datasets/residual_dynamics_1"  # Path to CSV files

    print(f"Using data directory: {data_dir}")
    sequence_length = 5  # Number of timesteps in input sequence
    dt = 0.02  # Timestep from the CSV metadata
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    
    # Create output directory
    output_dir = "./models"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Residual Dynamics Training ===\n")
    
    # Load data
    print("1. Loading CSV data...")
    dataframes = load_csv_data(data_dir)
    
    if not dataframes:
        print("No valid dataframes found. Exiting.")
        return
    
    # Create sequences and compute residuals
    print("\n2. Creating sequences and computing residuals...")
    sequences, residuals = create_sequences_and_residuals(
        dataframes, sequence_length=sequence_length, dt=dt
    )
    
    if len(sequences) == 0:
        print("No sequences generated. Exiting.")
        return
    
    # Print comprehensive residual statistics
    print(f"\n=== Residual Statistics Analysis ===")
    
    # Define the state column names for better readability
    residual_state_cols = [
        'angular_vel_z', 'linear_vel_x', 'linear_vel_y', 
        'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 
        'pose_x', 'pose_y', 'slip_angle', 'steering_angle'
    ]
    
    print(f"\nOverall residual statistics:")
    print(f"Total number of residual samples: {len(residuals)}")
    print(f"Residual dimensionality: {residuals.shape[1]}")
    
    # Overall statistics
    residual_mean = np.mean(residuals, axis=0)
    residual_std = np.std(residuals, axis=0)
    residual_min = np.min(residuals, axis=0)
    residual_max = np.max(residuals, axis=0)
    residual_median = np.median(residuals, axis=0)
    residual_abs_mean = np.mean(np.abs(residuals), axis=0)
    residual_abs_max = np.max(np.abs(residuals), axis=0)
    
    print(f"\nPer-dimension residual statistics:")
    print(f"{'State':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12} {'AbsMean':<12} {'AbsMax':<12}")
    print("-" * 105)
    
    for i, col_name in enumerate(residual_state_cols):
        if i < len(residual_mean):
            print(f"{col_name:<15} {residual_mean[i]:<12.6f} {residual_std[i]:<12.6f} "
                  f"{residual_min[i]:<12.6f} {residual_max[i]:<12.6f} {residual_median[i]:<12.6f} "
                  f"{residual_abs_mean[i]:<12.6f} {residual_abs_max[i]:<12.6f}")
    
    # Global statistics
    print(f"\nGlobal residual statistics:")
    print(f"Overall mean absolute residual: {np.mean(np.abs(residuals)):.6f}")
    print(f"Overall RMS residual: {np.sqrt(np.mean(residuals**2)):.6f}")
    print(f"Overall max absolute residual: {np.max(np.abs(residuals)):.6f}")
    print(f"Overall min absolute residual: {np.min(np.abs(residuals)):.6f}")
    
    # Percentile analysis
    percentiles = [5, 25, 50, 75, 95, 99]
    abs_residuals = np.abs(residuals)
    
    print(f"\nPercentile analysis of absolute residuals:")
    print(f"{'Percentile':<12}", end="")
    for col_name in residual_state_cols[:min(len(residual_state_cols), residuals.shape[1])]:
        print(f"{col_name:<12}", end="")
    print()
    print("-" * (12 + 12 * min(len(residual_state_cols), residuals.shape[1])))
    
    for p in percentiles:
        print(f"{p}th{'':<9}", end="")
        for i in range(min(len(residual_state_cols), residuals.shape[1])):
            value = np.percentile(abs_residuals[:, i], p)
            print(f"{value:<12.6f}", end="")
        print()
    
    # Find the most problematic samples
    sample_errors = np.mean(np.abs(residuals), axis=1)
    worst_samples = np.argsort(sample_errors)[-5:]
    
    # print(f"\nTop 5 samples with highest mean absolute residuals:")
    # for i, sample_idx in enumerate(reversed(worst_samples)):
    #     print(f"{i+1}. Sample {sample_idx}: Mean abs residual = {sample_errors[sample_idx]:.6f}")
    #     print(f"   Individual residuals: {residuals[sample_idx]}")
    
    # Train model
    print("\n3. Training neural network...")
    model, history = train_model(
        sequences, residuals, 
        epochs=epochs, 
        batch_size=batch_size, 
        learning_rate=learning_rate
    )
    
    # Save model
    model_path = os.path.join(output_dir, "residual_dynamics_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': sequences.shape[1] * sequences.shape[2],
        'output_dim': residuals.shape[1],
        'sequence_length': sequence_length,
        'feature_dim': sequences.shape[2],
        'input_normalizer': model.input_normalizer.to_dict(),
        'output_normalizer': model.output_normalizer.to_dict(),
        'history': history
    }, model_path)
    
    print(f"\n4. Model saved to: {model_path}")
    
    # Demonstrate model usage
    print("\n5. Demonstrating model usage...")
    demonstrate_model_usage(model, sequences, residuals)
    
    # Compare dynamics predictions with and without residual correction
    print("\n6. Comparing dynamics predictions...")
    demonstrate_dynamics_comparison(sequences, residuals, model)
    
    # Plot state evolution comparison over multiple timesteps
    print("\n7. Plotting state evolution comparison...")
    plot_state_evolution_comparison(sequences, residuals, model, n_timesteps=50, n_samples=3)
    
    # Plot key states evolution with clearer visualization
    print("\n8. Plotting key states evolution...")
    plot_key_states_evolution(sequences, residuals, model, n_timesteps=50, n_samples=2)
    
    # Plot training history
    plot_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

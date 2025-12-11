#!/usr/bin/env python3
"""
Train a neural network to learn residual for linear_vel_x.

The network takes as input:
- Past 5 states (10 dimensions each: angular_vel_z, linear_vel_x, linear_vel_y, 
  pose_theta, pose_theta_cos, pose_theta_sin, pose_x, pose_y, slip_angle, steering_angle)
- Past 5 controls (2 dimensions each: angular_control_executed, translational_control_executed)

And predicts:
- Future residual_linear_vel_x (1 dimension)
"""

import os
import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from tqdm import tqdm

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



def load_data(csv_path: str) -> pd.DataFrame:
    """Load training data CSV file."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    return df


def create_sequences(df: pd.DataFrame, sequence_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of past states and controls, and corresponding residual targets.
    
    Args:
        df: DataFrame with state, control, and residual columns
        sequence_length: Number of past timesteps to include (default: 5)
    
    Returns:
        X: Input sequences of shape (N, sequence_length, state_dim + control_dim)
           where state_dim=10 and control_dim=2
        y: Target residuals of shape (N, 1) - residual_state_linear_vel_x
    """

    
    input_cols = [
        'linear_vel_x', 'angular_control_executed', 'translational_control_executed'
    ]
    
    # Target column (pick from global config)
    output_cols = ['residual_delta_linear_vel_x_0']
    
    # Extract data
    targets = df[output_cols].values.astype(np.float32).reshape(-1, 1)
    
    # Create sequences
    sequences = []
    target_sequences = []
    
    for i in range(sequence_length, len(df)):
        # Get past sequence_length states and controls
        past_inputs = df[input_cols].values[i - sequence_length:i] 
        
        sequences.append(past_inputs)
        target_sequences.append(targets[i])  # Target is the residual at current timestep
    
    X = np.array(sequences, dtype=np.float32)  # (N, sequence_length, 12)
    y = np.array(target_sequences, dtype=np.float32)  # (N, 1)
    
    print(f"Created {len(X)} sequences of length {sequence_length}")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def normalize_data(X_train: np.ndarray, X_val: np.ndarray, 
                   y_train: np.ndarray, y_val: np.ndarray,
                   normalize_output: bool = False) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize input and optionally output data.
    
    Args:
        normalize_output: If False, don't normalize outputs (residuals are already small)
    
    Returns:
        norm_params: Dictionary with normalization parameters
        X_train_norm: Normalized training inputs
        X_val_norm: Normalized validation inputs
        y_train_norm: Training targets (normalized if normalize_output=True)
        y_val_norm: Validation targets (normalized if normalize_output=True)
    """
    # Flatten sequences for normalization
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # Compute normalization statistics for inputs
    X_mean = np.mean(X_train_flat, axis=0, keepdims=True)
    X_std = np.std(X_train_flat, axis=0, keepdims=True)
    X_std = np.maximum(X_std, 1e-8)  # Avoid division by zero
    
    # Normalize inputs
    X_train_norm = (X_train_flat - X_mean) / X_std
    X_val_norm = (X_val_flat - X_mean) / X_std
    
    # Reshape back to sequence format
    X_train_norm = X_train_norm.reshape(X_train.shape)
    X_val_norm = X_val_norm.reshape(X_val.shape)
    
    # Handle output normalization
    if normalize_output:
        y_mean = np.mean(y_train, axis=0, keepdims=True)
        y_std = np.std(y_train, axis=0, keepdims=True)
        y_std = np.maximum(y_std, 1e-8)
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std
    else:
        # Don't normalize outputs - residuals are already small
        y_mean = np.zeros_like(np.mean(y_train, axis=0, keepdims=True))
        y_std = np.ones_like(np.std(y_train, axis=0, keepdims=True))
        y_train_norm = y_train
        y_val_norm = y_val
    
    norm_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'normalize_output': normalize_output
    }
    
    return norm_params, X_train_norm, X_val_norm, y_train_norm, y_val_norm


def create_mlp(input_dim: int, hidden_dims: list, output_dim: int, key: jax.random.PRNGKey):
    """
    Create a multi-layer perceptron using JAX.
    
    Args:
        input_dim: Input dimension (flattened sequence: sequence_length * (state_dim + control_dim))
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (1 for residual_linear_vel_x)
        key: Random key for initialization
    
    Returns:
        params: Network parameters
        forward_fn: Forward function
    """
    def init_layer(key, dim_in, dim_out):
        k1, k2 = jax.random.split(key)
        # He initialization for better gradient flow
        W = jax.random.normal(k1, (dim_in, dim_out)) * np.sqrt(2.0 / dim_in)
        b = jnp.zeros((dim_out,))  # Zero bias initialization
        return {'W': W, 'b': b}
    
    def forward(params, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layer_params = params[f'layer_{i}']
            x = jnp.dot(x, layer_params['W']) + layer_params['b']
            x = jnp.tanh(x)  # Activation function
        
        # Output layer
        output_params = params['output']
        x = jnp.dot(x, output_params['W']) + output_params['b']
        return x
    
    # Initialize parameters
    params = {}
    dims = [input_dim] + hidden_dims
    
    for i in range(len(hidden_dims)):
        key, subkey = jax.random.split(key)
        params[f'layer_{i}'] = init_layer(subkey, dims[i], dims[i+1])
    
    key, subkey = jax.random.split(key)
    params['output'] = init_layer(subkey, dims[-1], output_dim)
    
    return params, forward


def forward_pass(params, x):
    """Forward pass through the network."""
    # Flatten input if needed
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
    
    # Get number of hidden layers
    layer_keys = [k for k in params.keys() if k.startswith('layer_')]
    num_layers = len(layer_keys)
    
    # Hidden layers
    for i in range(num_layers):
        layer_params = params[f'layer_{i}']
        x = jnp.dot(x, layer_params['W']) + layer_params['b']
        x = jnp.tanh(x)  # Activation function
    
    # Output layer
    output_params = params['output']
    x = jnp.dot(x, output_params['W']) + output_params['b']
    return x


@jax.jit
def loss_fn(params, X, y):
    """Compute MSE loss."""
    pred = forward_pass(params, X)
    return jnp.mean((pred - y) ** 2)


def train_step(params, opt_state, X, y, optimizer):
    """Single training step. Loss computation is JIT-compiled."""
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                norm_params: Dict,
                hidden_dims: list = [128, 128, 64],
                learning_rate: float = 1e-3,
                batch_size: int = 64,
                num_epochs: int = 100,
                seed: int = 42,
                use_lr_schedule: bool = False) -> Tuple[Dict, Dict]:
    """
    Train the neural network.
    
    Returns:
        params: Trained model parameters
        history: Training history dictionary
    """
    # Set random seed
    key = jax.random.PRNGKey(seed)
    
    # Input dimension: sequence_length * (state_dim + control_dim) = 5 * 12 = 60
    input_dim = X_train.shape[1] * X_train.shape[2]
    output_dim = 1
    
    print(f"\nInitializing model...")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Output dimension: {output_dim}")
    
    # Create model
    key, subkey = jax.random.split(key)
    params, _ = create_mlp(input_dim, hidden_dims, output_dim, subkey)
    
    # Convert to JAX arrays
    X_train_jax = jnp.array(X_train)
    y_train_jax = jnp.array(y_train)
    X_val_jax = jnp.array(X_val)
    y_val_jax = jnp.array(y_val)
    
    # Calculate number of batches (needed for learning rate schedule)
    num_batches = (len(X_train) + batch_size - 1) // batch_size
    
    # Create optimizer with optional learning rate scheduling
    if use_lr_schedule:
        # Cosine decay schedule: start at learning_rate, decay to learning_rate/10
        # decay_steps should be total number of steps, not epochs
        total_steps = num_epochs * num_batches
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps,
            alpha = 0.1 
        )
        optimizer = optax.adam(lr_schedule)
        print(f"Using cosine decay learning rate schedule: {learning_rate} -> {learning_rate/10.0} over {total_steps} steps")
    else:
        optimizer = optax.adam(learning_rate)
        print(f"Using fixed learning rate: {learning_rate}")
    
    opt_state = optimizer.init(params)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"\nStarting training...")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, Batches per epoch: {num_batches}")
    
    # Track best validation loss for early stopping
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_params = params
    
    for epoch in range(num_epochs):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(X_train))
        X_train_shuffled = X_train_jax[perm]
        y_train_shuffled = y_train_jax[perm]

        X_train_shuffled = X_train
        y_train_shuffled = y_train
        
        # Training loop
        epoch_train_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            params, opt_state, batch_loss = train_step(
                params, opt_state, X_batch, y_batch, optimizer
            )
            epoch_train_loss += float(batch_loss)
        
        avg_train_loss = epoch_train_loss / num_batches
        
        # Validation loss (normalized)
        val_loss = float(loss_fn(params, X_val_jax, y_val_jax))
        
        # Compute loss in original scale for better interpretation
        # Get predictions
        pred = forward_pass(params, X_val_jax)
        
        if norm_params.get('normalize_output', False):
            # Denormalize predictions and targets
            y_std_jax = jnp.array(norm_params['y_std'])
            y_mean_jax = jnp.array(norm_params['y_mean'])
            y_val_denorm = y_val_jax * y_std_jax + y_mean_jax
            pred_denorm = pred * y_std_jax + y_mean_jax
        else:
            # Already in original scale
            y_val_denorm = y_val_jax
            pred_denorm = pred
        
        # Compute RMSE in original scale
        val_rmse_denorm = float(jnp.sqrt(jnp.mean((pred_denorm - y_val_denorm) ** 2)))
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse_denorm'] = history.get('val_rmse_denorm', [])
        history['val_rmse_denorm'].append(val_rmse_denorm)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_params = params  # Save best params
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            loss_label = "norm" if norm_params.get('normalize_output', False) else "raw"
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss ({loss_label}): {avg_train_loss:.6f}, Val Loss ({loss_label}): {val_loss:.6f}, Val RMSE: {val_rmse_denorm:.6f}", end="")
            if val_loss == best_val_loss:
                print(" *")
            else:
                print()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            params = best_params
            break
    
    # Use best params if early stopping occurred
    if patience_counter >= patience:
        params = best_params
    
    return params, history


def reconstruct_forward_fn(params: Dict):
    """
    Reconstruct the forward function from saved parameters.
    This allows loading the model without pickling the function itself.
    """
    def forward(params, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Get number of hidden layers
        layer_keys = [k for k in params.keys() if k.startswith('layer_')]
        num_layers = len(layer_keys)
        
        # Hidden layers
        for i in range(num_layers):
            layer_params = params[f'layer_{i}']
            x = jnp.dot(x, layer_params['W']) + layer_params['b']
            x = jnp.tanh(x)  # Activation function
        
        # Output layer
        output_params = params['output']
        x = jnp.dot(x, output_params['W']) + output_params['b']
        return x
    
    return forward


def save_model(params: Dict, norm_params: Dict, history: Dict, 
               model_dir: str, model_name: str = "residual_model"):
    """Save model parameters, normalization params, and training history."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model parameters
    model_path = os.path.join(model_dir, f"{model_name}_params.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved model parameters to: {model_path}")
    
    # Save normalization parameters (convert numpy arrays to lists for better compatibility)
    # Keep original arrays but also save as lists
    norm_params_save = {
        'X_mean': norm_params['X_mean'],
        'X_std': norm_params['X_std'],
        'y_mean': norm_params['y_mean'],
        'y_std': norm_params['y_std'],
        'normalize_output': norm_params.get('normalize_output', True),
        'X_mean_list': norm_params['X_mean'].tolist(),
        'X_std_list': norm_params['X_std'].tolist(),
        'y_mean_list': norm_params['y_mean'].tolist(),
        'y_std_list': norm_params['y_std'].tolist(),
    }
    norm_path = os.path.join(model_dir, f"{model_name}_norm.pkl")
    with open(norm_path, 'wb') as f:
        pickle.dump(norm_params_save, f)
    print(f"Saved normalization parameters to: {norm_path}")
    print(f"  - Input normalization: X_mean shape={norm_params['X_mean'].shape}, X_std shape={norm_params['X_std'].shape}")
    print(f"  - Output normalization: y_mean={norm_params['y_mean'].flatten()}, y_std={norm_params['y_std'].flatten()}")
    print(f"  - Normalize output: {norm_params.get('normalize_output', True)}")
    
    # Save training history
    history_path = os.path.join(model_dir, f"{model_name}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to: {history_path}")
    
    # Save forward function info (we can't pickle the function, but we can save metadata)
    layer_keys = [k for k in params.keys() if k.startswith('layer_')]
    hidden_dims = [params[f'layer_{i}']['W'].shape[1] for i in range(len(layer_keys))]
    
    metadata = {
        'input_dim': params['layer_0']['W'].shape[0],
        'hidden_dims': hidden_dims,
        'output_dim': params['output']['W'].shape[1],
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved model metadata to: {metadata_path}")


def main():
    """Main training function."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        script_dir, 
        'training_data',
        'processed_data.csv'
    )
    model_dir = os.path.join(script_dir, 'models')
    
    # Training hyperparameters
    sequence_length = 10
    hidden_dims = [64, 64]  # Larger and deeper model
    learning_rate = 1e-3  # Slightly higher initial LR
    batch_size = 16
    num_epochs = 100  # More epochs for better convergence
    test_size = 0.2
    seed = 42
    use_lr_schedule = True  # Use learning rate scheduling
    
    print("=" * 60)
    print(f"Training Residual Dynamics Model")
    print("=" * 60)
    
    # Load data
    df = load_data(data_path)
    
    # Create sequences
    X, y = create_sequences(df, sequence_length=sequence_length)
    
    # Split into train/validation
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=test_size, random_state=seed, shuffle=False
    # )

    n = len(X)
    split = int(0.6 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Normalize data (including outputs)
    print("\nNormalizing data...")
    norm_params, X_train_norm, X_val_norm, y_train_norm, y_val_norm = normalize_data(
        X_train, X_val, y_train, y_val, normalize_output=True
    )
    print(f"Input normalization: mean shape={norm_params['X_mean'].shape}, std shape={norm_params['X_std'].shape}")
    print(f"Output normalization: mean={norm_params['y_mean'].flatten()}, std={norm_params['y_std'].flatten()}")
    
    # Train model
    params, history = train_model(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        norm_params,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        seed=seed,
        use_lr_schedule=use_lr_schedule
    )
    
    
    evaluation_file = "/home/florian/Documents/INI/f1tenth_development_gym/TrainingLite/dynamic_residual_jax/training_data/processed_data.csv"
    df_eval = load_data(evaluation_file)
    X_eval, y_eval = create_sequences(df_eval, sequence_length=sequence_length)
    _, X_eval_norm, _, y_eval_norm, _ = normalize_data(
        X_eval, X_eval, y_eval, y_eval, normalize_output=True
    )
    eval_loss = float(loss_fn(params, jnp.array(X_eval_norm), jnp.array(y_eval_norm)))
    print(f"\nEvaluation loss on full dataset (normalized): {eval_loss:.6f}")   
    
    # Save model outputs into evaluation dataframe
    y_eval_pred_norm = forward_pass(params, jnp.array(X_eval_norm))
    y_std_jax = jnp.array(norm_params['y_std'])
    y_mean_jax = jnp.array(norm_params['y_mean'])
    y_eval_pred = y_eval_pred_norm * y_std_jax + y_mean_jax
    df_eval = df_eval.iloc[sequence_length:]  # Align with sequences
    df_eval = df_eval.reset_index(drop=True)
    
    # Save three predictions: actual, predicted (denormalized), and normalized
    df_eval['actual_residual_delta_linear_vel_x_0'] = y_eval.flatten()
    df_eval['predicted_residual_delta_linear_vel_x_0'] = np.array(y_eval_pred).flatten()
    df_eval['predicted_residual_normalized'] = np.array(y_eval_pred_norm).flatten()
    
    # Calculate prediction error
    prediction_error = np.array(y_eval_pred).flatten() - y_eval.flatten()
    df_eval['prediction_error'] = prediction_error
    
    # Calculate absolute error
    df_eval['absolute_error'] = np.abs(prediction_error)
    
    df_eval['corrected_delta_linear_vel_x_0'] = df_eval['predicted_delta_linear_vel_x_0'] + df_eval['predicted_residual_delta_linear_vel_x_0']
    
    eval_output_path = os.path.join(model_dir, "evaluation_with_predictions.csv")
    df_eval.to_csv(eval_output_path, index=False)
    print(f"Saved evaluation predictions to: {eval_output_path}")
    
    # Print evaluation statistics
    print(f"\nEvaluation Statistics:")
    print(f"  Mean prediction error: {np.mean(prediction_error):.6f}")
    print(f"  Std prediction error: {np.std(prediction_error):.6f}")
    print(f"  Mean absolute error: {np.mean(np.abs(prediction_error)):.6f}")
    print(f"  Max absolute error: {np.max(np.abs(prediction_error)):.6f}")
    print(f"  Columns saved: actual_residual_delta_linear_vel_x_0, predicted_residual_delta_linear_vel_x_0, predicted_residual_normalized, prediction_error, absolute_error")


        
    
    # Save model
    print("\nSaving model...")
    save_model(params, norm_params, history, model_dir)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    loss_label = "normalized" if norm_params.get('normalize_output', False) else "raw"
    print(f"Final training loss ({loss_label}): {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss ({loss_label}): {history['val_loss'][-1]:.6f}")
    if 'val_rmse_denorm' in history and len(history['val_rmse_denorm']) > 0:
        print(f"Final validation RMSE: {history['val_rmse_denorm'][-1]:.6f}")
        print(f"  (Target residual RMSE: ~0.138)")
        


if __name__ == "__main__":
    main()


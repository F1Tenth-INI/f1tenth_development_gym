import os
import pickle
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from typing import Union


class Predictor:
    """Minimal predictor class for loading and using trained residual dynamics models."""
    
    def __init__(self, model_dir: str, model_name: str = "residual_model"):
        """
        Initialize predictor by loading model parameters and normalization parameters.
        
        Args:
            model_dir: Directory containing saved model files
            model_name: Name of the model (default: "residual_model")
        """
        # Load model parameters
        params_path = os.path.join(model_dir, f"{model_name}_params.pkl")
        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)
        
        # Load normalization parameters
        norm_path = os.path.join(model_dir, f"{model_name}_norm.pkl")
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        
        # Extract normalization parameters
        self.X_mean = np.array(norm_params['X_mean'])
        self.X_std = np.array(norm_params['X_std'])
        self.y_mean = np.array(norm_params['y_mean'])
        self.y_std = np.array(norm_params['y_std'])
        self.normalize_output = norm_params.get('normalize_output', True)
    
    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Get number of hidden layers
        layer_keys = [k for k in self.params.keys() if k.startswith('layer_')]
        num_layers = len(layer_keys)
        
        # Hidden layers
        for i in range(num_layers):
            layer_params = self.params[f'layer_{i}']
            x = jnp.dot(x, layer_params['W']) + layer_params['b']
            x = jnp.tanh(x)
        
        # Output layer
        output_params = self.params['output']
        x = jnp.dot(x, output_params['W']) + output_params['b']
        return x
    
    def get_params_jax(self):
        """Get predictor parameters as JAX-compatible structures."""
        predictor_params_jax = {}
        for key, value in self.params.items():
            if isinstance(value, dict):
                predictor_params_jax[key] = {
                    k: jnp.array(v) for k, v in value.items()
                }
            else:
                predictor_params_jax[key] = jnp.array(value)
        return predictor_params_jax
    
    def get_norm_params_jax(self):
        """Get normalization parameters as JAX-compatible structures."""
        return {
            'X_mean': jnp.array(self.X_mean),
            'X_std': jnp.array(self.X_std),
            'y_mean': jnp.array(self.y_mean),
            'y_std': jnp.array(self.y_std),
            'normalize_output': self.normalize_output
        }
    
    def predict(self, input: jnp.ndarray) -> jnp.ndarray:
        """
        Predict residual given input sequence.
        
        Args:
            input: Input array of shape (batch_size, sequence_length, features) 
                   or (sequence_length, features) for single prediction
        
        Returns:
            Predicted residual as numpy array
        """
        input = jnp.array(input)
        if input.ndim == 2:
            input = input[np.newaxis, :]
        predictor_params = self.get_params_jax()
        norm_params = self.get_norm_params_jax()
        return predictor_forward_jit(input, predictor_params, norm_params)
    



# JIT-compatible forward function
@partial(jax.jit)
def predictor_forward_jit(x: jnp.ndarray, predictor_params: dict, norm_params: dict) -> jnp.ndarray:
    """JIT-compatible forward pass through the predictor network."""
    # Flatten input if needed: (batch_size, history_length, features) -> (batch_size, history_length * features)
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
    
    # Normalize input
    input_norm = (x - norm_params['X_mean']) / norm_params['X_std']
    
    # Get number of hidden layers
    layer_keys = [k for k in predictor_params.keys() if k.startswith('layer_')]
    num_layers = len(layer_keys)
    
    # Hidden layers
    for i in range(num_layers):
        layer_params = predictor_params[f'layer_{i}']
        input_norm = jnp.dot(input_norm, layer_params['W']) + layer_params['b']
        input_norm = jnp.tanh(input_norm)
    
    # Output layer
    output_params = predictor_params['output']
    output_norm = jnp.dot(input_norm, output_params['W']) + output_params['b']
    
    # Denormalize output 
    output = output_norm * norm_params['y_std'] + norm_params['y_mean']

    
    return output


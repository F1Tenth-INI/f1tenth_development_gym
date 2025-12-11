import os
import pickle
import numpy as np
import jax.numpy as jnp
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
    
    def predict(self, input: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
        """
        Predict residual given input sequence.
        
        Args:
            input: Input array of shape (batch_size, sequence_length, features) 
                   or (sequence_length, features) for single prediction
        
        Returns:
            Predicted residual as numpy array
        """
        # Convert to numpy if needed
        if isinstance(input, jnp.ndarray):
            input = np.array(input)
        
        # Ensure 3D shape: (batch_size, sequence_length, features)
        if len(input.shape) == 2:
            input = input[np.newaxis, :]
        
        # Normalize input
        input_flat = input.reshape(input.shape[0], -1)
        input_norm = (input_flat - self.X_mean) / self.X_std
        input_norm = input_norm.reshape(input.shape)
        
        # Forward pass
        output_norm = self._forward(jnp.array(input_norm))
        
        # Denormalize output if needed
        if self.normalize_output:
            output = np.array(output_norm) * self.y_std + self.y_mean
        else:
            output = np.array(output_norm)
        
        # Return single value if single input was provided
        if output.shape[0] == 1:
            return output[0]
        return output


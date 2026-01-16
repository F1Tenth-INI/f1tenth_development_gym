#!/usr/bin/env python3
"""
Common utilities for state comparison visualization.
Shared between interactive UI and offline batch plotting.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple

# Configure matplotlib defaults
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['figure.titlesize'] = 16

# Add paths for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'sim/f110_sim/envs'))
sys.path.append(os.path.join(parent_dir, 'utilities'))

from sim.f110_sim.envs.car_model_jax import CarModelJAX
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import STATE_VARIABLES, STATE_INDICES

# Control column names
STEERING_CONTROL_COLUMN = 'angular_control_executed'
ACCELERATION_CONTROL_COLUMN = 'translational_control_executed'

# Available car models mapping
AVAILABLE_MODELS = {
    'pacejka': 'Pure Pacejka Model',
    'ks_jax': 'KS jax',
    'direct': 'Direct Dynamics Neural Network',
    'residual': 'Residual Dynamics Model',
}


class VisualizationCommon:
    """Common utilities for visualization shared between UI and offline modes."""
    
    def __init__(self):
        self.data = None
        self.csv_file_path = None
        self.time_column = 'time'
        self.state_columns = list(STATE_VARIABLES)
        self.control_columns = [STEERING_CONTROL_COLUMN, ACCELERATION_CONTROL_COLUMN]
        self.state_indices = STATE_INDICES
        
        # Car models - will be initialized when needed
        self.car_models = None
        self._current_start_index = None
        
        # Available car parameter files
        self.available_car_params = {}
        car_files_dir = os.path.join(parent_dir, 'utilities', 'car_files')
        if os.path.exists(car_files_dir):
            for filename in os.listdir(car_files_dir):
                if filename.endswith('.yml') or filename.endswith('.yaml'):
                    self.available_car_params[filename] = filename
    
    def reload_car_models(self):
        """Force reload modules and recreate car model instances."""
        import importlib
        import sys
        
        # Reload modules
        for module_name in ['sim.f110_sim.envs.dynamic_model_pacejka_jax',
                            'sim.f110_sim.envs.dynamic_model_ks_jax',
                            'TrainingLite.dynamic_residual_jax.dynamics_model_residual',
                            'TrainingLite.dynamic_residual_jax.predictor',
                            'sim.f110_sim.envs.car_model_jax']:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
        
        # Reimport and recreate models
        from sim.f110_sim.envs.car_model_jax import CarModelJAX
        self.car_models = {
            'pacejka': CarModelJAX(model_type='pacejka', dt=0.04, intermediate_steps=4),
            'pacejka_custom': CarModelJAX(model_type='pacejka_custom', dt=0.04, intermediate_steps=4),
            'ks_jax': CarModelJAX(model_type='ks', dt=0.04, intermediate_steps=4),
            'residual': CarModelJAX(model_type='residual', dt=0.04, intermediate_steps=4),
        }
    
    def load_car_parameters(self, param_file):
        """Load car parameters from YAML file."""
        try:
            vehicle_params = VehicleParameters(param_file)
            params_array = vehicle_params.to_np_array()
            print(f"Loaded car parameters from {param_file} - array length: {len(params_array)}")
            return params_array
        except Exception as e:
            print(f"Error loading car parameters: {e}")
            return None
    
    def get_model_key(self, display_name):
        """Get the model key from display name."""
        for key, name in AVAILABLE_MODELS.items():
            if name == display_name:
                return key
        return list(AVAILABLE_MODELS.keys())[0]  # Default
    
    def get_car_parameters_filename(self, display_name):
        """Get the actual filename from display name."""
        for filename, name in self.available_car_params.items():
            if name == display_name:
                return filename
        return list(self.available_car_params.keys())[0]  # Default
    
    def extract_initial_state_at_index(self, index):
        """Extract initial state from the data at a specific index."""
        if self.data is None or index >= len(self.data):
            return jnp.zeros(10, dtype=jnp.float32)
            
        initial_state = np.zeros(10, dtype=np.float32)
        for col_name, idx in self.state_indices.items():
            if col_name in self.data.columns:
                initial_state[idx] = self.data[col_name].iloc[index]
                
        return jnp.array(initial_state)
    
    def extract_control_sequence_at_index(self, start_index, horizon, steering_delay=0, acceleration_delay=0):
        """Extract control sequence from the data starting at a specific index with separate control delays."""
        if self.data is None:
            return jnp.zeros((horizon, 2), dtype=jnp.float32)
        
        # Control order: [desired_steering_angle, acceleration]
        control_sequence = np.zeros((horizon, 2), dtype=np.float32)
        
        # Handle steering control with its own delay
        steering_start_index = start_index - steering_delay
        steering_end_index = min(steering_start_index + horizon, len(self.data))
        
        if steering_start_index >= 0 and steering_start_index < len(self.data):
            actual_horizon = max(0, steering_end_index - steering_start_index)
            if actual_horizon > 0 and STEERING_CONTROL_COLUMN in self.data.columns:
                control_data = self.data[STEERING_CONTROL_COLUMN].iloc[steering_start_index:steering_end_index].values
                control_sequence[:actual_horizon, 0] = control_data
        else:
            available_start = max(0, steering_start_index)
            available_end = min(available_start + horizon, len(self.data))
            if available_end > available_start and STEERING_CONTROL_COLUMN in self.data.columns:
                first_control = self.data[STEERING_CONTROL_COLUMN].iloc[0]
                offset = max(0, -steering_start_index)
                actual_length = min(horizon - offset, available_end - available_start)
                if actual_length > 0:
                    control_sequence[offset:offset + actual_length, 0] = self.data[STEERING_CONTROL_COLUMN].iloc[available_start:available_start + actual_length].values
                if offset > 0:
                    control_sequence[:offset, 0] = first_control
        
        # Handle acceleration control with its own delay
        acceleration_start_index = start_index - acceleration_delay
        acceleration_end_index = min(acceleration_start_index + horizon, len(self.data))
        
        if acceleration_start_index >= 0 and acceleration_start_index < len(self.data):
            actual_horizon = max(0, acceleration_end_index - acceleration_start_index)
            if actual_horizon > 0 and ACCELERATION_CONTROL_COLUMN in self.data.columns:
                control_data = self.data[ACCELERATION_CONTROL_COLUMN].iloc[acceleration_start_index:acceleration_end_index].values
                control_sequence[:actual_horizon, 1] = control_data
        else:
            available_start = max(0, acceleration_start_index)
            available_end = min(available_start + horizon, len(self.data))
            if available_end > available_start and ACCELERATION_CONTROL_COLUMN in self.data.columns:
                first_control = self.data[ACCELERATION_CONTROL_COLUMN].iloc[0]
                offset = max(0, -acceleration_start_index)
                actual_length = min(horizon - offset, available_end - available_start)
                if actual_length > 0:
                    control_sequence[offset:offset + actual_length, 1] = self.data[ACCELERATION_CONTROL_COLUMN].iloc[available_start:available_start + actual_length].values
                if offset > 0:
                    control_sequence[:offset, 1] = first_control
        
        return jnp.array(control_sequence)
    
    def get_timestep(self):
        """Get timestep from data or use default."""
        if self.data is not None and self.time_column in self.data.columns and len(self.data) > 1:
            return float(self.data[self.time_column].iloc[1] - self.data[self.time_column].iloc[0])
        return 0.04  # Default 25Hz
    
    def run_model_prediction(self, model_name, initial_state, control_sequence, car_params, dt, horizon, data=None):
        """Run model prediction based on model name."""
        if self.car_models is None:
            self.reload_car_models()
        
        try:
            if model_name not in self.car_models:
                print(f"Unknown model: {model_name}")
                return None
            
            car_model = self.car_models[model_name]
            
            # Update car params if provided
            if car_params is not None:
                car_model.car_params = jnp.array(car_params)
            
            # Prepare history for residual model
            state_history = None
            control_history = None
            if model_name == 'residual':
                history_length = 10
                if self._current_start_index is not None and data is not None and self._current_start_index >= history_length:
                    start_idx = self._current_start_index
                    # Extract state and control history
                    state_history = np.array([self.extract_initial_state_at_index(start_idx - history_length + i) for i in range(history_length)], dtype=np.float32)
                    control_history = np.array([self.extract_control_sequence_at_index(start_idx - history_length + i, 1, 0, 0)[0] for i in range(history_length)], dtype=np.float32)
                else:
                    state_history = np.array(car_model.state_history)
                    control_history = np.array(car_model.control_history)
            
            # Run prediction
            control_seq = control_sequence[:horizon]
            return np.array(car_model.car_steps_sequential(initial_state, control_seq, state_history=state_history, control_history=control_history))
        except Exception as e:
            print(f"Model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def convert_predictions_to_dict(self, predicted_states):
        """Convert JAX predictions to dictionary format matching CSV columns."""
        comparison_data = {}
        for col_name, idx in self.state_indices.items():
            if col_name in self.state_columns:
                comparison_data[col_name] = predicted_states[:, idx]
        return comparison_data
    
    def run_single_comparison(self, start_index, model_name, param_file, horizon, steering_delay=0, acceleration_delay=0):
        """Run model comparison for a single start index."""
        try:
            # Load car parameters
            car_params = self.load_car_parameters(param_file)
            if car_params is None:
                return None
            
            # Prepare initial state and controls with delay
            initial_state = self.extract_initial_state_at_index(start_index)
            control_sequence = self.extract_control_sequence_at_index(start_index, horizon, steering_delay, acceleration_delay)
            
            # Store start_index for residual model
            self._current_start_index = start_index
            
            # Get timestep
            dt = self.get_timestep()
            
            # Run model prediction
            predicted_states = self.run_model_prediction(model_name, initial_state, control_sequence, car_params, dt, horizon, self.data)
            if predicted_states is None:
                return None
            
            # Convert and return
            return self.convert_predictions_to_dict(predicted_states)
            
        except Exception as e:
            print(f"Single comparison failed for index {start_index}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_prediction_with_gradient(self, ax, time_data, prediction_data, horizon, label=None, trajectory_index=0):
        """Plot prediction with color gradient over horizon (pure matplotlib, no UI dependencies)."""
        try:
            from matplotlib.colors import LinearSegmentedColormap
            colors = ["#5200F5", "#FF00BF", "#FF0000"]
            alpha_base = 0.4
            cmap = LinearSegmentedColormap.from_list("gradient", colors, N=100)
        except ImportError:
            if trajectory_index == 0:
                start_color = np.array([1.0, 0.4, 0.4])
                end_color = np.array([0.5, 0.0, 0.0])
                alpha_base = 0.8
            else:
                start_color = np.array([1.0, 0.6, 0.0])
                end_color = np.array([0.5, 0.3, 0.1])
                alpha_base = 0.4
            cmap = None
        
        # Convert time_data to numpy array
        if hasattr(time_data, 'iloc'):
            time_data = time_data.values
        elif hasattr(time_data, 'to_numpy'):
            time_data = time_data.to_numpy()
        time_data = np.asarray(time_data)
        
        # Plot prediction as segments with gradient
        if len(prediction_data) == 1:
            if cmap is not None:
                color = cmap(0.0)
            else:
                color = start_color
            alpha = alpha_base
            ax.plot(time_data[0], prediction_data[0], 'o', color=color, markersize=3, alpha=alpha, label=label)
        else:
            for i in range(len(prediction_data) - 1):
                color_position = i / max(1, len(prediction_data) - 1)
                if cmap is not None:
                    color = cmap(color_position)
                else:
                    color = start_color * (1 - color_position) + end_color * color_position
                alpha = alpha_base * (1.0 - 0.3 * color_position)
                ax.plot(time_data[i:i+2], prediction_data[i:i+2], 'o', color=color, markersize=3, 
                       alpha=alpha, label=label if i == 0 else None)
    
    def calculate_error_metrics(self, ground_truth, prediction, state_name):
        """Calculate error metrics between ground truth and prediction."""
        if len(ground_truth) == 0 or len(prediction) == 0:
            return None
        
        min_length = min(len(ground_truth), len(prediction))
        gt_data = np.array(ground_truth[:min_length])
        pred_data = np.array(prediction[:min_length])
        
        error = pred_data - gt_data
        
        return {
            'mean_error': np.mean(np.abs(error)),
            'max_error': np.max(np.abs(error)),
            'error_std': np.std(error),
            'rmse': np.sqrt(np.mean(error**2))
        }

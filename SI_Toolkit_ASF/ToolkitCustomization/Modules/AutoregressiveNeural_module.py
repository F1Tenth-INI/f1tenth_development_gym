"""
AutoregressiveNeural_module.py

A custom Keras model that wraps a Dense neural network and trains it autoregressively
over multiple time steps. Instead of predicting single-step derivatives with teacher forcing,
this module:
1. Takes an initial state and a sequence of control inputs
2. Runs the network in a loop, feeding predictions back as inputs (closed-loop)
3. Computes loss over the entire trajectory

This enables training the network to minimize trajectory errors, not just single-step errors.

Usage in config_training.yml:
    modeling:
        NET_NAME: "Custom-AutoregressiveDense"  # or AutoregressiveDenseXXH1-XXH2 for custom architecture
    training_default:
        POST_WASH_OUT_LEN: 20  # The horizon length for autoregressive training
        WASH_OUT_LEN: 0
"""

import re
import copy
import tensorflow as tf
import numpy as np

from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.TF.Network import compose_net_from_net_name
from SI_Toolkit.Functions.General.Normalising import (
    get_normalization_function,
    get_denormalization_function,
    get_scaling_function_for_output_of_differential_network,
)
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net, get_net
from SI_Toolkit.Predictors.autoregression import autoregression_loop, differential_model_autoregression_helper

# Default base network architecture if not specified in module name
DEFAULT_BASE_NET_NAME = "Dense-64H1-64H2"


class AutoregressiveDense(tf.keras.Model):
    """
    A Keras model that wraps a Dense network for autoregressive training.
    
    During training:
    - Input: [batch, horizon, num_inputs] where inputs are [control_inputs..., state_inputs...]
    - Output: [batch, horizon, num_outputs] predicted states at each timestep
    
    The model runs the base network in a loop:
    1. Takes initial state from inputs[:, 0, state_indices]
    2. For each timestep, concatenates control with current state prediction
    3. Feeds through network to get derivatives
    4. Integrates derivatives to get next state
    5. Feeds predicted state back for next timestep
    """
    
    def __init__(self, time_series_length, batch_size, net_info, base_net_name=None, name=None, **kwargs):
        super().__init__(**kwargs)
        
        self.lib = TensorFlowLibrary()
        self.batch_size = batch_size
        self.horizon = time_series_length  # This is wash_out_len + post_wash_out_len
        self.original_net_info = net_info
        
        # Get timestep from Settings or default
        try:
            from utilities.Settings import Settings
            self.dt = Settings.TIMESTEP_PLANNER
        except (ImportError, AttributeError):
            self.dt = 0.02  # Default 50Hz
            print(f"[AutoregressiveDense] Using default dt={self.dt}s")
        
        # Determine the base network architecture
        if base_net_name is None:
            base_net_name = DEFAULT_BASE_NET_NAME
        self.base_net_name = base_net_name
        
        # Setup normalization
        # Handle missing attributes that get_norm_info_for_net expects
        if not hasattr(net_info, 'parent_net_name'):
            net_info.parent_net_name = 'Network trained from scratch'
        if not hasattr(net_info, 'path_to_net'):
            net_info.path_to_net = None
        
        try:
            self.normalization_info = get_norm_info_for_net(net_info, copy_files=False)
        except Exception as e:
            print(f"[AutoregressiveDense] Could not load normalization info: {e}")
            self.normalization_info = None
        
        # Check if this is a differential network (outputs start with D_)
        self.differential_network = any('D_' in out for out in net_info.outputs)
        
        # Identify state vs control inputs
        # State inputs are those that don't have time suffixes like _-1
        self._setup_input_output_mapping(net_info)
        
        # Create the base Dense network
        self._create_base_network(net_info, base_net_name)
        
        # Setup autoregression loop
        self._setup_autoregression(net_info)
        
        # Normalization/denormalization functions
        self._setup_normalization(net_info)
        
    def _setup_input_output_mapping(self, net_info):
        """Identify which inputs are states vs controls."""
        self.all_inputs = net_info.inputs
        self.all_outputs = net_info.outputs
        
        # Strip time suffixes for comparison
        def strip_suffix(name):
            return re.sub(r'_-?\d+$', '', name)
        
        # Determine state variables from outputs (strip D_ prefix and time suffix)
        if self.differential_network:
            self.state_variables = [strip_suffix(x[2:]) if x.startswith('D_') else strip_suffix(x) 
                                    for x in self.all_outputs]
        else:
            self.state_variables = [strip_suffix(x) for x in self.all_outputs]
        
        # Identify control inputs (those not matching state variables)
        self.control_inputs = []
        self.state_inputs = []
        for inp in self.all_inputs:
            base_name = strip_suffix(inp)
            if base_name in self.state_variables:
                self.state_inputs.append(inp)
            else:
                self.control_inputs.append(inp)
        
        self.num_controls = len(self.control_inputs)
        self.num_state_inputs = len(self.state_inputs)
        self.num_outputs = len(self.all_outputs)
        
        # Build index mappings
        # control_indices: positions of control inputs in the full input vector
        # state_indices: positions of state inputs in the full input vector  
        self.control_indices = [self.all_inputs.index(c) for c in self.control_inputs]
        self.state_indices = [self.all_inputs.index(s) for s in self.state_inputs]
        
        print(f"[AutoregressiveDense] Controls ({self.num_controls}): {self.control_inputs}")
        print(f"[AutoregressiveDense] State inputs ({self.num_state_inputs}): {self.state_inputs}")
        print(f"[AutoregressiveDense] Outputs ({self.num_outputs}): {self.all_outputs}")
        print(f"[AutoregressiveDense] Differential network: {self.differential_network}")
        
    def _create_base_network(self, net_info, base_net_name):
        """Create the underlying Dense network."""
        # Create a copy of net_info for the base network
        base_net_info = copy.deepcopy(net_info)
        base_net_info.net_name = base_net_name
        
        # Build the Dense network manually to avoid batch_size issues with newer Keras
        # Parse network architecture from name
        names = base_net_name.split('-')
        h_sizes = []
        for name in names:
            for layer_suffix in ['H1', 'H2', 'H3', 'H4', 'H5']:
                if layer_suffix in name:
                    h_sizes.append(int(name[:-2]))
                    break
        
        if not h_sizes:
            h_sizes = [64, 64]  # Default architecture
        
        inputs_len = len(net_info.inputs)
        outputs_len = len(net_info.outputs)
        
        # Build Sequential model
        self.base_net = tf.keras.Sequential()
        self.base_net.add(tf.keras.Input(shape=(inputs_len,)))  # No time dimension
        
        for i, units in enumerate(h_sizes):
            self.base_net.add(tf.keras.layers.Dense(
                units=units,
                name=f'layers_{i}',
            ))
            self.base_net.add(tf.keras.layers.Activation('tanh'))
        
        # Output layer
        self.base_net.add(tf.keras.layers.Dense(
            units=outputs_len,
            activation='linear',
            name='layers_output',
        ))
        
        # Store net_info for compatibility
        self.base_net_info = base_net_info
        self.base_net_info.net_type = 'Dense'
        
        print(f"[AutoregressiveDense] Created base network: {base_net_name} with hidden layers {h_sizes}")
        
    def _setup_autoregression(self, net_info):
        """Setup the autoregression loop and differential model helper."""
        if self.differential_network:
            # Create differential model helper for state integration
            self.dmah = differential_model_autoregression_helper(
                inputs=net_info.inputs,
                outputs=net_info.outputs,
                normalization_info=self.normalization_info,
                dt=self.dt,
                lib=self.lib,
                state_variables=np.array(self.state_variables),
            )
        else:
            self.dmah = None
        
        # Create a wrapper that handles shape conversion from [batch, 1, features] to [batch, features]
        # The autoregression loop passes 3D tensors, but our Dense network expects 2D
        def model_wrapper(x):
            # x shape: [batch, 1, features] -> squeeze to [batch, features]
            x_squeezed = tf.squeeze(x, axis=1)
            output = self.base_net(x_squeezed)
            return output
            
        # Create autoregression loop with the wrapper
        self.AL = autoregression_loop(
            model=model_wrapper,
            model_inputs_len=len(net_info.inputs),
            model_outputs_len=len(net_info.outputs),
            lib=self.lib,
            differential_model_autoregression_helper_instance=self.dmah,
        )
        
    def _setup_normalization(self, net_info):
        """Setup normalization functions."""
        if self.normalization_info is not None:
            # Strip time suffixes for normalization lookup
            strip_suffix = lambda name: re.sub(r'_-?\d+$', '', name)
            
            # Normalize state inputs
            self.normalize_state = get_normalization_function(
                self.normalization_info,
                [strip_suffix(s) for s in self.state_inputs],
                self.lib
            )
            
            # Normalize control inputs  
            self.normalize_control = get_normalization_function(
                self.normalization_info,
                [strip_suffix(c) for c in self.control_inputs],
                self.lib
            )
            
            # Denormalize outputs
            if self.differential_network:
                output_names = [strip_suffix(x[2:]) if x.startswith('D_') else strip_suffix(x) 
                               for x in self.all_outputs]
            else:
                output_names = [strip_suffix(x) for x in self.all_outputs]
            self.denormalize_output = get_denormalization_function(
                self.normalization_info,
                output_names,
                self.lib
            )
        else:
            self.normalize_state = lambda x: x
            self.normalize_control = lambda x: x
            self.denormalize_output = lambda x: x
            
    def call(self, x, training=None, mask=None):
        """
        Forward pass: run autoregressive prediction.
        
        Args:
            x: Input tensor of shape [batch, horizon, num_inputs]
               where inputs are ordered as in net_info.inputs (controls first, then states)
               
        Returns:
            outputs: Tensor of shape [batch, horizon, num_outputs]
                    Predicted states at each timestep
        """
        # x shape: [batch, horizon, num_inputs]
        batch_size = tf.shape(x)[0]
        horizon = tf.shape(x)[1]
        
        # Extract initial state from first timestep
        # The state features are at state_indices positions in the input
        state_indices_tf = tf.constant(self.state_indices, dtype=tf.int32)
        initial_state = tf.gather(x[:, 0, :], state_indices_tf, axis=1)  # [batch, num_state_inputs]
        
        # Extract control inputs for all timesteps
        # Controls are at control_indices positions
        control_indices_tf = tf.constant(self.control_indices, dtype=tf.int32)
        control_sequence = tf.gather(x, control_indices_tf, axis=2)  # [batch, horizon, num_controls]
        
        # Prepare initial model input (state features that go into the network)
        model_initial_input = initial_state  # [batch, num_state_inputs]
        
        # Prepare initial dm_state for differential models
        # For differential networks, dm_state tracks the integrated state
        # The dm_state must match output dimension, so we need to build a full state vector
        if self.differential_network and self.dmah is not None:
            # Build full dm_state with zeros for outputs not present in inputs
            # outputs: pose_theta_cos, pose_theta_sin, pose_x, pose_y, linear_vel_x, angular_vel_z, slip_angle, steering_angle
            # inputs:  pose_theta_cos, pose_theta_sin, linear_vel_x, angular_vel_z, slip_angle, steering_angle (6)
            
            # Map from state_inputs to full output state
            # Create a mapping from output names to indices
            dm_state_list = []
            for out_var in self.state_variables:  # These are output state variable names
                # Check if this output has a corresponding input
                found = False
                for i, inp in enumerate(self.state_inputs):
                    inp_base = re.sub(r'_-?\d+$', '', inp)
                    if inp_base == out_var:
                        # Use tf.gather to get the correct feature
                        dm_state_list.append(initial_state[:, i:i+1])
                        found = True
                        break
                if not found:
                    # This output variable is not in inputs - initialize with zeros
                    dm_state_list.append(tf.zeros([batch_size, 1], dtype=initial_state.dtype))
            
            dm_state_init = tf.concat(dm_state_list, axis=1)  # [batch, num_outputs]
        else:
            dm_state_init = None
            
        # Run autoregression loop
        # external_input_left: controls [batch, horizon, num_controls]
        outputs = self.AL.run(
            initial_input=model_initial_input,
            external_input_left=control_sequence,
            dm_state_init=dm_state_init,
        )
        
        # outputs shape: [batch, horizon, num_outputs]
        return outputs
    
    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'base_net_name': self.base_net_name,
            'horizon': self.horizon,
            'dt': self.dt,
        })
        return config
    
    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Save the model weights."""
        # Save base network weights
        self.base_net.save(filepath, overwrite=overwrite, save_format=save_format, **kwargs)
        
    def save_weights(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Save just the weights."""
        self.base_net.save_weights(filepath, overwrite=overwrite, save_format=save_format, **kwargs)
        
    def load_weights(self, filepath, **kwargs):
        """Load weights into base network."""
        self.base_net.load_weights(filepath, **kwargs)


# Convenience classes with different architectures
class AutoregressiveDense64H1_64H2(AutoregressiveDense):
    """Autoregressive Dense network with 64-64 hidden layers."""
    def __init__(self, time_series_length, batch_size, net_info, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info, 
                        base_net_name="Dense-64H1-64H2", name=name, **kwargs)


class AutoregressiveDense128H1_128H2(AutoregressiveDense):
    """Autoregressive Dense network with 128-128 hidden layers."""
    def __init__(self, time_series_length, batch_size, net_info, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info,
                        base_net_name="Dense-128H1-128H2", name=name, **kwargs)


class AutoregressiveDense64H1_128H2_64H3(AutoregressiveDense):
    """Autoregressive Dense network with 64-128-64 hidden layers."""
    def __init__(self, time_series_length, batch_size, net_info, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info,
                        base_net_name="Dense-64H1-128H2-64H3", name=name, **kwargs)


class AutoregressiveDense32H1_64H2_32H3(AutoregressiveDense):
    """Autoregressive Dense network with 32-64-32 hidden layers."""
    def __init__(self, time_series_length, batch_size, net_info, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info,
                        base_net_name="Dense-32H1-64H2-32H3", name=name, **kwargs)


class AutoregressivePretrained(tf.keras.Model):
    """
    Wraps an existing pretrained network for autoregressive fine-tuning.
    
    This class loads a pretrained network and wraps it with an autoregression 
    loop for trajectory-based training.
    
    Usage in config_training.yml - encode pretrained model name in NET_NAME:
        modeling:
            # Format: Custom-AutoregressivePretrained-<pretrained_model_name>
            NET_NAME: "Custom-AutoregressivePretrained-Dense-9IN-64H1-64H2-8OUT-0"
        training_default:
            POST_WASH_OUT_LEN: 20  # Trajectory horizon
    
    The pretrained model will be loaded from PATH_TO_EXPERIMENT_FOLDERS/Models/
    """
    
    def __init__(self, time_series_length, batch_size, net_info, name=None, **kwargs):
        super().__init__(**kwargs)
        
        self.lib = TensorFlowLibrary()
        self.batch_size = batch_size
        self.horizon = time_series_length
        self.original_net_info = net_info
        
        # Get timestep from Settings or default
        try:
            from utilities.Settings import Settings
            self.dt = Settings.TIMESTEP_PLANNER
        except (ImportError, AttributeError):
            self.dt = 0.02
            print(f"[AutoregressivePretrained] Using default dt={self.dt}s")
        
        # Extract pretrained model name from net_info.net_name
        # Expected format: "Custom-AutoregressivePretrained-<pretrained_model_name>"
        # or just the pretrained model name passed directly
        pretrained_model_name = self._extract_pretrained_model_name(net_info)
                
        if pretrained_model_name is None:
            raise ValueError(
                "No pretrained model specified. Use NET_NAME format:\n"
                "  NET_NAME: 'Custom-AutoregressivePretrained-Dense-9IN-64H1-64H2-8OUT-0'\n"
                "Where 'Dense-9IN-64H1-64H2-8OUT-0' is your pretrained model name."
            )
        
        print(f"[AutoregressivePretrained] Loading pretrained model: {pretrained_model_name}")
    
    def _extract_pretrained_model_name(self, net_info):
        """Extract pretrained model name from net_info.net_name or other sources."""
        pretrained_model_name = None
        
        # Source 1: Parse from net_name (format: "Custom-AutoregressivePretrained-<model_name>")
        if hasattr(net_info, 'net_name') and net_info.net_name:
            net_name = net_info.net_name
            # Check for format: Custom-AutoregressivePretrained-ModelName
            prefix = "AutoregressivePretrained-"
            if prefix in net_name:
                # Extract everything after "AutoregressivePretrained-"
                idx = net_name.index(prefix) + len(prefix)
                pretrained_model_name = net_name[idx:]
                if pretrained_model_name:
                    return pretrained_model_name
        
        # Source 2: net_info attribute (if set programmatically)
        if hasattr(net_info, 'pretrained_model') and net_info.pretrained_model:
            return net_info.pretrained_model
        
        # Source 3: Check if parent_net_name is set (from previous training)
        if hasattr(net_info, 'parent_net_name'):
            if net_info.parent_net_name and net_info.parent_net_name != 'Network trained from scratch':
                return net_info.parent_net_name
        
        return None
        
        # Load the pretrained network
        self._load_pretrained_network(net_info, pretrained_model_name)
        
        # Check if this is a differential network
        self.differential_network = any('D_' in out for out in net_info.outputs)
        
        # Setup input/output mapping
        self._setup_input_output_mapping(net_info)
        
        # Setup autoregression loop
        self._setup_autoregression(net_info)
        
        # Setup normalization
        self._setup_normalization(net_info)
        
    def _load_pretrained_network(self, net_info, pretrained_model_name):
        """Load the pretrained network."""
        from types import SimpleNamespace
        
        # Create args for get_net
        a = SimpleNamespace()
        a.net_name = pretrained_model_name
        a.path_to_models = net_info.path_to_models if hasattr(net_info, 'path_to_models') else None
        a.wash_out_len = 0
        a.post_wash_out_len = 1  # Single step for the base network
        
        # Try to load the pretrained network
        try:
            self.base_net, loaded_net_info = get_net(
                a,
                time_series_length=1,
                batch_size=None,
                stateful=False,
                remove_redundant_dimensions=True,
            )
            
            # Use the loaded network's normalization info
            if hasattr(loaded_net_info, 'path_to_normalization_info'):
                net_info.path_to_normalization_info = loaded_net_info.path_to_normalization_info
            
            # Store base network info
            self.base_net_info = loaded_net_info
            self.base_net_name = pretrained_model_name
            
            print(f"[AutoregressivePretrained] Loaded pretrained network: {loaded_net_info.net_full_name}")
            print(f"[AutoregressivePretrained] Inputs: {loaded_net_info.inputs}")
            print(f"[AutoregressivePretrained] Outputs: {loaded_net_info.outputs}")
            
            # Update net_info with loaded network's inputs/outputs
            net_info.inputs = loaded_net_info.inputs
            net_info.outputs = loaded_net_info.outputs
            
        except Exception as e:
            raise ValueError(f"Failed to load pretrained model '{pretrained_model_name}': {e}")
    
    def _setup_input_output_mapping(self, net_info):
        """Identify which inputs are states vs controls."""
        self.all_inputs = net_info.inputs
        self.all_outputs = net_info.outputs
        
        def strip_suffix(name):
            return re.sub(r'_-?\d+$', '', name)
        
        if self.differential_network:
            self.state_variables = [strip_suffix(x[2:]) if x.startswith('D_') else strip_suffix(x) 
                                    for x in self.all_outputs]
        else:
            self.state_variables = [strip_suffix(x) for x in self.all_outputs]
        
        self.control_inputs = []
        self.state_inputs = []
        for inp in self.all_inputs:
            base_name = strip_suffix(inp)
            if base_name in self.state_variables:
                self.state_inputs.append(inp)
            else:
                self.control_inputs.append(inp)
        
        self.num_controls = len(self.control_inputs)
        self.num_state_inputs = len(self.state_inputs)
        self.num_outputs = len(self.all_outputs)
        
        self.control_indices = [self.all_inputs.index(c) for c in self.control_inputs]
        self.state_indices = [self.all_inputs.index(s) for s in self.state_inputs]
        
        print(f"[AutoregressivePretrained] Controls ({self.num_controls}): {self.control_inputs}")
        print(f"[AutoregressivePretrained] State inputs ({self.num_state_inputs}): {self.state_inputs}")
        
    def _setup_autoregression(self, net_info):
        """Setup the autoregression loop."""
        # Get normalization info
        try:
            self.normalization_info = get_norm_info_for_net(net_info, copy_files=False)
        except Exception:
            self.normalization_info = None
        
        if self.differential_network:
            self.dmah = differential_model_autoregression_helper(
                inputs=net_info.inputs,
                outputs=net_info.outputs,
                normalization_info=self.normalization_info,
                dt=self.dt,
                lib=self.lib,
                state_variables=np.array(self.state_variables),
            )
        else:
            self.dmah = None
        
        # Create wrapper for shape conversion
        def model_wrapper(x):
            x_squeezed = tf.squeeze(x, axis=1)
            output = self.base_net(x_squeezed)
            return output
            
        self.AL = autoregression_loop(
            model=model_wrapper,
            model_inputs_len=len(net_info.inputs),
            model_outputs_len=len(net_info.outputs),
            lib=self.lib,
            differential_model_autoregression_helper_instance=self.dmah,
        )
        
    def _setup_normalization(self, net_info):
        """Setup normalization functions."""
        if self.normalization_info is not None:
            strip_suffix = lambda name: re.sub(r'_-?\d+$', '', name)
            
            self.normalize_state = get_normalization_function(
                self.normalization_info,
                [strip_suffix(s) for s in self.state_inputs],
                self.lib
            )
            self.normalize_control = get_normalization_function(
                self.normalization_info,
                [strip_suffix(c) for c in self.control_inputs],
                self.lib
            )
            
            if self.differential_network:
                output_names = [strip_suffix(x[2:]) if x.startswith('D_') else strip_suffix(x) 
                               for x in self.all_outputs]
            else:
                output_names = [strip_suffix(x) for x in self.all_outputs]
            self.denormalize_output = get_denormalization_function(
                self.normalization_info,
                output_names,
                self.lib
            )
        else:
            self.normalize_state = lambda x: x
            self.normalize_control = lambda x: x
            self.denormalize_output = lambda x: x
            
    def call(self, x, training=None, mask=None):
        """Forward pass: run autoregressive prediction."""
        batch_size = tf.shape(x)[0]
        
        state_indices_tf = tf.constant(self.state_indices, dtype=tf.int32)
        initial_state = tf.gather(x[:, 0, :], state_indices_tf, axis=1)
        
        control_indices_tf = tf.constant(self.control_indices, dtype=tf.int32)
        control_sequence = tf.gather(x, control_indices_tf, axis=2)
        
        model_initial_input = initial_state
        
        if self.differential_network and self.dmah is not None:
            dm_state_list = []
            for out_var in self.state_variables:
                found = False
                for i, inp in enumerate(self.state_inputs):
                    inp_base = re.sub(r'_-?\d+$', '', inp)
                    if inp_base == out_var:
                        dm_state_list.append(initial_state[:, i:i+1])
                        found = True
                        break
                if not found:
                    dm_state_list.append(tf.zeros([batch_size, 1], dtype=initial_state.dtype))
            dm_state_init = tf.concat(dm_state_list, axis=1)
        else:
            dm_state_init = None
            
        outputs = self.AL.run(
            initial_input=model_initial_input,
            external_input_left=control_sequence,
            dm_state_init=dm_state_init,
        )
        
        return outputs
    
    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.base_net.save(filepath, overwrite=overwrite, save_format=save_format, **kwargs)
        
    def save_weights(self, filepath, overwrite=True, save_format=None, **kwargs):
        self.base_net.save_weights(filepath, overwrite=overwrite, save_format=save_format, **kwargs)
        
    def load_weights(self, filepath, **kwargs):
        self.base_net.load_weights(filepath, **kwargs)


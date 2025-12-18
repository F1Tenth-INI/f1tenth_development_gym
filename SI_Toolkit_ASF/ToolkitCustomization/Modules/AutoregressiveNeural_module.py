"""
AutoregressiveNeural_module.py

A custom Keras model for autoregressive training of neural networks.
Supports both training from scratch and fine-tuning pretrained models.

Usage in config_training.yml:
    modeling:
        # Train new network from scratch (architecture after module name):
        NET_NAME: "Custom-AutoregressiveNeural_module-Dense64H1_64H2"
        NET_NAME: "Custom-AutoregressiveNeural_module-Dense128H1_128H2"
        NET_NAME: "Custom-AutoregressiveNeural_module-Dense64H1_128H2_64H3"
        
        # Fine-tune pretrained network (Pretrained-<model_name>):
        NET_NAME: "Custom-AutoregressiveNeural_module-Pretrained-Dense-9IN-64H1-64H2-8OUT-0"
        
    training_default:
        POST_WASH_OUT_LEN: 20  # The horizon length for autoregressive training
        WASH_OUT_LEN: 0
"""

import os
import re
import glob
import numpy as np
import tensorflow as tf

from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net
from SI_Toolkit.Predictors.autoregression import autoregression_loop, differential_model_autoregression_helper


class AutoregressiveBase(tf.keras.Model):
    """
    Base class for autoregressive training of neural networks.
    
    Handles both:
    - Training new networks from scratch
    - Fine-tuning pretrained networks
    
    The autoregression logic is shared (same as predictor_autoregressive_neural).
    """
    
    def __init__(self, time_series_length, batch_size, net_info, name=None, **kwargs):
        super().__init__(**kwargs)
        
        self.lib = TensorFlowLibrary()
        self.batch_size = batch_size
        self.horizon = time_series_length
        self.net_info = net_info
        
        # Get timestep
        try:
            from utilities.Settings import Settings
            self.dt = Settings.TIMESTEP_PLANNER
        except (ImportError, AttributeError):
            self.dt = 0.02
            print(f"[{self.__class__.__name__}] Using default dt={self.dt}s")
        
        # Handle missing attributes for normalization
        if not hasattr(net_info, 'parent_net_name'):
            net_info.parent_net_name = 'Network trained from scratch'
        if not hasattr(net_info, 'path_to_net'):
            net_info.path_to_net = None
        
        # Subclasses must set self.base_net before calling _setup_autoregression
        self._create_or_load_network(net_info)
        
        # Load normalization info
        try:
            self.normalization_info = get_norm_info_for_net(net_info, copy_files=False)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Could not load normalization info: {e}")
            self.normalization_info = None
        
        # Setup autoregression (shared logic)
        self._setup_autoregression(net_info)
    
    def _create_or_load_network(self, net_info):
        """Override in subclasses to create or load the base network."""
        raise NotImplementedError("Subclasses must implement _create_or_load_network")
    
    def _setup_autoregression(self, net_info):
        """
        Setup autoregression components.
        Uses same logic as predictor_autoregressive_neural.
        Pre-computes all indices as TF constants for graph efficiency.
        """
        # Check if differential network (outputs start with D_)
        self.differential_network = any('D_' in out for out in net_info.outputs)
        
        # Strip time suffixes for comparison
        def strip_suffix(name):
            return re.sub(r'_-?\d+$', '', name)
        
        # Determine state variables from outputs
        if self.differential_network:
            self.state_variables = [strip_suffix(x[2:]) if x.startswith('D_') else strip_suffix(x) 
                                   for x in net_info.outputs]
        else:
            self.state_variables = [strip_suffix(x) for x in net_info.outputs]
        
        # Identify control vs state inputs (same logic as predictor)
        self.control_inputs = []
        self.state_inputs = []
        for inp in net_info.inputs:
            base_name = strip_suffix(inp)
            if base_name in self.state_variables:
                self.state_inputs.append(inp)
            else:
                self.control_inputs.append(inp)
        
        self.control_indices = [list(net_info.inputs).index(c) for c in self.control_inputs]
        self.state_indices = [list(net_info.inputs).index(s) for s in self.state_inputs]
        
        # Pre-compute indices as TF constants for graph efficiency
        self._state_indices_tf = tf.constant(self.state_indices, dtype=tf.int32)
        self._control_indices_tf = tf.constant(self.control_indices, dtype=tf.int32)
        
        # Pre-compute dm_state mapping: for each output state variable, 
        # find index in state_inputs or -1 if not found (will be zero-padded)
        if self.differential_network:
            dm_state_mapping = []
            state_inputs_base = [strip_suffix(inp) for inp in self.state_inputs]
            for out_var in self.state_variables:
                if out_var in state_inputs_base:
                    dm_state_mapping.append(state_inputs_base.index(out_var))
                else:
                    dm_state_mapping.append(-1)  # Sentinel for zero-padding
            self._dm_state_mapping = dm_state_mapping
            self._dm_state_has_missing = any(m == -1 for m in dm_state_mapping)
            # Pre-compute gather indices for known state variables
            self._dm_known_indices = tf.constant(
                [m for m in dm_state_mapping if m != -1], dtype=tf.int32
            )
            self._dm_missing_count = sum(1 for m in dm_state_mapping if m == -1)
        
        print(f"[{self.__class__.__name__}] Controls ({len(self.control_inputs)}): {self.control_inputs}")
        print(f"[{self.__class__.__name__}] State inputs ({len(self.state_inputs)}): {self.state_inputs}")
        print(f"[{self.__class__.__name__}] State variables: {self.state_variables}")
        print(f"[{self.__class__.__name__}] Differential network: {self.differential_network}")
        
        # Setup differential model helper (same as predictor)
        self.dmah = None
        if self.differential_network:
            self.dmah = differential_model_autoregression_helper(
                inputs=net_info.inputs,
                outputs=net_info.outputs,
                normalization_info=self.normalization_info,
                dt=self.dt,
                lib=self.lib,
                state_variables=np.array(self.state_variables),
            )
        
        # Check if base network expects time dimension
        try:
            expected_shape = self.base_net.input_shape
            self.base_net_expects_time_dim = len(expected_shape) == 3
            print(f"[{self.__class__.__name__}] Base net input shape: {expected_shape}")
        except Exception:
            self.base_net_expects_time_dim = False
        
        # Create model wrapper for autoregression loop
        # Capture the flag as a Python bool to avoid graph issues
        expects_time_dim = self.base_net_expects_time_dim
        base_net = self.base_net
        
        if expects_time_dim:
            def model_wrapper(x):
                output = base_net(x)
                return tf.squeeze(output, axis=1) if len(output.shape) == 3 else output
        else:
            def model_wrapper(x):
                return base_net(tf.squeeze(x, axis=1))
        
        # Create autoregression loop (same as predictor)
        self.AL = autoregression_loop(
            model=model_wrapper,
            model_inputs_len=len(net_info.inputs),
            model_outputs_len=len(net_info.outputs),
            lib=self.lib,
            differential_model_autoregression_helper_instance=self.dmah,
        )
    
    def call(self, x, training=None, mask=None):
        """
        Forward pass: run autoregressive prediction.
        Optimized for graph compilation - no Python loops or re.sub calls.
        
        Args:
            x: Input tensor [batch, horizon, num_inputs]
               
        Returns:
            outputs: Tensor [batch, horizon, num_outputs]
        """
        # Extract initial state from first timestep (using pre-computed indices)
        initial_state = tf.gather(x[:, 0, :], self._state_indices_tf, axis=1)
        
        # Extract control inputs for all timesteps  
        control_sequence = tf.gather(x, self._control_indices_tf, axis=2)
        
        # Prepare dm_state for differential models (vectorized, no Python loops)
        dm_state_init = None
        if self.differential_network and self.dmah is not None:
            if self._dm_state_has_missing:
                # Build dm_state by gathering known values and inserting zeros for missing
                # This is done using tf.tensor_scatter_nd_update for efficiency
                batch_size = tf.shape(initial_state)[0]
                num_outputs = len(self.state_variables)
                
                # Start with zeros
                dm_state_init = tf.zeros([batch_size, num_outputs], dtype=initial_state.dtype)
                
                # Gather known values
                if len(self._dm_known_indices) > 0:
                    known_values = tf.gather(initial_state, self._dm_known_indices, axis=1)
                    # Find positions where we need to insert known values
                    insert_indices = tf.constant(
                        [[i] for i, m in enumerate(self._dm_state_mapping) if m != -1], 
                        dtype=tf.int32
                    )
                    # Use transpose to update columns
                    dm_state_init = tf.transpose(
                        tf.tensor_scatter_nd_update(
                            tf.transpose(dm_state_init),
                            insert_indices,
                            tf.transpose(known_values)
                        )
                    )
            else:
                # All state variables are in inputs - simple gather
                dm_state_mapping_tf = tf.constant(self._dm_state_mapping, dtype=tf.int32)
                dm_state_init = tf.gather(initial_state, dm_state_mapping_tf, axis=1)
        
        # Run autoregression loop
        outputs = self.AL.run(
            initial_input=initial_state,
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


class AutoregressiveDense(AutoregressiveBase):
    """
    Creates a new Dense network from scratch for autoregressive training.
    
    Usage: NET_NAME: "Custom-AutoregressiveDense64H1_64H2"
    """
    
    # Default architecture if not specified
    DEFAULT_ARCHITECTURE = [64, 64]
    
    def __init__(self, time_series_length, batch_size, net_info, 
                 architecture=None, name=None, **kwargs):
        """
        Args:
            architecture: List of hidden layer sizes, e.g., [64, 64] or [128, 256, 128]
                         If None, uses DEFAULT_ARCHITECTURE
        """
        self.architecture = architecture or self.DEFAULT_ARCHITECTURE
        super().__init__(time_series_length, batch_size, net_info, name=name, **kwargs)
    
    def _create_or_load_network(self, net_info):
        """Create a new Dense network from scratch."""
        inputs_len = len(net_info.inputs)
        outputs_len = len(net_info.outputs)
        
        # Build Sequential model (no time dimension for Dense)
        self.base_net = tf.keras.Sequential()
        self.base_net.add(tf.keras.Input(shape=(inputs_len,)))
        
        for i, units in enumerate(self.architecture):
            self.base_net.add(tf.keras.layers.Dense(units=units, name=f'layers_{i}'))
            self.base_net.add(tf.keras.layers.Activation('tanh'))
        
        self.base_net.add(tf.keras.layers.Dense(
            units=outputs_len, activation='linear', name='layers_output'
        ))
        
        print(f"[{self.__class__.__name__}] Created network with hidden layers {self.architecture}")


class AutoregressivePretrained(AutoregressiveBase):
    """
    Loads a pretrained network for autoregressive fine-tuning.
    
    Usage: NET_NAME: "Custom-AutoregressivePretrained-Dense-9IN-64H1-64H2-8OUT-0"
    
    The model name is extracted from NET_NAME after "AutoregressivePretrained-".
    """
    
    def __init__(self, time_series_length, batch_size, net_info, name=None, **kwargs):
        # Extract pretrained model name before calling super().__init__
        self.pretrained_model_name = self._extract_pretrained_model_name(net_info)
        
        if self.pretrained_model_name is None:
            raise ValueError(
                "No pretrained model specified. Use NET_NAME format:\n"
                "  NET_NAME: 'Custom-AutoregressivePretrained-Dense-9IN-64H1-64H2-8OUT-0'\n"
                "Where 'Dense-9IN-64H1-64H2-8OUT-0' is your pretrained model name."
            )
        
        super().__init__(time_series_length, batch_size, net_info, name=name, **kwargs)
    
    def _extract_pretrained_model_name(self, net_info):
        """Extract pretrained model name from net_info.net_name."""
        if hasattr(net_info, 'net_name') and net_info.net_name:
            net_name = net_info.net_name
            prefix = "AutoregressivePretrained-"
            if prefix in net_name:
                idx = net_name.index(prefix) + len(prefix)
                return net_name[idx:]
        
        # Fallback options
        if hasattr(net_info, 'pretrained_model') and net_info.pretrained_model:
            return net_info.pretrained_model
        if hasattr(net_info, 'parent_net_name'):
            if net_info.parent_net_name and net_info.parent_net_name != 'Network trained from scratch':
                return net_info.parent_net_name
        return None
    
    def _create_or_load_network(self, net_info):
        """Load a pretrained network."""
        import yaml
        
        print(f"[{self.__class__.__name__}] Loading pretrained model: {self.pretrained_model_name}")
        
        # Find model directory
        if hasattr(net_info, 'path_to_models') and net_info.path_to_models:
            model_dir = net_info.path_to_models
        elif hasattr(net_info, 'path_to_experiment_folders'):
            model_dir = os.path.join(net_info.path_to_experiment_folders, 'Models')
        else:
            model_dir = './SI_Toolkit_ASF/Experiments/Experiments_05_12_2025_BackToFront_shifted/Models'
        
        model_folder = os.path.join(model_dir, self.pretrained_model_name)
        
        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"Pretrained model folder not found at: {model_folder}")
        
        # Load model (try .keras, .h5, or weights)
        keras_path = os.path.join(model_folder, f"{self.pretrained_model_name}.keras")
        h5_path = os.path.join(model_folder, f"{self.pretrained_model_name}.h5")
        weights_path = os.path.join(model_folder, "ckpt.weights.h5")
        
        if os.path.exists(keras_path):
            print(f"[{self.__class__.__name__}] Loading .keras model from: {keras_path}")
            self.base_net = tf.keras.models.load_model(keras_path, compile=False)
        elif os.path.exists(h5_path):
            print(f"[{self.__class__.__name__}] Loading .h5 model from: {h5_path}")
            self.base_net = tf.keras.models.load_model(h5_path, compile=False)
        elif os.path.exists(weights_path):
            print(f"[{self.__class__.__name__}] Loading weights from: {weights_path}")
            self._build_and_load_weights(net_info, model_folder, weights_path)
        else:
            raise FileNotFoundError(f"No model file found in {model_folder}")
        
        # Load normalization info
        ni_files = glob.glob(os.path.join(model_folder, 'NI_*.csv'))
        if ni_files:
            net_info.path_to_normalization_info = ni_files[0]
            net_info.parent_net_name = self.pretrained_model_name
            print(f"[{self.__class__.__name__}] Found normalization info: {ni_files[0]}")
        
        # Load inputs/outputs from training config
        config_path = os.path.join(model_folder, 'config_training.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            training_config = config.get('training_default', {})
            
            state_inputs = training_config.get('state_inputs', [])
            control_inputs = training_config.get('control_inputs', [])
            outputs = training_config.get('outputs', [])
            
            if state_inputs or control_inputs:
                net_info.inputs = control_inputs + state_inputs
            if outputs:
                net_info.outputs = outputs
        
        print(f"[{self.__class__.__name__}] Loaded: {self.pretrained_model_name}")
        print(f"[{self.__class__.__name__}]   Inputs: {net_info.inputs}")
        print(f"[{self.__class__.__name__}]   Outputs: {net_info.outputs}")
    
    def _build_and_load_weights(self, net_info, model_folder, weights_path):
        """Build model architecture from name and load weights."""
        parts = self.pretrained_model_name.split('-')
        
        inputs_len = None
        outputs_len = None
        h_sizes = []
        
        for part in parts:
            if part.endswith('IN'):
                inputs_len = int(part[:-2])
            elif part.endswith('OUT'):
                outputs_len = int(part[:-3])
            elif any(part.endswith(f'H{i}') for i in range(1, 6)):
                h_sizes.append(int(part[:-2]))
        
        if inputs_len is None or outputs_len is None:
            raise ValueError(f"Could not parse architecture from: {self.pretrained_model_name}")
        
        self.base_net = tf.keras.Sequential()
        self.base_net.add(tf.keras.Input(shape=(inputs_len,)))
        
        for i, units in enumerate(h_sizes):
            self.base_net.add(tf.keras.layers.Dense(units=units, name=f'layers_{i}'))
            self.base_net.add(tf.keras.layers.Activation('tanh'))
        
        self.base_net.add(tf.keras.layers.Dense(units=outputs_len, activation='linear', name='layers_output'))
        self.base_net.load_weights(weights_path)
        
        print(f"[{self.__class__.__name__}] Built model: {inputs_len}IN -> {h_sizes} -> {outputs_len}OUT")


# =============================================================================
# Convenience classes for common architectures
# These are thin wrappers that just specify the architecture
# =============================================================================

class AutoregressiveDense64H1_64H2(AutoregressiveDense):
    """64-64 hidden layers."""
    DEFAULT_ARCHITECTURE = [64, 64]

class AutoregressiveDense128H1_128H2(AutoregressiveDense):
    """128-128 hidden layers."""
    DEFAULT_ARCHITECTURE = [128, 128]

class AutoregressiveDense64H1_128H2_64H3(AutoregressiveDense):
    """64-128-64 hidden layers."""
    DEFAULT_ARCHITECTURE = [64, 128, 64]

class AutoregressiveDense32H1_64H2_32H3(AutoregressiveDense):
    """32-64-32 hidden layers."""
    DEFAULT_ARCHITECTURE = [32, 64, 32]

class AutoregressiveDense256H1_256H2(AutoregressiveDense):
    """256-256 hidden layers."""
    DEFAULT_ARCHITECTURE = [256, 256]


# =============================================================================
# Dispatcher class - this is what the module loader finds
# Parses NET_NAME and creates the appropriate class
# =============================================================================

# Architecture name to hidden layer sizes mapping
ARCHITECTURE_MAP = {
    'Dense64H1_64H2': [64, 64],
    'Dense128H1_128H2': [128, 128],
    'Dense256H1_256H2': [256, 256],
    'Dense64H1_128H2_64H3': [64, 128, 64],
    'Dense32H1_64H2_32H3': [32, 64, 32],
    'Dense32H1_32H2': [32, 32],
    'Dense128H1_256H2_128H3': [128, 256, 128],
}


def AutoregressiveNeural_module(time_series_length, batch_size, net_info, **kwargs):
    """
    Dispatcher function that the module loader calls.
    
    Parses net_info.net_name to determine what to create:
    - "Custom-AutoregressiveNeural_module-Dense64H1_64H2" 
        → AutoregressiveDense with [64, 64]
    - "Custom-AutoregressiveNeural_module-Pretrained-Dense-9IN-64H1-64H2-8OUT-0"
        → AutoregressivePretrained loading that model
    
    Returns:
        An instance of the appropriate autoregressive model class
    """
    net_name = net_info.net_name
    
    # Parse: Custom-AutoregressiveNeural_module-<suffix>
    parts = net_name.split('-')
    if len(parts) < 3:
        raise ValueError(
            f"Invalid NET_NAME format: {net_name}\n"
            "Expected: Custom-AutoregressiveNeural_module-<Dense64H1_64H2|Pretrained-...>"
        )
    
    # Get everything after "AutoregressiveNeural_module-"
    suffix_start = net_name.find('AutoregressiveNeural_module-')
    if suffix_start == -1:
        raise ValueError(f"Could not parse NET_NAME: {net_name}")
    
    suffix = net_name[suffix_start + len('AutoregressiveNeural_module-'):]
    
    print(f"[AutoregressiveNeural_module] Parsing suffix: '{suffix}'")
    
    # Case 1: Pretrained model
    if suffix.startswith('Pretrained-'):
        pretrained_model_name = suffix[len('Pretrained-'):]
        print(f"[AutoregressiveNeural_module] Mode: Fine-tune pretrained '{pretrained_model_name}'")
        
        # Create a modified net_info with the pretrained model name embedded
        # AutoregressivePretrained expects format: Custom-AutoregressivePretrained-<model>
        net_info.net_name = f"Custom-AutoregressivePretrained-{pretrained_model_name}"
        
        return AutoregressivePretrained(time_series_length, batch_size, net_info, **kwargs)
    
    # Case 2: New network from scratch
    else:
        architecture_name = suffix
        print(f"[AutoregressiveNeural_module] Mode: Train from scratch '{architecture_name}'")
        
        # Look up architecture
        if architecture_name in ARCHITECTURE_MAP:
            architecture = ARCHITECTURE_MAP[architecture_name]
        else:
            # Try to parse architecture from name (e.g., Dense64H1_64H2)
            architecture = _parse_architecture_from_name(architecture_name)
        
        if architecture is None:
            available = ', '.join(ARCHITECTURE_MAP.keys())
            raise ValueError(
                f"Unknown architecture: {architecture_name}\n"
                f"Available: {available}\n"
                f"Or use format like Dense64H1_64H2, Dense128H1_256H2_128H3, etc."
            )
        
        return AutoregressiveDense(
            time_series_length, batch_size, net_info, 
            architecture=architecture, **kwargs
        )


def _parse_architecture_from_name(name):
    """
    Parse hidden layer sizes from architecture name.
    
    Examples:
        'Dense64H1_64H2' → [64, 64]
        'Dense128H1_256H2_128H3' → [128, 256, 128]
    """
    if not name.startswith('Dense'):
        return None
    
    # Extract parts like "64H1", "128H2", etc.
    import re
    pattern = r'(\d+)H\d+'
    matches = re.findall(pattern, name)
    
    if matches:
        return [int(m) for m in matches]
    return None

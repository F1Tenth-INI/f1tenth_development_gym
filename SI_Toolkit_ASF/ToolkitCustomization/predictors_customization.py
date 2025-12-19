import numpy as np
import re

from utilities.Settings import Settings
from utilities.state_utilities import STATE_VARIABLES, STATE_INDICES, CONTROL_INPUTS

from SI_Toolkit.computation_library import NumpyLibrary

environment_name = Settings.ENVIRONMENT_NAME
model_of_car_dynamics = Settings.ODE_MODEL_OF_CAR_DYNAMICS
car_parameter_file = Settings.CONTROLLER_CAR_PARAMETER_FILE

# Full state order (alphabetically sorted, as expected by car_model)
FULL_STATE_ORDER = list(STATE_VARIABLES)



class next_state_predictor_ODE():

    def __init__(self,
                 dt,
                 intermediate_steps,
                 lib,
                 batch_size=1,
                 variable_parameters=None,
                 control_inputs=None,
                 state_variables=None,
                 disable_individual_compilation=False,
                 **kwargs,
                 ):
        self.lib = lib
        self.intermediate_steps = int(intermediate_steps)
        self.t_step = float(dt / float(self.intermediate_steps))
        self.variable_parameters = variable_parameters

        # State variables (from config_predictors.yml)
        if state_variables is None:
            raise ValueError("state_variables must be provided (define in config_predictors.yml)")
        self.state_variables = np.array(state_variables)
        self.state_indices = {x: i for i, x in enumerate(self.state_variables)}

        # Control inputs (from config_predictors.yml)
        if control_inputs is None:
            raise ValueError("control_inputs must be provided (define in config_predictors.yml)")
        self.control_inputs = np.array(control_inputs)
        self.control_indices = {x: i for i, x in enumerate(self.control_inputs)}
        self.num_control_inputs = len(self.control_inputs)
        
        # Check if mu is in control inputs and pre-compute all indices
        self.mu_from_control_input = 'mu' in self.control_indices
        if self.mu_from_control_input:
            self.mu_control_idx = self.control_indices['mu']
            # Pre-compute base control indices (handles time-suffixed names like 'angular_control_-4')
            self.angular_control_idx = self._get_control_idx('angular_control')
            self.translational_control_idx = self._get_control_idx('translational_control')

        if "core_dynamics_only" in kwargs and kwargs["core_dynamics_only"] is True:
            self.core_dynamics_only = True
        else:
            self.core_dynamics_only = False

        if environment_name == 'Car':
            from SI_Toolkit_ASF.car_model import car_model
            self.env = car_model(
                model_of_car_dynamics=model_of_car_dynamics,
                batch_size=batch_size,
                car_parameter_file=car_parameter_file,
                dt=dt,
                computation_lib=lib,
                intermediate_steps=intermediate_steps,
                                 )  # Environment model, keeping car ODEs
        else:
            raise NotImplementedError('{} not yet implemented in next_state_predictor_ODE_tf'.format(Settings.ENVIRONMENT_NAME))

        self.params = self.env.car_parameters

        if disable_individual_compilation:
            self.step = self._step
        else:
            from SI_Toolkit.Compile import CompileAdaptive  # Lazy import
            self.step = CompileAdaptive(self._step)

    def _step(self, s, Q):
        # Handle mu from different sources (indices pre-computed in __init__)
        if self.mu_from_control_input:
            # mu from control inputs (e.g., CSV for Brunton test)
            self.env._mu = Q[:, self.mu_control_idx]
            # Extract only base controls for car_model (using pre-computed indices)
            Q_for_car = self.lib.stack([Q[:, self.angular_control_idx], Q[:, self.translational_control_idx]], axis=1)
        elif self.variable_parameters is not None and hasattr(self.variable_parameters, 'mu'):
            # mu from MPC variable_parameters
            self.env._mu = self.variable_parameters.mu
            Q_for_car = Q
        else:
            Q_for_car = Q

        if self.core_dynamics_only:
            s_next = self.env.step_dynamics_core(s, Q_for_car)
        else:
            s_next = self.env.step_dynamics(s, Q_for_car)

        return s_next

    def _get_control_idx(self, base_name):
        """Get index for control, handling time-suffixed names like 'angular_control_-4'."""
        if base_name in self.control_indices:
            return self.control_indices[base_name]
        for key in self.control_indices:
            if base_name in key:
                return self.control_indices[key]
        raise KeyError(f"Control '{base_name}' not found in control_inputs: {list(self.control_indices.keys())}")



class StateAugmenter:
    """
    Unified state augmentation for both NumPy batch processing and TF autoregression.
    
    Uses computation library abstraction to work with NumPy, TensorFlow, or PyTorch.
    
    Two usage modes:
    1. For autoregression (append mode): augment() appends missing features to output
    2. For batch reordering (reorder mode): augment_to_target_order() reorders to target feature order
    """
    
    def __init__(self, input_features, lib=None, target_features=None, 
                 disable_individual_compilation=False, strip_derivative_prefix=False):
        """
        Args:
            input_features: list/array of input feature names
            lib: computation library (NumpyLibrary, TensorFlowLibrary, etc.). If None, uses NumpyLibrary
            target_features: target feature order. If None, uses FULL_STATE_ORDER
            disable_individual_compilation: if True, don't compile augment method
            strip_derivative_prefix: if True, strip 'D_' prefix and time suffixes from input_features
        """
        if lib is None:
            lib = NumpyLibrary()
        self.lib = lib
        
        # Process input features
        if strip_derivative_prefix:
            # Strip D_ prefix and time suffixes (e.g., _-1) for differential networks
            self.input_features = [re.sub(r'_-?\d+$', '', x[2:] if x.startswith('D_') else x) 
                                   for x in input_features]
        else:
            self.input_features = list(input_features)
        
        self.input_indices = {key: idx for idx, key in enumerate(self.input_features)}
        self.target_features = target_features if target_features is not None else FULL_STATE_ORDER
        
        # Determine what needs augmentation
        self.features_augmentation = []
        self.indices_augmentation = []  # For backward compatibility with predictor_output_augmentation
        
        for feat in self.target_features:
            if feat not in self.input_features:
                # Check if we can compute this feature
                can_compute = False
                if feat == 'pose_theta' and 'pose_theta_sin' in self.input_features and 'pose_theta_cos' in self.input_features:
                    can_compute = True
                elif feat == 'pose_theta_sin' and 'pose_theta' in self.input_features:
                    can_compute = True
                elif feat == 'pose_theta_cos' and 'pose_theta' in self.input_features:
                    can_compute = True
                elif feat in ['linear_vel_y', 'angular_vel_z', 'linear_vel_x', 'slip_angle', 'steering_angle']:
                    can_compute = True  # Set to zero
                
                if can_compute and feat in STATE_INDICES:
                    self.features_augmentation.append(feat)
                    self.indices_augmentation.append(STATE_INDICES[feat])
        
        self.augmentation_len = len(self.features_augmentation)
        
        # Pre-compute indices for trigonometric conversions
        if 'pose_theta' in self.input_features:
            self.index_pose_theta = self.lib.to_tensor(self.input_indices['pose_theta'], self.lib.int32)
        if 'pose_theta_sin' in self.input_features:
            self.index_pose_theta_sin = self.lib.to_tensor(self.input_indices['pose_theta_sin'], self.lib.int32)
        if 'pose_theta_cos' in self.input_features:
            self.index_pose_theta_cos = self.lib.to_tensor(self.input_indices['pose_theta_cos'], self.lib.int32)
        
        # Pre-compute indices for linear_vel_y reconstruction from slip_angle and v_x
        self.can_compute_linear_vel_y = ('slip_angle' in self.input_features and 'linear_vel_x' in self.input_features)
        if self.can_compute_linear_vel_y:
            self.index_slip_angle = self.lib.to_tensor(self.input_indices['slip_angle'], self.lib.int32)
            self.index_linear_vel_x = self.lib.to_tensor(self.input_indices['linear_vel_x'], self.lib.int32)
        
        # Set up compilation
        if disable_individual_compilation or self.lib.lib == 'Numpy':
            self.augment = self._augment
        else:
            from SI_Toolkit.Compile import CompileAdaptive
            self.augment = CompileAdaptive(self._augment)
    
    def get_indices_augmentation(self):
        return self.indices_augmentation
    
    def get_features_augmentation(self):
        return self.features_augmentation
    
    def _augment(self, states):
        """
        Append missing features to states (for autoregression compatibility).
        
        Args:
            states: tensor of shape [batch, time, features]
            
        Returns:
            augmented states with features appended
        """
        output = states
        
        if 'angular_vel_z' in self.features_augmentation:
            angular_vel_z = self.lib.zeros_like(states[:, :, -1:])
            output = self.lib.concat([output, angular_vel_z], axis=-1)
        
        if 'linear_vel_x' in self.features_augmentation:
            linear_vel_x = self.lib.zeros_like(states[:, :, -1:])
            output = self.lib.concat([output, linear_vel_x], axis=-1)
        
        if 'linear_vel_y' in self.features_augmentation:
            # Compute from slip_angle and v_x if available: v_y = tan(beta) * v_x
            if self.can_compute_linear_vel_y:
                v_x = states[..., self.index_linear_vel_x]
                beta = states[..., self.index_slip_angle]
                # Avoid division issues when v_x is near zero
                v_x_safe = self.lib.where(self.lib.abs(v_x) < 1.0e-3, 
                                          self.lib.to_tensor(1.0e-3, v_x.dtype), v_x)
                linear_vel_y = (self.lib.tan(beta) * v_x_safe)[:, :, self.lib.newaxis]
            else:
                linear_vel_y = self.lib.zeros_like(states[:, :, -1:])
            output = self.lib.concat([output, linear_vel_y], axis=-1)
        
        if 'pose_theta' in self.features_augmentation:
            pose_theta = self.lib.atan2(
                states[..., self.index_pose_theta_sin],
                states[..., self.index_pose_theta_cos]
            )[:, :, self.lib.newaxis]
            output = self.lib.concat([output, pose_theta], axis=-1)
        
        if 'pose_theta_sin' in self.features_augmentation:
            pose_theta_sin = self.lib.sin(states[..., self.index_pose_theta])[:, :, self.lib.newaxis]
            output = self.lib.concat([output, pose_theta_sin], axis=-1)
        
        if 'pose_theta_cos' in self.features_augmentation:
            pose_theta_cos = self.lib.cos(states[..., self.index_pose_theta])[:, :, self.lib.newaxis]
            output = self.lib.concat([output, pose_theta_cos], axis=-1)
        
        if 'slip_angle' in self.features_augmentation:
            slip_angle = self.lib.zeros_like(states[:, :, -1:])
            output = self.lib.concat([output, slip_angle], axis=-1)
        
        if 'steering_angle' in self.features_augmentation:
            steering_angle = self.lib.zeros_like(states[:, :, -1:])
            output = self.lib.concat([output, steering_angle], axis=-1)
        
        return output
    
    def augment_to_target_order(self, states, verbose=False):
        """
        Augment and reorder states to match target_features order.
        
        This is for batch processing where output needs specific feature ordering.
        
        Args:
            states: numpy array of shape [batch, time, features] or [time, features]
            verbose: print debug info
            
        Returns:
            augmented_states: array with target_features order
            target_features: list of feature names
        """
        # Check if augmentation needed
        missing = [f for f in self.target_features if f not in self.input_features]
        if not missing:
            return states, list(self.input_features)
        
        if verbose:
            print(f"  Augmenting states with missing features: {missing}")
        
        # Handle 2D vs 3D
        is_3d = len(states.shape) == 3
        if not is_3d:
            states = states[np.newaxis, :, :]
        
        batch_size, time_len, _ = states.shape
        augmented = np.zeros((batch_size, time_len, len(self.target_features)), dtype=np.float32)
        
        for target_idx, feat in enumerate(self.target_features):
            if feat in self.input_indices:
                augmented[:, :, target_idx] = states[:, :, self.input_indices[feat]]
            elif feat == 'pose_theta' and 'pose_theta_sin' in self.input_indices and 'pose_theta_cos' in self.input_indices:
                augmented[:, :, target_idx] = np.arctan2(
                    states[:, :, self.input_indices['pose_theta_sin']],
                    states[:, :, self.input_indices['pose_theta_cos']]
                )
                if verbose:
                    print(f"    Added pose_theta = atan2(sin, cos) at index {target_idx}")
            elif feat == 'pose_theta_sin' and 'pose_theta' in self.input_indices:
                augmented[:, :, target_idx] = np.sin(states[:, :, self.input_indices['pose_theta']])
                if verbose:
                    print(f"    Added pose_theta_sin = sin(theta) at index {target_idx}")
            elif feat == 'pose_theta_cos' and 'pose_theta' in self.input_indices:
                augmented[:, :, target_idx] = np.cos(states[:, :, self.input_indices['pose_theta']])
                if verbose:
                    print(f"    Added pose_theta_cos = cos(theta) at index {target_idx}")
            elif feat == 'linear_vel_y':
                # Compute from slip_angle and linear_vel_x if available: v_y = tan(beta) * v_x
                # This is critical for Pacejka dynamics which use v_y directly.
                if 'slip_angle' in self.input_indices and 'linear_vel_x' in self.input_indices:
                    v_x = states[:, :, self.input_indices['linear_vel_x']]
                    beta = states[:, :, self.input_indices['slip_angle']]
                    # Avoid division issues when v_x is near zero
                    v_x_safe = np.where(np.abs(v_x) < 1.0e-3, 1.0e-3, v_x)
                    augmented[:, :, target_idx] = np.tan(beta) * v_x_safe
                    if verbose:
                        print(f"    Added linear_vel_y = tan(slip_angle) * v_x at index {target_idx}")
                else:
                    augmented[:, :, target_idx] = 0.0
                    if verbose:
                        print(f"    Added linear_vel_y = 0 at index {target_idx} (slip_angle or v_x unavailable)")
            else:
                augmented[:, :, target_idx] = 0.0
                if verbose:
                    print(f"    Added {feat} = 0 at index {target_idx}")
        
        if verbose:
            print(f"    Augmented shape: {augmented.shape}")
        
        if not is_3d:
            augmented = augmented[0]
        
        return augmented, list(self.target_features)


# Backward compatibility alias
class predictor_output_augmentation(StateAugmenter):
    """Backward compatible wrapper for StateAugmenter."""
    
    def __init__(self, net_info, lib=NumpyLibrary(), disable_individual_compilation=False, differential_network=False):
        super().__init__(
            input_features=net_info.outputs,
            lib=lib,
            target_features=None,  # Use FULL_STATE_ORDER
            disable_individual_compilation=disable_individual_compilation,
            strip_derivative_prefix=True  # Differential networks have D_ prefix
        )


def augment_states_numpy(states, input_features, target_features=None, verbose=False):
    """
    Convenience function for NumPy batch augmentation.
    
    Args:
        states: numpy array of shape [batch, time, features] or [time, features]
        input_features: list/array of feature names in states
        target_features: list of target feature names (default: FULL_STATE_ORDER)
        verbose: print debug info
        
    Returns:
        augmented_states: numpy array with target features
        augmented_features: list of feature names after augmentation
    """
    augmenter = StateAugmenter(
        input_features=input_features,
        lib=NumpyLibrary(),
        target_features=target_features,
        disable_individual_compilation=True
    )
    return augmenter.augment_to_target_order(states, verbose=verbose)


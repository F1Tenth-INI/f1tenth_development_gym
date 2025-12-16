import os
import sys

# Add path this files parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from sim.f110_sim.envs.car_model_jax import car_steps_sequential_jax
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import *
from predictor import Predictor, predictor_forward_jit
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import time

from train import INPUT_COLS, OUTPUT_COLS

class DynamicsModelResidual:
    def __init__(self):
        self.car_params = VehicleParameters().to_np_array()
        self.dt = 0.04
        self.horizon = 1
        self.history_length = 10
        # Load residual neural network model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'models')
        self.predictor = Predictor(model_dir)

        # Init rolling window of state and control history
        self.state_history = np.zeros((self.history_length, 10), dtype=np.float32)
        self.control_history = np.zeros((self.history_length, 2), dtype=np.float32)


    def set_history(self, state_history, control_history):
        self.state_history = state_history
        self.control_history = control_history


    def predict_sequence(self, initial_state, control_sequence, initial_state_history=None, initial_control_history=None):
        """JIT-compiled sequence prediction using jax.lax.scan."""
        # Get predictor params once
        predictor_params = self.predictor.get_params_jax()
        norm_params = self.predictor.get_norm_params_jax()
        
        # Convert to JAX arrays
        initial_state_jax = jnp.array(initial_state)
        control_sequence_jax = jnp.array(control_sequence)
        
        # Use provided history or default to instance history
      
        state_history_jax = jnp.array(self.state_history) 
        control_history_jax = jnp.array(self.control_history)
        car_params_jax = jnp.array(self.car_params)
        
        return predict_sequence_jax(
            initial_state_jax, control_sequence_jax,
            state_history_jax, control_history_jax,
            predictor_params, norm_params, car_params_jax,
            self.history_length, self.dt
        )
        
    def predict(self, state, control):
        """Non-JIT wrapper for single step prediction."""
        next_states = self.predict_sequence(state, control[None, :])
        return next_states[0]


@partial(jax.jit, static_argnames=["history_length", "dt"])
def predict_sequence_jax(initial_state, control_sequence, state_history, control_history,
                         predictor_params, norm_params, car_params, history_length, dt):
    """JIT-compiled sequence prediction."""
    carry = (initial_state, state_history, control_history)
    
    def step(carry, control):
        state, sh, ch = carry
        # Update history - do operations separately to avoid shape issues
        sh_rolled = jnp.roll(sh, -1, axis=0)
        sh = sh_rolled.at[-1, :].set(state)
        ch_rolled = jnp.roll(ch, -1, axis=0)
        ch = ch_rolled.at[-1, :].set(control)
        # Build predictor input
        input_seq = jnp.stack([sh[:, LINEAR_VEL_X_IDX], ch[:, 0], ch[:, 1]], axis=-1)
        # Predict residual
        residual = predictor_forward_jit(input_seq[None, :, :], predictor_params, norm_params)[0]
        # Base dynamics
        next_state = car_steps_sequential_jax(state, control[None, :], car_params, dt, 1)[-1]
        # Apply residual
        next_state = next_state.at[LINEAR_VEL_X_IDX].add(residual[OUTPUT_COLS.index('residual_delta_linear_vel_x_0')] * dt)
        return (next_state, sh, ch), next_state
    
    _, trajectory = jax.lax.scan(step, carry, control_sequence)
    return trajectory




# Test function
if __name__ == "__main__":
    dynamics_model_residual = DynamicsModelResidual()
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    control = np.array([0.0, 0.0], dtype=np.float32)


    # Test sequence prediction with JIT
    control_sequence = np.array([control] * 10)  # Shape: (10, 2)
    next_states = dynamics_model_residual.predict_sequence(state, control_sequence)
    print(next_states)




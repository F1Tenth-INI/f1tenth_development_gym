from sim.f110_sim.envs.car_model_jax import car_steps_sequential_jax
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import *
from predictor import Predictor
import numpy as np
import jax.numpy as jnp

import os
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
        # self.state_history = 10 * [10 * [0.0]]
        # self.control_history = 10 * [2 * [0.0]]

    def set_history(self, state_history, control_history):
        self.state_history = state_history
        self.control_history = control_history

    def predict(self, state, control):

        # Update history
        self.state_history = np.roll(self.state_history, shift=-1, axis=0)
        self.control_history = np.roll(self.control_history, shift=-1, axis=0)
        self.state_history[-1] = state
        self.control_history[-1] = control

        # Build input sequence for predictor: (10, 3) - [linear_vel_x, angular_control, translational_control]
        input_sequence = np.zeros((self.history_length, 3), dtype=np.float32)
        input_sequence[:, 0] = self.state_history[:, LINEAR_VEL_X_IDX]  # linear_vel_x
        input_sequence[:, 1] = self.control_history[:, 0]  # angular_control_executed
        input_sequence[:, 2] = self.control_history[:, 1]  # translational_control_executed

        # Predict residual
        predicted_residual = self.predictor.predict(input_sequence)
        predicted_residual = float(np.array(predicted_residual).item())

        # Run base dynamics
        # Convert to JAX arrays and reshape control to (horizon, 2)
        state_jax = jnp.array(state)
        control_jax = jnp.array(control)
        if control_jax.ndim == 1:
            control_sequence = control_jax[None, :]  # Reshape (2,) to (1, 2)
        else:
            control_sequence = control_jax
        
        next_states = car_steps_sequential_jax(state_jax, control_sequence, self.car_params, self.dt, self.horizon)
        next_state = next_states[-1]

        # Apply residual correction to linear_vel_x
        next_state = next_state.at[LINEAR_VEL_X_IDX].add(predicted_residual * self.dt)

        return next_state

    def predict_sequence(self, state_sequence, control_sequence):
        next_states = []
        for state, control in zip(state_sequence, control_sequence):
            next_state = self.predict(state, control)
            next_states.append(next_state)
        return next_states




# Test function
if __name__ == "__main__":
    dynamics_model_residual = DynamicsModelResidual()
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    control = np.array([0.0, 0.0], dtype=np.float32)
    next_state = dynamics_model_residual.predict(state, control)
    print(next_state)


    # Test sequence prediction
    state_sequence = [state] * 10
    control_sequence = [control] * 10
    next_states = dynamics_model_residual.predict_sequence(state_sequence, control_sequence)
    print(next_states)


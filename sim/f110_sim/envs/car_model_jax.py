from TrainingLite.dynamic_residual_jax.dynamics_model_residual import DynamicsModelResidual, predict_sequence_jax, predict_single_step_jax
from .dynamic_model_pacejka_jax import car_dynamics_pacejka_jax
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

residual_model = DynamicsModelResidual()


HISTORY_LENGTH = 10

@partial(jax.jit, static_argnames=["intermediate_steps"])
def car_dynamics_pacejka_jax_with_customization(state, control, car_params, dt, intermediate_steps=1):
    # Customization logic here
    next_state = car_dynamics_pacejka_jax(state, control, car_params, dt, intermediate_steps)
    
    next_state = car_step_customization(state, control, next_state)
    
    return next_state


# This function runs on RPGD JAX (Fast). Dont replace. Dont change.
@partial(jax.jit, static_argnames=["dt", "horizon", "model_type", "intermediate_steps"])
def car_steps_sequential_jax(s0, Q_sequence, car_params, dt, horizon, model_type='pacejka', intermediate_steps=1, state_history=None, control_history=None):
    """
    Run car dynamics for a single car sequentially.

    Args:
        s0: (10,) initial state
        Q_sequence: (H, 2) sequence of H controls applied to the car
        car_params: array of car parameters
        dt: time step (float)
        horizon: number of steps to run
        model_type: 'pacejka' or 'ks_pacejka' or 'residual'
        intermediate_steps: number of intermediate integration steps per evaluation (default: 1)
        state_history: (10, 10) history of last 10 states
        control_history: (10, 2) history of last 10 controls

    Returns:
        trajectory: (H, 10) trajectory of states during applying the H controls
    """
   
    if model_type == 'residual':
        # For residual model, we need to update history during sequential evaluation
        # Get predictor params once
        predictor_params = residual_model.predictor.get_params_jax()
        norm_params = residual_model.predictor.get_norm_params_jax()
        car_params_jax = jnp.array(car_params)
        
        # Initialize histories
        if state_history is None:
            state_history = jnp.zeros((HISTORY_LENGTH, 10))
        if control_history is None:
            control_history = jnp.zeros((HISTORY_LENGTH, 2))
        
        state_history_jax = jnp.array(state_history)
        control_history_jax = jnp.array(control_history)
        
        def rollout_fn(carry, control):
            state, sh, ch = carry
            # Predict next state and get updated histories
            next_state, new_sh, new_ch = predict_single_step_jax(
                state, control,
                sh, ch,
                predictor_params, norm_params, car_params_jax,
                dt
            )
            return (next_state, new_sh, new_ch), next_state
        
        carry = (s0, state_history_jax, control_history_jax)
        _, trajectory = jax.lax.scan(rollout_fn, carry, Q_sequence)
        return trajectory
    
    elif model_type == 'pacejka':
        dynamics_fn = lambda s, c: car_dynamics_pacejka_jax(s, c, car_params, dt, intermediate_steps)
        def rollout_fn(state, control):
            next_state = dynamics_fn(state, control)
            return next_state, next_state
        _, trajectory = jax.lax.scan(rollout_fn, s0, Q_sequence)
        return trajectory
    
    elif model_type == 'pacejka_custom':
        dynamics_fn = lambda s, c: car_dynamics_pacejka_jax_with_customization(s, c, car_params, dt, intermediate_steps)
        def rollout_fn(state, control):
            next_state = dynamics_fn(state, control)
            return next_state, next_state
        _, trajectory = jax.lax.scan(rollout_fn, s0, Q_sequence)
        return trajectory
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")





@partial(jax.jit)
def car_step_customization(state, control, next_state):

    # Customization logic goes here
    next_state = next_state.at[1].set(next_state[1] + 0.0)  # Add 0.1 to v_x
    return next_state


class CarModelJAX:
    """Wrapper class for JAX car models to provide a consistent interface."""
    
    def __init__(self, model_type='pacejka', dt=0.04, intermediate_steps=4):
        self.model_type = model_type
        self.dt = dt
        self.intermediate_steps = intermediate_steps
        from utilities.car_files.vehicle_parameters import VehicleParameters
        self.car_params = VehicleParameters().to_np_array()
        
        # For residual model, store history
        if model_type == 'residual':
            from TrainingLite.dynamic_residual_jax.dynamics_model_residual import DynamicsModelResidual
            self.residual_model = DynamicsModelResidual(dt=dt)
            self.state_history = self.residual_model.state_history
            self.control_history = self.residual_model.control_history
    
    def car_steps_sequential(self, initial_state, control_sequence, state_history=None, control_history=None):
        """Run car dynamics sequentially.
        
        Args:
            initial_state: (10,) initial state
            control_sequence: (H, 2) sequence of controls
            state_history: optional state history (for residual model)
            control_history: optional control history (for residual model)
        
        Returns:
            trajectory: (H, 10) trajectory of states
        """
        # Convert to JAX arrays
        initial_state_jax = jnp.array(initial_state)
        control_sequence_jax = jnp.array(control_sequence)
        car_params_jax = jnp.array(self.car_params)
        
        # For residual model, use provided history or instance history
        if self.model_type == 'residual':
            if state_history is None:
                state_history = self.state_history
            if control_history is None:
                control_history = self.control_history
            state_history_jax = jnp.array(state_history)
            control_history_jax = jnp.array(control_history)
        else:
            state_history_jax = None
            control_history_jax = None
        
        # Call the JAX function
        trajectory = car_steps_sequential_jax(
            initial_state_jax, control_sequence_jax, car_params_jax,
            self.dt, len(control_sequence), self.model_type, self.intermediate_steps,
            state_history_jax, control_history_jax
        )
        
        return np.array(trajectory)
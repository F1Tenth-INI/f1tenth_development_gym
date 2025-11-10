from .dynamic_model_pacejka_jax import car_dynamics_pacejka_jax
from functools import partial
import jax


@partial(jax.jit, static_argnames=["intermediate_steps"])
def car_dynamics_pacejka_jax_with_customization(state, control, car_params, dt, intermediate_steps=1):
    # Customization logic here
    next_state = car_dynamics_pacejka_jax(state, control, car_params, dt, intermediate_steps)
    
    next_state = car_step_customization(state, control, next_state)
    
    return next_state

@partial(jax.jit, static_argnames=["dt", "horizon", "model_type", "intermediate_steps"])
def car_steps_sequential_jax(s0, Q_sequence, car_params, dt, horizon, model_type='pacejka', intermediate_steps=1):
    """
    Run car dynamics for a single car sequentially.

    Args:
        s0: (10,) initial state
        Q_sequence: (H, 2) sequence of H controls applied to the car
        car_params: array of car parameters
        dt: time step (float)
        horizon: number of steps to run
        model_type: 'pacejka' or 'ks_pacejka'
        intermediate_steps: number of intermediate integration steps per evaluation (default: 1)

    Returns:
        trajectory: (H, 10) trajectory of states during applying the H controls
    """
   
    if model_type == 'pacejka':
        dynamics_fn = lambda s, c: car_dynamics_pacejka_jax(s, c, car_params, dt, intermediate_steps)
    
    elif model_type == 'pacejka_custom':
        dynamics_fn = lambda s, c: car_dynamics_pacejka_jax_with_customization(s, c, car_params, dt, intermediate_steps)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    
    
    def rollout_fn(state, control):
        next_state = dynamics_fn(state, control)
        return next_state, next_state
    
    _, trajectory = jax.lax.scan(rollout_fn, s0, Q_sequence)
    return trajectory





@partial(jax.jit)
def car_step_customization(state, control, next_state):

    # Customization logic goes here
    next_state = next_state.at[1].set(next_state[1] + 0.0)  # Add 0.1 to v_x
    return next_state


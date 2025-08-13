import jax
import jax.numpy as jnp
from functools import partial




class StateIndices:
    yaw_rate = 0
    v_x = 1
    v_y = 2
    yaw_angle = 3
    yaw_angle_cos = 4
    yaw_angle_sin = 5
    pose_x = 6
    pose_y = 7
    slip_angle = 8
    steering_angle = 9

    number_of_states = 10
    
    
class ControlIndices:
    desired_steering_angle = 0
    acceleration = 1


@partial(jax.jit, static_argnames=['intermediate_steps'])
def car_dynamics_pacejka_jax(state, control, car_params, dt, intermediate_steps=1):
    """Advance car dynamics using JAX-optimized Pacejka model with kinematic blending.
    
    Args:
        state (jnp.ndarray): Car state as defined in state_utilities,
        
        control (jnp.ndarray): Control input vector with the following elements:
            - control[0]: desired_steering_angle (desired steering angle input)
            - control[1]: acceleration (translational control input)

        car_params (jnp.ndarray): Array of car parameters as defined in VehicleParameters.to_np_array().

        dt (float): Time step for integration.
        
        intermediate_steps (int): Number of intermediate integration steps per evaluation (default: 1).

    Returns:
        jnp.ndarray: updated car state
          
    """
    
    # Unpack car parameters from JAX array
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
    servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch = car_params

    # Calculate sub-timestep for intermediate integration
    dt_sub = dt / intermediate_steps
    
    # Define single integration step function
    def single_step(carry, _):
        state_input = carry
        # Unpack state
        psi_dot, v_x, v_y, psi, _, _, s_x, s_y, _, delta = state_input

        # Unpack control inputs
        desired_steering_angle, translational_control = control

        # Apply Servo PID Control (match original exactly)
        steering_angle_difference = desired_steering_angle - delta
        steering_diff_low = 0.001  # Match original threshold
        delta_dot = jnp.where(jnp.abs(steering_angle_difference) > steering_diff_low,
                              steering_angle_difference * servo_p,
                              0.0)
        
        # Apply steering constraints (prevent movement when at limits)
        at_min_limit = jnp.logical_and(delta <= s_min, delta_dot <= 0.0)
        at_max_limit = jnp.logical_and(delta >= s_max, delta_dot >= 0.0)
        delta_dot = jnp.where(jnp.logical_or(at_min_limit, at_max_limit), 0.0, delta_dot)
        
        delta_dot = jnp.clip(delta_dot, sv_min, sv_max)

        # Apply acceleration constraints (match original exactly)
        v_x_dot = translational_control
        
        # Velocity constraints (stop acceleration when at velocity limits)
        v_too_low = jnp.logical_and(v_x < v_min, v_x_dot < 0)
        v_too_high = jnp.logical_and(v_x > v_max, v_x_dot > 0)
        v_x_dot = jnp.where(jnp.logical_or(v_too_low, v_too_high), 0.0, v_x_dot)
        
        # Motor power limits
        pos_limit = jnp.where(v_x > v_switch, a_max * v_switch / v_x, a_max)
        v_x_dot = jnp.clip(v_x_dot, a_min, pos_limit)
        
        # Friction limits
        max_a_friction = mu * g_
        v_x_dot = jnp.clip(v_x_dot, -max_a_friction, max_a_friction)

        # Prevent division by zero in tire model
        v_x_safe = jnp.where(v_x == 0.0, 1e-5, v_x)
        
        # Compute slip angles
        alpha_f = -jnp.arctan((v_y + psi_dot * lf) / v_x_safe) + delta
        alpha_r = -jnp.arctan((v_y - psi_dot * lr) / v_x_safe)

        # Compute vertical tire forces
        F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
        F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

        # Compute lateral forces using Pacejka's formula
        Fy_f = mu * F_zf * D_f * jnp.sin(C_f * jnp.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - jnp.arctan(B_f * alpha_f))))
        Fy_r = mu * F_zr * D_r * jnp.sin(C_r * jnp.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - jnp.arctan(B_r * alpha_r))))
                
        # Compute state derivatives
        d_s_x = v_x * jnp.cos(psi) - v_y * jnp.sin(psi)
        d_s_y = v_x * jnp.sin(psi) + v_y * jnp.cos(psi)
        d_psi = psi_dot
        d_v_x = v_x_dot
        d_v_y = (Fy_r + Fy_f) / m - v_x * psi_dot
        d_psi_dot = (-lr * Fy_r + lf * Fy_f) / I_z

        # Integrate using Euler's method
        s_x = s_x + dt_sub * d_s_x
        s_y = s_y + dt_sub * d_s_y
        delta = jnp.clip(delta + dt_sub * delta_dot, s_min, s_max)
        v_x = v_x + dt_sub * d_v_x
        v_y = v_y + dt_sub * d_v_y
        psi = psi + dt_sub * d_psi
        psi_dot = psi_dot + dt_sub * d_psi_dot

        # Kinematic blending for low speeds
        low_speed_threshold, high_speed_threshold = 0.5, 3.0
        weight = (v_x - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
        weight = jnp.clip(weight, 0.0, 1.0)

        # Simple kinematic model for low speeds
        l_wb = lf + lr  # Use correct wheelbase
        s_x_ks = s_x + dt_sub * (v_x * jnp.cos(psi))
        s_y_ks = s_y + dt_sub * (v_x * jnp.sin(psi))
        psi_ks = psi + dt_sub * (v_x / l_wb * jnp.tan(delta))
        v_y_ks = 0.0

        # Weighted interpolation between kinematic and dynamic models
        s_x = (1.0 - weight) * s_x_ks + weight * s_x
        s_y = (1.0 - weight) * s_y_ks + weight * s_y
        psi = (1.0 - weight) * psi_ks + weight * psi
        v_y = (1.0 - weight) * v_y_ks + weight * v_y

        # Recalculate derived values
        psi_sin = jnp.sin(psi)
        psi_cos = jnp.cos(psi)
        
        # Calculate slip angle properly like original
        v_x_safe = jnp.where(v_x < 1e-3, 1e-3, v_x)
        slip_angle = jnp.arctan(v_y / v_x_safe)

        next_state = jnp.array([psi_dot, v_x, v_y, psi, psi_cos, psi_sin, s_x, s_y, slip_angle, delta], dtype=jnp.float32)
        return next_state, None
    
    # Apply intermediate steps using lax.scan for differentiability
    final_state, _ = jax.lax.scan(single_step, state, None, length=intermediate_steps)
    
    return final_state


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
   
    dynamics_fn = lambda s, c: car_dynamics_pacejka_jax(s, c, car_params, dt, intermediate_steps)
    
    def rollout_fn(state, control):
        next_state = dynamics_fn(state, control)
        return next_state, next_state
    
    _, trajectory = jax.lax.scan(rollout_fn, s0, Q_sequence)
    return trajectory


@partial(jax.jit, static_argnames=["horizon", "model_type", "intermediate_steps"])
def car_steps_sequential_with_dt_array_jax(s0, Q_sequence, car_params, dt_array, horizon, model_type='pacejka', intermediate_steps=1):
    """
    Run car dynamics with variable time steps.

    Args:
        s0: (10,) initial state  
        Q_sequence: (H, 2) sequence of H controls
        car_params: array of car parameters
        dt_array: (H,) array of time steps
        horizon: number of steps
        model_type: 'pacejka' or 'ks_pacejka'
        intermediate_steps: number of intermediate integration steps per evaluation (default: 1)

    Returns:
        trajectory: (H, 10) trajectory of states
    """

    dynamics_fn = car_dynamics_pacejka_jax
    
    def rollout_fn(carry, step_idx):
        state = carry
        control = Q_sequence[step_idx]
        dt = dt_array[step_idx]
        next_state = dynamics_fn(state, control, car_params, dt, intermediate_steps)
        return next_state, next_state
    
    _, trajectory = jax.lax.scan(rollout_fn, s0, jnp.arange(horizon))
    return trajectory


@partial(jax.jit, static_argnames=["model_type", "intermediate_steps"])
def car_batch_sequence_jax(s_batch, Q_batch_sequence, car_params, dt_array, model_type='pacejka', intermediate_steps=1):
    """
    Run car dynamics for multiple cars in parallel.

    Args:
        s_batch: (N, 10) batch of initial states
        Q_batch_sequence: (N, H, 2) batch of control sequences 
        car_params: array of car parameters
        dt_array: (H,) array of time steps
        model_type: 'pacejka' or 'ks_pacejka'
        intermediate_steps: number of intermediate integration steps per evaluation (default: 1)

    Returns:
        trajectories: (N, H, 10) batch of trajectories
    """
    horizon = Q_batch_sequence.shape[1]
    rollout_fn = lambda s, Q: car_steps_sequential_with_dt_array_jax(
        s, Q, car_params, dt_array, horizon=horizon, model_type=model_type, intermediate_steps=intermediate_steps)
    return jax.vmap(rollout_fn)(s_batch, Q_batch_sequence)


@partial(jax.jit, static_argnames=["model_type", "intermediate_steps"])
def car_step_parallel_jax(states, controls, car_params, dt, model_type='pacejka', intermediate_steps=1):
    """
    Run car dynamics for multiple cars in parallel for a single time step.
    
    Args:
        states: (N, 10) array where N is number of cars
        controls: (N, 2) array where N is number of cars
        car_params: array of car parameters
        dt: time step (float)
        model_type: 'pacejka' or 'ks_pacejka'
        intermediate_steps: number of intermediate integration steps per evaluation (default: 1)

    Returns:
        new_states: (N, 10) updated states array
    """


    dynamics_fn = car_dynamics_pacejka_jax
    
    # Use vmap to vectorize over the batch dimension
    return jax.vmap(lambda s, c: dynamics_fn(s, c, car_params, dt, intermediate_steps))(states, controls)


# Legacy function names for backward compatibility
def car_step_jax(state, control, car_params, dt, intermediate_steps=1):
    """Legacy function name - use car_dynamics_pacejka_jax instead"""
    return car_dynamics_pacejka_jax(state, control, car_params, dt, intermediate_steps)


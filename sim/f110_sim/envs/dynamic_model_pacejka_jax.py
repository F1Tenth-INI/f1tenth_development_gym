import jax
import jax.numpy as jnp
from functools import partial



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
    servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch, c_rr, \
    v_dead, curve_resistance_factor, brake_multiplier = car_params

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

        # Apply acceleration constraints with asymmetric braking
        # Use brake_multiplier parameter for asymmetric acceleration/braking
        v_x_dot = jnp.where(
            translational_control >= 0,
            translational_control,  # Acceleration unchanged
            translational_control * brake_multiplier  # Scale down braking
        )
        
        # Velocity constraints (stop acceleration when at velocity limits)
        v_too_low = jnp.logical_and(v_x < v_min, v_x_dot < 0)
        v_too_high = jnp.logical_and(v_x > v_max, v_x_dot > 0)
        v_x_dot = jnp.where(jnp.logical_or(v_too_low, v_too_high), 0.0, v_x_dot)
        
        # Motor power limits
        pos_limit = jnp.where(v_x > v_switch, a_max * v_switch / v_x, a_max)
        v_x_dot = jnp.clip(v_x_dot, a_min, pos_limit)
        
        
        # --- Enhanced resistance forces ---
        # Smooth sign near 0 to avoid pulling car backwards at rest
        smooth_sign = v_x / jnp.sqrt(v_x*v_x + v_dead*v_dead)
        
        # 1. Rolling resistance (proportional to normal force)
        a_roll = -c_rr * g_ * smooth_sign
        v_x_dot += a_roll
        
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


        # Kinematic blending for low speeds
        low_speed_threshold, high_speed_threshold = 0.5, 1.0
        weight = (v_x - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
        weight = jnp.clip(weight, 0.0, 1.0)

        Fy_f *= weight
        Fy_r *= weight
        
        # # 3. Curve resistance (tire scrub during cornering)
        # # Additional rolling resistance due to lateral forces
        # # This creates the "slowing down in curves" effect
        # lateral_force_magnitude = jnp.sqrt(Fy_f*Fy_f + Fy_r*Fy_r)
        # a_curve = -curve_resistance_factor * lateral_force_magnitude / m * smooth_sign
        # v_x_dot += a_curve
                
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

        # Recalculate derived values (wrap angle)
        psi_sin = jnp.sin(psi)
        psi_cos = jnp.cos(psi)
        psi = jnp.arctan2(psi_sin, psi_cos)
        
        # Calculate slip angle properly like original
        v_x_safe = jnp.where(v_x < 1e-3, 1e-3, v_x)
        slip_angle = jnp.arctan(v_y / v_x_safe)

        next_state = jnp.array([psi_dot, v_x, v_y, psi, psi_cos, psi_sin, s_x, s_y, slip_angle, delta], dtype=jnp.float32)
        
        return next_state, None
    
    # Apply intermediate steps using lax.scan for differentiability
    final_state, _ = jax.lax.scan(single_step, state, None, length=intermediate_steps)
    
    return final_state



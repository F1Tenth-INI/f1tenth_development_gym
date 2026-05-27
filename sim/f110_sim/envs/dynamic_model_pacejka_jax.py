"""JAX Pacejka car dynamics aligned with SI_Toolkit_ASF/car_model.py (ODE_TF reference).

Matches ODE:pacejka, ODE:ks, and ODE:ks_pacejka from car_model when used with the same
servo PID, acceleration constraints, and control interpretation as base_classes.ODE_TF.
"""
import jax
import jax.numpy as jnp
from functools import partial
from utilities.Settings import Settings


def _wrap_angle_rad(angle):
    """Wrap heading to [-pi, pi]."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _steering_constraints(delta, delta_dot, s_min, s_max, sv_min, sv_max):
    at_min = jnp.logical_and(delta <= s_min, delta_dot <= 0.0)
    at_max = jnp.logical_and(delta >= s_max, delta_dot >= 0.0)
    allowed = jnp.logical_not(jnp.logical_or(at_min, at_max))
    delta_dot = delta_dot * allowed.astype(delta_dot.dtype)
    return jnp.clip(delta_dot, sv_min, sv_max)


def _accl_constraints(v_x, accl, a_min, a_max, v_min, v_max, v_switch, mu, g_):
    vel = jnp.where(v_x == 0.0, 1.0e-4, v_x)
    pos_limit = jnp.where(
        vel > v_switch,
        a_max * v_switch / vel,
        a_max,
    )
    v_too_low = jnp.logical_and(v_x < v_min, accl < 0.0)
    v_too_high = jnp.logical_and(v_x > v_max, accl > 0.0)
    accl = jnp.where(jnp.logical_or(v_too_low, v_too_high), 0.0, accl)
    accl = jnp.clip(accl, a_min, pos_limit)
    max_a_friction = mu * g_
    return jnp.clip(accl, -max_a_friction, max_a_friction)


def _servo_proportional(desired_steering_angle, delta, servo_p, steering_diff_low, sv_min, sv_max):
    steering_angle_difference = desired_steering_angle - delta
    active = jnp.abs(steering_angle_difference) > steering_diff_low
    delta_dot = active.astype(jnp.float32) * (steering_angle_difference * servo_p)
    return jnp.clip(delta_dot, sv_min, sv_max)


def _pacejka_step(s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                  lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r,
                  mu, s_min, s_max, dt_sub):
    v_x_safe = jnp.where(v_x < 1.0e-3, 1.0e-3, v_x)
    alpha_f = -jnp.arctan((v_y + psi_dot * lf) / v_x_safe) + delta
    alpha_r = -jnp.arctan((v_y - psi_dot * lr) / v_x_safe)

    F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
    F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

    F_yf = mu * F_zf * D_f * jnp.sin(
        C_f * jnp.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - jnp.arctan(B_f * alpha_f))))
    F_yr = mu * F_zr * D_r * jnp.sin(
        C_r * jnp.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - jnp.arctan(B_r * alpha_r))))

    d_s_x = v_x * jnp.cos(psi) - v_y * jnp.sin(psi)
    d_s_y = v_x * jnp.sin(psi) + v_y * jnp.cos(psi)
    d_psi = psi_dot
    d_v_x = v_x_dot
    d_v_y = (F_yr + F_yf) / m - v_x * psi_dot
    d_psi_dot = (-lr * F_yr + lf * F_yf) / I_z

    s_x = s_x + dt_sub * d_s_x
    s_y = s_y + dt_sub * d_s_y
    delta = jnp.clip(delta + dt_sub * delta_dot, s_min, s_max)
    v_x = v_x + dt_sub * d_v_x
    v_y = v_y + dt_sub * d_v_y
    psi = _wrap_angle_rad(psi + dt_sub * d_psi)
    psi_dot = psi_dot + dt_sub * d_psi_dot
    return s_x, s_y, delta, v_x, v_y, psi, psi_dot


def _ks_step(s_x, s_y, delta, v_x, psi, angular_vel_z, delta_dot, v_x_dot,
             l_wb, s_min, s_max, dt_sub):
    s_x_dot = v_x * jnp.cos(psi)
    s_y_dot = v_x * jnp.sin(psi)
    psi_dot = (v_x / l_wb) * jnp.tan(delta)
    psi_dot_dot = (v_x_dot * jnp.tan(delta) / l_wb) + v_x * delta_dot / (l_wb * jnp.cos(delta) ** 2)

    s_x = s_x + dt_sub * s_x_dot
    s_y = s_y + dt_sub * s_y_dot
    delta = jnp.clip(delta + dt_sub * delta_dot, s_min, s_max)
    v_x = v_x + dt_sub * v_x_dot
    psi = _wrap_angle_rad(psi + dt_sub * psi_dot)
    angular_vel_z = angular_vel_z + dt_sub * psi_dot_dot
    return s_x, s_y, delta, v_x, psi, angular_vel_z


def _next_step_output(psi_dot, v_x, v_y, psi, s_x, s_y, delta):
    v_x_safe = jnp.where(v_x < 1.0e-3, 1.0e-3, v_x)
    slip_angle = jnp.arctan(v_y / v_x_safe)
    return jnp.array([
        psi_dot, v_x, v_y, psi, jnp.cos(psi), jnp.sin(psi),
        s_x, s_y, slip_angle, delta,
    ], dtype=jnp.float32)


@partial(jax.jit, static_argnames=['intermediate_steps', 'ode_model'])
def car_dynamics_pacejka_jax(state, control, car_params, dt, intermediate_steps=1,
                             ode_model='ODE:ks_pacejka'):
    """Advance car dynamics (ODE_TF-equivalent).

    Args:
        state: [psi_dot, v_x, v_y, psi, cos, sin, s_x, s_y, slip, delta]
        control: [desired_steering_angle, translational_control]
                 translational_control is acceleration unless MOTOR_PID_IN_CAR_MODEL.
        car_params: VehicleParameters.to_np_array()
        dt: integration timestep
        intermediate_steps: substeps per call (matches car_model.intermediate_steps)
        ode_model: 'ODE:pacejka', 'ODE:ks', or 'ODE:ks_pacejka' (Settings.ODE_MODEL_OF_CAR_DYNAMICS)
    """
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
        servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch = car_params[:25]
    steering_diff_low = car_params[29] if car_params.shape[0] > 29 else jnp.float32(0.0)
    l_wb = lf + lr
    dt_sub = dt / intermediate_steps
    use_speed_pi = Settings.MOTOR_PID_IN_CAR_MODEL
    use_ks = ode_model in ('ODE:ks', 'ODE:ks_pacejka')
    use_pacejka = ode_model in ('ODE:pacejka', 'ODE:ks_pacejka')
    use_blend = ode_model == 'ODE:ks_pacejka'

    def single_step(carry, _):
        state_in, speed_error_integral = carry
        psi_dot, v_x, v_y, psi, _, _, s_x, s_y, _, delta = state_in
        desired_steering_angle, translational_control = control

        delta_dot = _servo_proportional(
            desired_steering_angle, delta, servo_p, steering_diff_low, sv_min, sv_max)
        delta_dot = _steering_constraints(delta, delta_dot, s_min, s_max, sv_min, sv_max)

        if use_speed_pi:
            speed_kp = 10.0 * a_max / jnp.maximum(v_max, 1e-3)
            speed_ki = 0.5 * speed_kp
            speed_error = translational_control - v_x
            speed_error_integral = jnp.clip(
                speed_error_integral + speed_error * dt_sub, -5.0, 5.0)
            v_x_dot = speed_kp * speed_error + speed_ki * speed_error_integral
        else:
            v_x_dot = translational_control
            speed_error_integral = speed_error_integral

        v_x_dot = _accl_constraints(v_x, v_x_dot, a_min, a_max, v_min, v_max, v_switch, mu, g_)

        if use_pacejka:
            p_s_x, p_s_y, p_delta, p_v_x, p_v_y, p_psi, p_psi_dot = _pacejka_step(
                s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r,
                mu, s_min, s_max, dt_sub)
            s_pacejka = _next_step_output(p_psi_dot, p_v_x, p_v_y, p_psi, p_s_x, p_s_y, p_delta)
        else:
            s_pacejka = state_in

        if use_ks:
            angular_vel_z = psi_dot
            k_s_x, k_s_y, k_delta, k_v_x, k_psi, k_angular_vel_z = _ks_step(
                s_x, s_y, delta, v_x, psi, angular_vel_z, delta_dot, v_x_dot,
                l_wb, s_min, s_max, dt_sub)
            s_ks = _next_step_output(k_angular_vel_z, k_v_x, 0.0, k_psi, k_s_x, k_s_y, k_delta)
        else:
            s_ks = state_in

        if use_blend:
            weight = 1.0 / (1.0 + jnp.exp(-4.8 * (v_x - 1.75)))
            next_state = (1.0 - weight) * s_ks + weight * s_pacejka
        elif use_pacejka:
            next_state = s_pacejka
        else:
            next_state = s_ks

        return (next_state, speed_error_integral), None

    init_carry = (state, jnp.array(0.0, dtype=jnp.float32))
    final_carry, _ = jax.lax.scan(single_step, init_carry, None, length=intermediate_steps)
    return final_carry[0]


def car_dynamics_pacejka_jax_from_settings(state, control, car_params, dt, intermediate_steps=1):
    """Wrapper using Settings.ODE_MODEL_OF_CAR_DYNAMICS."""
    return car_dynamics_pacejka_jax(
        state, control, car_params, dt,
        intermediate_steps=intermediate_steps,
        ode_model=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
    )

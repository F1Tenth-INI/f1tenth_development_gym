"""
Single-track Pacejka bicycle model (JAX).

Public control input is normalized: ``control = [steer_norm, throttle_norm]``
in ``[-1, 1]``. The YAML-defined ``drive_mode`` decides what the throttle
stick maps to physically inside the dynamics:
- ``torque`` : throttle * tau_wheel_max_nm (or * tau_wheel_regen_max_nm)
- ``accel``  : throttle * a_max (legacy RL / MPC direct-acceleration path)
- ``current``: throttle * motor_current_max_a * K_t * gear_ratio (VESC I_q)

There is intentionally no in-dynamics speed-PI. Controllers that want a
desired-speed frontend should call ``utilities.controller_utilities`` to
convert speed -> acceleration / normalized throttle BEFORE handing the
command to the dynamics. Keeping the PI outside the model means the
dynamics describe physics only, and switching cars / drive modes does not
silently change the controller behavior.

Independently of ``drive_mode``, ``use_longitudinal_slip`` selects whether
the integrator runs the slip-aware path (Pacejka Fx(kappa), friction
ellipse, wheel-speed state ``wheel_omega`` for 4WD-locked axles) or the
legacy direct-acceleration path.

See: https://manuals.plus/trampa/vesc-6-mk-vi-vedder-esc-for-dc-and-bldc-motors-manual
"""

import jax
import jax.numpy as jnp
from functools import partial


def _pacejka_force(mu, fz, b, c, d, e, slip):
    """Magic formula (lateral slip angle or longitudinal slip ratio)."""
    return (
        mu
        * fz
        * d
        * jnp.sin(c * jnp.arctan(b * slip - e * (b * slip - jnp.arctan(b * slip))))
    )


@partial(jax.jit, static_argnames=["intermediate_steps"])
def car_dynamics_pacejka_jax(state, control, car_params, dt, intermediate_steps=1):
    """Advance car dynamics (11 states when slip model on; same layout with omega for legacy path).

    State (11): [psi_dot, v_x, v_y, psi, cos, sin, s_x, s_y, slip_angle, delta, wheel_omega].

    control[0]: normalized steering stick in [-1, 1] (-1 = s_min, +1 = s_max).
    control[1]: normalized throttle stick in [-1, 1]; mapping decided by drive_mode.
    """
    # --- unpack params (see VehicleParameters.to_np_array) ---
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
    servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch, c_rr, \
    v_dead, curve_resistance_factor, brake_multiplier, steering_diff_low, \
    motor_tau, motor_drop, use_slip_f, R_w, I_w, tau_max, tau_regen_max, omega_visc, kappa_den, \
    B_xf, C_xf, D_xf, E_xf, B_xr, C_xr, D_xr, E_xr, drive_mode_id, \
    i_max, kt_motor, gear_ratio = car_params

    use_slip = use_slip_f > 0.5
    dt_sub = dt / intermediate_steps
    # drive_mode encoding (kept in sync with vehicle_parameters._DRIVE_MODE_IDS):
    # 0: torque, 1: accel, 2: current. There is no "speed" id -- speed-PID
    # lives in the controller, not in the dynamics.
    mode_torque = drive_mode_id < 0.5
    mode_accel = jnp.logical_and(drive_mode_id >= 0.5, drive_mode_id < 1.5)
    mode_current = drive_mode_id >= 1.5
    l_wb = lf + lr
    mR = m * jnp.maximum(R_w, 1e-6)

    def single_step(state_input, _):
        psi_dot, v_x, v_y, psi, _, _, s_x, s_y, _, delta, omega = state_input
        steer_norm, throttle_norm = control
        steer_norm = jnp.clip(steer_norm, -1.0, 1.0)
        throttle_norm = jnp.clip(throttle_norm, -1.0, 1.0)

        # --- normalized -> physical command per drive_mode ---
        # Asymmetric mapping for forward / reverse so e.g. tau_max != tau_regen_max
        # is honored without changing the public stick range.
        desired_steering_angle = jnp.where(
            steer_norm >= 0.0, steer_norm * s_max, steer_norm * (-s_min)
        )
        cmd_torque = jnp.where(
            throttle_norm >= 0.0, throttle_norm * tau_max, throttle_norm * tau_regen_max
        )
        cmd_accel = jnp.where(
            throttle_norm >= 0.0, throttle_norm * a_max, throttle_norm * (-a_min)
        )
        # symmetric current cap (regen done by VESC at the same |I_q|)
        cmd_current_torque = throttle_norm * i_max * kt_motor * gear_ratio

        # --- steering servo ---
        steering_angle_difference = desired_steering_angle - delta
        steer_eps = jnp.maximum(steering_diff_low, 1e-5)
        delta_dot = jnp.where(
            jnp.abs(steering_angle_difference) > steer_eps,
            steering_angle_difference * servo_p,
            0.0,
        )
        at_min_limit = jnp.logical_and(delta <= s_min, delta_dot <= 0.0)
        at_max_limit = jnp.logical_and(delta >= s_max, delta_dot >= 0.0)
        delta_dot = jnp.where(jnp.logical_or(at_min_limit, at_max_limit), 0.0, delta_dot)
        delta_dot = jnp.clip(delta_dot, sv_min, sv_max)

        # Per-mode equivalent acceleration command (used by legacy v_x path).
        a_cmd_from_torque = cmd_torque / mR
        a_cmd_from_current = cmd_current_torque / mR
        a_cmd = jnp.where(
            mode_torque, a_cmd_from_torque,
            jnp.where(mode_accel, cmd_accel, a_cmd_from_current),
        )
        translational_control = a_cmd

        v_norm = jnp.clip(v_x / jnp.maximum(v_max, 1e-3), 0.0, 1.0)
        forward_scale = jnp.maximum(1.0 - motor_drop * v_norm, 0.15)

        # Torque target [N·m] at shaft (4WD locked) for the slip path. Modes
        # that don't natively command torque go through accel * m * R first
        # and then get clipped to the torque envelope; that gives consistent
        # saturation behavior across modes.
        tau_from_accel = a_cmd * mR
        tau_target = jnp.where(
            mode_torque, cmd_torque,
            jnp.where(mode_current, cmd_current_torque, tau_from_accel),
        )
        tau_target = jnp.clip(tau_target, -tau_regen_max, tau_max)
        tau_target = jnp.where(tau_target > 0.0, tau_target * forward_scale, tau_target)
        tau_motor = jnp.where(use_slip, tau_target, 0.0)

        # --- legacy direct acceleration path ---
        v_x_dot_accel = jnp.where(
            translational_control >= 0.0,
            translational_control,
            translational_control * brake_multiplier,
        )
        v_x_dot_accel = jnp.where(v_x_dot_accel > 0.0, v_x_dot_accel * forward_scale, v_x_dot_accel)
        v_too_low = jnp.logical_and(v_x < v_min, v_x_dot_accel < 0.0)
        v_too_high = jnp.logical_and(v_x > v_max, v_x_dot_accel > 0.0)
        v_x_dot_accel = jnp.where(jnp.logical_or(v_too_low, v_too_high), 0.0, v_x_dot_accel)
        pos_limit = jnp.where(v_x > v_switch, a_max * v_switch / jnp.maximum(v_x, 1e-6), a_max)
        v_x_dot_accel = jnp.clip(v_x_dot_accel, a_min, pos_limit)
        smooth_sign = v_x / jnp.sqrt(v_x * v_x + v_dead * v_dead)
        a_roll = -c_rr * g_ * smooth_sign
        v_x_dot_accel = v_x_dot_accel + a_roll
        max_a_friction = mu * g_
        v_x_dot_accel = jnp.clip(v_x_dot_accel, -max_a_friction, max_a_friction)

        # --- slip + tire forces ---
        vx_alpha = jnp.where(jnp.abs(v_x) < 1e-5, jnp.sign(v_x + 1e-9) * 1e-5, v_x)
        kappa = (R_w * omega - v_x) / jnp.maximum(jnp.abs(v_x), kappa_den)

        # Two fixed-point passes for load transfer vs longitudinal acceleration
        v_x_dot_guess = jnp.zeros((), dtype=v_x.dtype)

        def fz_fx_pass(vxd_in):
            F_zf = m * (-vxd_in * h_cg + g_ * lr) / l_wb
            F_zr = m * (vxd_in * h_cg + g_ * lf) / l_wb
            F_zf = jnp.maximum(F_zf, 1.0)
            F_zr = jnp.maximum(F_zr, 1.0)

            alpha_f = -jnp.arctan((v_y + psi_dot * lf) / vx_alpha) + delta
            alpha_r = -jnp.arctan((v_y - psi_dot * lr) / vx_alpha)

            Fy_f = _pacejka_force(mu, F_zf, B_f, C_f, D_f, E_f, alpha_f)
            Fy_r = _pacejka_force(mu, F_zr, B_r, C_r, D_r, E_r, alpha_r)

            Fx_f0 = _pacejka_force(mu, F_zf, B_xf, C_xf, D_xf, E_xf, kappa)
            Fx_r0 = _pacejka_force(mu, F_zr, B_xr, C_xr, D_xr, E_xr, kappa)

            mu_fz_f = mu * F_zf + 1e-6
            mu_fz_r = mu * F_zr + 1e-6
            avail_f = jnp.sqrt(jnp.maximum(0.0, mu_fz_f * mu_fz_f - Fy_f * Fy_f))
            avail_r = jnp.sqrt(jnp.maximum(0.0, mu_fz_r * mu_fz_r - Fy_r * Fy_r))
            Fx_f = jnp.clip(Fx_f0, -avail_f, avail_f)
            Fx_r = jnp.clip(Fx_r0, -avail_r, avail_r)

            vxd_out = (Fx_f + Fx_r) / jnp.maximum(m, 1e-6) + a_roll
            return Fy_f, Fy_r, Fx_f, Fx_r, vxd_out

        Fy_f1, Fy_r1, Fx_f1, Fx_r1, vxd1 = fz_fx_pass(v_x_dot_guess)
        Fy_f2, Fy_r2, Fx_f2, Fx_r2, vxd2 = fz_fx_pass(vxd1)
        Fy_f = Fy_f2
        Fy_r = Fy_r2
        Fx_f = Fx_f2
        Fx_r = Fx_r2
        v_x_dot_slip = vxd2

        sum_fx = Fx_f + Fx_r
        d_omega = (tau_motor - omega_visc * omega - R_w * sum_fx) / jnp.maximum(I_w, 1e-9)

        # Low-speed lateral blend (same schedule for Fy and kinematic pose)
        v_dyn_lo, v_dyn_hi = 0.4, 2.6
        w_dyn = jnp.clip((v_x - v_dyn_lo) / (v_dyn_hi - v_dyn_lo), 0.0, 1.0)
        Fy_f = Fy_f * w_dyn
        Fy_r = Fy_r * w_dyn

        d_v_x_slip = v_x_dot_slip
        d_v_y = (Fy_r + Fy_f) / m - v_x * psi_dot
        d_psi_dot = (-lr * Fy_r + lf * Fy_f) / I_z

        d_v_x = jnp.where(use_slip, d_v_x_slip, v_x_dot_accel)
        d_s_x = v_x * jnp.cos(psi) - v_y * jnp.sin(psi)
        d_s_y = v_x * jnp.sin(psi) + v_y * jnp.cos(psi)
        d_psi = psi_dot

        s_x = s_x + dt_sub * d_s_x
        s_y = s_y + dt_sub * d_s_y
        delta = jnp.clip(delta + dt_sub * delta_dot, s_min, s_max)
        v_x = v_x + dt_sub * d_v_x
        v_y = v_y + dt_sub * d_v_y
        psi = psi + dt_sub * d_psi
        psi_dot = psi_dot + dt_sub * d_psi_dot
        omega = jnp.where(
            use_slip,
            omega + dt_sub * d_omega,
            v_x / jnp.maximum(R_w, 1e-6),
        )

        s_x_ks = s_x + dt_sub * (v_x * jnp.cos(psi))
        s_y_ks = s_y + dt_sub * (v_x * jnp.sin(psi))
        psi_ks = psi + dt_sub * (v_x / l_wb * jnp.tan(delta))
        v_y_ks = 0.0
        s_x = (1.0 - w_dyn) * s_x_ks + w_dyn * s_x
        s_y = (1.0 - w_dyn) * s_y_ks + w_dyn * s_y
        psi = (1.0 - w_dyn) * psi_ks + w_dyn * psi
        v_y = (1.0 - w_dyn) * v_y_ks + w_dyn * v_y

        psi_sin = jnp.sin(psi)
        psi_cos = jnp.cos(psi)
        psi = jnp.arctan2(psi_sin, psi_cos)
        v_x_safe2 = jnp.where(jnp.abs(v_x) < 1e-3, 1e-3, v_x)
        slip_angle = jnp.arctan(v_y / v_x_safe2)

        next_state = jnp.array(
            [psi_dot, v_x, v_y, psi, psi_cos, psi_sin, s_x, s_y, slip_angle, delta, omega],
            dtype=jnp.float32,
        )
        return next_state, None

    final_state, _ = jax.lax.scan(single_step, state, None, length=intermediate_steps)
    return final_state

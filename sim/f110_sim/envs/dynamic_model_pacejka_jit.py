import numpy as np
from numba import njit, prange


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
    wheel_omega = 10

    number_of_states = 11


class ControlIndices:
    desired_steering_angle = 0
    acceleration = 1


@njit(fastmath=True)
def _pacejka_f(mu, fz, b, c, d, e, slip):
    return mu * fz * d * np.sin(c * np.arctan(b * slip - e * (b * slip - np.arctan(b * slip))))


@njit(fastmath=True)
def _speed_pi_numba(desired_speed, v_x, integ, dt, a_max, v_max):
    speed_kp = 10.0 * a_max / max(v_max, 1e-3)
    speed_ki = 0.5 * speed_kp
    lim = 5.0
    err = desired_speed - v_x
    integ = integ + err * dt
    if integ > lim:
        integ = lim
    elif integ < -lim:
        integ = -lim
    return speed_kp * err + speed_ki * integ, integ


@njit(fastmath=True)
def car_dynamics_pacejka_jit(s, Q, car_params, t_step):
    """State (11); car_params (51) — see VehicleParameters.to_np_array.

    DEPRECATED: this Numba path predates the normalized [-1, 1] control input.
    It interprets ``Q[1]`` in legacy physical units (accel / speed when
    drive_mode_id == 2 / torque when slip and drive_mode_id == 0). New code
    paths use the JAX dynamics, which translate normalized -> physical
    internally per ``drive_mode``. Kept here only to avoid breaking older
    setups that still select ``SIM_ODE_IMPLEMENTATION = 'jit_Pacejka'``.
    """
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
    servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch, c_rr, \
    v_dead, curve_resistance_factor, brake_multiplier, steering_diff_low, \
    motor_tau, motor_drop, use_slip_f, R_w, I_w, tau_max, tau_regen_max, omega_visc, kappa_den, \
    B_xf, C_xf, D_xf, E_xf, B_xr, C_xr, D_xr, E_xr, drive_mode_id, \
    i_max, kt_motor, gear_ratio = car_params

    use_slip = use_slip_f > 0.5
    # Approximate legacy semantics: speed-PI is the only multi-mode case the
    # JIT path was ever wired up for, so map drive_mode_id == 2 (speed) -> PI.
    use_speed_pi = drive_mode_id > 1.5 and drive_mode_id < 2.5
    l_wb = lf + lr

    psi_dot = s[0]
    v_x = s[1]
    v_y = s[2]
    psi = s[3]
    s_x = s[6]
    s_y = s[7]
    delta = s[9]
    omega = s[10]

    desired_steering_angle = Q[0]
    u_long = Q[1]

    steer_eps = max(steering_diff_low, 1e-5)
    steering_angle_difference = desired_steering_angle - delta
    if abs(steering_angle_difference) > steer_eps:
        delta_dot = steering_angle_difference * servo_p
    else:
        delta_dot = 0.0
    if (delta <= s_min and delta_dot <= 0.0) or (delta >= s_max and delta_dot >= 0.0):
        delta_dot = 0.0
    delta_dot = max(min(delta_dot, sv_max), sv_min)

    smooth_sign = v_x / np.sqrt(v_x * v_x + v_dead * v_dead)
    a_roll = -c_rr * g_ * smooth_sign

    if use_slip:
        v_norm = min(max(v_x / max(v_max, 1e-3), 0.0), 1.0)
        forward_scale = max(1.0 - motor_drop * v_norm, 0.15)
        if use_speed_pi:
            tc_pi, _ = _speed_pi_numba(u_long, v_x, 0.0, t_step, a_max, v_max)
            tau_from_pi = max(min(tc_pi * m * max(R_w, 1e-6), tau_max), -tau_regen_max)
            if tau_from_pi > 0.0:
                tau_from_pi *= forward_scale
            tau_motor = tau_from_pi
        else:
            tau_from_cmd = max(min(u_long, tau_max), -tau_regen_max)
            if tau_from_cmd > 0.0:
                tau_from_cmd *= forward_scale
            tau_motor = tau_from_cmd

        v_x_safe = max(abs(v_x), 1e-3)
        kappa = (R_w * omega - v_x) / max(abs(v_x), kappa_den)

        v_x_dot_guess = 0.0
        Fx_f = 0.0
        Fx_r = 0.0
        Fy_f = 0.0
        Fy_r = 0.0
        for _ in range(2):
            F_zf = m * (-v_x_dot_guess * h_cg + g_ * lr) / l_wb
            F_zr = m * (v_x_dot_guess * h_cg + g_ * lf) / l_wb
            if F_zf < 1.0:
                F_zf = 1.0
            if F_zr < 1.0:
                F_zr = 1.0

            if abs(v_x) >= 1e-5:
                vx_a = v_x
            else:
                vx_a = 1e-5 if v_x >= 0.0 else -1e-5
            alpha_f = -np.arctan((v_y + psi_dot * lf) / vx_a) + delta
            alpha_r = -np.arctan((v_y - psi_dot * lr) / vx_a)

            Fy_f = _pacejka_f(mu, F_zf, B_f, C_f, D_f, E_f, alpha_f)
            Fy_r = _pacejka_f(mu, F_zr, B_r, C_r, D_r, E_r, alpha_r)

            Fx_f0 = _pacejka_f(mu, F_zf, B_xf, C_xf, D_xf, E_xf, kappa)
            Fx_r0 = _pacejka_f(mu, F_zr, B_xr, C_xr, D_xr, E_xr, kappa)

            mu_fz_f = mu * F_zf + 1e-6
            mu_fz_r = mu * F_zr + 1e-6
            avail_f = np.sqrt(max(mu_fz_f * mu_fz_f - Fy_f * Fy_f, 0.0))
            avail_r = np.sqrt(max(mu_fz_r * mu_fz_r - Fy_r * Fy_r, 0.0))
            Fx_f = max(min(Fx_f0, avail_f), -avail_f)
            Fx_r = max(min(Fx_r0, avail_r), -avail_r)

            v_x_dot_guess = (Fx_f + Fx_r) / m + a_roll

        v_x_dot = v_x_dot_guess
        sum_fx = Fx_f + Fx_r
        d_omega = (tau_motor - omega_visc * omega - R_w * sum_fx) / max(I_w, 1e-9)

        v_dyn_lo, v_dyn_hi = 0.4, 2.6
        w_dyn = (v_x - v_dyn_lo) / (v_dyn_hi - v_dyn_lo)
        if w_dyn < 0.0:
            w_dyn = 0.0
        elif w_dyn > 1.0:
            w_dyn = 1.0
        Fy_f *= w_dyn
        Fy_r *= w_dyn

        d_s_x = v_x * np.cos(psi) - v_y * np.sin(psi)
        d_s_y = v_x * np.sin(psi) + v_y * np.cos(psi)
        d_psi = psi_dot
        d_v_y = (Fy_r + Fy_f) / m - v_x * psi_dot
        d_psi_dot = (-lr * Fy_r + lf * Fy_f) / I_z

        s_x += t_step * d_s_x
        s_y += t_step * d_s_y
        delta = max(min(delta + t_step * delta_dot, s_max), s_min)
        v_x += t_step * v_x_dot
        v_y += t_step * d_v_y
        psi += t_step * d_psi
        psi_dot += t_step * d_psi_dot
        omega += t_step * d_omega

        s_x_ks = s_x + t_step * (v_x * np.cos(psi))
        s_y_ks = s_y + t_step * (v_x * np.sin(psi))
        psi_ks = psi + t_step * (v_x / l_wb * np.tan(delta))
        v_y_ks = 0.0
        s_x = (1.0 - w_dyn) * s_x_ks + w_dyn * s_x
        s_y = (1.0 - w_dyn) * s_y_ks + w_dyn * s_y
        psi = (1.0 - w_dyn) * psi_ks + w_dyn * psi
        v_y = (1.0 - w_dyn) * v_y_ks + w_dyn * v_y

    else:
        translational_control = u_long
        if translational_control >= 0:
            v_x_dot = translational_control
        else:
            v_x_dot = translational_control * brake_multiplier
        if (v_x < v_min and v_x_dot < 0) or (v_x > v_max and v_x_dot > 0):
            v_x_dot = 0.0
        if v_x > v_switch:
            pos_limit = a_max * v_switch / v_x
        else:
            pos_limit = a_max
        v_x_dot = max(min(v_x_dot, pos_limit), a_min)
        v_x_dot += a_roll
        max_a_friction = mu * g_
        v_x_dot = min(max(v_x_dot, -max_a_friction), max_a_friction)

        for _ in range(1):
            if v_x == 0:
                v_x = 1e-5
            alpha_f = -np.arctan((v_y + psi_dot * lf) / v_x) + delta
            alpha_r = -np.arctan((v_y - psi_dot * lr) / v_x)
            F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
            F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)
            F_yf = mu * F_zf * D_f * np.sin(C_f * np.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - np.arctan(B_f * alpha_f))))
            F_yr = mu * F_zr * D_r * np.sin(C_r * np.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - np.arctan(B_r * alpha_r))))
            lateral_force_magnitude = np.sqrt(F_yf * F_yf + F_yr * F_yr)
            a_curve = -curve_resistance_factor * lateral_force_magnitude / m * smooth_sign
            v_x_dot += a_curve
            d_s_x = v_x * np.cos(psi) - v_y * np.sin(psi)
            d_s_y = v_x * np.sin(psi) + v_y * np.cos(psi)
            d_psi = psi_dot
            d_v_x = v_x_dot
            d_v_y = (F_yr + F_yf) / m - v_x * psi_dot
            d_psi_dot = (-lr * F_yr + lf * F_yf) / I_z
            s_x += t_step * d_s_x
            s_y += t_step * d_s_y
            delta = max(min(delta + t_step * delta_dot, s_max), s_min)
            v_x += t_step * d_v_x
            v_y += t_step * d_v_y
            psi += t_step * d_psi
            psi_dot += t_step * d_psi_dot

        omega = v_x / max(R_w, 1e-6)

    psi_sin = np.sin(psi)
    psi_cos = np.cos(psi)
    v_x_safe = v_x if v_x >= 1e-3 else 1e-3
    slip_angle = np.arctan(v_y / v_x_safe)
    return np.array(
        [psi_dot, v_x, v_y, psi, psi_cos, psi_sin, s_x, s_y, slip_angle, delta, omega],
        dtype=np.float32,
    )


@njit(fastmath=True)
def car_steps_sequential(s, Q_sequence, car_params, t_step, num_steps):
    state_trajectory = np.zeros((num_steps, 11), dtype=np.float32)
    for i in range(num_steps):
        s = car_dynamics_pacejka_jit(s, Q_sequence[i], car_params, t_step)
        state_trajectory[i] = s
    return state_trajectory


@njit(parallel=True, fastmath=True)
def car_step_parallel(states, Qs, car_params, t_step):
    num_cars = states.shape[0]
    new_states = np.empty_like(states)
    for i in prange(num_cars):
        new_states[i] = car_dynamics_pacejka_jit(states[i], Qs[i], car_params, t_step)
    return new_states

"""JAX Pacejka car dynamics with optional longitudinal slip / motor driveline.

State (11): [psi_dot, v_x, v_y, psi, cos(psi), sin(psi), s_x, s_y, slip_angle, delta, motor_angular_vel]

When use_longitudinal_slip is enabled in car_params, translational control drives motor torque and
wheel slip is resolved via Pacejka Fx; otherwise the legacy direct-acceleration path is used.
"""
import jax
import jax.numpy as jnp
from functools import partial
from utilities.Settings import Settings


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


def _apply_rolling_resistance(v_x, v_x_dot, c_rr, v_dead, g_, mu):
    """Rolling drag and friction re-limit (matches dynamic_model_pacejka_jit)."""
    smooth_sign = v_x / jnp.sqrt(v_x * v_x + v_dead * v_dead)
    v_x_dot = v_x_dot - c_rr * g_ * smooth_sign
    max_a_friction = mu * g_
    return jnp.clip(v_x_dot, -max_a_friction, max_a_friction)


def _servo_proportional(desired_steering_angle, delta, servo_p, steering_diff_low, sv_min, sv_max):
    steering_angle_difference = desired_steering_angle - delta
    active = jnp.abs(steering_angle_difference) > steering_diff_low
    delta_dot = active.astype(jnp.float32) * (steering_angle_difference * servo_p)
    return jnp.clip(delta_dot, sv_min, sv_max)


def _pacejka_tire_force(slip, B, C, D, E, mu, F_z):
    Bs = B * slip
    return mu * F_z * D * jnp.sin(
        C * jnp.arctan(Bs - E * (Bs - jnp.arctan(Bs))))


def _friction_circle_scale(Fx, Fy, mu, F_z):
    """Scale Fx/Fy jointly so sqrt(Fx^2 + Fy^2) <= mu * F_z per axle."""
    f_mag = jnp.sqrt(Fx * Fx + Fy * Fy)
    f_max = mu * F_z
    return jnp.where(f_mag > 1.0e-6, jnp.minimum(1.0, f_max / f_mag), 1.0)


def _unpack_car_params(car_params):
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
        servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch = car_params[:25]
    c_rr = car_params[25] if car_params.shape[0] > 25 else jnp.float32(0.0)
    v_dead = car_params[26] if car_params.shape[0] > 26 else jnp.float32(0.05)
    curve_resistance_factor = car_params[27] if car_params.shape[0] > 27 else jnp.float32(0.0)
    brake_multiplier = car_params[28] if car_params.shape[0] > 28 else jnp.float32(1.0)
    steering_diff_low = car_params[29] if car_params.shape[0] > 29 else jnp.float32(0.0)

    use_slip = (car_params[30] > 0.5) if car_params.shape[0] > 30 else False
    wheel_radius = car_params[31] if car_params.shape[0] > 31 else jnp.float32(0.05)
    wheel_inertia = car_params[32] if car_params.shape[0] > 32 else jnp.float32(1.0e-4)
    tau_wheel_max = car_params[33] if car_params.shape[0] > 33 else jnp.float32(0.5)
    tau_wheel_regen = car_params[34] if car_params.shape[0] > 34 else jnp.float32(0.5)
    omega_viscous = car_params[35] if car_params.shape[0] > 35 else jnp.float32(0.0)
    kappa_den_min = car_params[36] if car_params.shape[0] > 36 else jnp.float32(0.12)
    motor_tau_s = car_params[37] if car_params.shape[0] > 37 else jnp.float32(0.0)
    motor_torque_drop = car_params[38] if car_params.shape[0] > 38 else jnp.float32(0.0)
    gear_ratio = car_params[39] if car_params.shape[0] > 39 else jnp.float32(1.0)
    motor_K_t = car_params[40] if car_params.shape[0] > 40 else jnp.float32(0.01)
    motor_I_max = car_params[41] if car_params.shape[0] > 41 else jnp.float32(60.0)
    drive_mode = car_params[42] if car_params.shape[0] > 42 else jnp.float32(0.0)

    if car_params.shape[0] > 50:
        B_xf, C_xf, D_xf, E_xf = car_params[43], car_params[44], car_params[45], car_params[46]
        B_xr, C_xr, D_xr, E_xr = car_params[47], car_params[48], car_params[49], car_params[50]
    else:
        B_xf = C_xf = D_xf = E_xf = B_xr = C_xr = D_xr = E_xr = jnp.float32(0.0)

    kappa_long_peak = car_params[51] if car_params.shape[0] > 51 else jnp.float32(0.12)
    kappa_long_falloff = car_params[52] if car_params.shape[0] > 52 else jnp.float32(0.14)
    kappa_torque_cut_start = car_params[53] if car_params.shape[0] > 53 else jnp.float32(0.16)
    kappa_torque_cut_gain = car_params[54] if car_params.shape[0] > 54 else jnp.float32(4.0)
    kappa_spin_drag = car_params[55] if car_params.shape[0] > 55 else jnp.float32(0.02)

    return {
        'mu': mu, 'lf': lf, 'lr': lr, 'h_cg': h_cg, 'm': m, 'I_z': I_z, 'g_': g_,
        'B_f': B_f, 'C_f': C_f, 'D_f': D_f, 'E_f': E_f,
        'B_r': B_r, 'C_r': C_r, 'D_r': D_r, 'E_r': E_r,
        'servo_p': servo_p, 's_min': s_min, 's_max': s_max, 'sv_min': sv_min, 'sv_max': sv_max,
        'a_min': a_min, 'a_max': a_max, 'v_min': v_min, 'v_max': v_max, 'v_switch': v_switch,
        'c_rr': c_rr, 'v_dead': v_dead, 'curve_resistance_factor': curve_resistance_factor,
        'brake_multiplier': brake_multiplier, 'steering_diff_low': steering_diff_low,
        'use_slip': use_slip, 'wheel_radius': wheel_radius, 'wheel_inertia': wheel_inertia,
        'tau_wheel_max': tau_wheel_max, 'tau_wheel_regen': tau_wheel_regen,
        'omega_viscous': omega_viscous, 'kappa_den_min': kappa_den_min,
        'motor_tau_s': motor_tau_s, 'motor_torque_drop': motor_torque_drop,
        'gear_ratio': gear_ratio, 'motor_K_t': motor_K_t, 'motor_I_max': motor_I_max,
        'drive_mode': drive_mode,
        'B_xf': B_xf, 'C_xf': C_xf, 'D_xf': D_xf, 'E_xf': E_xf,
        'B_xr': B_xr, 'C_xr': C_xr, 'D_xr': D_xr, 'E_xr': E_xr,
        'kappa_long_peak': kappa_long_peak,
        'kappa_long_falloff': kappa_long_falloff,
        'kappa_torque_cut_start': kappa_torque_cut_start,
        'kappa_torque_cut_gain': kappa_torque_cut_gain,
        'kappa_spin_drag': kappa_spin_drag,
    }


def _longitudinal_slip_falloff(kappa, p):
    """Reduce Fx beyond peak slip so spin does not sustain unrealistic push."""
    excess = jnp.maximum(jnp.abs(kappa) - p['kappa_long_peak'], 0.0)
    width = jnp.maximum(p['kappa_long_falloff'], 1.0e-3)
    return jnp.exp(-jnp.square(excess / width))


def _wheelspin_active(v_x, v_wheel, p):
    """True when wheels outrun the chassis (not normal low-speed launch)."""
    # Below launch_speed, motor/chassis mismatch is expected (forward and reverse).
    launch_speed = jnp.maximum(p['kappa_den_min'] * 3.0, 0.35)
    launching = jnp.abs(v_x) < launch_speed
    v_ground = jnp.maximum(jnp.abs(v_x), launch_speed)
    spinning = jnp.abs(v_wheel) > 1.5 * v_ground + 0.1
    return jnp.logical_and(spinning, jnp.logical_not(launching))


def _drive_torque_spin_scale(kappa, tau_motor, p):
    """Cut drive torque when |kappa| is large; keep full regen for recovery."""
    excess = jnp.maximum(jnp.abs(kappa) - p['kappa_torque_cut_start'], 0.0)
    scale = 1.0 / (1.0 + p['kappa_torque_cut_gain'] * excess)
    return jnp.where(tau_motor > 0.0, scale, 1.0)


def _motor_torque_command(translational_control, v_x, p):
    """Map planner translational_control to commanded motor-shaft torque [N·m]."""
    gear = jnp.maximum(p['gear_ratio'], 1.0e-3)
    accel_cmd = translational_control * 1.4
    throttle = jnp.clip(accel_cmd / jnp.maximum(p['a_max'], 1.0e-3), -1.0, 1.0)

    speed_scale = 1.0 - p['motor_torque_drop'] * jnp.clip(
        jnp.abs(v_x) / jnp.maximum(p['v_max'], 1.0e-3), 0.0, 1.0)
    speed_scale = jnp.maximum(speed_scale, 0.15)

    tau_from_torque = jnp.where(
        throttle >= 0.0,
        throttle * p['tau_wheel_max'] * speed_scale,
        throttle * p['tau_wheel_regen'],
    ) / gear

    tau_from_current = throttle * p['motor_I_max'] * p['motor_K_t']

    tau_from_accel = p['m'] * p['wheel_radius'] * accel_cmd / gear

    mode = p['drive_mode']
    tau_cmd = jnp.where(
        mode > 1.5,
        tau_from_current,
        jnp.where(mode > 0.5, tau_from_torque, tau_from_accel),
    )
    return tau_cmd


def _filter_motor_torque(tau_cmd, tau_motor, dt_sub, motor_tau_s):
    alpha = jnp.where(motor_tau_s > 1.0e-5, 1.0 - jnp.exp(-dt_sub / motor_tau_s), 1.0)
    return tau_motor + alpha * (tau_cmd - tau_motor)


def _sync_motor_omega(omega_motor, v_x, kappa, spinning, dt_sub, p):
    """Pull motor speed toward road speed; only decouple during deep sustained spin."""
    omega_kin = v_x * p['gear_ratio'] / jnp.maximum(p['wheel_radius'], 1.0e-4)
    control_dt = jnp.float32(Settings.TIMESTEP_CONTROL)
    base_alpha = jnp.minimum(dt_sub / control_dt, 1.0)
    deep_spin = jnp.logical_and(
        spinning,
        jnp.abs(kappa) > p['kappa_long_peak'] * 1.25,
    )
    # Low-speed driveline must stay tied to chassis or Fx oscillates (surging / boaty feel).
    low_speed = jnp.abs(v_x) < 1.5
    alpha = jnp.where(deep_spin, jnp.where(low_speed, 0.35 * base_alpha, 0.0), base_alpha)
    return omega_motor + alpha * (omega_kin - omega_motor)


def _clip_pacejka_state(psi_dot, v_x, v_y, psi, s_x, s_y, delta, omega_motor, p):
    """Keep states finite; cap motor revs from current speed (not v_max)."""
    v_cap = jnp.maximum(p['v_max'], 1.0)
    v_x = jnp.clip(v_x, p['v_min'], v_cap)
    v_y = jnp.clip(v_y, -v_cap, v_cap)
    psi_dot = jnp.clip(psi_dot, -10.0, 10.0)
    omega_kin = jnp.abs(v_x) * p['gear_ratio'] / jnp.maximum(p['wheel_radius'], 1.0e-4)
    omega_cap = jnp.maximum(omega_kin * 1.4 + 30.0, 50.0)
    omega_abs_max = jnp.minimum(
        omega_cap,
        v_cap * p['gear_ratio'] / jnp.maximum(p['wheel_radius'], 1.0e-4) * 1.2,
    )
    omega_motor = jnp.clip(omega_motor, -omega_abs_max, omega_abs_max)
    return psi_dot, v_x, v_y, psi, s_x, s_y, delta, omega_motor


def _longitudinal_speed_for_tires(v_x, eps=1.0e-3):
    """Signed minimum speed for tire slip angles (smooth through v_x ≈ 0)."""
    return jnp.sign(v_x + 1.0e-8) * jnp.maximum(jnp.abs(v_x), eps)


def _longitudinal_slip_kappa(v_x, omega_motor, wheel_radius, gear_ratio, kappa_den_min):
    omega_wheel = omega_motor / jnp.maximum(gear_ratio, 1.0e-3)
    v_wheel = omega_wheel * wheel_radius
    denom = jnp.maximum(jnp.maximum(jnp.abs(v_x), jnp.abs(v_wheel)), kappa_den_min)
    kappa = (v_wheel - v_x) / denom
    return jnp.clip(kappa, -1.5, 1.5), v_wheel


def _pacejka_step(s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                  omega_motor, tau_motor, p, dt_sub):
    mu, lf, lr, h_cg, m, I_z, g_ = p['mu'], p['lf'], p['lr'], p['h_cg'], p['m'], p['I_z'], p['g_']
    s_min, s_max = p['s_min'], p['s_max']
    v_dead, curve_resistance_factor = p['v_dead'], p['curve_resistance_factor']
    wheel_radius, wheel_inertia = p['wheel_radius'], p['wheel_inertia']
    gear_ratio, omega_viscous = p['gear_ratio'], p['omega_viscous']

    v_x_safe = _longitudinal_speed_for_tires(v_x)
    alpha_f = -jnp.arctan((v_y + psi_dot * lf) / v_x_safe) + delta
    alpha_r = -jnp.arctan((v_y - psi_dot * lr) / v_x_safe)

    F_zf = jnp.maximum(m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf), 0.0)
    F_zr = jnp.maximum(m * (v_x_dot * h_cg + g_ * lf) / (lr + lf), 0.0)

    F_yf_raw = _pacejka_tire_force(alpha_f, p['B_f'], p['C_f'], p['D_f'], p['E_f'], mu, F_zf)
    F_yr_raw = _pacejka_tire_force(alpha_r, p['B_r'], p['C_r'], p['D_r'], p['E_r'], mu, F_zr)

    kappa, v_wheel = _longitudinal_slip_kappa(v_x, omega_motor, wheel_radius, gear_ratio, p['kappa_den_min'])
    spinning = _wheelspin_active(v_x, v_wheel, p)
    Fx_f_raw = _pacejka_tire_force(kappa, p['B_xf'], p['C_xf'], p['D_xf'], p['E_xf'], mu, F_zf)
    Fx_r_raw = _pacejka_tire_force(kappa, p['B_xr'], p['C_xr'], p['D_xr'], p['E_xr'], mu, F_zr)
    slip_falloff = jnp.where(spinning, _longitudinal_slip_falloff(kappa, p), 1.0)

    fc_f = _friction_circle_scale(Fx_f_raw, F_yf_raw, mu, F_zf)
    fc_r = _friction_circle_scale(Fx_r_raw, F_yr_raw, mu, F_zr)
    Fx_f = Fx_f_raw * fc_f * slip_falloff
    Fx_r = Fx_r_raw * fc_r * slip_falloff
    F_yf = F_yf_raw * fc_f
    F_yr = F_yr_raw * fc_r
    Fx = Fx_f + Fx_r

    smooth_sign = v_x / jnp.sqrt(v_x * v_x + v_dead * v_dead)
    lateral_force_magnitude = jnp.sqrt(F_yf * F_yf + F_yr * F_yr)
    v_x_dot = Fx / m - curve_resistance_factor * lateral_force_magnitude / m * smooth_sign
    v_x_dot = _apply_rolling_resistance(v_x, v_x_dot, p['c_rr'], v_dead, g_, mu)

    spin_scale = jnp.where(spinning, _drive_torque_spin_scale(kappa, tau_motor, p), 1.0)
    tau_motor_eff = tau_motor * spin_scale
    tau_tire_at_motor = Fx * wheel_radius / jnp.maximum(gear_ratio, 1.0e-3)
    spin_excess = jnp.maximum(jnp.abs(kappa) - p['kappa_long_peak'], 0.0)
    spin_drag = jnp.where(
        spinning,
        p['kappa_spin_drag'] * spin_excess * jnp.abs(omega_motor),
        0.0,
    )
    d_omega_m = (
        tau_motor_eff - tau_tire_at_motor - omega_viscous * omega_motor - spin_drag
    ) / jnp.maximum(wheel_inertia, 1.0e-8)

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
    psi = psi + dt_sub * d_psi
    psi_dot = psi_dot + dt_sub * d_psi_dot
    omega_motor = omega_motor + dt_sub * d_omega_m
    omega_motor = _sync_motor_omega(omega_motor, v_x, kappa, spinning, dt_sub, p)
    psi_dot, v_x, v_y, psi, s_x, s_y, delta, omega_motor = _clip_pacejka_state(
        psi_dot, v_x, v_y, psi, s_x, s_y, delta, omega_motor, p,
    )
    return s_x, s_y, delta, v_x, v_y, psi, psi_dot, omega_motor


def _pacejka_step_legacy(s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                         lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r,
                         mu, s_min, s_max, dt_sub, v_dead, curve_resistance_factor):
    v_x_safe = _longitudinal_speed_for_tires(v_x)
    alpha_f = -jnp.arctan((v_y + psi_dot * lf) / v_x_safe) + delta
    alpha_r = -jnp.arctan((v_y - psi_dot * lr) / v_x_safe)

    F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
    F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

    F_yf = mu * F_zf * D_f * jnp.sin(
        C_f * jnp.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - jnp.arctan(B_f * alpha_f))))
    F_yr = mu * F_zr * D_r * jnp.sin(
        C_r * jnp.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - jnp.arctan(B_r * alpha_r))))

    smooth_sign = v_x / jnp.sqrt(v_x * v_x + v_dead * v_dead)
    lateral_force_magnitude = jnp.sqrt(F_yf * F_yf + F_yr * F_yr)
    v_x_dot = v_x_dot - curve_resistance_factor * lateral_force_magnitude / m * smooth_sign

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
    psi = psi + dt_sub * d_psi
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
    psi = psi + dt_sub * psi_dot
    angular_vel_z = angular_vel_z + dt_sub * psi_dot_dot
    return s_x, s_y, delta, v_x, psi, angular_vel_z


def _kinematic_motor_omega(v_x, gear_ratio, wheel_radius):
    return v_x * jnp.maximum(gear_ratio, 1.0e-3) / jnp.maximum(wheel_radius, 1.0e-4)


def _next_step_output(psi_dot, v_x, v_y, psi, s_x, s_y, delta, omega_motor):
    v_x_safe = _longitudinal_speed_for_tires(v_x)
    slip_angle = jnp.arctan(v_y / v_x_safe)
    return jnp.array([
        psi_dot, v_x, v_y, psi, jnp.cos(psi), jnp.sin(psi),
        s_x, s_y, slip_angle, delta, omega_motor,
    ], dtype=jnp.float32)


def _read_omega_motor(state_in, v_x, gear_ratio, wheel_radius):
    if state_in.shape[0] > 10:
        return state_in[10]
    return _kinematic_motor_omega(v_x, gear_ratio, wheel_radius)


@partial(jax.jit, static_argnames=['intermediate_steps', 'ode_model'])
def car_dynamics_pacejka_jax(state, control, car_params, dt, intermediate_steps=1,
                             ode_model='ODE:ks_pacejka'):
    """Advance car dynamics (ODE_TF-equivalent with optional longitudinal slip).

    Args:
        state: [psi_dot, v_x, v_y, psi, cos, sin, s_x, s_y, slip, delta, motor_angular_vel]
        control: [desired_steering_angle, translational_control]
        car_params: VehicleParameters.to_np_array()
        dt: integration timestep
        intermediate_steps: substeps per call
        ode_model: 'ODE:pacejka', 'ODE:ks', or 'ODE:ks_pacejka'
    """
    p = _unpack_car_params(car_params)
    l_wb = p['lf'] + p['lr']
    dt_sub = dt / intermediate_steps
    use_speed_pi = Settings.MOTOR_PID_IN_CAR_MODEL
    use_ks = ode_model in ('ODE:ks', 'ODE:ks_pacejka')
    use_pacejka = ode_model in ('ODE:pacejka', 'ODE:ks_pacejka')
    use_blend = ode_model == 'ODE:ks_pacejka'

    def single_step(carry, _):
        state_in, speed_error_integral, tau_motor = carry
        psi_dot, v_x, v_y, psi, _, _, s_x, s_y, _, delta = state_in[:10]
        omega_motor = _read_omega_motor(state_in, v_x, p['gear_ratio'], p['wheel_radius'])
        desired_steering_angle, translational_control = control

        delta_dot = _servo_proportional(
            desired_steering_angle, delta, p['servo_p'], p['steering_diff_low'], p['sv_min'], p['sv_max'])
        delta_dot = _steering_constraints(delta, delta_dot, p['s_min'], p['s_max'], p['sv_min'], p['sv_max'])

        def _longitudinal_slip_branch(_):
            tau_cmd = _motor_torque_command(translational_control, v_x, p)
            sei = speed_error_integral
            if use_speed_pi:
                speed_kp = 10.0 * p['a_max'] / jnp.maximum(p['v_max'], 1e-3)
                speed_ki = 0.5 * speed_kp
                speed_error = translational_control - v_x
                sei = jnp.clip(speed_error_integral + speed_error * dt_sub, -5.0, 5.0)
                accel_pi = speed_kp * speed_error + speed_ki * sei
                tau_cmd = _motor_torque_command(accel_pi / 1.4, v_x, p)
            tau_m = _filter_motor_torque(tau_cmd, tau_motor, dt_sub, p['motor_tau_s'])
            return jnp.float32(0.0), tau_m, sei

        def _longitudinal_legacy_branch(_):
            if use_speed_pi:
                speed_kp = 10.0 * p['a_max'] / jnp.maximum(p['v_max'], 1e-3)
                speed_ki = 0.5 * speed_kp
                speed_error = translational_control - v_x
                sei = jnp.clip(speed_error_integral + speed_error * dt_sub, -5.0, 5.0)
                vx_dot = speed_kp * speed_error + speed_ki * sei
            else:
                vx_dot = translational_control
                sei = speed_error_integral
            vx_dot = _accl_constraints(
                v_x, vx_dot, p['a_min'], p['a_max'], p['v_min'], p['v_max'], p['v_switch'], p['mu'], p['g_'])
            vx_dot = _apply_rolling_resistance(v_x, vx_dot, p['c_rr'], p['v_dead'], p['g_'], p['mu'])
            return vx_dot, jnp.float32(0.0), sei

        v_x_dot, tau_motor, speed_error_integral = jax.lax.cond(
            p['use_slip'], _longitudinal_slip_branch, _longitudinal_legacy_branch, None)

        if use_pacejka:
            def _pacejka_slip_step(_):
                return _pacejka_step(
                    s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                    omega_motor, tau_motor, p, dt_sub)

            def _pacejka_legacy_step(_):
                leg = _pacejka_step_legacy(
                    s_x, s_y, delta, v_x, v_y, psi, psi_dot, delta_dot, v_x_dot,
                    p['lf'], p['lr'], p['h_cg'], p['m'], p['I_z'], p['g_'],
                    p['B_f'], p['C_f'], p['D_f'], p['E_f'], p['B_r'], p['C_r'], p['D_r'], p['E_r'],
                    p['mu'], p['s_min'], p['s_max'], dt_sub, p['v_dead'], p['curve_resistance_factor'])
                omega = _kinematic_motor_omega(leg[3], p['gear_ratio'], p['wheel_radius'])
                return leg[0], leg[1], leg[2], leg[3], leg[4], leg[5], leg[6], omega

            p_s_x, p_s_y, p_delta, p_v_x, p_v_y, p_psi, p_psi_dot, omega_motor = jax.lax.cond(
                p['use_slip'], _pacejka_slip_step, _pacejka_legacy_step, None)
            s_pacejka = _next_step_output(p_psi_dot, p_v_x, p_v_y, p_psi, p_s_x, p_s_y, p_delta, omega_motor)
        else:
            s_pacejka = state_in if state_in.shape[0] > 10 else jnp.concatenate(
                [state_in, jnp.array([omega_motor], dtype=jnp.float32)])

        if use_ks:
            angular_vel_z = psi_dot
            k_s_x, k_s_y, k_delta, k_v_x, k_psi, k_angular_vel_z = _ks_step(
                s_x, s_y, delta, v_x, psi, angular_vel_z, delta_dot, v_x_dot,
                l_wb, p['s_min'], p['s_max'], dt_sub)
            omega_ks = _kinematic_motor_omega(k_v_x, p['gear_ratio'], p['wheel_radius'])
            s_ks = _next_step_output(k_angular_vel_z, k_v_x, 0.0, k_psi, k_s_x, k_s_y, k_delta, omega_ks)
        else:
            s_ks = state_in if state_in.shape[0] > 10 else jnp.concatenate(
                [state_in, jnp.array([omega_motor], dtype=jnp.float32)])

        if use_blend:
            def _blend_ks_pacejka(_):
                weight = 1.0 / (1.0 + jnp.exp(-4.8 * (v_x - 1.75)))
                return (1.0 - weight) * s_ks + weight * s_pacejka

            def _blend_slip_lateral_ks(_):
                """Slip driveline for v_x; KS lateral dynamics below ~1 m/s (Pacejka singularity)."""
                w = 1.0 / (1.0 + jnp.exp(-6.0 * (jnp.abs(v_x) - 1.0)))
                blended = w * s_pacejka + (1.0 - w) * s_ks
                return blended.at[1].set(s_pacejka[1]).at[10].set(s_pacejka[10])

            next_state = jax.lax.cond(
                p['use_slip'], _blend_slip_lateral_ks, _blend_ks_pacejka, None)
        elif use_pacejka:
            next_state = s_pacejka
        else:
            next_state = s_ks

        return (next_state, speed_error_integral, tau_motor), None

    init_omega = _read_omega_motor(state, state[1], p['gear_ratio'], p['wheel_radius'])
    init_carry = (state if state.shape[0] > 10 else jnp.concatenate(
        [state, jnp.array([init_omega], dtype=jnp.float32)]),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32))
    final_carry, _ = jax.lax.scan(single_step, init_carry, None, length=intermediate_steps)
    return final_carry[0]


def car_dynamics_pacejka_jax_from_settings(state, control, car_params, dt, intermediate_steps=1):
    """Wrapper using Settings.ODE_MODEL_OF_CAR_DYNAMICS."""
    return car_dynamics_pacejka_jax(
        state, control, car_params, dt,
        intermediate_steps=intermediate_steps,
        ode_model=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
    )

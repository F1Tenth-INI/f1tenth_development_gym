from typing import List, Optional
import yaml
import os
import numpy as np

from utilities.Settings import Settings


# Required in every car YAML used by the simulator / jax dynamics.
_REQUIRED_YAML_KEYS = (
    'mu', 'lf', 'lr', 'h', 'm', 'I_z', 'g', 'width', 'length',
    's_min', 's_max', 'sv_min', 'sv_max',
    'a_max', 'a_min', 'v_min', 'v_max', 'v_switch',
    'servo_p', 'steering_diff_low',
    'C_Pf', 'C_Pr', 'c_rr', 'v_dead', 'curve_resistance_factor', 'brake_multiplier',
)

# Legacy / MPC-only keys: loaded when present, otherwise defaults (not passed to jax).
_LEGACY_DEFAULTS = {
    'C_Sf': 4.718,
    'C_Sr': 5.4562,
    'l_wb': None,
    'h_cg': None,
    'min_speed_st': 0.5,
    'min_speed_pacejka': 0.7,
    'C_0d': 0.41117415569890003,
    'C_R': 3.693303119695026,
    'C_acc': 5.0,
    'C_d': 0.0,
    'C_dec': 6.0,
}

# Longitudinal slip / driveline (optional in YAML).
_LONGITUDINAL_DEFAULTS = {
    'use_longitudinal_slip': False,
    'drive_mode': 'accel',
    'wheel_radius_m': 0.05,
    'wheel_inertia_kg_m2': 0.00012,
    'tau_wheel_max_nm': 0.5,
    'tau_wheel_regen_max_nm': 0.5,
    'omega_viscous_damping': 2e-5,
    'kappa_den_min_m_s': 0.12,
    'motor_longitudinal_tau_s': 0.0,
    'motor_speed_torque_drop': 0.0,
    'motor_current_max_a': 60.0,
    'motor_K_t': 0.012,
    'gear_ratio': 6.6,
    'C_Pxf': [12.0, 1.9, 1.0, 0.97],
    'C_Pxr': [12.0, 1.9, 1.0, 0.97],
    # High-slip recovery (Fx falloff + drive-torque cut + wheel spin drag).
    'kappa_long_peak': 0.12,
    'kappa_long_falloff': 0.14,
    'kappa_torque_cut_start': 0.16,
    'kappa_torque_cut_gain': 4.0,
    'kappa_spin_drag': 0.02,
}

_DRIVE_MODE_ID = {'accel': 0.0, 'torque': 1.0, 'current': 2.0}


class VehicleParameters:
    mu: float
    lf: float
    lr: float
    h: float
    m: float
    I_z: float
    g: float
    width: float
    length: float
    s_min: float
    s_max: float
    sv_min: float
    sv_max: float
    a_max: float
    a_min: float
    v_min: float
    v_max: float
    v_switch: float
    servo_p: float
    steering_diff_low: float
    C_Pf: List[float]
    C_Pr: List[float]
    c_rr: float
    v_dead: float
    curve_resistance_factor: float
    brake_multiplier: float

    def __init__(self, param_file_name='gym_car_parameters.yml'):
        current_dir = os.path.dirname(__file__)
        yaml_file_path = os.path.join(current_dir, param_file_name)

        with open(yaml_file_path, 'r') as file:
            params = yaml.safe_load(file)

        for key in _REQUIRED_YAML_KEYS:
            if key not in params:
                raise ValueError(f"Parameter '{key}' not found in {param_file_name}.")

        for key in _REQUIRED_YAML_KEYS:
            setattr(self, key, params[key])

        for key, default in _LEGACY_DEFAULTS.items():
            setattr(self, key, params[key] if key in params else default)

        if getattr(self, 'l_wb') is None:
            self.l_wb = self.lf + self.lr
        if getattr(self, 'h_cg') is None:
            self.h_cg = self.h

        for key, default in _LONGITUDINAL_DEFAULTS.items():
            setattr(self, key, params[key] if key in params else default)

        if Settings.SURFACE_FRICTION is not None:
            self.mu = Settings.SURFACE_FRICTION

    def to_dict(self):
        d = {k: getattr(self, k) for k in _REQUIRED_YAML_KEYS}
        d.update({k: getattr(self, k) for k in _LEGACY_DEFAULTS})
        d.update({k: getattr(self, k) for k in _LONGITUDINAL_DEFAULTS})
        return d

    def to_np_array(self):
        """Flat parameter vector for jax/numba dynamics (see dynamic_model_pacejka_jax)."""
        drive_mode = _DRIVE_MODE_ID.get(str(self.drive_mode).lower(), 0.0)
        return np.array([
            self.mu,
            self.lf,
            self.lr,
            self.h,
            self.m,
            self.I_z,
            self.g,
            self.C_Pf[0], self.C_Pf[1], self.C_Pf[2], self.C_Pf[3],
            self.C_Pr[0], self.C_Pr[1], self.C_Pr[2], self.C_Pr[3],
            self.servo_p,
            self.s_min, self.s_max, self.sv_min, self.sv_max,
            self.a_min, self.a_max, self.v_min, self.v_max, self.v_switch,
            self.c_rr,
            self.v_dead,
            self.curve_resistance_factor,
            self.brake_multiplier,
            self.steering_diff_low,
            1.0 if self.use_longitudinal_slip else 0.0,
            self.wheel_radius_m,
            self.wheel_inertia_kg_m2,
            self.tau_wheel_max_nm,
            self.tau_wheel_regen_max_nm,
            self.omega_viscous_damping,
            self.kappa_den_min_m_s,
            self.motor_longitudinal_tau_s,
            self.motor_speed_torque_drop,
            self.gear_ratio,
            self.motor_K_t,
            self.motor_current_max_a,
            drive_mode,
            self.C_Pxf[0], self.C_Pxf[1], self.C_Pxf[2], self.C_Pxf[3],
            self.C_Pxr[0], self.C_Pxr[1], self.C_Pxr[2], self.C_Pxr[3],
            self.kappa_long_peak,
            self.kappa_long_falloff,
            self.kappa_torque_cut_start,
            self.kappa_torque_cut_gain,
            self.kappa_spin_drag,
        ], dtype=np.float32)

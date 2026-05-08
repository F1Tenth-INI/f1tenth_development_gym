from typing import List
import yaml
import os
import numpy as np

from utilities.Settings import Settings

_OPTIONAL_CAR_PARAM_DEFAULTS = {
    "motor_longitudinal_tau_s": 0.0,
    "motor_speed_torque_drop": 0.0,
    "use_longitudinal_slip": False,
    "wheel_radius_m": 0.033,
    "wheel_inertia_kg_m2": 0.00012,
    "tau_wheel_max_nm": 0.55,
    "tau_wheel_regen_max_nm": 0.45,
    "omega_viscous_damping": 0.0,
    "kappa_den_min_m_s": 0.12,
    "C_Pxf": [10.0, 1.9, 1.0, 0.97],
    "C_Pxr": [10.0, 1.9, 1.0, 0.97],
    # Public control input is normalized in [-1, 1]. ``drive_mode`` says how
    # the dynamics translate that normalized stick into a physical command.
    # ``auto`` picks a sensible default from legacy flags so older YAMLs
    # without ``drive_mode`` keep behaving as before:
    #   use_longitudinal_slip=True  -> "torque"
    #   otherwise                   -> "accel"
    # NOTE: there is intentionally no "speed" mode here. Desired-speed -> stick
    # conversion belongs to the controller (see
    # ``utilities.controller_utilities.ControllerUtilities.motor_pid``); the
    # dynamics describe physics only.
    "drive_mode": "auto",  # "torque" | "accel" | "current" | "auto"
    # Only used when drive_mode == "current" (VESC current/I_q control). The
    # equivalent wheel torque is throttle_norm * motor_current_max_a *
    # motor_K_t * gear_ratio. Defaults are reasonable RC-class placeholders.
    "motor_current_max_a": 60.0,
    "motor_K_t": 0.012,
    "gear_ratio": 6.6,
}


# Numeric IDs used in to_np_array so JIT/JAX kernels don't need to handle
# Python strings. Keep these in lock-step with the unpacking in the dynamics.
_DRIVE_MODE_IDS = {"torque": 0, "accel": 1, "current": 2}


def _drive_mode_id(name):
    if name not in _DRIVE_MODE_IDS:
        raise ValueError(
            f"Unknown drive_mode '{name}'. Allowed: {list(_DRIVE_MODE_IDS)}"
        )
    return float(_DRIVE_MODE_IDS[name])


class VehicleParameters:
    mu: float # Surface friction
    C_Sf: float # Front tire cornering stiffness [N/rad]
    C_Sr: float # Rear tire cornering stiffness [N/rad]
    lf: float # Distance from the center of mass to the front axle [m]
    lr: float # Distance from the center of mass to the rear axle [m]
    l_wb: float # Wheelbase [m]
    h: float # Height of the center of mass [m]
    m: float  # mass of the car [kg]
    I_z: float # Moment of inertia around the z-axis [kg*m^2]
    g: float # Gravitational acceleration [m/s^2]
    width: float # Width of the car [m]
    length: float # Length of the car [m]
    s_min: float # Minimum steering angle [rad]
    s_max: float # Maximum steering angle [rad]
    sv_min: float # Minimum steering velocity [rad/s]
    sv_max: float # Maximum steering velocity [rad/s]
    a_max: float # Maximum acceleration [m/s^2]
    a_min: float # Minimum acceleration [m/s^2]
    v_min: float # Minimum velocity [m/s]
    v_max: float # Maximum velocity [m/s]
    v_switch: float # Switching velocity [m/s]: From here accelerating is harder
    servo_p: float # Servo proportional gain
    steering_diff_low: float
    min_speed_st: float
    
    # Pacejka parameters
    C_0d: float 
    C_Pf: List[float] # Pacejka parameters for the front tires [B, C, D, E]
    C_Pr: List[float] # Pacejka parameters for the rear tires [B, C, D, E]
    
    c_rr: float # Rolling resistance coefficient
    
    # Additional resistance parameters
    v_dead: float # velocity deadband for smooth sign function [m/s]
    curve_resistance_factor: float # additional resistance factor for cornering
    brake_multiplier: float # asymmetric braking effectiveness multiplier
    motor_longitudinal_tau_s: float  # low-pass time constant on accel command (0 = off)
    motor_speed_torque_drop: float  # scales down max forward accel at v_max (0 = off)
    use_longitudinal_slip: bool  # Pacejka Fx + wheel omega state (4WD locked axles)
    wheel_radius_m: float
    wheel_inertia_kg_m2: float  # equivalent inertia at shaft (4 wheels, no center diff)
    tau_wheel_max_nm: float  # max drive torque command at wheels (sum both axles) [N·m]
    tau_wheel_regen_max_nm: float  # max braking torque magnitude [N·m]
    omega_viscous_damping: float  # N·m·s/rad viscous loss at shaft
    kappa_den_min_m_s: float  # regularizes slip ratio denominator near standstill
    C_Pxf: List[float]  # Pacejka longitudinal front (B,C,D,E)
    C_Pxr: List[float]  # Pacejka longitudinal rear (B,C,D,E)
    drive_mode: str  # "torque" | "accel" | "speed" | "current"
    motor_current_max_a: float  # for drive_mode="current": max motor current [A]
    motor_K_t: float  # motor torque constant [N·m / A]
    gear_ratio: float  # motor shaft -> wheel shaft (>1 means motor faster)

    """
    Initializes a new instance of the CarParameters class.

    This method sets the class variables based on the parameters
    defined in the specified YAML file. It also allows for the overwriting
    of the surface friction value if it is specified in the Settings.

    :param param_file_name: The name of the YAML file containing car parameters.
                            Defaults to 'gym_car_parameters.yaml'.
    """

    def __init__(self, param_file_name='gym_car_parameters.yml'):
        class_variable_names = list(VehicleParameters.__annotations__.keys())
        current_dir = os.path.dirname(__file__)
        yaml_file_path = os.path.join(current_dir, param_file_name)

        with open(yaml_file_path, 'r') as file:
          params = yaml.safe_load(file)
          for class_variable_name in class_variable_names:
            if class_variable_name not in params:
              if class_variable_name in _OPTIONAL_CAR_PARAM_DEFAULTS:
                setattr(
                    self,
                    class_variable_name,
                    _OPTIONAL_CAR_PARAM_DEFAULTS[class_variable_name],
                )
              else:
                raise ValueError(
                    f"Parameter '{class_variable_name}' not found in the YAML file."
                )
            else:
              setattr(self, class_variable_name, params[class_variable_name])

        # Overwrite Sufrace friction
        if Settings.SURFACE_FRICTION is not None:
            self.mu = Settings.SURFACE_FRICTION

        # Resolve "auto" drive_mode to a concrete one. This keeps older YAMLs
        # that predate this field behaving the same as before. The legacy
        # ``MOTOR_PID_IN_CAR_MODEL`` flag is intentionally not honored: a
        # speed-PID front-end now lives in the controller, not the dynamics.
        if isinstance(self.drive_mode, str) and self.drive_mode.lower() == "auto":
            if bool(self.use_longitudinal_slip):
                self.drive_mode = "torque"
            else:
                self.drive_mode = "accel"
        if self.drive_mode not in _DRIVE_MODE_IDS:
            if str(self.drive_mode).lower() == "speed":
                raise ValueError(
                    f"drive_mode='speed' is no longer supported. The speed-PID "
                    f"frontend has moved to the controller. In your planner, "
                    f"call ControllerUtilities.motor_pid(desired_speed, v_x) "
                    f"and use drive_mode='accel' (or 'torque') in {param_file_name}."
                )
            raise ValueError(
                f"Invalid drive_mode '{self.drive_mode}' in {param_file_name}. "
                f"Allowed: {list(_DRIVE_MODE_IDS)} (or 'auto')."
            )

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__annotations__}

    def to_np_array(self):
        return np.array([
            # Simulator engine Car parameters
            self.mu,  # mu (friction coefficient)
            self.lf,  # lf (distance from center of gravity to front axle)
            self.lr,  # lr (distance from center of gravity to rear axle)
            self.h,  # h_cg (center of gravity height of sprung mass)
            self.m,  # m (Total Mass of car)
            self.I_z,  # I_z (Moment of inertia about z-axis)
            self.g,  # g (Gravitation Constant)

            # Pacejka Magic Formula Parameters (Front Tire)
            self.C_Pf[0],  # B_f
            self.C_Pf[1],  # C_f
            self.C_Pf[2],  # D_f
            self.C_Pf[3],  # E_f

            # Pacejka Magic Formula Parameters (Rear Tire)
            self.C_Pr[0],  # B_r
            self.C_Pr[1],  # C_r
            self.C_Pr[2],  # D_r
            self.C_Pr[3],  # E_r

            # Steering Constraints
            self.servo_p,  # servo_p (proportional factor of servo PID)
            self.s_min,  # s_min (min steering angle)
            self.s_max,  # s_max (max steering angle)
            self.sv_min,  # sv_min (min steering velocity)
            self.sv_max,  # sv_max (max steering velocity)

            # Acceleration Constraints
            self.a_min,  # a_min (min acceleration)
            self.a_max,  # a_max (max acceleration)
            self.v_min,  # v_min (min velocity)
            self.v_max,  # v_max (max velocity)
            self.v_switch,  # v_switch (velocity threshold for model transition)
            
            self.c_rr, # c_rr (rolling resistance coefficient)
            
            # Additional resistance parameters
            self.v_dead, # v_dead (velocity deadband for smooth sign function)
            self.curve_resistance_factor, # curve_resistance_factor (cornering resistance)
            self.brake_multiplier, # brake_multiplier (asymmetric braking effectiveness)
            self.steering_diff_low,  # steering deadband / small-angle cutoff [rad]

            # RC-style driveline (consumed by jax_pacejka; jit_Pacejka leaves them as no-op)
            self.motor_longitudinal_tau_s,  # first-order accel-command time constant [s]
            self.motor_speed_torque_drop,   # fractional max-accel drop from 0 to v_max
            # Longitudinal slip + VESC-style torque path (4WD locked, one shaft speed)
            float(self.use_longitudinal_slip),
            self.wheel_radius_m,
            self.wheel_inertia_kg_m2,
            self.tau_wheel_max_nm,
            self.tau_wheel_regen_max_nm,
            self.omega_viscous_damping,
            self.kappa_den_min_m_s,
            self.C_Pxf[0],
            self.C_Pxf[1],
            self.C_Pxf[2],
            self.C_Pxf[3],
            self.C_Pxr[0],
            self.C_Pxr[1],
            self.C_Pxr[2],
            self.C_Pxr[3],
            # Public control input is normalized [-1, 1]; this id picks the
            # YAML-defined translation in the dynamics (torque / accel /
            # current). Replaces the old Settings.MOTOR_PID_IN_CAR_MODEL
            # flag, which was global and unit-inconsistent across cars.
            _drive_mode_id(self.drive_mode),
            self.motor_current_max_a,
            self.motor_K_t,
            self.gear_ratio,
        ], dtype=np.float32)
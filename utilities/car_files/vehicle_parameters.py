from typing import List
import yaml
import os
import numpy as np

from utilities.Settings import Settings

class VehicleParameters:
    mu: float  # Surface friction
    lf: float  # Distance from the center of mass to the front axle [m]
    lr: float  # Distance from the center of mass to the rear axle [m]
    l_wb: float  # Wheelbase [m] (used by SI_Toolkit KS model)
    h: float  # Height of the center of mass [m]
    m: float  # Mass of the car [kg]
    I_z: float  # Moment of inertia around the z-axis [kg*m^2]
    g: float  # Gravitational acceleration [m/s^2]
    width: float  # Width of the car [m]
    length: float  # Length of the car [m]
    s_min: float  # Minimum steering angle [rad]
    s_max: float  # Maximum steering angle [rad]
    sv_min: float  # Minimum steering velocity [rad/s]
    sv_max: float  # Maximum steering velocity [rad/s]
    a_max: float  # Maximum acceleration [m/s^2]
    a_min: float  # Minimum acceleration [m/s^2]
    v_min: float  # Minimum velocity [m/s]
    v_max: float  # Maximum velocity [m/s]
    v_switch: float  # Switching velocity [m/s]: from here accelerating is harder
    servo_p: float  # Servo proportional gain
    steering_diff_low: float  # Steering deadband [rad]
    C_Pf: List[float]  # Pacejka parameters for the front tires [B, C, D, E]
    C_Pr: List[float]  # Pacejka parameters for the rear tires [B, C, D, E]
    c_rr: float  # Rolling resistance coefficient (dimensionless)
    v_dead: float  # Velocity deadband for smooth sign [m/s]
    curve_resistance_factor: float  # Extra longitudinal drag scaling with lateral tire load
    brake_multiplier: float  # Scales commanded deceleration (values < 1 weaken braking)
    accel_multiplier: float  # Scales commanded acceleration (values < 1 weaken acceleration)
    steering_multiplier: float  # Scales commanded steering angle (values < 1 reduce steering)
    imu_x: float  # IMU offset from rear axle, body x forward [m]
    imu_y: float  # IMU offset from rear axle, body y left [m]
    wheel_radius: float  # Drive wheel radius [m]
    motor_current_gain: float  # Motor current model gain [A/(m/s)]
    motor_current_max_a: float  # Motor current saturation [A]

    def __init__(self, param_file_name='gym_car_parameters.yml'):
        class_variable_names = list(VehicleParameters.__annotations__.keys())
        optional_defaults = {
            "imu_x": 0.0,
            "imu_y": 0.0,
        }
        current_dir = os.path.dirname(__file__)
        yaml_file_path = os.path.join(current_dir, param_file_name)

        with open(yaml_file_path, 'r') as file:
          params = yaml.safe_load(file)
          for class_variable_name in class_variable_names:
            if class_variable_name not in params:
              if class_variable_name in optional_defaults:
                setattr(self, class_variable_name, optional_defaults[class_variable_name])
              else:
                raise ValueError(f"Parameter '{class_variable_name}' not found in the YAML file.")
            else:
              setattr(self, class_variable_name, params[class_variable_name])

        if Settings.SURFACE_FRICTION is not None:
            self.mu = Settings.SURFACE_FRICTION

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__annotations__}

    def to_np_array(self):
        """Flat array consumed by dynamic_model_pacejka_jax (indices 0-31)."""
        return np.array([
            self.mu,
            self.lf,
            self.lr,
            self.h,
            self.m,
            self.I_z,
            self.g,
            self.C_Pf[0],
            self.C_Pf[1],
            self.C_Pf[2],
            self.C_Pf[3],
            self.C_Pr[0],
            self.C_Pr[1],
            self.C_Pr[2],
            self.C_Pr[3],
            self.servo_p,
            self.s_min,
            self.s_max,
            self.sv_min,
            self.sv_max,
            self.a_min,
            self.a_max,
            self.v_min,
            self.v_max,
            self.v_switch,
            self.steering_diff_low,
            self.c_rr,
            self.v_dead,
            self.curve_resistance_factor,
            self.brake_multiplier,
            self.accel_multiplier,
            self.steering_multiplier,
        ], dtype=np.float32)

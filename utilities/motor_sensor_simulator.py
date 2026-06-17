import math

import numpy as np

from utilities.Settings import Settings
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    STEERING_ANGLE_IDX,
    TRANSLATIONAL_CONTROL_IDX,
)


class MotorSensorSimulator:
    """Simulate drivetrain / motor sensor readings from vehicle state and control."""

    MOTOR_SENSOR_KEYS = (
        "wheel_speed_rear_rad_s",
        "wheel_speed_front_rad_s",
        "wheel_speed_rear_rpm",
        "wheel_speed_front_rpm",
        "steering_rate",
        "longitudinal_accel",
        "motor_current_a",
        "motor_throttle",
        "motor_brake",
    )

    @staticmethod
    def from_states(state, prev_state, control, car_params, dt=None):
        """Compute motor sensor dict from current and previous vehicle states."""
        if dt is None:
            dt = Settings.TIMESTEP_SIM

        state = np.asarray(state, dtype=np.float64)
        prev_state = np.asarray(prev_state, dtype=np.float64)
        control = np.asarray(control, dtype=np.float64)

        wheel_radius = float(car_params.wheel_radius)
        motor_gain = float(car_params.motor_current_gain)
        motor_max_a = float(car_params.motor_current_max_a)

        v_x = float(state[LINEAR_VEL_X_IDX])
        delta = float(state[STEERING_ANGLE_IDX])
        prev_delta = float(prev_state[STEERING_ANGLE_IDX])
        prev_v_x = float(prev_state[LINEAR_VEL_X_IDX])
        cos_delta = max(abs(math.cos(delta)), 1e-3)

        rear_rad_s = v_x / wheel_radius
        front_rad_s = v_x / (wheel_radius * cos_delta)
        rpm_scale = 60.0 / (2.0 * math.pi)

        v_cmd = float(control[TRANSLATIONAL_CONTROL_IDX])
        v_max = max(float(car_params.v_max), 1e-3)
        speed_error = v_cmd - v_x
        motor_current = np.clip(motor_gain * speed_error, -motor_max_a, motor_max_a)
        if v_cmd >= 0.0:
            throttle = float(np.clip(v_cmd / v_max, 0.0, 1.0))
            brake = float(np.clip(max(-speed_error, 0.0) / v_max, 0.0, 1.0))
        else:
            throttle = 0.0
            brake = float(np.clip(abs(v_cmd) / v_max, 0.0, 1.0))

        steering_rate = (delta - prev_delta) / dt if dt > 0.0 else 0.0
        longitudinal_accel = (v_x - prev_v_x) / dt if dt > 0.0 else 0.0

        return {
            "wheel_speed_rear_rad_s": rear_rad_s,
            "wheel_speed_front_rad_s": front_rad_s,
            "wheel_speed_rear_rpm": rear_rad_s * rpm_scale,
            "wheel_speed_front_rpm": front_rad_s * rpm_scale,
            "steering_rate": float(steering_rate),
            "longitudinal_accel": float(longitudinal_accel),
            "motor_current_a": float(motor_current),
            "motor_throttle": throttle,
            "motor_brake": brake,
        }

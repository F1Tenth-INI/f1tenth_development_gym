import time

import pygame

from Control_Toolkit_ASF.Controllers import template_planner
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.my_joystick import UniversalJoystick
from utilities.Settings import Settings


def _expo(x: float, expo: float) -> float:
    """Symmetric exponential curve in [-1, 1]: y = (1 - expo) * x + expo * x**3."""
    return (1.0 - expo) * x + expo * (x * x * x)


class manual_planner(template_planner):
    """RC-style manual planner.

    When the car YAML has ``use_longitudinal_slip: true``, the longitudinal stick
    commands wheel torque [N·m] (VESC-style), scaled by ``tau_wheel_max_nm`` /
    ``tau_wheel_regen_max_nm``. Otherwise it commands acceleration [m/s²] (legacy).
    """

    STEER_MAX_RAD = 0.4
    STEER_EXPO = 0.55

    THROTTLE_MAX = 5.0
    BRAKE_MAX = 6.0
    THROTTLE_EXPO = 0.35
    BRAKE_EXPO = 0.20

    THROTTLE_RATE_UP = 25.0
    THROTTLE_RATE_DOWN = 60.0

    def __init__(self):
        super().__init__()

        self._car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
        self._use_slip = bool(self._car_params.use_longitudinal_slip)

        pygame.init()

        self.joystick = UniversalJoystick(
            index=0,
            deadzone=0.08,
            steering_invert=True,
            throttle_invert=True,
            auto_calibrate=False,
            prefer_detected_axis3_for_sony=True,
        )

        self.angular_control = None
        self.translational_control = None

        self.angular_control_normed = None
        self.translational_control_normed = None

        self._throttle_state = 0.0
        self._last_t = None

    def process_observation(self):
        self.angular_control_normed, self.translational_control_normed = self.joystick.read()

        steer_normed = self.angular_control_normed
        throttle_normed = self.translational_control_normed

        steer_shaped = _expo(steer_normed, self.STEER_EXPO)
        self.angular_control = self.STEER_MAX_RAD * steer_shaped

        if self._use_slip:
            if throttle_normed >= 0.0:
                shaped = _expo(throttle_normed, self.THROTTLE_EXPO)
                target = float(self._car_params.tau_wheel_max_nm) * shaped
            else:
                shaped = _expo(throttle_normed, self.BRAKE_EXPO)
                target = -float(self._car_params.tau_wheel_regen_max_nm) * abs(shaped)
        else:
            if throttle_normed >= 0.0:
                shaped = _expo(throttle_normed, self.THROTTLE_EXPO)
                target = self.THROTTLE_MAX * shaped
            else:
                shaped = _expo(throttle_normed, self.BRAKE_EXPO)
                target = self.BRAKE_MAX * shaped

        now = time.monotonic()
        if self._last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, min(now - self._last_t, 0.1))
        self._last_t = now

        slew = self.THROTTLE_RATE_UP if target >= self._throttle_state else self.THROTTLE_RATE_DOWN
        max_step = slew * dt
        delta = target - self._throttle_state
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step
        self._throttle_state += delta

        self.translational_control = float(self._throttle_state)

        return self.angular_control, self.translational_control

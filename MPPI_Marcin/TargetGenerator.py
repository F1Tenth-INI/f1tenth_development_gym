import numpy as np
from math import fmod

# Wraps the angle into range [-π, π]
def wrap_angle_rad(angle: float) -> float:
    Modulo = fmod(angle, 2 * np.pi)  # positive modulo
    if Modulo < -np.pi:
        angle = Modulo + 2 * np.pi
    elif Modulo > np.pi:
        angle = Modulo - 2 * np.pi
    else:
        angle = Modulo
    return angle


class TargetGenerator:
    def __init__(self):

        self.current_target = None

        self.tolerance = 1      # m

        self.x_lower_bound = -10.
        self.x_upper_bound = 10.
        self.y_upper_bound = 10.
        self.y_lower_bound = -10.


    def generate_random_target_position(self):
        target_x = np.random.uniform(self.x_lower_bound, self.x_upper_bound)
        target_y = np.random.uniform(self.y_lower_bound, self.y_upper_bound)
        return np.array([[target_x, target_y]])


    def step(self, pos):

        if (
                self.current_target is None
                or
                np.linalg.norm(pos-self.current_target) < self.tolerance
        ):
            self.current_target = self.generate_random_target_position()

        return self.current_target

    def angle_to_target(self, pos, theta):
        (x_target, y_target) = self.current_target[0]-pos
        angle_target = np.arctan2(y_target, x_target)

        return angle_target-wrap_angle_rad(theta)





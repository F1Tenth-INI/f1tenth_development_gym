import numpy as np


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




import numpy as np
from scipy.interpolate import BPoly, interp1d


class SpeedGenerator:
    def __init__(self):

        self.speed_min = 1.0
        self.speed_max = 8.0

        self.target_speed = None
        self.last_target_speed = 0.0

        self.counter = 0
        self.counter_max = 20

        self.random_speed_f = None

    def generate_target_speed(self):

        if self.target_speed is not None:
            self.last_target_speed = self.target_speed
        self.target_speed = np.random.uniform(self.speed_min, self.speed_max)


    def step(self):

        if self.target_speed is None or self.counter == self.counter_max-1:
            self.generate_target_speed()
            self.counter = 0

            y = np.array((self.last_target_speed, self.target_speed))

            t_init = [0.0, self.counter_max-1]

            yder = [[y[i], 0] for i in range(len(y))]
            self.random_speed_f = BPoly.from_derivatives(t_init, yder)

        current_speed = self.random_speed_f(self.counter)
        self.counter += 1

        return current_speed




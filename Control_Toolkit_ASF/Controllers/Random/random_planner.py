import random
import yaml

from utilities.Settings import Settings

class random_planner:
    "A planner for generating random controls. Great for gathering data on an empty map"

    def __init__(self):
        self.config = yaml.load(open('DataGen/config_data_gen.yml', "r"), Loader=yaml.FullLoader)
        self.mu_ac, self.sigma_ac = self.config['control_inputs']['angular_control_range']
        self.min_tc, self.max_tc = self.config['control_inputs']['translational_control_range']
        self.strategy = self.config['control_inputs']['strategy']

        self.step = 0

        if self.strategy == 'constant':
            self.angular_control = random.gauss(mu=self.mu_ac, sigma=self.sigma_ac)
            self.translational_control = random.uniform(self.min_tc, self.max_tc)

        elif self.strategy == 'ramp':
            self.start_translational_control = 1.0
            self.end_translational_control = 6.0
            self.ramp = (self.end_translational_control - self.start_translational_control) / Settings.EXPERIMENT_LENGTH

    def process_observation(self, ranges, ego_odom):
        if self.strategy == 'constant':
            angular_control = self.angular_control
            translational_control = self.translational_control

        elif self.strategy == 'random':
            # angular_control = random.gauss(mu=self.mu_ac, sigma=self.sigma_ac)
            angular_control = random.uniform(-0.4189, 0.4189)
            translational_control = random.uniform(self.min_tc, self.max_tc)

        elif self.strategy == 'ramp':
            translational_control = self.ramp * self.step + self.start_translational_control
            angular_control = random.uniform(-0.4189, 0.4189)

        self.step += 1
        return angular_control, translational_control

import random
import yaml

class random_planner:
    "A planner for generating random controls. Great for gathering data on an empty map"

    def __init__(self):
        self.config = yaml.load(open('DataGen/config_data_gen.yml', "r"), Loader=yaml.FullLoader)
        self.mu_ac, self.sigma_ac = self.config['control_inputs']['angular_control_range']
        self.min_tc, self.max_tc = self.config['control_inputs']['translational_control_range']
        self.strategy = self.config['control_inputs']['strategy']

    def process_observation(self, *_):
        if self.strategy == 'constant':
            angular_control = random.gauss(mu=self.mu_ac, sigma=self.sigma_ac)
            translational_control = random.uniform(self.min_tc, self.max_tc)

        elif self.strategy == 'random':
            angular_control = random.gauss(mu=self.mu_ac, sigma=self.sigma_ac)
            translational_control = random.uniform(self.min_tc, self.max_tc)
        
        return angular_control, translational_control
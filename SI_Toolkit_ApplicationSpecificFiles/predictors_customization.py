import numpy as np

STATE_VARIABLES=None
STATE_INDICES=None
CONTROL_INPUTS=None

class next_state_predictor_ODE():

    def __init__(self, dt, intermediate_steps):
        self.s = None

        self.intermediate_steps = intermediate_steps
        self.t_step = np.float32(dt / float(self.intermediate_steps))

    def step(self, s, Q, params):

        s_next = np.copy(s)

        return s_next


def augment_predictor_output(output_array, net_info):
    pass
    return output_array
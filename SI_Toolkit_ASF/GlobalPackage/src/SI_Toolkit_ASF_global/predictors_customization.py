import numpy as np

STATE_VARIABLES = np.sort(['pose_x', 'pose_y', 'pose_theta'])
STATE_INDICES = {x: np.where(STATE_VARIABLES == x)[0][0] for x in STATE_VARIABLES}
CONTROL_INPUTS = np.sort(['speed', 'steering'])
CONTROL_INDICES = {x: np.where(CONTROL_INPUTS == x)[0][0] for x in CONTROL_INPUTS}

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
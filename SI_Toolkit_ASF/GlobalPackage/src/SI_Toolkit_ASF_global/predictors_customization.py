import numpy as np

STATE_VARIABLES = np.sort([
    'angular_vel_z',
    'linear_vel_x',
    'linear_vel_y',
    'pose_theta',
    'pose_theta_cos',
    'pose_theta_sin',
    'pose_x',
    'pose_y',
])
STATE_INDICES = {x: np.where(STATE_VARIABLES == x)[0][0] for x in STATE_VARIABLES}
CONTROL_INPUTS = np.sort(['speed', 'steering'])
CONTROL_INDICES = {x: np.where(CONTROL_INPUTS == x)[0][0] for x in CONTROL_INPUTS}

POSE_THETA_IDX = STATE_INDICES['pose_theta']
POSE_THETA_COS_IDX = STATE_INDICES['pose_theta_cos']
POSE_THETA_SIN_IDX = STATE_INDICES['pose_theta_sin']
POSE_X_IDX = STATE_INDICES['pose_x']
POSE_Y_IDX = STATE_INDICES['pose_y']

LINEAR_VEL_X = STATE_INDICES['linear_vel_x']
LINEAR_VEL_Y = STATE_INDICES['linear_vel_y']
ANGULAR_VEL_Z = STATE_INDICES['angular_vel_z']

SPEED_IDX = CONTROL_INDICES['speed']
STEERING_IDX = CONTROL_INDICES['steering']


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

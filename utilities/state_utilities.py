from utilities.Settings import Settings
import numpy as np

ODOMETRY_VARIABLES = np.sort([
    'angular_vel_z',
    'linear_vel_x',
    'pose_theta',
    'pose_theta_cos',
    'pose_theta_sin',
    'pose_x',
    'pose_y',
])

def odometry_dict_to_state(odom):
    s = np.array([odom['angular_vel_z'], odom['linear_vel_x'],
                 odom['pose_theta'], odom['pose_theta_cos'], odom['pose_theta_sin'],
                 odom['pose_x'], odom['pose_y']])
    return s




FULL_STATE_VARIABLES = np.sort([
    'angular_vel_z',  # x5: yaw rate
    'linear_vel_x',   # x3: velocity in x direction
    'pose_theta',  # x4: yaw angle
    'pose_theta_cos',
    'pose_theta_sin',
    'pose_x',  # x0: x position in global coordinates
    'pose_y',  # x1: y position in global coordinates
    'slip_angle',  # x6: slip angle at vehicle center
    'steering_angle'  # x2: steering angle of front wheels
])

if Settings.ONLY_ODOMETRY_AVAILABLE:
    STATE_VARIABLES = ODOMETRY_VARIABLES
else:
    STATE_VARIABLES = FULL_STATE_VARIABLES

STATE_INDICES = {x: np.where(STATE_VARIABLES == x)[0][0] for x in STATE_VARIABLES}


CONTROL_INPUTS = np.sort(['angular_control', 'translational_control'])
CONTROL_INDICES = {x: np.where(CONTROL_INPUTS == x)[0][0] for x in CONTROL_INPUTS}

POSE_THETA_IDX = STATE_INDICES['pose_theta']
POSE_THETA_COS_IDX = STATE_INDICES['pose_theta_cos']
POSE_THETA_SIN_IDX = STATE_INDICES['pose_theta_sin']
POSE_X_IDX = STATE_INDICES['pose_x']
POSE_Y_IDX = STATE_INDICES['pose_y']

LINEAR_VEL_X_IDX = STATE_INDICES['linear_vel_x']
ANGULAR_VEL_Z_IDX = STATE_INDICES['angular_vel_z']

if not Settings.ONLY_ODOMETRY_AVAILABLE:
    SLIP_ANGLE_IDX = STATE_INDICES['slip_angle']
    STEERING_ANGLE_IDX = STATE_INDICES['steering_angle']
else:
    SLIP_ANGLE_IDX = None
    STEERING_ANGLE_IDX = None


def create_car_state(state: dict = {}, dtype=None) -> np.ndarray:
    """
    Constructor of car state from named arguments. The order of variables is fixed in STATE_VARIABLES.

    Input parameters are passed as a dict with the following possible keys. Other keys are ignored.
    Unset key-value pairs are initialized to 0.
    """
    state["pose_theta_cos"] = (
        np.cos(state["pose_theta"]) if "pose_theta" in state.keys() else np.cos(0.0)
    )
    state["pose_theta_sin"] = (
        np.sin(state["pose_theta"]) if "pose_theta" in state.keys() else np.sin(0.0)
    )

    if dtype is None:
        dtype = np.float32

    s = np.zeros_like(STATE_VARIABLES, dtype=np.float32)
    for i, v in enumerate(STATE_VARIABLES):
        s[i] = state.get(v) if v in state.keys() else s[i]
    return s


def full_state_original_to_alphabetical(o):
    alphabetical = np.array([o[5], o[3], o[4], np.cos(o[4]), np.sin(o[4]), o[0], o[1], o[6], o[2]])
    return alphabetical


def full_state_alphabetical_to_original(a):
    original = np.array([a[5], a[6], a[8], a[1], a[2], a[0], a[7]])
    return original


ANGULAR_CONTROL_IDX = CONTROL_INDICES['angular_control'] # 0
TRANSLATIONAL_CONTROL_IDX = CONTROL_INDICES['translational_control'] # 1

def get_control_limits(clip_control_input):
    if isinstance(clip_control_input[0], list):
        clip_control_input_low = np.array(clip_control_input[0])
        clip_control_input_high = np.array(clip_control_input[1])
    else:
        clip_control_input_high = np.array(clip_control_input)
        clip_control_input_low = -np.array(clip_control_input_high)

    return clip_control_input_low, clip_control_input_high

if Settings.ENVIRONMENT_NAME == 'Car':
    if not Settings.WITH_PID:  # MPC return velocity and steering angle
        control_limits_low, control_limits_high = get_control_limits([[-3.2, -9.5], [3.2, 9.5]])
    else:  # MPC returns acceleration and steering velocity
        control_limits_low, control_limits_high = get_control_limits([[-1.066, -2], [1.066, 8]])
else:
    raise NotImplementedError('{} mpc not implemented yet'.format(Settings.ENVIRONMENT_NAME))

control_limits_max_abs = np.max(np.vstack((np.abs(control_limits_low), np.abs(control_limits_high))), axis=0)

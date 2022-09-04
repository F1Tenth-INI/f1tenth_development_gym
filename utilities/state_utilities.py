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


CONTROL_INPUTS = np.sort(['translational_control', 'angular_control'])
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


def full_state_original_to_alphabetical(o):
    alphabetical = np.array([o[5], o[3], o[4], np.cos(o[4]), np.sin(o[4]), o[0], o[1], o[6], o[2]])
    return alphabetical


def full_state_alphabetical_to_original(a):
    original = np.array(a[5], a[6], a[8], a[1], a[2], a[0], a[7])
    return original


TRANSLATIONAL_CONTROL_IDX = CONTROL_INDICES['translational_control']
ANGULAR_CONTROL_IDX = CONTROL_INDICES['angular_control']


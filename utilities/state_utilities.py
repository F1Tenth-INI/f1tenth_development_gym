from utilities.Settings import Settings
import numpy as np
import math


STATE_VARIABLES = np.sort([
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

FULL_STATE_VARIABLES = STATE_VARIABLES

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
SLIP_ANGLE_IDX = STATE_INDICES['slip_angle']
STEERING_ANGLE_IDX = STATE_INDICES['steering_angle']

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
    
    from f110_gym.envs.dynamic_models_pacejka import StateIndices 
    
    slipping_angle = np.arctan(o[StateIndices.v_y] / o[StateIndices.v_x])
    alphabetical = np.zeros(9)
    
    alphabetical[ANGULAR_VEL_Z_IDX] = o[StateIndices.yaw_rate]
    alphabetical[LINEAR_VEL_X_IDX] = o[StateIndices.v_x]
    alphabetical[POSE_THETA_IDX] = o[StateIndices.yaw_angle]
    alphabetical[POSE_THETA_COS_IDX] = np.cos(o[StateIndices.yaw_angle])
    alphabetical[POSE_THETA_SIN_IDX] = np.sin(o[StateIndices.yaw_angle])
    alphabetical[POSE_X_IDX] = o[StateIndices.pose_x]
    alphabetical[POSE_Y_IDX] = o[StateIndices.pose_y]
    alphabetical[SLIP_ANGLE_IDX] = slipping_angle
    alphabetical[STEERING_ANGLE_IDX] = o[StateIndices.steering_angle]
    
    return alphabetical


def full_state_alphabetical_to_original(a):
    
    from f110_gym.envs.dynamic_models_pacejka import StateIndices

    lateral_velocity =  a[SLIP_ANGLE_IDX] * math.sin(a[LINEAR_VEL_X_IDX])    

    
    original = np.zeros(6)
    original[StateIndices.pose_x] = a[5]
    original[StateIndices.pose_y] = a[6]
    original[StateIndices.yaw_angle] = a[2]
    original[StateIndices.v_x] = a[1]
    original[StateIndices.v_y] = lateral_velocity
    original[StateIndices.yaw_rate] = a[0]
    original[StateIndices.steering_angle] = a[8]
    
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

control_limits_low, control_limits_high = get_control_limits([[-0.8 , -1], [0.8, 18]])
control_limits_max_abs = np.max(np.vstack((np.abs(control_limits_low), np.abs(control_limits_high))), axis=0)

"""Car-state channel indices of ``utilities.state_utilities`` as Python ints.

The originals are numpy integers, which ``torch.compile`` traces as tensors and then
cannot use in list indices such as ``state[:, [POSE_X_IDX, POSE_Y_IDX]]``.
"""

from utilities import state_utilities as _su

ANGULAR_VEL_Z_IDX = int(_su.ANGULAR_VEL_Z_IDX)
LINEAR_VEL_X_IDX = int(_su.LINEAR_VEL_X_IDX)
LINEAR_VEL_Y_IDX = int(_su.LINEAR_VEL_Y_IDX)
POSE_THETA_IDX = int(_su.POSE_THETA_IDX)
POSE_THETA_COS_IDX = int(_su.POSE_THETA_COS_IDX)
POSE_THETA_SIN_IDX = int(_su.POSE_THETA_SIN_IDX)
POSE_X_IDX = int(_su.POSE_X_IDX)
POSE_Y_IDX = int(_su.POSE_Y_IDX)
SLIP_ANGLE_IDX = int(_su.SLIP_ANGLE_IDX)
STEERING_ANGLE_IDX = int(_su.STEERING_ANGLE_IDX)

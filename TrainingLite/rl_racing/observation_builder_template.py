from __future__ import annotations

from typing import Dict, Any

import numpy as np

from utilities.state_utilities import *  # indices like LINEAR_VEL_X_IDX, WP_KAPPA_IDX, etc.
from utilities.waypoint_utils import WP_KAPPA_IDX, WP_VX_IDX


def build_observation(super_obs: Dict[str, np.ndarray], planner: Any = None) -> np.ndarray:
    """
    Default observation builder *template*.

    This file is the source that `learner_server.py` copies into each model's:
      `TrainingLite/rl_racing/models/<model_name>/client/observation_builder.py`
    """

    next_waypoints = super_obs["next_waypoints"].astype(np.float32)
    state_history_len = 1
    state_history = super_obs["state_history"].astype(np.float32)
    state_indices = [
        LINEAR_VEL_X_IDX,
        LINEAR_VEL_Y_IDX,
        ANGULAR_VEL_Z_IDX,
        STEERING_ANGLE_IDX,
    ]
    state_slice = state_history[-state_history_len:, state_indices]
    if state_slice.shape[0] < state_history_len:
        pad = np.zeros(
            (state_history_len - state_slice.shape[0], len(state_indices)),
            dtype=np.float32,
        )
        state_slice = np.concatenate([pad, state_slice], axis=0)
    state_features = state_slice.reshape(-1).astype(np.float32)
    last_actions = super_obs["last_actions"].astype(np.float32)

    curvatures = super_obs["next_waypoints"][:, WP_KAPPA_IDX].astype(np.float32)
    border_points = super_obs["border_points"].astype(np.float32)

    border_points_left, border_points_right = border_points
    border_points_left = border_points_left[::3]
    border_points_right = border_points_right[::3]

    border_points = np.concatenate([border_points_left.flatten(), border_points_right.flatten()])

    _, d, e, _ = super_obs["frenet_coordinates"].astype(np.float32)
    d = np.atleast_1d(d).astype(np.float32)
    e = np.atleast_1d(e).astype(np.float32)

    target_speeds = next_waypoints[:, WP_VX_IDX].astype(np.float32)
    target_speeds = target_speeds[::5]

    pp_action = super_obs["pp_action"].astype(np.float32)

    lidar_history_len = 3
    lidar_history_stride = 5  # 1 = consecutive frames; m>1 = latest, m steps ago, 2m ago, ...
    # stride = 4, len = 5 → scans at 16, 12, 8, 4, 0 steps back from lates
    
    lidar_history = super_obs["lidar_history"].astype(np.float32)
    n_rows, n_beams = lidar_history.shape
    lidar_frames = []
    for i in range(lidar_history_len):
        steps_back = (lidar_history_len - 1 - i) * lidar_history_stride
        idx = n_rows - 1 - steps_back
        if idx >= 0:
            lidar_frames.append(lidar_history[idx])
        else:
            lidar_frames.append(np.zeros(n_beams, dtype=np.float32))
    lidar_features = np.stack(lidar_frames, axis=0).reshape(-1).astype(np.float32)

    motor_angular_velocity = np.asarray(
        super_obs["motor_sensors"]["motor_angular_velocity"], dtype=np.float32
    ).reshape(-1)
    imu_x = np.asarray(super_obs["imu"]["imu_a_x"], dtype=np.float32).reshape(-1)
    imu_y = np.asarray(super_obs["imu"]["imu_a_y"], dtype=np.float32).reshape(-1)


    obs = np.concatenate(
        [
            np.tile(np.array([0.1, 1.0, 0.3, 2.5], dtype=np.float32), state_history_len) * state_features,
            1.0 * curvatures,
            0.1* border_points,
            1.0 * last_actions,
            1.0 * np.concatenate([d, e]),
            0.1 * target_speeds,
            [1.0, 0.1] * pp_action,
            0.1 * lidar_features,
            # 1/10000 * motor_angular_velocity,
            0.2 * imu_x,
            0.2 * imu_y,
        ]
    ).astype(np.float32)

    return obs

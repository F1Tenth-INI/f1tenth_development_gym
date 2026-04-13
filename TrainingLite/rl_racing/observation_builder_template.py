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

    state_features = super_obs["car_state"][
        [LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX, ANGULAR_VEL_Z_IDX, STEERING_ANGLE_IDX]
    ].astype(np.float32)
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

    obs = np.concatenate(
        [
            [0.1, 1, 1, 2.5] * state_features,
            1.0 * curvatures,
            0.2 * border_points,
            1.0 * last_actions,
            1.0 * np.concatenate([d, e]),
            0.1 * target_speeds,
            [0.1, 1.0] * pp_action,
        ]
    ).astype(np.float32)

    return obs


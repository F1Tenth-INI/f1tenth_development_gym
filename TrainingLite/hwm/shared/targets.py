"""Tracked waypoint and its target state, shared by the progress reward and the actor observation."""

from __future__ import annotations

import torch

from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from TrainingLite.hwm.shared.state_indices import (
    LINEAR_VEL_X_IDX,
    POSE_THETA_COS_IDX,
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)


def build_target_states(raceline: torch.Tensor) -> torch.Tensor:
    """Target state ``(N, state_dim)`` of every raceline point ``(N, 2)``.

    The car standing on the waypoint, heading straight at the next one, at
    ``HWM_TARGET_SPEED`` with no lateral or yaw motion and zero steering and slip.
    """
    direction = torch.roll(raceline, -1, dims=0) - raceline
    heading = torch.atan2(direction[:, 1], direction[:, 0])
    target = raceline.new_zeros(int(raceline.shape[0]), int(Settings.HWM_STATE_DIM))
    target[:, POSE_X_IDX] = raceline[:, 0]
    target[:, POSE_Y_IDX] = raceline[:, 1]
    target[:, POSE_THETA_IDX] = heading
    target[:, POSE_THETA_COS_IDX] = torch.cos(heading)
    target[:, POSE_THETA_SIN_IDX] = torch.sin(heading)
    target[:, LINEAR_VEL_X_IDX] = float(Settings.HWM_TARGET_SPEED)
    return target


def target_index(positions: torch.Tensor, raceline: torch.Tensor) -> torch.Tensor:
    """Index ``(B,)`` of the tracked waypoint for positions ``(B, 2)``.

    The ``HWM_WAYPOINT_LOOKAHEAD``-th waypoint ahead, counting the next one ahead as
    the first: that is the nearest waypoint, or the one after it once the position has
    passed the line through the nearest waypoint perpendicular to its outgoing segment.
    It moves on with the car every step, so the target is never reached.
    """
    lookahead = int(Settings.HWM_WAYPOINT_LOOKAHEAD)
    if lookahead < 1:
        raise ValueError(f"HWM_WAYPOINT_LOOKAHEAD must be >= 1, got {lookahead}")
    n = int(raceline.shape[0])
    nearest = WallGeometry._nearest_indices(positions, raceline)
    segment = raceline[(nearest + 1) % n] - raceline[nearest]
    past_nearest = ((positions - raceline[nearest]) * segment).sum(dim=-1) > 0
    return (nearest + past_nearest.long() + lookahead - 1) % n

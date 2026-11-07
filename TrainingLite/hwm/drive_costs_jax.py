"""Per-sample drive costs of the HW planner, compiled with JAX.

The same terms as ``HirarchicalPlanner.rollout_waypoint_states``, ``_matched_waypoint_costs`` and the crash
and wall part of ``MPC_reward``, for every rollout on its own: matching the upcoming waypoints to rollout
states, the matched-waypoint cost, the first body-corner crash and the corner clearance. The terms that need the whole batch (the crash cost scale, the softmax) stay with
the caller. float32 like the torch code, and like it the body corners and the wall lookup in float64:
``drive_costs`` must be called under ``jax.experimental.enable_x64`` (thread-local).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)

# Columns of the ``rows`` argument.
ROW_X, ROW_Y, ROW_HEADING, ROW_SPEED, ROW_DRIFT = range(5)

# Order of the ``consts`` argument.
CONST_NAMES = (
    "target_speed_factor", "overspeed_weight", "arrival_weight", "border", "clearance", "clearance_span",
    "origin_x", "origin_y", "origin_cos", "origin_sin", "resolution", "map_width", "map_height",
)


def _cells(x, y, c):
    x_t = x - c["origin_x"]
    y_t = y - c["origin_y"]
    x_rot = x_t * c["origin_cos"] + y_t * c["origin_sin"]
    y_rot = -x_t * c["origin_sin"] + y_t * c["origin_cos"]
    inside = (
        (x_rot >= 0) & (x_rot < c["map_width"] * c["resolution"])
        & (y_rot >= 0) & (y_rot < c["map_height"] * c["resolution"])
    )
    cols = jnp.clip(jnp.nan_to_num(jnp.floor(x_rot / c["resolution"])), 0, c["map_width"] - 1).astype(jnp.int32)
    rows = jnp.clip(jnp.nan_to_num(jnp.floor(y_rot / c["resolution"])), 0, c["map_height"] - 1).astype(jnp.int32)
    return rows, cols, inside


def _one(states, dist, rows, count, grid, corner_x, corner_y, c):
    """Costs of one rollout ``states`` ``(T, D)`` with squared distances ``dist`` ``(T, J)`` to the waypoints."""
    steps = states.shape[0]
    width = dist.shape[1]

    # Waypoint j takes the closest state not taken by an earlier waypoint.
    in_range = jnp.arange(width) < count
    dist = jnp.where(in_range[None], dist, jnp.inf)

    def match(taken, j):
        column = jnp.where(taken, jnp.inf, dist[:, j])
        best = jnp.argmin(column)
        reachable = in_range[j] & jnp.isfinite(column[best])
        return taken.at[best].set(taken[best] | reachable), (best, reachable)

    _, (best, valid) = jax.lax.scan(match, jnp.zeros(steps, dtype=bool), jnp.arange(width))

    count_valid = valid.sum()
    indices = jnp.where(valid, best, 0)
    matched = states[indices]  # (J, D)

    position = matched[:, jnp.array([POSE_X_IDX, POSE_Y_IDX])] - rows[:, jnp.array([ROW_X, ROW_Y])]
    distance = jnp.where(valid, jnp.linalg.norm(position, axis=-1), 0.0).sum()
    heading_error = matched[:, POSE_THETA_IDX] - rows[:, ROW_HEADING]
    heading = jnp.where(valid, jnp.abs(jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))), 0.0)
    speed = matched[:, LINEAR_VEL_X_IDX]
    target_speed = rows[:, ROW_SPEED]
    drift_error = jnp.where(valid, jnp.abs(matched[:, LINEAR_VEL_Y_IDX] - rows[:, ROW_DRIFT]), 0.0).sum()
    drive_target = target_speed * c["target_speed_factor"]
    drive_shortfall = jnp.where(valid, jnp.maximum(drive_target - speed, 0.0), 0.0)
    drive_overspeed = jnp.where(valid, jnp.maximum(speed - drive_target, 0.0), 0.0)

    index_f = indices.astype(jnp.float32)
    fast_driving = jnp.where(valid, index_f, -jnp.inf).max()
    counts = jnp.maximum(count_valid, 1).astype(jnp.float32)
    drive = (
        10 * distance + 2 * heading.sum() + drive_shortfall.sum() + drive_overspeed.sum() * c["overspeed_weight"]
        + 0.25 * drift_error + c["arrival_weight"] * fast_driving
    ) / counts
    min_speed = states[1:, LINEAR_VEL_X_IDX].min()
    reverse = jnp.where(min_speed < 0, 100 * jnp.abs(min_speed), 0.0)
    drive = drive + reverse

    # Walls: a body corner within ``border`` of a wall or off the map is a crash.
    pose = states[:, jnp.array([POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX])].astype(jnp.float64)
    cos_t, sin_t = jnp.cos(pose[:, 2:3]), jnp.sin(pose[:, 2:3])
    cx = pose[:, 0:1] + cos_t * corner_x - sin_t * corner_y  # (T, 4)
    cy = pose[:, 1:2] + sin_t * corner_x + cos_t * corner_y
    r, k, inside = _cells(cx, cy, c)
    corner_wall = grid[r, k]
    crashed = (~inside | (corner_wall < c["border"])).any(axis=1)
    corner_distance = jnp.where(inside, corner_wall, 0.0).min(axis=1)
    late = steps - jnp.where(crashed.any(), jnp.argmax(crashed), steps)
    closeness = jnp.clip((c["clearance"] - corner_distance) / c["clearance_span"], 0.0, 1.0).sum()
    return drive, count_valid > 0, late, closeness


@partial(jax.jit, static_argnames=("consts",))
def drive_costs(states, dist, rows, count, grid, corner_x, corner_y, consts):
    """Per-rollout ``(drive, active, late, closeness)`` for ``states`` ``(B, T, D)``.

    ``dist`` ``(B, T, J)`` are the squared distances of the states to the upcoming waypoints, from
    ``squared_distances``: XLA fuses the multiply-add, and the matching must break exact ties as the torch
    code does. ``rows`` ``(J, 5)`` are those waypoints, ``count`` how many of them are
    matched (the rest is padding), ``grid`` the wall distance map, ``corner_x``/``corner_y`` ``(4,)`` the body
    corners and ``consts`` the values named in ``CONST_NAMES``.
    """
    c = dict(zip(CONST_NAMES, consts))
    return jax.vmap(_one, in_axes=(0, 0, None, None, None, None, None, None))(
        states, dist, rows, count, grid, corner_x, corner_y, c
    )


def squared_distances(states: np.ndarray, match_xy: np.ndarray) -> np.ndarray:
    """``(B, T, J)`` float32 squared distances of the positions in ``states`` ``(B, T, D)`` to ``match_xy`` ``(J, 2)``,
    rounded like ``rollout_waypoint_states``."""
    dx = states[:, :, None, POSE_X_IDX] - match_xy[:, 0]
    dy = states[:, :, None, POSE_Y_IDX] - match_xy[:, 1]
    return dx * dx + dy * dy

"""Observation builders shared by planner and learner.

``build_actor_observation`` feeds the actor, ``build_LLD_observation_inference`` the LLD.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_COS_IDX,
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    SLIP_ANGLE_IDX,
    STEERING_ANGLE_IDX,
)

if TYPE_CHECKING:
    from TrainingLite.hwm.shared.memory_manager import MemoryManager


def lld_direct_delta() -> bool:
    """Test switch ``HWM_LLD_DIRECT_DELTA``. False is the previous reduced-delta LLD."""
    return bool(getattr(Settings, "HWM_LLD_DIRECT_DELTA", False))


def lld_predict_displacement() -> bool:
    """``HWM_LLD_PREDICT_DISPLACEMENT``: the delta also contains the step's car-frame displacement."""
    return bool(getattr(Settings, "HWM_LLD_PREDICT_DISPLACEMENT", False))


def lld_correction_alone() -> bool:
    """``HWM_LLD_CORRECTION_ALONE``: g matches the recorded delta; the encoder is trained to help it.

    g gets no gradient from the retrieved samples. The
    retrieval encoder is trained on the residual prediction with g held fixed. The residuals
    ``delta - g(key)`` depend on the current weights of g, so they are evaluated in the forward
    pass and not cached.
    """
    return bool(getattr(Settings, "HWM_LLD_CORRECTION_ALONE", False))


def lld_kinematic_residual() -> bool:
    """``HWM_LLD_KINEMATIC_RESIDUAL``: yaw rate and displacement are kinematics plus the residual net."""
    return bool(getattr(Settings, "HWM_LLD_KINEMATIC_RESIDUAL", False))


def lld_displacement_channels() -> bool:
    """The delta ends with car-frame forward and left displacement."""
    return lld_predict_displacement() or lld_kinematic_residual()


def lld_pose_free() -> bool:
    """True when actor superstates drop global pose: ``HWM_LLD_DIRECT_DELTA`` or ``HWM_LLD_REFLECT``.

    LLD superstates drop it always (``lld_rows_for_super``). ``HWM_LLD_REFLECT`` still
    mirrors each LLD row. The value tokens and the next-state reconstruction follow
    ``HWM_LLD_DIRECT_DELTA``.
    """
    return lld_direct_delta() or bool(getattr(Settings, "HWM_LLD_REFLECT", False))


# Global pose. Never an input of the LLD: position and heading, and sin/cos of heading.
_GLOBAL_POSE = (POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX)


def _global_pose_channels() -> tuple[int, ...]:
    # Plain ints: the index constants are numpy ints, and torch.compile mis-evaluates ``i in (numpy ints)``.
    return tuple(int(i) for i in _GLOBAL_POSE + (POSE_THETA_COS_IDX, POSE_THETA_SIN_IDX))


def lld_dynamic_indices() -> list[int]:
    """Channels whose change the LLD predicts.

    Yaw rate, body velocity and steering, plus the heading change unless
    ``HWM_LLD_DIRECT_DELTA`` (then heading is the mean yaw rate over the step).
    With ``HWM_LLD_PREDICT_DISPLACEMENT`` the last two channels are the step's
    displacement in the car frame, forward then left (stored under the pose x/y
    slots so they scale and mirror like those). Otherwise position is integrated
    from velocity. Sin, cos and slip are never predicted.
    """
    if lld_direct_delta():
        indices = [ANGULAR_VEL_Z_IDX, LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX, STEERING_ANGLE_IDX]
    else:
        indices = [
            ANGULAR_VEL_Z_IDX,
            LINEAR_VEL_X_IDX,
            LINEAR_VEL_Y_IDX,
            STEERING_ANGLE_IDX,
            POSE_THETA_IDX,
        ]
    if lld_displacement_channels():
        indices += [POSE_X_IDX, POSE_Y_IDX]
    return indices


def lld_kept_state_index(device: torch.device) -> torch.Tensor:
    """State channels kept in an LLD superstate row. Position and heading are never kept."""
    state_dim = int(Settings.HWM_STATE_DIM)
    dropped = set(_global_pose_channels())
    keep = tuple(index for index in range(state_dim) if index not in dropped)
    return _constant(keep, torch.long, device)


def lld_super_row_dim() -> int:
    """Width of one ``[kept state, action]`` row inside an LLD superstate."""
    return int(lld_kept_state_index(torch.device("cpu")).shape[0]) + int(Settings.HWM_ACTION_DIM)


def lld_super_flat_dim() -> int:
    """Width of a flattened LLD superstate (``S`` packed rows)."""
    return int(Settings.HWM_SUPER_STATE_SIZE) * lld_super_row_dim()


def lld_reflect_on() -> bool:
    """Test switch ``HWM_LLD_REFLECT``. It also makes the actor superstate pose-free (``lld_pose_free``)."""
    return bool(getattr(Settings, "HWM_LLD_REFLECT", False))


# State channels that change sign when the car is mirrored about its own x-axis
# (heading, lateral velocity, yaw rate, slip, steering). Speed and the motor do not.
_ODD_STATE = (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_Y_IDX,
    SLIP_ANGLE_IDX,
    STEERING_ANGLE_IDX,
    POSE_Y_IDX,
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
)


def _mirror_sign(states: torch.Tensor) -> torch.Tensor:
    """-1 where the raw or scaled ``states`` have a negative steering angle, else +1: ``(...)``."""
    steering = states[..., STEERING_ANGLE_IDX]
    return torch.where(steering < 0, -torch.ones_like(steering), torch.ones_like(steering))


def _reflect_rows(rows: torch.Tensor, state_dim: int, sign: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mirror ``[state, action]`` rows.

    By default each row is mirrored on its own, by the sign of its own steering angle, so
    a window can contain both orientations. With ``sign`` (broadcast against
    ``rows[..., 0]``) every row uses that sign, which mirrors a whole scene. The steering
    action is odd like the steering angle; the throttle is even.
    """
    action_dim = int(rows.shape[-1]) - state_dim
    odd = [0.0] * (state_dim + action_dim)
    for index in _ODD_STATE:
        odd[index] = 1.0
    if action_dim > 0:
        odd[state_dim] = 1.0  # steering action
    odd_mask = _constant(tuple(odd), rows.dtype, rows.device)
    if sign is None:
        sign = _mirror_sign(rows[..., :state_dim])
    return rows * (1.0 + (sign[..., None].to(rows.dtype) - 1.0) * odd_mask)


def _reflect_values(values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """``lld_reflect`` without the switch."""
    odd = _constant(tuple(1.0 if i in _ODD_STATE else 0.0 for i in lld_dynamic_indices()), values.dtype, values.device)
    sign = _mirror_sign(states.to(values.device))[..., None].to(values.dtype)
    return values * (1.0 + (sign - 1.0) * odd)


def lld_reflect(values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Mirror LLD ``values`` ``(..., d)`` (``lld_dynamic_indices`` order) by the steering sign of ``states`` ``(..., state_dim)``.

    A delta, a noise vector in standard deviations, or a mean: each channel is odd
    or even exactly as its state channel. Applying it twice gives the original, so it
    maps canonical to true and true to canonical. No-op unless ``HWM_LLD_REFLECT``.
    """
    if not lld_reflect_on():
        return values
    return _reflect_values(values, states)


def lld_rows_for_super(rows: torch.Tensor, state_dim: int, reflect: bool = True) -> torch.Tensor:
    """Drop position and heading from ``[state, action]`` rows. Always, for every LLD input.

    With ``HWM_LLD_REFLECT`` every row is also mirrored so its steering angle is not
    negative. The actor does not use this when it keeps pose.
    """
    if reflect and lld_reflect_on():
        rows = _reflect_rows(rows, state_dim)
    state = rows[..., :state_dim].index_select(-1, lld_kept_state_index(rows.device))
    return torch.cat((state, rows[..., state_dim:]), dim=-1)


@functools.lru_cache(maxsize=64)
def _constant(values: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``torch.tensor(values)``, built once per dtype and device. Callers must not modify it."""
    return torch.tensor(values, dtype=dtype, device=device)


def _channel_scale(values: list, name: str, reference: torch.Tensor) -> torch.Tensor:
    if len(values) < reference.shape[-1]:
        raise ValueError(
            f"Settings.{name} has {len(values)} entries, need at least {reference.shape[-1]}"
        )
    return _constant(tuple(values), reference.dtype, reference.device)


def _scale(reference: torch.Tensor) -> torch.Tensor:
    """Per-channel divisors from ``Settings.HWM_STATE_SCALE`` (STATE_VARIABLES order)."""
    return _channel_scale(list(Settings.HWM_STATE_SCALE), "HWM_STATE_SCALE", reference)


def _diff_scale(reference: torch.Tensor) -> torch.Tensor:
    """Per-channel divisors from ``Settings.HWM_STATE_DIFF_SCALE`` (STATE_VARIABLES order)."""
    return _channel_scale(list(Settings.HWM_STATE_DIFF_SCALE), "HWM_STATE_DIFF_SCALE", reference)


def lld_delta_scale(reference: torch.Tensor) -> torch.Tensor:
    """Raw size of one unit of the LLD's delta, ``lld_dynamic_indices`` order.

    The unit of the neighbor tokens, the training target and ``reconstruct_next_state``.
    """
    return (_scale(reference) * _diff_scale(reference)).index_select(-1, _dynamic_index(reference.device))


def _dynamic_index(device: torch.device) -> torch.Tensor:
    return _constant(tuple(lld_dynamic_indices()), torch.long, device)


def car_frame_displacement(
    dx_world: torch.Tensor, dy_world: torch.Tensor, heading: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """World displacement into the car frame at ``heading``: x forward, y left."""
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    return cos_h * dx_world + sin_h * dy_world, -sin_h * dx_world + cos_h * dy_world


def world_frame_displacement(
    dx_car: torch.Tensor, dy_car: torch.Tensor, heading: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Car-frame displacement (x forward, y left) into the world frame."""
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    return cos_h * dx_car - sin_h * dy_car, sin_h * dx_car + cos_h * dy_car


def average_velocity_displacement(
    vx: torch.Tensor,
    vy: torch.Tensor,
    vx_next: torch.Tensor,
    vy_next: torch.Tensor,
    heading: torch.Tensor,
    d_heading: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World displacement over one step: the mean body velocity, rotated by the mid-step heading, times ``dt``."""
    mid = heading + 0.5 * d_heading
    cos_m = torch.cos(mid)
    sin_m = torch.sin(mid)
    ax = 0.5 * (vx + vx_next)
    ay = 0.5 * (vy + vy_next)
    return (ax * cos_m - ay * sin_m) * dt, (ax * sin_m + ay * cos_m) * dt


def lld_delta(states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
    """Change from raw ``states`` to raw ``next_states`` ``(..., state_dim)`` in LLD delta units.

    ``lld_dynamic_indices`` order; the heading change is wrapped to ``(-π, π]``.
    With ``HWM_LLD_REFLECT`` the delta is mirrored by the steering sign of ``states``,
    the frame the LLD works in. ``reconstruct_next_state`` mirrors it back.

    The last two channels, with ``HWM_LLD_PREDICT_DISPLACEMENT``, are the car-frame
    displacement. With ``HWM_LLD_KINEMATIC_RESIDUAL`` they are what the average-velocity
    displacement misses: the true displacement minus that integral, in the car frame.
    """
    diff = (next_states - states).clone()
    theta = diff[..., POSE_THETA_IDX]
    diff[..., POSE_THETA_IDX] = torch.atan2(torch.sin(theta), torch.cos(theta))
    scale = lld_delta_scale(states)
    delta = diff.index_select(-1, _dynamic_index(states.device)) / scale
    if lld_displacement_channels():
        heading = states[..., POSE_THETA_IDX]
        world_x, world_y = diff[..., POSE_X_IDX], diff[..., POSE_Y_IDX]
        if lld_kinematic_residual():
            dt = float(Settings.TIMESTEP_CONTROL)
            if lld_direct_delta():
                # Heading is not predicted: it turns by the mean yaw rate over the step.
                d_heading = 0.5 * (states[..., ANGULAR_VEL_Z_IDX] + next_states[..., ANGULAR_VEL_Z_IDX]) * dt
            else:
                d_heading = diff[..., POSE_THETA_IDX]
            base_x, base_y = average_velocity_displacement(
                states[..., LINEAR_VEL_X_IDX], states[..., LINEAR_VEL_Y_IDX],
                next_states[..., LINEAR_VEL_X_IDX], next_states[..., LINEAR_VEL_Y_IDX],
                heading, d_heading, dt,
            )
            world_x, world_y = world_x - base_x, world_y - base_y
        # Rotate into the car frame at the current heading. Lateral sits in the pose-y
        # slot, which reflect treats as odd.
        forward, left = car_frame_displacement(world_x, world_y, heading)
        delta = delta.clone()
        delta[..., -2] = forward / scale[..., -2]
        delta[..., -1] = left / scale[..., -1]
    return lld_reflect(delta, states)


def _normalize_states(states: torch.Tensor) -> torch.Tensor:
    """``states`` ends in the state dimension. Actions are not passed here."""
    return states / _scale(states)[: states.shape[-1]]


def _normalize_rows(rows: torch.Tensor, state_dim: int) -> torch.Tensor:
    """Rows are ``[state, action]``. Only the state half is scaled."""
    out = rows.clone()
    out[..., :state_dim] = _normalize_states(out[..., :state_dim])
    return out


def _as_batch(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _normalized_history(
    memory: MemoryManager,
    episode_id: int,
    past_states: Optional[torch.Tensor],
    batch: int,
) -> torch.Tensor:
    """Previous ``S-1`` rows ``(B, S-1, row_dim)``, state half scaled.

    From ``past_states`` if given, else the last rows of the open episode in ``memory``.
    """
    width = int(Settings.HWM_SUPER_STATE_SIZE)
    if past_states is None:
        history, _ = memory.recent(episode_id, width - 1)
        history = history.unsqueeze(0).expand(batch, -1, -1)
    else:
        history = past_states if torch.is_tensor(past_states) else torch.as_tensor(past_states)
        history = history.to(dtype=torch.float32)
        if history.ndim == 2:
            history = history.unsqueeze(0)
    return _normalize_rows(history.to(memory.device), int(Settings.HWM_STATE_DIM))


def actor_super_state_dim() -> int:
    """Width of the actor's ``current_super_state``.

    Global pose is dropped from every row only while ``lld_pose_free``. The LLD drops it
    always. The current row is state only (no action yet).
    """
    state_dim = int(Settings.HWM_STATE_DIM)
    if lld_pose_free():
        kept = int(lld_kept_state_index(torch.device("cpu")).shape[0])
        past_row = kept + int(Settings.HWM_ACTION_DIM)
        return (int(Settings.HWM_SUPER_STATE_SIZE) - 1) * past_row + kept
    row_dim = state_dim + int(Settings.HWM_ACTION_DIM)
    return (int(Settings.HWM_SUPER_STATE_SIZE) - 1) * row_dim + state_dim


def _actor_superstate(history: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Flattened actor superstate. ``history`` is normalized ``[state, action]`` rows."""
    state_dim = int(Settings.HWM_STATE_DIM)
    current = _normalize_states(state)
    if lld_pose_free():
        past = lld_rows_for_super(history, state_dim, reflect=False)
        current = current.index_select(-1, lld_kept_state_index(current.device))
    else:
        past = history
    return torch.cat((past.flatten(start_dim=1), current), dim=-1)


def build_actor_observation(
    state: np.ndarray | torch.Tensor,
    memory: MemoryManager,
    wall_geometry: WallGeometry,
    episode_id: int,
    past_states: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Build the actor observation for ``state`` ``(state_dim,)`` or ``(B, state_dim)``.

    ``current_super_state`` ``(B, actor_super_state_dim())``: the previous ``S-1``
    ``[state, action]`` rows, then the current state, scaled like the LLD superstate.
    The current action is what the actor chooses, so it is not part of it. Inference
    reads the previous rows from ``memory`` (the current row must not be stored yet);
    training and planning pass them as ``past_states`` ``(B, S-1, row_dim)``.

    ``wall_points`` ``(B, 2 * HWM_WALL_POINTS, 3)``: the upcoming left wall points, then
    the right ones, as ``[x, y, side]`` with ``x, y`` in the car frame (x forward, y
    left) in meters and ``side`` +1 for the left wall, -1 for the right wall.
    """
    device = memory.device
    state_b = _as_batch(state).to(device)
    history = _normalized_history(memory, episode_id, past_states, int(state_b.shape[0]))
    current_super_state = _actor_superstate(history, state_b)

    pose = state_b[:, [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]].to(wall_geometry.device)
    walls = wall_geometry.get_relative_wall_positions(pose, int(Settings.HWM_WALL_POINTS))
    left, right = walls["left"].to(device), walls["right"].to(device)
    wall_points = torch.cat(
        (
            torch.cat((left, torch.ones_like(left[..., :1])), dim=-1),
            torch.cat((right, -torch.ones_like(right[..., :1])), dim=-1),
        ),
        dim=1,
    )
    return {"current_super_state": current_super_state, "wall_points": wall_points}


def build_LLD_observation_inference(
    state: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    memory: MemoryManager,
    episode_id: int,
    past_states: Optional[torch.Tensor] = None,
    before_steps: Optional[torch.Tensor] = None,
    future_margin: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Build the LLD observation.

    Inference reads the previous ``super_state_size - 1`` rows from ``memory``; the
    current row must not be stored yet. Training passes those rows as ``past_states``
    of shape ``(B, S-1, row_dim)`` and the samples' global memory indices as
    ``before_steps`` ``(B,)``.

    Retrieved windows were stored completely before the first row of the current
    superstate, so they never share a row with it and never come from its future.

    ``future_margin`` (training only): besides the past, windows that start more than that
    many rows after the sample's next row are allowed, so the LLD sees the whole memory
    except the sample's immediate future (see ``retrieve_LLD_neighbors``).

    ``n_soft`` ``(B,)`` is ``Σ exp(-d / temperature)`` from the memory embeddings.
    """
    current_super_state, current_state_action = lld_super_state(state, action, memory, episode_id, past_states)
    temperature = float(Settings.HWM_LLD_ATTENTION_TEMPERATURE)
    if temperature <= 0:
        raise ValueError(f"HWM_LLD_ATTENTION_TEMPERATURE must be positive, got {temperature}")
    if before_steps is None and future_margin is None and getattr(Settings, "HWM_LLD_RESIDUAL_ONLY", False):
        # Memory-free inference (HWM_LLD_RESIDUAL_ONLY): the LLD uses only the current superstate.
        batch = int(current_super_state.shape[0])
        empty = current_super_state.new_zeros((0, current_super_state.shape[-1]))
        return {
            "current_super_state": current_super_state,
            "similar_superstates": empty,
            "similar_superstates_diff": current_super_state.new_zeros((0, 0)),
            "similar_valid": current_super_state.new_zeros((batch, 0)),
            "similar_density": current_super_state.new_zeros((0,)),
            "current_state_action": current_state_action,
            "n_soft": current_super_state.new_ones((batch,)),
        }
    neighbors = retrieve_LLD_neighbors(current_super_state, memory, before_steps, future_margin=future_margin)
    n_soft = torch.exp(-neighbors["distances"] / temperature).sum(dim=-1)
    return {
        "current_super_state": current_super_state,
        "similar_superstates": neighbors["similar_superstates"],
        "similar_superstates_diff": neighbors["similar_superstates_diff"],
        "similar_valid": neighbors["valid"],
        "similar_density": neighbors["density"],
        "current_state_action": current_state_action,
        "n_soft": n_soft,
    }


def lld_super_state(
    state: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    memory: MemoryManager,
    episode_id: int,
    past_states: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(current_super_state (B, S * row_dim), current_state_action (B, row_dim))``, normalized.

    The previous rows come from ``memory`` or ``past_states`` as in
    ``build_LLD_observation_inference``.
    """
    device = memory.device
    state_b = _as_batch(state).to(device)
    action_b = _as_batch(action).to(device)
    history = _normalized_history(memory, episode_id, past_states, int(state_b.shape[0]))
    current_state_action = torch.cat((_normalize_states(state_b), action_b), dim=-1)
    # Past [state, action] rows, then the current [state, action]. Keys use that same order.
    # Position and heading are always dropped. current_state_action stays full.
    state_dim = int(Settings.HWM_STATE_DIM)
    past = lld_rows_for_super(history, state_dim)
    current = lld_rows_for_super(current_state_action, state_dim)
    return torch.cat((past.flatten(start_dim=1), current), dim=-1), current_state_action


def retrieve_LLD_neighbors(
    current_super_state: torch.Tensor,
    memory: MemoryManager,
    before_steps: Optional[torch.Tensor] = None,
    future_margin: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Every key of the memory's bank for the normalized superstates ``(B, S * row_dim)``.

    No hard nearest-neighbor choice is made: the LLD's attention can move differentiably
    between all keys as the live state or action changes.

    ``before_steps`` as in ``build_LLD_observation_inference``; None means everything
    stored so far. ``future_margin`` (training): windows from the far future are allowed
    again. Keys whose first row is more than ``future_margin`` rows after the sample's
    next row stay available; the windows in between do not.
    Returns ``similar_superstates`` ``(N, S * row_dim)`` (normalized),
    ``similar_superstates_diff`` ``(N, prediction_dim)`` (the step leaving each window) and
    ``density`` ``(N,)`` (the stored n_soft of each key): one bank shared by every row,
    not copied per row. ``valid`` ``(B, N)`` (0 for keys this row may not see) and the
    embedding ``distances`` ``(B, N)`` to the query are per row.
    """
    width = int(Settings.HWM_SUPER_STATE_SIZE)
    state_dim = int(Settings.HWM_STATE_DIM)
    batch = int(current_super_state.shape[0])
    if before_steps is None:
        current_step = torch.full((batch,), memory.next_step, dtype=torch.long)
    else:
        current_step = torch.as_tensor(before_steps).to(dtype=torch.long).reshape(-1)
    # The current superstate starts S-1 rows before the current one.
    superstate_start = current_step - (width - 1)
    # A key window starting after the sample's next row (current_step + 1) plus the margin.
    after = None if future_margin is None else current_step + 2 + int(future_margin)
    similar_rows, valid, similar_distances, density = memory.get_all_matches(
        current_super_state, before=superstate_start, after=after
    )
    valid = valid.to(similar_distances.dtype)
    # Step from the matched state to the next stored state.
    similar_superstates_diff = lld_delta(similar_rows[..., -2, :state_dim], similar_rows[..., -1, :state_dim])
    similar_rows = lld_rows_for_super(_normalize_rows(similar_rows, state_dim), state_dim)
    similar_superstates = similar_rows[..., :-1, :].flatten(start_dim=-2)
    return {
        "similar_superstates": similar_superstates,
        "similar_superstates_diff": similar_superstates_diff,
        "valid": valid,
        "distances": similar_distances,
        "density": density,
    }

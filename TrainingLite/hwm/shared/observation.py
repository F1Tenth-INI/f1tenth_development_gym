"""Observation builders shared by planner and learner.

``build_actor_observation`` feeds the actor, ``build_LLD_observation_inference`` the LLD.
``build_observation`` is a placeholder for the value network; keep ``observation_dim``
in sync with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    STEERING_ANGLE_IDX,
)

if TYPE_CHECKING:
    from TrainingLite.hwm.shared.memory_manager import MemoryManager


def lld_dynamic_indices() -> list[int]:
    """Channels whose change the LLD predicts.

    ``pose_x``, ``pose_y`` and ``pose_theta`` changes follow from the current
    velocity and yaw rate. ``slip_angle`` is ``atan(vy / vx)``. Sin and cos of
    yaw are functions of yaw.
    """
    return [
        ANGULAR_VEL_Z_IDX,
        LINEAR_VEL_X_IDX,
        LINEAR_VEL_Y_IDX,
        STEERING_ANGLE_IDX,
    ]


_BODY_IDX = [LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX, ANGULAR_VEL_Z_IDX, STEERING_ANGLE_IDX]


def _channel_scale(values: list, name: str, reference: torch.Tensor) -> torch.Tensor:
    if len(values) < reference.shape[-1]:
        raise ValueError(
            f"Settings.{name} has {len(values)} entries, "
            f"need at least {reference.shape[-1]}"
        )
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def _scale(reference: torch.Tensor) -> torch.Tensor:
    """Per-channel divisors from ``Settings.HWM_STATE_SCALE`` (STATE_VARIABLES order)."""
    return _channel_scale(list(Settings.HWM_STATE_SCALE), "HWM_STATE_SCALE", reference)


def _diff_scale(reference: torch.Tensor) -> torch.Tensor:
    """Per-channel divisors from ``Settings.HWM_STATE_DIFF_SCALE`` (STATE_VARIABLES order)."""
    return _channel_scale(list(Settings.HWM_STATE_DIFF_SCALE), "HWM_STATE_DIFF_SCALE", reference)


def _normalize_states(states: torch.Tensor) -> torch.Tensor:
    """``states`` ends in the state dimension. Actions are not passed here."""
    return states / _scale(states)[: states.shape[-1]]


def _normalize_rows(rows: torch.Tensor, state_dim: int) -> torch.Tensor:
    """Rows are ``[state, action]``. Only the state half is scaled."""
    out = rows.clone()
    out[..., :state_dim] = _normalize_states(out[..., :state_dim])
    return out


def observation_dim() -> int:
    """Flat observation size produced by ``build_observation``."""
    return len(_BODY_IDX) + 2 * int(Settings.HWM_WALL_POINTS) * 2


def build_observation(
    state: np.ndarray,
    memory: MemoryManager,
    wall_geometry: WallGeometry,
    episode_id: int,
    goal: Optional[np.ndarray] = None,
) -> dict[str, torch.Tensor]:
    """Builds and composes all observation types for inference"""
    state_t = torch.as_tensor(np.asarray(state, dtype=np.float32))
    body = _normalize_states(state_t)[_BODY_IDX]

    pose = torch.stack(
        (state_t[POSE_X_IDX], state_t[POSE_Y_IDX], state_t[POSE_THETA_IDX])
    ).reshape(1, 3)
    walls = wall_geometry.get_relative_wall_positions(pose, int(Settings.HWM_WALL_POINTS))
    wall_features = torch.cat((walls["left"].reshape(-1), walls["right"].reshape(-1)))

    # ``memory.recent(episode_id, Settings.HWM_CONTEXT_LEN)`` and ``goal`` are available here
    # for history- or goal-conditioned observations.
    return torch.cat((body, wall_features)).to(torch.float32)

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
    """Width of the actor's ``current_super_state``."""
    state_dim = int(Settings.HWM_STATE_DIM)
    row_dim = state_dim + int(Settings.HWM_ACTION_DIM)
    return (int(Settings.HWM_SUPER_STATE_SIZE) - 1) * row_dim + state_dim


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
    current_super_state = torch.cat((history.flatten(start_dim=1), _normalize_states(state_b)), dim=-1)

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
) -> dict[str, torch.Tensor]:
    """Build the LLD observation.

    Inference reads the previous ``super_state_size - 1`` rows from ``memory``; the
    current row must not be stored yet. Training passes those rows as ``past_states``
    of shape ``(B, S-1, row_dim)`` and the samples' global memory indices as
    ``before_steps`` ``(B,)``.

    Retrieved windows were stored completely before the first row of the current
    superstate, so they never share a row with it and never come from its future.
    Retrieval ranks by distance in the memory's cached projector embeddings.

    ``n_soft`` ``(B,)`` is ``1 + Σ exp(-d)`` over the retrieved neighbors, ``d`` being
    their embedding distance to the current superstate. It is the kernel count the LLD
    forward computes, without gradients, and equal to it while the memory embeddings
    match the LLD weights.
    """
    width = int(Settings.HWM_SUPER_STATE_SIZE)
    state_dim = int(Settings.HWM_STATE_DIM)
    device = memory.device
    state_b = _as_batch(state).to(device)
    action_b = _as_batch(action).to(device)
    history = _normalized_history(memory, episode_id, past_states, int(state_b.shape[0]))
    current_state_action = torch.cat((_normalize_states(state_b), action_b), dim=-1)

    # Past [state, action] rows, then the current [state, action]. Keys use that same order.
    current_super_state = torch.cat((history.flatten(start_dim=1), current_state_action), dim=-1)

    if before_steps is None:
        current_step = torch.full((int(state_b.shape[0]),), memory.next_step, dtype=torch.long)
    else:
        current_step = torch.as_tensor(before_steps).to(dtype=torch.long).reshape(-1)
    # The current superstate starts S-1 rows before the current one.
    superstate_start = current_step - (width - 1)
    # One row after the match. The value token is the step that leaves it.
    similar_rows, similar_distances = memory.get_similars(
        current_super_state,
        num_similars=10,
        before=superstate_start,
    )
    similar_rows = _normalize_rows(similar_rows, state_dim)
    similar_superstates = similar_rows[:, :, :-1, :].flatten(start_dim=2)

    states = similar_rows[..., :state_dim]
    dynamic = lld_dynamic_indices()
    # Step from the matched state to the next stored state.
    state_diff = (states[:, :, -1, :] - states[:, :, -2, :]) / _diff_scale(states)
    similar_superstates_diff = state_diff[..., dynamic]
    return {
        "current_super_state": current_super_state,
        "similar_superstates": similar_superstates,
        "similar_superstates_diff": similar_superstates_diff,
        "current_state_action": current_state_action,
        "n_soft": 1.0 + torch.exp(-similar_distances).sum(dim=-1),
    }
    
    

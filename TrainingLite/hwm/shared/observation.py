"""Observation builder shared by planner and learner.

This is a placeholder. Replace ``build_observation`` with the real feature
construction; keep the signature and keep ``observation_dim`` in sync so the
actor input size matches on both client and server.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from TrainingLite.hwm.shared.memory_manager import MemoryManager
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

_BODY_IDX = [LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX, ANGULAR_VEL_Z_IDX, STEERING_ANGLE_IDX]


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
    body = state_t[_BODY_IDX]

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


def build_LLD_observation_inference(
    state: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    memory: MemoryManager,
    episode_id: int,
    past_states: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Build the LLD observation.

    Inference reads the previous ``super_state_size - 1`` rows from ``memory``.
    Training passes those rows as ``past_states`` of shape ``(B, S-1, row_dim)``.
    """
    width = int(Settings.HWM_SUPER_STATE_SIZE)
    state_dim = int(Settings.HWM_STATE_DIM)
    state_b = _as_batch(state)
    action_b = _as_batch(action)
    if past_states is None:
        history, _ = memory.recent(episode_id, width - 1)
        history = history.unsqueeze(0)
    else:
        history = past_states if torch.is_tensor(past_states) else torch.as_tensor(past_states)
        history = history.to(dtype=torch.float32)
        if history.ndim == 2:
            history = history.unsqueeze(0)

    current_state_action = torch.cat((state_b, action_b), dim=-1)
    # One extra step so each neighbor has a following row for the state difference.
    similar_rows = memory.get_similars(
        current_state_action.detach().cpu().numpy(),
        num_similars=10,
        superstate_size=width + 1,
    )
    device = similar_rows.device
    history = history.to(device)
    current_state_action = current_state_action.to(device)

    # Past [state, action] rows, then the current [state, action]. Keys use the same packing.
    current_super_state = torch.cat((history.flatten(start_dim=1), current_state_action), dim=-1)
    similar_superstates = similar_rows[:, :, :width, :].flatten(start_dim=2)

    states = similar_rows[..., :state_dim]
    state_diff = states[:, :, 1:, :] - states[:, :, :-1, :]
    # State changes only, then the matched state. Actions stay out of the value token.
    similar_superstates_diff = torch.cat((state_diff.flatten(start_dim=2), states[:, :, 0, :]), dim=-1)
    return {
        "current_super_state": current_super_state,
        "similar_superstates": similar_superstates,
        "similar_superstates_diff": similar_superstates_diff,
        "current_state_action": current_state_action,
    }
    
    

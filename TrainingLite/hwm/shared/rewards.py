"""Batched low-level rewards for the HWM planner.

Placeholders: weights, the target state and the hardwired speed will change.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from TrainingLite.hwm.shared.observation import _normalize_states, _scale
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    POSE_THETA_COS_IDX,
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)


class RewardOutput(NamedTuple):
    reward: torch.Tensor
    """``(B,)`` sum of the learned terms, the value head's target reward."""
    discount: torch.Tensor
    """``(B,)`` estimated probability that the step does not end in a wall."""
    uncertainty: torch.Tensor
    """``(B,)`` ``1 - 1/sqrt(n_soft)``, for planning only; not learned by the value head."""
    terms: dict[str, torch.Tensor]
    """``(B,)`` each learned term before summation."""


class LowLevelRewards:
    # Per-unit penalties on the change of the network-unit action, [steering, acceleration].
    JITTER_WEIGHTS = (1.5, 0.1)
    # Speed of the waypoint target state, m/s.
    TARGET_SPEED = 2.0

    def __init__(self, wall_geometry: WallGeometry, lld: torch.nn.Module):
        self.wall_geometry = wall_geometry
        self.lld = lld
        self.raceline = wall_geometry.raceline_abs

    def __call__(
        self,
        lld_observation: dict[str, torch.Tensor],
        lld_prediction: Optional[torch.distributions.MultivariateNormal] = None,
    ) -> RewardOutput:
        """All rewards and the wall discount for the step the LLD observation describes.

        ``lld_observation`` comes from ``build_LLD_observation_inference``. It provides
        the current state and action, the previous action (last history row of the
        superstate) and ``n_soft``. ``lld_prediction`` is the LLD output for that
        observation; it is computed here without gradients if not given.
        """
        state_dim = int(Settings.HWM_STATE_DIM)
        row_dim = state_dim + int(Settings.HWM_ACTION_DIM)
        current = lld_observation["current_state_action"]
        state = current[:, :state_dim] * _scale(current[:, :state_dim])
        action = current[:, state_dim:]

        super_state = lld_observation["current_super_state"]
        if super_state.shape[-1] > row_dim:
            prev_row = super_state[:, -2 * row_dim : -row_dim]
            # All-zero rows pad the history before the episode start; no jitter there.
            started = prev_row.abs().sum(dim=-1, keepdim=True) > 0
            prev_action = torch.where(started, prev_row[:, state_dim:], action)
        else:
            prev_action = action

        terms = {
            "action_jitter": self.action_jitter(action, prev_action),
            "waypoint_tracking": self.waypoint_tracking(state),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)

        if lld_prediction is None:
            with torch.no_grad():
                lld_prediction = self.lld(lld_observation)
        mean, cov = self.next_pose_gaussian(state, lld_prediction)
        crash = self.wall_geometry.crash_probability(mean.detach(), cov.detach())
        discount = 1.0 - crash.to(device=state.device, dtype=state.dtype)

        uncertainty = 1.0 - torch.rsqrt(lld_observation["n_soft"])
        return RewardOutput(reward, discount, uncertainty, terms)

    def next_pose_gaussian(
        self, state: torch.Tensor, lld_prediction: torch.distributions.MultivariateNormal
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian over the next ``[x, y, theta]``: mean ``(B, 3)``, covariance ``(B, 3, 3)``.

        ``state`` is the raw current state ``(B, state_dim)``. The mean is the pose
        reconstructed from the predicted mean delta. Each column of the delta's Cholesky
        factor is pushed through the reconstruction at ``±1`` and half the pose difference
        becomes a column of the pose covariance's square root.
        """
        mu = lld_prediction.mean
        L = lld_prediction.scale_tril
        d = int(mu.shape[-1])
        deltas = torch.cat((mu[:, None], mu[:, None] + L.mT, mu[:, None] - L.mT), dim=1)
        states = state[:, None].expand(-1, 1 + 2 * d, -1)
        poses = self.lld.reconstruct_next_state(states, deltas)[..., [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
        diff = poses[:, 1 : 1 + d] - poses[:, 1 + d :]
        diff[..., 2] = torch.atan2(torch.sin(diff[..., 2]), torch.cos(diff[..., 2]))
        half = 0.5 * diff
        return poses[:, 0], half.mT @ half

    @classmethod
    def action_jitter(cls, action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """``-Σ w_i |a_i - a_prev_i|`` for actions ``(..., 2)`` in network units. Returns ``(...)``."""
        weights = torch.tensor(cls.JITTER_WEIGHTS, dtype=action.dtype, device=action.device)
        return -((action - prev_action).abs() * weights).sum(dim=-1)

    def waypoint_target_states(self, states: torch.Tensor) -> torch.Tensor:
        """Target state ``(B, state_dim)`` for raw states ``(B, state_dim)``.

        The car standing on the next raceline waypoint ahead of it, heading straight at
        the waypoint after that, at ``TARGET_SPEED`` with no lateral or yaw motion and
        zero steering and slip.
        """
        raceline = self.raceline.to(device=states.device, dtype=states.dtype)
        n = raceline.shape[0]
        pos = states[:, [POSE_X_IDX, POSE_Y_IDX]]
        nearest = WallGeometry._nearest_indices(pos, raceline)
        segment = raceline[(nearest + 1) % n] - raceline[nearest]
        past_nearest = ((pos - raceline[nearest]) * segment).sum(dim=-1) > 0
        nxt = (nearest + past_nearest.long()) % n
        direction = raceline[(nxt + 1) % n] - raceline[nxt]
        heading = torch.atan2(direction[:, 1], direction[:, 0])

        target = states.new_zeros(states.shape)
        target[:, POSE_X_IDX] = raceline[nxt, 0]
        target[:, POSE_Y_IDX] = raceline[nxt, 1]
        target[:, POSE_THETA_IDX] = heading
        target[:, POSE_THETA_COS_IDX] = torch.cos(heading)
        target[:, POSE_THETA_SIN_IDX] = torch.sin(heading)
        target[:, LINEAR_VEL_X_IDX] = self.TARGET_SPEED
        return target

    def waypoint_tracking(self, states: torch.Tensor) -> torch.Tensor:
        """``-||(s - s_target) / HWM_STATE_SCALE||`` for raw states ``(B, state_dim)``. Returns ``(B,)``.

        The heading difference is wrapped to ``(-π, π]``.
        """
        diff = states - self.waypoint_target_states(states)
        theta = diff[:, POSE_THETA_IDX]
        diff[:, POSE_THETA_IDX] = torch.atan2(torch.sin(theta), torch.cos(theta))
        return -torch.linalg.norm(_normalize_states(diff), dim=-1)

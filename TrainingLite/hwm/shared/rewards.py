"""Batched low-level rewards for the HWM planner.

Placeholders: weights, the target state and the hardwired speed will change.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from TrainingLite.hwm.shared.observation import _scale, lld_cosine_attention, lld_reflect_on, lld_super_row_dim
from TrainingLite.hwm.shared.targets import build_target_states, target_index
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from TrainingLite.hwm.shared.state_indices import (
    LINEAR_VEL_X_IDX,
    POSE_THETA_IDX,
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
    JITTER_WEIGHTS = (0,0) #(0.1, 0.3)
    # Multiplier of the progress term. Its raw unit is ``state_distance``; with
    # HWM_TARGET_POSITION_WEIGHT = 0.1 one step at 2 m/s is about 0.008.
    PROGRESS_WEIGHT = 10.0
    # A step whose target moves by more than this many waypoints switched to another part
    # of the track that runs close by (behind a wall) and earns no progress.
    MAX_TARGET_STEP = 20

    def __init__(self, wall_geometry: WallGeometry, lld: torch.nn.Module):
        self.wall_geometry = wall_geometry
        self.lld = lld
        self.raceline = wall_geometry.raceline_abs
        # Target state of every waypoint, and the route length (summed scaled distance
        # between consecutive targets) from waypoint 0 to each waypoint and once around.
        self.targets = build_target_states(self.raceline)
        segments = self.state_distance(self.targets, torch.roll(self.targets, -1, dims=0))
        self.route = torch.cat((segments.new_zeros(1), torch.cumsum(segments[:-1], dim=0)))
        self.lap = segments.sum()

    def __call__(
        self,
        lld_observation: dict[str, torch.Tensor],
        lld_prediction: Optional[torch.distributions.MultivariateNormal] = None,
        next_state: Optional[torch.Tensor] = None,
    ) -> RewardOutput:
        """All rewards and the wall discount for the step the LLD observation describes.

        ``lld_observation`` comes from ``build_LLD_observation_inference``. It provides
        the current state and action, the previous action (last history row of the
        superstate) and ``n_soft``. ``lld_prediction`` is the LLD output for that
        observation; it is computed here without gradients if not given. ``next_state``
        ``(B, state_dim)`` is the raw state the step reached, for the progress term;
        if not given, the state reconstructed from the predicted mean delta is used.
        """
        state_dim = int(Settings.HWM_STATE_DIM)
        action_dim = int(Settings.HWM_ACTION_DIM)
        row_dim = lld_super_row_dim()
        current = lld_observation["current_state_action"]
        state = current[:, :state_dim] * _scale(current[:, :state_dim])
        action = current[:, state_dim:]

        super_state = lld_observation["current_super_state"]
        if lld_reflect_on():
            # HWM_LLD_REFLECT stores each history row mirrored by its own sign, so the
            # previous steering action cannot be read back. No jitter term in that case.
            prev_action = action
        elif super_state.shape[-1] > row_dim:
            prev_row = super_state[:, -2 * row_dim : -row_dim]
            # All-zero rows pad the history before the episode start; no jitter there.
            # The action is the tail of each packed superstate row.
            started = prev_row.abs().sum(dim=-1, keepdim=True) > 0
            prev_action = torch.where(started, prev_row[:, -action_dim:], action)
        else:
            prev_action = action

        if lld_prediction is None:
            with torch.no_grad():
                lld_prediction = self.lld(lld_observation)
        if next_state is None:
            next_state = self.lld.reconstruct_next_state(state, lld_prediction.mean)

        terms = {
            "action_jitter": self.action_jitter(action, prev_action),
            "waypoint_progress": self.waypoint_progress(state, next_state),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)

        mean, cov = self.next_pose_gaussian(state, lld_prediction)
        crash = self.wall_geometry.crash_probability(mean.detach(), cov.detach())
        discount = 1.0 - crash.to(device=state.device, dtype=state.dtype)

        # Dot-product attention does not define n_soft, so this term is zero there.
        if lld_cosine_attention():
            uncertainty = torch.zeros(int(state.shape[0]), device=state.device, dtype=state.dtype)
        else:
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
        # A new tensor: writing the wrapped heading into ``diff`` would break backprop through it.
        heading = torch.atan2(torch.sin(diff[..., 2]), torch.cos(diff[..., 2]))
        diff = torch.stack((diff[..., 0], diff[..., 1], heading), dim=-1)
        half = 0.5 * diff
        return poses[:, 0], half.mT @ half

    @classmethod
    def action_jitter(cls, action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """``-Σ w_i |a_i - a_prev_i|`` for actions ``(..., 2)`` in network units. Returns ``(...)``."""
        weights = torch.tensor(cls.JITTER_WEIGHTS, dtype=action.dtype, device=action.device)
        return -((action - prev_action).abs() * weights).sum(dim=-1)

    @staticmethod
    def state_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Weighted distance between raw states ``(..., state_dim)``: position, heading and forward speed.

        ``||(w_p |Δxy|, w_θ Δθ, w_v Δv_x)||`` with the weights ``HWM_TARGET_*_WEIGHT`` and
        ``Δθ`` wrapped. All other channels are ignored.
        """
        diff = a - b
        position = torch.linalg.norm(diff[..., [POSE_X_IDX, POSE_Y_IDX]], dim=-1)
        theta = diff[..., POSE_THETA_IDX]
        heading = torch.atan2(torch.sin(theta), torch.cos(theta))
        speed = diff[..., LINEAR_VEL_X_IDX]
        return torch.linalg.norm(
            torch.stack(
                (
                    float(Settings.HWM_TARGET_POSITION_WEIGHT) * position,
                    float(Settings.HWM_TARGET_HEADING_WEIGHT) * heading,
                    float(Settings.HWM_TARGET_SPEED_WEIGHT) * speed,
                ),
                dim=-1,
            ),
            dim=-1,
        )

    def target_index(self, states: torch.Tensor) -> torch.Tensor:
        """Index ``(B,)`` of the tracked waypoint (``targets.target_index``) for raw states ``(B, state_dim)``."""
        raceline = self.raceline.to(device=states.device, dtype=states.dtype)
        return target_index(states[:, [POSE_X_IDX, POSE_Y_IDX]], raceline)

    def waypoint_target_states(self, states: torch.Tensor) -> torch.Tensor:
        """Target state ``(B, state_dim)`` of the tracked waypoint for raw states ``(B, state_dim)``."""
        return self.targets.to(device=states.device, dtype=states.dtype)[self.target_index(states)]

    def route_potential(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Φ (B,), target index (B,))``, ``Φ`` = route length to the target minus the distance to it."""
        index = self.target_index(states)
        targets = self.targets.to(device=states.device, dtype=states.dtype)
        route = self.route.to(device=states.device, dtype=states.dtype)
        return route[index] - self.state_distance(states, targets[index]), index

    def waypoint_progress(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """``PROGRESS_WEIGHT · (Φ(s') - Φ(s))`` for raw states ``(B, state_dim)``. Returns ``(B,)``.

        Approaching the target state earns its distance decrease. Driving over a
        waypoint moves the target on by one; the route length between the two targets
        is added, so the switch costs nothing (by the triangle inequality it can only
        add, a little in curves) and passing several waypoints within a step still
        counts as progress. Crossing the lap start adds or removes one lap of route.

        The nearest waypoint can jump to another part of the track where it runs close
        behind a wall. Such a step (target moving by more than ``MAX_TARGET_STEP``
        waypoints) earns nothing; otherwise its route difference would be free reward.
        """
        phi, index = self.route_potential(states)
        next_phi, next_index = self.route_potential(next_states)
        n = int(self.targets.shape[0])
        signed = torch.remainder(next_index - index + n // 2, n) - n // 2
        lap = self.lap.to(device=states.device, dtype=states.dtype)
        wrap = ((signed > 0) & (next_index < index)).to(states.dtype) - ((signed < 0) & (next_index > index)).to(states.dtype)
        progress = self.PROGRESS_WEIGHT * (next_phi - phi + wrap * lap)
        return torch.where(signed.abs() <= self.MAX_TARGET_STEP, progress, torch.zeros_like(progress))

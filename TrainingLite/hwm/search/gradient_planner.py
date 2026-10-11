"""Gradient-descent planning over the coefficient feedback law and the LLD.

Used for acting and for the actor's training targets when ``HWM_PLAN_MODE`` is ``"gd"``.
One plan is batched over ``B`` independent roots.

The decision variables are the coefficients of ``actions_from_coefficients``, not a
sequence of actions. They are rolled out for ``HWM_GD_HORIZON`` control steps. Each
step recomputes the features at the predicted state, so the same coefficients steer
differently as the car moves. ``HWM_GD_INFERENCE_SAMPLES`` (training:
``HWM_GD_TRAIN_SAMPLES``) trajectories are drawn from the LLD with the reparameterization
trick. The noise is drawn once per plan and kept for every gradient step, and sample 0
is the mean trajectory. The reward is the mean over those trajectories of
``-|d|_min + mean(wall) + mean(min(v_x, 0))``, minus ``HWM_GD_COEFF_REG`` times the
squared distance from the coefficients the plan started at.

``HWM_GD_WARM_START`` starts from the previous plan's coefficients (zeros on the first
plan of an episode). Otherwise the actor's coefficients are the start. Inference stops
on ``HWM_GD_TIME_BUDGET_S``; training runs ``HWM_GD_STEPS``. The car drives the action
the optimized coefficients produce at the root state.
"""

from __future__ import annotations

import time
from typing import NamedTuple, Optional

import torch

from TrainingLite.hwm.search.coefficients import (
    COEFF_DIM,
    actions_from_coefficients,
    frozen_reference,
)
from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.observation import (
    _normalize_rows,
    _normalize_states,
    apply_rollout_window,
    build_actor_observation,
    lld_rows_for_super,
    lld_super_state,
    repeat_neighbor_rows,
    retrieve_LLD_neighbors,
)
from TrainingLite.hwm.shared.rewards import LowLevelRewards
from TrainingLite.hwm.shared.state_indices import LINEAR_VEL_X_IDX, POSE_X_IDX, POSE_Y_IDX
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings

PLAN_MODES = ("actor", "gd", "traj")


def plan_mode() -> str:
    """``HWM_PLAN_MODE``, checked against ``PLAN_MODES``."""
    mode = str(Settings.HWM_PLAN_MODE)
    if mode not in PLAN_MODES:
        raise ValueError(f"HWM_PLAN_MODE must be one of {PLAN_MODES}, got {mode!r}")
    return mode


class GradientPlan(NamedTuple):
    action: torch.Tensor
    """``(B, action_dim)`` action the optimized coefficients produce at the root. Network units."""
    coefficients: torch.Tensor
    """``(B, COEFF_DIM)`` the optimized feedback law."""
    states: torch.Tensor
    """``(B, horizon, state_dim)`` raw state of the mean trajectory before each step."""
    histories: torch.Tensor
    """``(B, horizon, S-1, row_dim)`` raw rows preceding each of those states."""
    reward: torch.Tensor
    """``(B,)`` mean over samples of ``-|d|_min + mean wall punishment + mean predicted reverse speed``."""
    distance: torch.Tensor
    """``(B,)`` mean over samples of the smallest weighted distance to the target."""
    crash: torch.Tensor
    """``(B,)`` mean over samples of the mean wall punishment, in meters, zero or negative."""
    steps: int
    """Gradient steps taken."""


class GradientPlanner:
    def __init__(
        self,
        actor: torch.nn.Module,
        lld: torch.nn.Module,
        rewards: LowLevelRewards,
        memory: MemoryManager,
        wall_geometry: WallGeometry,
    ):
        self.actor = actor
        self.lld = lld
        self.rewards = rewards
        self.memory = memory
        self.wall_geometry = wall_geometry
        self.horizon = int(Settings.HWM_GD_HORIZON)
        self.lr = float(Settings.HWM_GD_LR)
        self.step_cap = float(Settings.HWM_GD_COEFF_STEP)
        self.train_steps = int(Settings.HWM_GD_STEPS)
        self.resample = bool(Settings.HWM_GD_RESAMPLE_NEIGHBORS)
        self.wall_weight = float(Settings.HWM_GD_WALL_WEIGHT)
        self.coeff_reg = float(Settings.HWM_GD_COEFF_REG)
        self.smoothing = float(Settings.HWM_GD_CRASH_SMOOTHING_M)
        self.last_grad_abs: Optional[torch.Tensor] = None
        if self.horizon < 1:
            raise ValueError(f"HWM_GD_HORIZON must be at least 1, got {self.horizon}")
        if self.lr <= 0:
            raise ValueError(f"HWM_GD_LR must be positive, got {self.lr}")
        if self.step_cap <= 0:
            raise ValueError(f"HWM_GD_COEFF_STEP must be positive, got {self.step_cap}")
        if self.train_steps < 0:
            raise ValueError(f"HWM_GD_STEPS must be at least 0, got {self.train_steps}")
        if self.coeff_reg < 0:
            raise ValueError(f"HWM_GD_COEFF_REG must be at least 0, got {self.coeff_reg}")
        if self.smoothing <= 0:
            raise ValueError(f"HWM_GD_CRASH_SMOOTHING_M must be positive, got {self.smoothing}")

    def optimize(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        before_steps: Optional[torch.Tensor] = None,
        deadline: Optional[float] = None,
        max_steps: Optional[int] = None,
        initial_coefficients: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> GradientPlan:
        """Optimize coefficients from raw ``state`` ``(B, state_dim)`` after raw ``history`` ``(B, S-1, row_dim)``.

        ``before_steps`` ``(B,)`` are the roots' global memory indices (learner): every
        retrieval only sees windows stored before that root. None (planner): everything
        stored so far. ``deadline`` is a ``time.perf_counter()`` value. No retrieval
        and no gradient step starts after it; the coefficients so far are returned
        without a scoring rollout. ``max_steps`` caps the steps (training). With
        neither, ``HWM_GD_STEPS`` steps are run.

        ``initial_coefficients`` ``(B, COEFF_DIM)`` is what gradient descent starts from.
        Without it, the actor's coefficients at ``state`` are the start. ``num_samples``
        defaults to ``HWM_GD_INFERENCE_SAMPLES`` when a deadline is set and to
        ``HWM_GD_TRAIN_SAMPLES`` otherwise. Sample 0 is always the mean trajectory.
        """
        actor_training, lld_training = self.actor.training, self.lld.training
        lld_grad = [parameter.requires_grad for parameter in self.lld.parameters()]
        self.actor.eval()
        self.lld.eval()
        for parameter in self.lld.parameters():
            parameter.requires_grad_(False)
        try:
            return self._optimize(
                state, history, before_steps, deadline, max_steps, initial_coefficients, num_samples
            )
        finally:
            self.actor.train(actor_training)
            self.lld.train(lld_training)
            for parameter, flag in zip(self.lld.parameters(), lld_grad):
                parameter.requires_grad_(flag)

    def _optimize(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        before_steps: Optional[torch.Tensor],
        deadline: Optional[float],
        max_steps: Optional[int],
        initial_coefficients: Optional[torch.Tensor],
        num_samples: Optional[int],
    ) -> GradientPlan:
        device = self.memory.device
        state = state.to(device=device, dtype=torch.float32)
        history = history.to(device=device, dtype=torch.float32)
        batch = int(state.shape[0])
        if max_steps is None and deadline is None:
            max_steps = self.train_steps
        if num_samples is None:
            num_samples = int(
                Settings.HWM_GD_INFERENCE_SAMPLES if deadline is not None else Settings.HWM_GD_TRAIN_SAMPLES
            )
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be at least 1, got {num_samples}")
        target, lookahead = frozen_reference(state, self.wall_geometry)

        def out_of_time() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        if initial_coefficients is None:
            if not getattr(self.actor, "coefficient", False):
                raise RuntimeError("gradient planning without initial coefficients needs HWM_PLAN_MODE 'gd'")
            with torch.no_grad():
                obs = build_actor_observation(state, self.memory, self.wall_geometry, 0, past_states=history)
                initial_coefficients, _ = self.actor(obs)
        anchor = initial_coefficients.to(device=device, dtype=torch.float32).reshape(batch, COEFF_DIM).detach()
        anchor = anchor.clamp(-1.0, 1.0)
        if out_of_time():
            return self._unscored(anchor, state, history, target, lookahead, 0)

        noise = self._noise(num_samples, device)
        neighbors = self._retrieve(state, history, before_steps, anchor, target, lookahead, deadline)
        if neighbors is None:
            return self._unscored(anchor, state, history, target, lookahead, 0)
        coeffs = anchor.detach().requires_grad_(True)
        taken = 0
        while True:
            if out_of_time():
                break
            if max_steps is not None and taken >= max_steps:
                break
            if self.resample and taken > 0:
                refreshed = self._retrieve(
                    state, history, before_steps, coeffs.detach(), target, lookahead, deadline
                )
                if refreshed is None:
                    break
                neighbors = refreshed
            reward = self._rollout(state, history, coeffs, neighbors, target, lookahead, noise).reward
            penalty = self.coeff_reg * (coeffs - anchor).square().sum()
            grad = torch.autograd.grad(reward.sum() - penalty, coeffs)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            self.last_grad_abs = grad.detach().abs().mean(dim=0)
            with torch.no_grad():
                coeffs += (self.lr * grad).clamp(-self.step_cap, self.step_cap)
                coeffs.clamp_(-1.0, 1.0)
            taken += 1

        final = coeffs.detach()
        # Inference only needs the action at the root. Scoring is another full rollout.
        if deadline is not None:
            return self._unscored(final, state, history, target, lookahead, taken)
        if self.resample:
            neighbors = self._retrieve(state, history, before_steps, final, target, lookahead)
        with torch.no_grad():
            scored = self._rollout(state, history, final, neighbors, target, lookahead, noise)
        action = actions_from_coefficients(final, state, target, lookahead, self.wall_geometry)
        return scored._replace(action=action, coefficients=final, steps=taken)

    def _unscored(
        self,
        coefficients: torch.Tensor,
        state: torch.Tensor,
        history: torch.Tensor,
        target: torch.Tensor,
        lookahead: torch.Tensor,
        steps: int,
    ) -> GradientPlan:
        """The coefficients and the root action, with the scored fields left at zero."""
        batch = int(state.shape[0])
        zeros = state.new_zeros(batch)
        action = actions_from_coefficients(coefficients, state, target, lookahead, self.wall_geometry)
        return GradientPlan(
            action=action.detach(),
            coefficients=coefficients.detach(),
            states=state.new_zeros(batch, self.horizon, int(state.shape[-1])),
            histories=history.new_zeros(batch, self.horizon, *history.shape[1:]),
            reward=zeros,
            distance=zeros,
            crash=zeros,
            steps=steps,
        )

    @torch.no_grad()
    def mean_trajectory(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        coefficients: torch.Tensor,
        before_steps: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """``(B, steps + 1, state_dim)`` raw states of the noise-free LLD rollout of ``coefficients``.

        Starts with ``state``. The reference is frozen at ``state`` as in ``optimize``,
        and the windows are retrieved the same way, so this is the plan's expected path.
        ``steps`` defaults to the planning horizon.
        """
        device = self.memory.device
        state = state.to(device=device, dtype=torch.float32)
        history = history.to(device=device, dtype=torch.float32)
        coefficients = coefficients.to(device=device, dtype=torch.float32)
        target, lookahead = frozen_reference(state, self.wall_geometry)
        lld_training = self.lld.training
        self.lld.eval()
        try:
            bank = None
            states = [state]
            for _ in range(self.horizon if steps is None else int(steps)):
                action = actions_from_coefficients(coefficients, state, target, lookahead, self.wall_geometry)
                if bank is None or not Settings.HWM_LLD_WHOLE_MEMORY_ATTENTION:
                    query, _ = lld_super_state(state, action, self.memory, 0, past_states=history)
                    bank = retrieve_LLD_neighbors(query, self.memory, before_steps)
                _, next_state = self._predict(state, action, history, bank)
                history = torch.cat((history, torch.cat((state, action), dim=-1)[:, None]), dim=1)[:, 1:]
                state = next_state
                states.append(state)
        finally:
            self.lld.train(lld_training)
        return torch.stack(states, dim=1)

    def _noise(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """``(horizon, samples, prediction_dim)``. Sample 0 is zero, so that trajectory is the mean."""
        pred_dim = int(self.lld.prediction_dim)
        noise = torch.randn(self.horizon, num_samples, pred_dim, device=device)
        noise[:, 0] = 0
        clip = float(Settings.HWM_GD_NOISE_CLIP)
        if clip > 0:
            noise.clamp_(-clip, clip)
        return noise

    @torch.no_grad()
    def _retrieve(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        before_steps: Optional[torch.Tensor],
        coefficients: torch.Tensor,
        target: torch.Tensor,
        lookahead: torch.Tensor,
        deadline: Optional[float] = None,
    ) -> Optional[list[dict[str, torch.Tensor]]]:
        """Neighbor windows along the mean rollout of ``coefficients``. None if ``deadline`` wins."""
        neighbors: list[dict[str, torch.Tensor]] = []
        if Settings.HWM_LLD_WHOLE_MEMORY_ATTENTION:
            # The full bank does not depend on the query; the LLD computes live,
            # differentiable weights from each rollout state's query. Gather and
            # mask it once rather than copying the same bank through a mean rollout.
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            action = actions_from_coefficients(coefficients, state, target, lookahead, self.wall_geometry)
            query, _ = lld_super_state(state, action, self.memory, 0, past_states=history)
            found = retrieve_LLD_neighbors(query, self.memory, before_steps)
            return [found] * self.horizon
        for _ in range(self.horizon):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            action = actions_from_coefficients(coefficients, state, target, lookahead, self.wall_geometry)
            query, _ = lld_super_state(state, action, self.memory, 0, past_states=history)
            found = retrieve_LLD_neighbors(query, self.memory, before_steps)
            neighbors.append(found)
            _, next_state = self._predict(state, action, history, found)
            history = torch.cat((history, torch.cat((state, action), dim=-1)[:, None]), dim=1)[:, 1:]
            state = next_state
        return neighbors

    def _rollout(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        coefficients: torch.Tensor,
        neighbors: list[dict[str, torch.Tensor]],
        target: torch.Tensor,
        lookahead: torch.Tensor,
        noise: torch.Tensor,
    ) -> GradientPlan:
        """Mean reward of the sampled rollouts. Differentiable in ``coefficients``."""
        batch = int(state.shape[0])
        samples = int(noise.shape[1])
        pred_dim = int(noise.shape[-1])
        state = state.repeat_interleave(samples, dim=0)
        history = history.repeat_interleave(samples, dim=0)
        coefficients = coefficients.repeat_interleave(samples, dim=0)
        target = target.repeat_interleave(samples, dim=0)
        lookahead = lookahead.repeat_interleave(samples, dim=0)
        start_xy = state[:, [POSE_X_IDX, POSE_Y_IDX]]
        states, histories, reached = [], [], []
        for t in range(self.horizon):
            states.append(state.detach())
            histories.append(history.detach())
            action = actions_from_coefficients(coefficients, state, target, lookahead, self.wall_geometry)
            found = repeat_neighbor_rows(neighbors[t], samples)
            prediction, _ = self._predict(state, action, history, found)
            eps = noise[t].unsqueeze(0).expand(batch, samples, pred_dim).reshape(batch * samples, pred_dim)
            delta = prediction.mean + torch.matmul(prediction.scale_tril, eps.unsqueeze(-1)).squeeze(-1)
            next_state = self.lld.reconstruct_next_state(state, delta)
            reached.append(next_state)
            history = torch.cat((history, torch.cat((state, action), dim=-1)[:, None]), dim=1)[:, 1:]
            state = next_state
        centers = torch.stack([step[:, [POSE_X_IDX, POSE_Y_IDX]] for step in reached], dim=1)
        wall = self.wall_geometry.trajectory_center_punishment(start_xy, centers)
        distance = torch.stack(
            [self.rewards.state_distance(step, target) for step in reached], dim=1
        ).abs()
        nearest = distance.min(dim=1).values.view(batch, samples).mean(dim=1)
        punishment = wall.mean(dim=1).view(batch, samples).mean(dim=1)
        reverse = (
            torch.stack([step[:, LINEAR_VEL_X_IDX] for step in reached], dim=1)
            .clamp(max=0)
            .mean(dim=1)
            .view(batch, samples)
            .mean(dim=1)
        )
        mean_states = torch.stack(states, dim=1).view(batch, samples, self.horizon, -1)[:, 0]
        mean_histories = torch.stack(histories, dim=1).view(batch, samples, self.horizon, *histories[0].shape[1:])[:, 0]
        root_action = actions_from_coefficients(
            coefficients.view(batch, samples, COEFF_DIM)[:, 0],
            mean_states[:, 0],
            target.view(batch, samples, -1)[:, 0],
            lookahead.view(batch, samples)[:, 0],
            self.wall_geometry,
        )
        return GradientPlan(
            action=root_action,
            coefficients=coefficients.view(batch, samples, COEFF_DIM)[:, 0],
            states=mean_states.detach(),
            histories=mean_histories.detach(),
            reward=-nearest + self.wall_weight * punishment + reverse,
            distance=nearest.detach(),
            crash=punishment.detach(),
            steps=0,
        )

    def _predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor,
        neighbors: dict[str, torch.Tensor],
    ) -> tuple[torch.distributions.MultivariateNormal, torch.Tensor]:
        """``(LLD predictive, next raw state from its mean delta)`` for one control step."""
        state_dim = int(Settings.HWM_STATE_DIM)
        current = lld_rows_for_super(torch.cat((_normalize_states(state), action), dim=-1), state_dim)
        past = lld_rows_for_super(_normalize_rows(history, state_dim), state_dim)
        superstate = torch.cat((past.flatten(start_dim=1), current), dim=-1)
        neighbors = apply_rollout_window(self.lld, superstate.detach(), neighbors)
        prediction = self.lld(
            {
                "current_super_state": superstate,
                "similar_superstates": neighbors["similar_superstates"],
                "similar_superstates_diff": neighbors["similar_superstates_diff"],
                "similar_valid": neighbors["valid"],
                **({"similar_density": neighbors["density"]} if neighbors.get("density") is not None else {}),
            }
        )
        return prediction, self.lld.reconstruct_next_state(state, prediction.mean)

"""Fixed-noise trajectory optimization over the LLD, the actor's training targets in ``"traj"`` mode.

From every root ``HWM_TRAJ_SAMPLES`` (``K``) standard-normal vectors ``ε`` are drawn, one
per trajectory, and each is the reparameterization noise of every step of its trajectory:
``delta_t = μ_t + L_t ε``. Each trajectory is first driven ``HWM_TRAJ_HORIZON`` (``n``)
steps by the actor, closed loop through the LLD, the actor seeing that same ``ε`` as its
noise input. Those actions are the start of ``HWM_TRAJ_GD_STEPS`` gradient steps on each
trajectory's own action sequence, under its own ``ε``. The reward of a trajectory is

    Σ_t waypoint_progress(s_t, s_{t+1}) + HWM_TRAJ_WALL_WEIGHT · Σ_t wall(s_{t+1}) + mean_t min(v_x, 0)

where the progress target is the waypoint tracked from each state, so it moves along
the trajectory, and ``wall`` is the center punishment in meters (zero or negative).
Gradient descent maximizes it minus ``HWM_TRAJ_ACTION_REG`` times the squared distance
from the actor's actions. The optimized actions are rolled out once more; the visited
states with their optimized actions are the actor's regression targets.

The LLD attends over the whole memory (``HWM_LLD_WHOLE_MEMORY_ATTENTION``): the bank is
gathered once per root, and every prediction computes live, differentiable attention
weights from its own superstate.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.observation import (
    _normalize_rows,
    _normalize_states,
    build_traj_actor_observation,
    apply_rollout_window,
    lld_reflect,
    lld_rows_for_super,
    lld_super_state,
    repeat_neighbor_rows,
    retrieve_LLD_neighbors,
    traj_noise_dim,
)
from TrainingLite.hwm.shared.rewards import LowLevelRewards
from TrainingLite.hwm.shared.state_indices import LINEAR_VEL_X_IDX, POSE_X_IDX, POSE_Y_IDX
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings


class TrajectoryPlan(NamedTuple):
    states: torch.Tensor
    """``(B * K, n, state_dim)`` raw state before each step of the optimized rollout, root first."""
    histories: torch.Tensor
    """``(B * K, n, S-1, row_dim)`` raw rows preceding each of those states."""
    actions: torch.Tensor
    """``(B * K, n, action_dim)`` optimized actions, network units in ``[-1, 1]``."""
    noise: torch.Tensor
    """``(B * K, d)`` the trajectory's fixed noise, also the actor's noise input along it."""
    reward: torch.Tensor
    """``(B * K,)`` reward of the optimized actions."""
    initial_reward: torch.Tensor
    """``(B * K,)`` reward of the actor's actions, before gradient descent."""
    progress: torch.Tensor
    """``(B * K,)`` summed waypoint progress of the optimized rollout."""
    wall: torch.Tensor
    """``(B * K,)`` summed wall punishment of the optimized rollout, meters, zero or negative."""
    steps: int
    """Gradient steps taken."""


class _Rollout(NamedTuple):
    states: torch.Tensor
    histories: torch.Tensor
    reward: torch.Tensor
    progress: torch.Tensor
    wall: torch.Tensor


class TrajectoryOptimizer:
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
        self.horizon = int(Settings.HWM_TRAJ_HORIZON)
        self.samples = int(Settings.HWM_TRAJ_SAMPLES)
        self.gd_steps = int(Settings.HWM_TRAJ_GD_STEPS)
        self.lr = float(Settings.HWM_TRAJ_GD_LR)
        self.step_cap = float(Settings.HWM_TRAJ_ACTION_STEP)
        self.noise_clip = float(Settings.HWM_TRAJ_NOISE_CLIP)
        self.wall_weight = float(Settings.HWM_TRAJ_WALL_WEIGHT)
        self.action_reg = float(Settings.HWM_TRAJ_ACTION_REG)
        if not Settings.HWM_LLD_WHOLE_MEMORY_ATTENTION:
            raise ValueError("HWM_PLAN_MODE 'traj' needs HWM_LLD_WHOLE_MEMORY_ATTENTION = True")
        if self.horizon < 1:
            raise ValueError(f"HWM_TRAJ_HORIZON must be at least 1, got {self.horizon}")
        if self.samples < 1:
            raise ValueError(f"HWM_TRAJ_SAMPLES must be at least 1, got {self.samples}")
        if self.gd_steps < 0:
            raise ValueError(f"HWM_TRAJ_GD_STEPS must be at least 0, got {self.gd_steps}")
        if self.lr <= 0:
            raise ValueError(f"HWM_TRAJ_GD_LR must be positive, got {self.lr}")
        if self.step_cap <= 0:
            raise ValueError(f"HWM_TRAJ_ACTION_STEP must be positive, got {self.step_cap}")
        if self.action_reg < 0:
            raise ValueError(f"HWM_TRAJ_ACTION_REG must be at least 0, got {self.action_reg}")

    def optimize(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        before_steps: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> TrajectoryPlan:
        """Optimize ``K`` trajectories from each raw root ``state`` ``(B, state_dim)`` after ``history`` ``(B, S-1, row_dim)``.

        ``before_steps`` ``(B,)`` are the roots' global memory indices: the LLD only
        attends to windows stored before them. None: everything stored so far. Rows of
        the result are root-major, ``K`` trajectories per root. ``num_samples`` overrides
        ``K``. ``noise`` ``(K, d)`` or ``(d,)`` replaces the random draw; a zero vector is
        the mean dynamics.
        """
        actor_training, lld_training = self.actor.training, self.lld.training
        lld_grad = [parameter.requires_grad for parameter in self.lld.parameters()]
        self.actor.eval()
        self.lld.eval()
        for parameter in self.lld.parameters():
            parameter.requires_grad_(False)
        try:
            return self._optimize(state, history, before_steps, num_samples, noise)
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
        num_samples: Optional[int],
        noise: Optional[torch.Tensor],
    ) -> TrajectoryPlan:
        device = self.memory.device
        state = state.to(device=device, dtype=torch.float32)
        history = history.to(device=device, dtype=torch.float32)
        samples = self.samples if num_samples is None else int(num_samples)
        bank = repeat_neighbor_rows(self._bank(state, history, before_steps), samples)
        state = state.repeat_interleave(samples, dim=0)
        history = history.repeat_interleave(samples, dim=0)
        if noise is None:
            noise = torch.randn(int(state.shape[0]), traj_noise_dim(), device=device)
        else:
            noise = noise.to(device=device, dtype=torch.float32).reshape(-1, traj_noise_dim())
            noise = noise.expand(int(state.shape[0]), -1).contiguous()
        if self.noise_clip > 0:
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

        with torch.no_grad():
            _, initial = self._actor_rollout(state, history, noise, bank, self.horizon)
        actions = initial.clone().requires_grad_(True)
        initial_reward: Optional[torch.Tensor] = None
        for _ in range(self.gd_steps):
            out = self._rollout(state, history, actions, noise, bank)
            if initial_reward is None:
                initial_reward = out.reward.detach()
            objective = out.reward.sum() - self.action_reg * (actions - initial).square().sum()
            grad = torch.autograd.grad(objective, actions)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                actions += (self.lr * grad).clamp(-self.step_cap, self.step_cap)
                actions.clamp_(-1.0, 1.0)

        final_actions = actions.detach()
        with torch.no_grad():
            final = self._rollout(state, history, final_actions, noise, bank)
        return TrajectoryPlan(
            states=final.states,
            histories=final.histories,
            actions=final_actions,
            noise=noise,
            reward=final.reward,
            initial_reward=final.reward if initial_reward is None else initial_reward,
            progress=final.progress,
            wall=final.wall,
            steps=self.gd_steps,
        )

    @torch.no_grad()
    def _bank(
        self, state: torch.Tensor, history: torch.Tensor, before_steps: Optional[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """The whole-memory bank and each root's valid mask; the bank does not depend on the query."""
        action = state.new_zeros(int(state.shape[0]), int(Settings.HWM_ACTION_DIM))
        query, _ = lld_super_state(state, action, self.memory, 0, past_states=history)
        return retrieve_LLD_neighbors(query, self.memory, before_steps)

    def _actor_rollout(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        noise: torch.Tensor,
        bank: dict[str, torch.Tensor],
        steps: int,
        dynamics_noise: Optional[torch.Tensor] = None,
        noise_decay: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(states (R, steps + 1, state_dim), actions (R, steps, action_dim))`` of the actor under ``noise``.

        ``dynamics_noise`` is the LLD reparameterization, defaulting to ``noise`` (training).
        ``noise_decay`` multiplies both by that factor once per imagined step (step k uses
        ``decay ** k``); 1 holds them constant, as in training.
        """
        if dynamics_noise is None:
            dynamics_noise = noise
        states, actions = [state], []
        for k in range(steps):
            factor = noise_decay**k
            obs = build_traj_actor_observation(
                state, self.memory, self.wall_geometry, 0, noise * factor, past_states=history
            )
            action, _ = self.actor(obs)
            next_state = self._step(state, action, history, dynamics_noise * factor, bank)
            history = torch.cat((history, torch.cat((state, action), dim=-1)[:, None]), dim=1)[:, 1:]
            state = next_state
            states.append(state)
            actions.append(action)
        return torch.stack(states, dim=1), torch.stack(actions, dim=1)

    def _rollout(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor,
        bank: dict[str, torch.Tensor],
    ) -> _Rollout:
        """Open-loop rollout of ``actions`` ``(R, n, action_dim)`` under ``noise``. Differentiable in ``actions``."""
        start_xy = state[:, [POSE_X_IDX, POSE_Y_IDX]]
        states, histories, reached = [], [], []
        progress = state.new_zeros(int(state.shape[0]))
        for t in range(int(actions.shape[1])):
            states.append(state.detach())
            histories.append(history.detach())
            action = actions[:, t]
            next_state = self._step(state, action, history, noise, bank)
            progress = progress + self.rewards.waypoint_progress(state, next_state)
            reached.append(next_state)
            history = torch.cat((history, torch.cat((state, action), dim=-1)[:, None]), dim=1)[:, 1:]
            state = next_state
        centers = torch.stack([step[:, [POSE_X_IDX, POSE_Y_IDX]] for step in reached], dim=1)
        wall = self.wall_geometry.trajectory_center_punishment(start_xy, centers).sum(dim=1)
        reverse = torch.stack([step[:, LINEAR_VEL_X_IDX] for step in reached], dim=1).clamp(max=0).mean(dim=1)
        return _Rollout(
            states=torch.stack(states, dim=1),
            histories=torch.stack(histories, dim=1),
            reward=progress + self.wall_weight * wall + reverse,
            progress=progress.detach(),
            wall=wall.detach(),
        )

    def _step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor,
        noise: torch.Tensor,
        bank: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Next raw state from the LLD's reparameterized sample ``μ + L ε``."""
        state_dim = int(Settings.HWM_STATE_DIM)
        current = lld_rows_for_super(torch.cat((_normalize_states(state), action), dim=-1), state_dim)
        past = lld_rows_for_super(_normalize_rows(history, state_dim), state_dim)
        superstate = torch.cat((past.flatten(start_dim=1), current), dim=-1)
        # Ranked once, on the first query of this rollout, then held.
        bank = apply_rollout_window(self.lld, superstate.detach(), bank)
        prediction = self.lld(
            {
                "current_super_state": superstate,
                "similar_superstates": bank["similar_superstates"],
                "similar_superstates_diff": bank["similar_superstates_diff"],
                "similar_valid": bank["valid"],
                **({"similar_density": bank["density"]} if bank.get("density") is not None else {}),
            }
        )
        # ``noise`` is in the true frame. The LLD works in the mirrored frame of ``state``
        # (HWM_LLD_REFLECT), where the same noise is S ε; reconstruct mirrors the result back,
        # so the true-frame step is S μ + (S L S) ε.
        mirrored = lld_reflect(noise, state)
        delta = prediction.mean + torch.matmul(prediction.scale_tril, mirrored.unsqueeze(-1)).squeeze(-1)
        return self.lld.reconstruct_next_state(state, delta)

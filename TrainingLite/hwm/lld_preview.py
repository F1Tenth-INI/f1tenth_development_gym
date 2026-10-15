"""Live check of the LLD against the driving controller's plan (``Settings.HWM_LLD_PREVIEW``).

The controller that drives the car (the actor, Pure Pursuit, or on the test track its scripted
open-loop command) is rolled out closed loop for ``HWM_DISPLAY_PATH_STEPS``
through the simulator's own dynamics, with the simulator's control delay: that is its
plan, states and actions. The LLD replays the plan's actions open loop from the same
start, with ``HWM_LLD_PREVIEW_SAMPLES`` sampled rollouts. Rendered as scene layers: the
plan as black dots, the LLD's mean path as orange dots and its 2σ position ellipses as a
translucent orange band. A blue path is what the residual net alone would predict (no memory).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
import torch

from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax_from_settings
from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.observation import (
    build_actor_observation,
    build_LLD_observation_inference,
    lld_super_state,
)
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.Settings import Settings
from utilities.state_utilities import POSE_THETA_COS_IDX, POSE_THETA_IDX, POSE_THETA_SIN_IDX, POSE_X_IDX, POSE_Y_IDX
from utilities.test_track import command as test_track_command
from utilities.waypoint_utils import WaypointUtils, jit_update_next_waypoints

_ELLIPSE_POINTS = 24


class LLDPreview:
    PLAN_COLOR = (20, 20, 20)
    LLD_COLOR = (230, 90, 0)
    RESIDUAL_COLOR = (0, 90, 230)

    def __init__(self, lld: torch.nn.Module, actor: torch.nn.Module, memory: MemoryManager, wall_geometry: WallGeometry):
        self.lld = lld
        self.actor = actor
        self.memory = memory
        self.wall_geometry = wall_geometry
        self.horizon = int(Settings.HWM_DISPLAY_PATH_STEPS)
        self.samples = int(Settings.HWM_LLD_PREVIEW_SAMPLES)
        # Control steps between computing a command and the car applying it.
        self.delay = int(round(float(Settings.CONTROL_DELAY) / float(Settings.TIMESTEP_CONTROL)))
        self.substeps = int(round(float(Settings.TIMESTEP_CONTROL) / float(Settings.TIMESTEP_SIM)))
        self.car_params = jnp.asarray(VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_np_array())
        self.denorm = np.asarray(Settings.HWM_ACTION_DENORM, dtype=np.float32)
        self.pp = PurePursuitPlanner()
        self._drawn: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
        self._offset = 0

    def past_rows(self, episode_id: int, local_rows: Optional[list] = None) -> torch.Tensor:
        """The rows ``compute`` needs before the current, not yet stored, state of ``episode_id``.

        ``local_rows`` (the drive's own ``[state, action]`` rows, oldest first) replace the
        memory's when the drive is not stored there. Missing rows are zeros.
        """
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        length = max(width - 1, self.delay)
        if local_rows is None:
            past, _ = self.memory.recent(episode_id, length)
            return past
        past = torch.zeros((length, int(Settings.HWM_STATE_DIM) + 2), dtype=torch.float32, device=self.memory.device)
        rows = local_rows[-length:]
        if rows:
            past[length - len(rows) :] = torch.as_tensor(np.stack(rows), device=past.device)
        return past

    def compute(
        self,
        state: np.ndarray,
        past: torch.Tensor,
        controller_observation: dict[str, Any],
        waypoint_utils: WaypointUtils,
        nearest_index: int,
        correcting_index: int,
        use_actor: bool,
        scripted_time: Optional[float] = None,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plan from ``state`` after ``past`` (``past_rows``) and the LLD's replay of it.

        Returns the plan's states
        ``(H + 1, state_dim)``, the sampled LLD states ``(samples, H + 1, state_dim)`` for the
        band, the noise-free LLD mean states ``(H + 1, state_dim)`` for the orange line, and the
        residual net's own states ``(H + 1, state_dim)`` for the blue line.
        """
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        state_dim = int(Settings.HWM_STATE_DIM)
        history = past[past.shape[0] - (width - 1) :]
        past_actions = past[:, state_dim:].cpu().numpy()
        plan_states, plan_actions = self.plan(
            np.asarray(state, dtype=np.float32),
            past_actions[past_actions.shape[0] - self.delay :],
            history,
            controller_observation,
            waypoint_utils,
            nearest_index,
            correcting_index,
            use_actor,
            scripted_time,
        )
        start = torch.as_tensor(plan_states[0])
        actions = torch.as_tensor(plan_actions)
        lld_states = self.predict(start, actions, history)
        mean_states = self.mean_rollout(start, actions, history)
        residual_states = self.residual_rollout(start, actions, history)
        return plan_states, lld_states, mean_states, residual_states

    def plan(
        self,
        state: np.ndarray,
        past_actions: np.ndarray,
        history: torch.Tensor,
        controller_observation: dict[str, Any],
        waypoint_utils: WaypointUtils,
        nearest_index: int,
        correcting_index: int,
        use_actor: bool,
        scripted_time: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The actor or Pure Pursuit closed loop in the simulator dynamics (or the scripted test-track command).

        ``past_actions`` ``(delay, 2)`` are the commands still in the delay line (zeros
        before the episode start, like the simulator's buffer). ``history`` ``(S-1, row_dim)``
        are the rows before ``state``, for the actor's superstate. Pure Pursuit gets the
        waypoint window ``waypoint_utils`` would compute for every planned state. The actor
        samples like the planner (``HWM_SAMPLE_ACTIONS``). Returns states
        ``(H + 1, state_dim)`` and the computed actions ``(H, 2)`` in network units.
        """
        self.pp.correcting_index = correcting_index
        nearest = 0 if nearest_index is None else int(nearest_index)
        # Copy now. Episode reset clears waypoint_utils.waypoints while this rollout runs,
        # and the numba search then crashes on None.
        waypoints = None
        sectors = None
        ignore_steps = 0
        look_ahead_steps = 0
        decrease_resolution_factor = 1
        if waypoint_utils is not None and getattr(waypoint_utils, "waypoints", None) is not None:
            waypoints = np.array(waypoint_utils.waypoints, dtype=np.float64, copy=True)
            if getattr(waypoint_utils, "sectors", None) is not None:
                sectors = np.array(waypoint_utils.sectors, copy=True)
            ignore_steps = int(waypoint_utils.ignore_steps)
            look_ahead_steps = int(waypoint_utils.look_ahead_steps)
            decrease_resolution_factor = int(waypoint_utils.decrease_resolution_factor)
        history = history.to(self.memory.device)[None]
        pending = [a for a in past_actions]
        states = [state]
        actions = []
        s = state
        dt = float(Settings.TIMESTEP_CONTROL)
        for k in range(self.horizon):
            if scripted_time is not None:
                steer, accel = test_track_command(float(scripted_time) + k * dt)
                a = np.array([steer, accel], dtype=np.float32) / self.denorm
            elif use_actor:
                a = self._actor_action(s, history)
                row = torch.cat((torch.as_tensor(s), torch.as_tensor(a))).to(history)
                history = torch.cat((history, row[None, None]), dim=1)[:, 1:]
            else:
                if k == 0 or waypoints is None:
                    observation = controller_observation
                else:
                    nearest, next_waypoints, _, _ = jit_update_next_waypoints(
                        s.astype(np.float64),
                        waypoints,
                        nearest,
                        ignore_steps,
                        look_ahead_steps,
                        decrease_resolution_factor,
                        sectors,
                        float(Settings.GLOBAL_WAYPOINTS_SEARCH_THRESHOLD) ** 2,
                    )
                    observation = {
                        **controller_observation,
                        "car_state": s,
                        "next_waypoints": np.asarray(next_waypoints, dtype=np.float32),
                    }
                a = np.asarray(self.pp.process_observation(observation), dtype=np.float32) / self.denorm
            actions.append(a)
            pending.append(a)
            applied = pending.pop(0) * self.denorm
            applied[1] = np.clip(applied[1], -Settings.HWM_MAX_TRANSLATIONAL_CONTROL, Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
            s = np.array(
                car_dynamics_pacejka_jax_from_settings(
                    jnp.asarray(s),
                    jnp.asarray(applied),
                    self.car_params,
                    float(Settings.TIMESTEP_CONTROL),
                    intermediate_steps=self.substeps,
                ),
                dtype=np.float32,
            )
            theta = float(np.arctan2(s[POSE_THETA_SIN_IDX], s[POSE_THETA_COS_IDX]))
            s[POSE_THETA_IDX] = theta
            states.append(s)
        return np.stack(states), np.stack(actions)

    @torch.no_grad()
    def _actor_action(self, state: np.ndarray, history: torch.Tensor) -> np.ndarray:
        """The actor's action ``(2,)`` in network units for raw ``state`` after ``history`` ``(1, S-1, row_dim)``."""
        obs = build_actor_observation(torch.as_tensor(state), self.memory, self.wall_geometry, 0, past_states=history)
        dist = self.actor(obs)
        action = dist.sample() if Settings.HWM_SAMPLE_ACTIONS else self.actor.deterministic_action(dist)
        return action[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict(self, state: torch.Tensor, actions: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """``samples`` LLD rollouts of ``actions`` ``(H, 2)``: states ``(samples, H + 1, state_dim)``."""
        device = self.memory.device
        s = state.to(device).expand(self.samples, -1)
        history = history.to(device).expand(self.samples, -1, -1)
        states = [s]
        for a in actions.to(device):
            a = a.expand(self.samples, -1)
            obs = build_LLD_observation_inference(s, a, self.memory, 0, past_states=history)
            nxt = self.lld.reconstruct_next_state(s, self.lld(obs).sample())
            history = torch.cat((history, torch.cat((s, a), dim=-1)[:, None]), dim=1)[:, 1:]
            s = nxt
            states.append(s)
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def mean_rollout(self, state: torch.Tensor, actions: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """The LLD's mean trajectory of ``actions`` ``(H, 2)``: states ``(H + 1, state_dim)``.

        No noise is drawn, so the line does not move between redraws of the same plan.
        """
        device = self.memory.device
        s = state.to(device).reshape(1, -1)
        history = history.to(device).reshape(1, history.shape[-2], history.shape[-1])
        states = [s[0]]
        for a in actions.to(device):
            a = a.reshape(1, -1)
            obs = build_LLD_observation_inference(s, a, self.memory, 0, past_states=history)
            nxt = self.lld.reconstruct_next_state(s, self.lld(obs).mean)
            history = torch.cat((history, torch.cat((s, a), dim=-1)[:, None]), dim=1)[:, 1:]
            s = nxt
            states.append(s[0])
        return torch.stack(states)

    @torch.no_grad()
    def residual_rollout(
        self, state: torch.Tensor, actions: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        """The residual net alone as the LLD (no memory) replaying ``actions``: states ``(H + 1, state_dim)``."""
        device = self.memory.device
        s = state.to(device).reshape(1, -1)
        history = history.to(device).reshape(1, history.shape[-2], history.shape[-1])
        states = [s[0]]
        for a in actions.to(device):
            a = a.reshape(1, -1)
            query, _ = lld_super_state(s, a, self.memory, 0, past_states=history)
            delta = self.lld.residual_mean(query)
            nxt = self.lld.reconstruct_next_state(s, delta)
            history = torch.cat((history, torch.cat((s, a), dim=-1)[:, None]), dim=1)[:, 1:]
            s = nxt
            states.append(s[0])
        return torch.stack(states)

    def render(
        self,
        render_utils: Any,
        plan_states: np.ndarray,
        lld_states: torch.Tensor,
        age: int = 0,
        mean_states: Optional[torch.Tensor] = None,
        residual_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Draw a ``compute`` result whose start state is ``age`` control steps old."""
        xy = lld_states[..., [POSE_X_IDX, POSE_Y_IDX]].double().cpu()
        # Monte Carlo estimate of E[position]: the average of the sampled rollouts that also
        # make the band. The noise-free ``mean_states`` rollout is no longer drawn.
        sample_mean = xy.mean(dim=0)
        mean = sample_mean
        centered = xy - sample_mean
        cov = torch.einsum("kti,ktj->tij", centered, centered) / max(1, xy.shape[0] - 1)
        cov = cov + 1e-8 * torch.eye(2, dtype=cov.dtype)
        w, v = torch.linalg.eigh(cov)
        scale = v * w.clamp_min(0.0).sqrt()[:, None, :]
        angle = torch.linspace(0.0, 2.0 * np.pi, _ELLIPSE_POINTS, dtype=cov.dtype)
        circle = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        # Same orientation for every ellipse (det of the square root made positive), so
        # one nonzero-rule fill shows their union at a single opacity.
        scale[:, :, 0] *= torch.where(torch.linalg.det(scale) < 0, -1.0, 1.0).to(scale.dtype)[:, None]
        ellipses = mean[1:, None, :] + 2.0 * torch.einsum("pj,tij->tpi", circle, scale[1:])

        plan_xy = plan_states[:, [POSE_X_IDX, POSE_Y_IDX]]
        residual_xy = None if residual_states is None else residual_states[:, [POSE_X_IDX, POSE_Y_IDX]].double().cpu().numpy()
        self._drawn = (plan_xy, mean.numpy(), ellipses.numpy(), residual_xy)
        self._offset = min(max(int(age), 0), plan_xy.shape[0] - 1)
        self._draw(render_utils)
        render_utils.set_label_dict({"LLD end error [m]": float(np.linalg.norm(mean[-1].numpy() - plan_xy[-1]))})

    def advance(self, render_utils: Optional[Any]) -> None:
        """One control step since the last ``render``: hide the points the car has passed."""
        if self._drawn is None or render_utils is None:
            return
        self._offset = min(self._offset + 1, self._drawn[0].shape[0] - 1)
        self._draw(render_utils)

    def _draw(self, render_utils: Any) -> None:
        plan_xy, mean, ellipses, residual_xy = self._drawn
        k = self._offset
        scene = render_utils.scene
        # The LLD travels less far than the car, so the same step index sits behind it.
        # Shift the orange path and its band so the point for this step lies on the car.
        here = min(k, plan_xy.shape[0] - 1, mean.shape[0] - 1)
        # Always anchored to the black path's current point, also in LLD-sim.
        shift = plan_xy[here] - mean[here]
        # The web renderer downsamples trajectory layers to these caps.
        caps = {"max_trajectories": int(ellipses.shape[0]), "max_points_per_trajectory": max(_ELLIPSE_POINTS, plan_xy.shape[0])}
        scene.set_trajectories("hwm.lld_band", ellipses[max(k - 1, 0):] + shift, color=self.LLD_COLOR, fill=True, alpha=0.25, **caps)
        scene.set_trajectories("hwm.pp_plan", plan_xy[None, k:], color=self.PLAN_COLOR, width=3.0, **caps)
        scene.set_trajectories("hwm.lld_mean", mean[None, k:] + shift, color=self.LLD_COLOR, width=3.0, **caps)
        if residual_xy is not None:
            there = min(k, plan_xy.shape[0] - 1, residual_xy.shape[0] - 1)
            scene.set_trajectories(
                "hwm.lld_residual", residual_xy[None, k:] + (plan_xy[there] - residual_xy[there]), color=self.RESIDUAL_COLOR, width=3.0, **caps
            )


class PreviewResult(NamedTuple):
    generation: int
    """Episode generation of the preview's start state."""
    control_index: int
    """Control step of the preview's start state."""
    value: Any
    """What the computation returned; ``(plan_states, lld_states)`` for ``LLDPreview.compute``."""
    source: str
    """Which controller was previewed, for the label."""


class BackgroundPreview:
    """Runs ``compute`` (e.g. ``LLDPreview.compute``) on its own thread so the control loop never waits for it.

    The control loop submits a start state whenever the thread is idle and draws the
    newest finished result, aged by the control steps that passed while it was computed.
    The computation shares the planner's models; a weight update during it can only
    spoil that one drawing.
    """

    def __init__(self, compute: Callable[..., Any], name: str = "hwm-preview"):
        self.compute = compute
        self._condition = threading.Condition()
        self._job: Optional[tuple[int, int, str, dict[str, Any]]] = None
        self._busy = False
        self._result: Optional[PreviewResult] = None
        self._error: Optional[BaseException] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, generation: int, control_index: int, source: str, **compute_kwargs: Any) -> bool:
        """Start ``compute(**compute_kwargs)`` unless one is still running. True if started."""
        with self._condition:
            if self._busy or self._job is not None:
                return False
            self._job = (int(generation), int(control_index), source, compute_kwargs)
            self._condition.notify()
            return True

    def pop_result(self) -> Optional[PreviewResult]:
        """The result finished since the last call, if any."""
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"{self._thread.name} failed") from self._error
            result, self._result = self._result, None
            return result

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._job is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                (generation, control_index, source, kwargs), self._job = self._job, None
                self._busy = True
            try:
                value = self.compute(**kwargs)
            except BaseException as error:
                print(f"[{self._thread.name}] {type(error).__name__}: {error}")
                with self._condition:
                    self._busy = False
                continue
            with self._condition:
                self._result = PreviewResult(generation, control_index, value, source)
                self._busy = False

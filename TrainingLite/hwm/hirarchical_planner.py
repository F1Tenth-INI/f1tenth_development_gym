"""Hierarchical world-model planner (client side).

Mirrors the SAC actor lifecycle in ``TrainingLite/rl_racing/sac_agent_planner.py``:

* ``Settings.HWM_INFERENCE_MODEL_NAME`` is None  -> TRAINING: connect to the learner,
  stream ``[state, action]`` rows (10-d car state + the raw model action, before
  clipping or denormalization, nothing else from the observation dict), apply broadcast
  weights, Pure Pursuit until the first weights arrive.
* ``Settings.HWM_INFERENCE_MODEL_NAME`` is a name -> INFERENCE: load
  ``TrainingLite/hwm/models/<name>/`` from disk, no TCP.

Insertion points (the only methods meant to be edited):

    build_observation(state)          -> torch.Tensor      (delegates to shared/observation.py)
    select_action(obs, state)         -> np.ndarray [-1,1]^2

Everything else is transport, bookkeeping and fallback plumbing.
"""

from __future__ import annotations

import collections
import copy
import os
import signal
import sys
import threading
from typing import Any, Dict, Optional

import numpy as np
import torch

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
from TrainingLite.hwm.comm.tcp_client import HWMTCPClient
from TrainingLite.hwm.lld_preview import BackgroundPreview, LLDPreview
from TrainingLite.hwm.paths import resolve_model_dir
from TrainingLite.hwm.models.bundle import ModelBundle
from TrainingLite.hwm.models.networks import build_models
from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.observation import (
    _normalize_rows,
    _normalize_states,
    build_actor_observation,
    build_LLD_observation_inference,
    lld_rows_for_super,
)
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from utilities.test_track import command as test_track_command
from utilities.test_track import enabled as test_track_enabled
from utilities.test_track import store_memory as test_track_store_memory

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

WP_RACELINE_IDX = 0
WP_X_IDX = 1
WP_Y_IDX = 2
WP_HEADING_IDX = 3
WP_CURVATURE_IDX = 4
WP_TARGET_SPEED_IDX = 5
WP_TARGET_ACCEL_IDX = 6
WP_GLOBAL_IDX = 7
WP_DISTANCE_RIGH_WALL_IDX = 8
WP_DISTANCE_LEFT_WALL_IDX = 9
WP_TARGET_DRIFT_IDX = 10  # m/s. Target speed along the car y axis. HWM only; the shared raceline stays 10 columns.
WP_TARGET_STEER_IDX = 11  # rad. Steering angle of the car at the waypoint. HWM only.
WP_TARGET_YAW_RATE_IDX = 12  # rad/s. Yaw rate of the car at the waypoint. HWM only.
WP_TARGET_SLIP_IDX = 13  # rad. Slip angle of the car at the waypoint. HWM only.
WP_HWM_COLUMNS = (WP_TARGET_DRIFT_IDX, WP_TARGET_STEER_IDX, WP_TARGET_YAW_RATE_IDX, WP_TARGET_SLIP_IDX)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:  # keep the sim process single-threaded for torch; may already be fixed by earlier imports
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class _JaxRollout(torch.autograd.Function):
    """Torch op around a jitted JAX rollout, with its gradient from the JAX vector-Jacobian product."""

    @staticmethod
    def forward(ctx, applied, s0, forward_fn, backward_fn):
        import jax.numpy as jnp
        applied_np = applied.detach().cpu().numpy()
        ctx.s0, ctx.applied, ctx.backward_fn = s0, applied_np, backward_fn
        return torch.as_tensor(np.array(forward_fn(jnp.asarray(s0), jnp.asarray(applied_np)), dtype=np.float32, copy=True))

    @staticmethod
    def backward(ctx, grad):
        import jax.numpy as jnp
        g = np.array(ctx.backward_fn(jnp.asarray(ctx.s0), jnp.asarray(ctx.applied), jnp.asarray(grad.detach().cpu().numpy())), dtype=np.float32, copy=True)
        return torch.as_tensor(g), None, None, None


class HirarchicalPlanner(template_planner):
    BOOTSTRAP_ROWS = 2

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_info(self, message: str) -> None:
        print(message)

    def _log_debug(self, message: str) -> None:
        if Settings.HWM_AGENT_DEBUG:
            print(message)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._log_info("Initializing HirarchicalPlanner")

        self.inference_model_name: Optional[str] = Settings.HWM_INFERENCE_MODEL_NAME
        self.training_mode = self.inference_model_name is None

        # Shared components (identical construction on the learner, both from Settings).
        self.wall_geometry = WallGeometry()
        self.models: ModelBundle = build_models().to("cpu").eval()
        self.memory = MemoryManager(device="cpu")
        lld = self.models["LLD"]
        self.memory.attach_projector(lld.attention_norm_projector)
        # Bumped at every episode start so preview results from older episodes are ignored.
        self._generation = 0
        # The background preview rolls out its own copies of the actor and the LLD. The control thread
        # also runs the LLD every step, and a shared module produces garbage paths.
        self._display_lock = threading.Lock()
        self._display_actor = copy.deepcopy(self.models["actor"]).eval()
        self._display_lld = copy.deepcopy(self.models["LLD"]).eval()

        self.client: Optional[HWMTCPClient] = None
        self.latest_training_info: Optional[Dict[str, Any]] = None
        self._received_weights = not self.training_mode
        self._warned_no_weights = False
        self._server_handshake: Optional[dict] = None

        if self.training_mode:
            self._log_info("[HirarchicalPlanner] Mode: TRAINING (weights from learner over TCP)")
            host = str(getattr(Settings, "LEARNER_TCP_HOST", "127.0.0.1"))
            port = int(getattr(Settings, "LEARNER_TCP_PORT", 5555))
            self.client = HWMTCPClient(host=host, port=port)
            self.client.start()
            self._log_info(f"[HirarchicalPlanner] TCP client -> {host}:{port}")
        else:
            self._log_info(f"[HirarchicalPlanner] Mode: INFERENCE (model: {self.inference_model_name})")
            self._init_inference_models()
        if Settings.HWM_SIM_SPEED is not None:
            speed = float(Settings.HWM_SIM_SPEED)
            if speed <= 0:
                raise ValueError(f"HWM_SIM_SPEED must be positive or None, got {speed}")
            # The sim loop sleeps each control step up to 1 / MAX_SIM_FREQUENCY.
            Settings.MAX_SIM_FREQUENCY = speed / float(Settings.TIMESTEP_CONTROL)
            self._log_info(
                f"[HirarchicalPlanner] Simulation capped at {speed:g}x real time ({Settings.MAX_SIM_FREQUENCY:g} Hz)"
            )

        # Control plumbing (same as SAC actor).
        self.action_denormalization_array = np.asarray(Settings.HWM_ACTION_DENORM, dtype=np.float32)
        self.lowpass_alpha = 1.0
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        self.terminate_server_after_simulation = True
        self.autonomous_driving = not Settings.HWM_PP_DRIVES

        self.fallback_planner = PurePursuitPlanner()
        self._controller_observation: Optional[Dict[str, Any]] = None
        self.lld_preview: Optional[LLDPreview] = None
        self.preview_worker: Optional[BackgroundPreview] = None
        self._preview_submitted = -int(Settings.HWM_LLD_PREVIEW_EVERY)
        if Settings.HWM_LLD_PREVIEW:
            self.lld_preview = LLDPreview(self._display_lld, self._display_actor, self.memory, self.wall_geometry)
            self.preview_worker = BackgroundPreview(self._preview_job)

        # Episode / streaming bookkeeping.
        self.control_index = 0
        self.total_timesteps = 0
        self.episode_timesteps = 0
        self._episode_id = 0
        # Test track without memory storage: the drive's own [state, action] rows, which the
        # preview needs for its history and the control-delay line.
        self._test_rows: list[np.ndarray] = []
        self._stream_send_idx = 0
        self._stream_batch_size = int(Settings.HWM_STREAM_BATCH_SIZE)
        self._bootstrap_sent = False
        
        self.action_buffer = torch.zeros(Settings.MPC_TRAJECTORY_LENGTH, 2)
        self.former_nearest_waypoint = None
        # Turns on when the car leaves the raceline or the reward EMA drops. Updates stay
        # refused until the car has been close again for MPC_WAYPOINT_UPDATE_RESUME_DELAY.
        self._waypoint_updates_held = False
        self._waypoint_update_close_since = None
        self._waypoint_off_raceline = False
        self._reward_drop_warned = False
        self.recover_mode = False
        self.off_track = False
        self._reset_stall()
        self._wp_off_track_count = None
        self._wp_off_track_marked = None
        self._completed_laps = 0
        self._wp_speed_cap_applied = False
        self._last_opt_wp = None
        self._wp_recording = None
        self._reward_EMA = 1
        self.reset()

    def reset(self):
        self.control_index = 0
        self.episode_timesteps = 0
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        # Episode rows are kept until on_step_end() sees done; see _reset_episode_state().
        self._generation += 1
        self._preview_submitted = -int(Settings.HWM_LLD_PREVIEW_EVERY)
        self.action_buffer = torch.zeros(Settings.MPC_TRAJECTORY_LENGTH, 2)
        self._last_opt_wp = None
        self._reset_stall()

    def _blend_heading(self, old: torch.Tensor, new: torch.Tensor, gamma: float) -> torch.Tensor:
        """Circular blend. A linear mix of two angles flips when they sit on opposite sides of ±π."""
        return torch.atan2(
            (1.0 - gamma) * torch.sin(old) + gamma * torch.sin(new),
            (1.0 - gamma) * torch.cos(old) + gamma * torch.cos(new),
        )

    def _record_waypoint_action(self, state: np.ndarray, action_seq: torch.Tensor) -> None:
        """Store the plan from the closest state on this pass of a waypoint.

        The first pass writes that sequence directly. Later passes stage it, and the commit
        blends it with the same EMA as the waypoint state. A staged sequence is dropped by
        the same holds as a staged waypoint.
        """
        self._ensure_waypoints()
        if not hasattr(self, "_wp_actions"):
            return
        pos = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        dist = (self._waypoint_xy - pos).pow(2).sum(dim=-1)
        idx = int(dist.argmin())
        if self._wp_recording != idx:
            previous = self._wp_recording
            if previous is not None and not bool(self._wp_action_known[previous]):
                if float(self._wp_action_dist[previous]) < float("inf"):
                    self._wp_action_known[previous] = True
            self._wp_recording = idx
            self._wp_action_dist[idx] = float("inf")
        best = float(dist[idx])
        if best > float(self._wp_action_dist[idx]) or self._waypoint_frozen(idx):
            return
        seq = torch.as_tensor(action_seq, dtype=torch.float32).reshape(-1, 2)
        horizon = int(self._wp_actions.shape[1])
        if int(seq.shape[0]) < horizon:
            seq = torch.cat((seq, seq[-1:].expand(horizon - int(seq.shape[0]), -1)), dim=0)
        seq = seq[:horizon]
        if not bool(self._wp_action_known[idx]):
            if not bool(self._wp_action_snapped[idx]):
                self._wp_hist_actions[idx].append((self._wp_actions[idx].clone(), False))
                self._wp_action_snapped[idx] = True
            self._wp_actions[idx] = seq
            self._wp_action_dist[idx] = best
            return
        if (
            bool(getattr(self, "recover_mode", False))
            or bool(getattr(self, "_waypoint_updates_held", False))
            or self.episode_timesteps < 100
        ):
            return
        self._wp_actions_staged[idx] = seq
        self._wp_action_staged[idx] = True
        self._wp_action_dist[idx] = best

    def _raceline_segment(self, state: np.ndarray) -> tuple[int, float]:
        """``(start index, progress)`` of the raceline segment the car is on."""
        pos = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        xy = self._waypoint_xy
        segment = torch.roll(xy, shifts=-1, dims=0) - xy
        length2 = segment.pow(2).sum(dim=-1).clamp_min(1e-12)
        progress = ((pos - xy) * segment).sum(dim=-1) / length2
        progress = progress.clamp(0.0, 1.0)
        projection = xy + progress[:, None] * segment
        segment_index = int((pos - projection).pow(2).sum(dim=-1).argmin())
        return segment_index, float(progress[segment_index])

    def _interpolated_waypoint_actions(self, state: np.ndarray) -> torch.Tensor | None:
        """Blend the stored plans of the two waypoints on either side of the car.

        If only one of them has a stored plan, that plan is used as is. Returns ``None`` while
        neither has, so the first lap keeps planning from the previous action sequence.
        """
        if not hasattr(self, "_wp_action_known") or not bool(self._wp_action_known.any()):
            return None
        start, alpha = self._raceline_segment(state)
        end = (start + 1) % self._waypoints_n
        has_start = bool(self._wp_action_known[start])
        has_end = bool(self._wp_action_known[end])
        if not (has_start or has_end):
            return None
        if has_start != has_end:
            return self._wp_actions[start if has_start else end]
        return (1.0 - alpha) * self._wp_actions[start] + alpha * self._wp_actions[end]

    def _rollout_seeded_actions(self, state: np.ndarray, nominal: torch.Tensor) -> torch.Tensor:
        """Seed built from every waypoint along the path, not only the two around the car.

        ``nominal`` (the blend of the two neighbours' stored sequences) is rolled out with the
        model. Each rollout state ``k`` looks up the two raceline waypoints around it and takes
        the mix, by position between them, of the first action of their stored sequences as
        action ``k``. A waypoint without a stored sequence is skipped; if neither has one, action
        ``k`` of the running plan (``action_buffer``) is used.
        """
        horizon = int(nominal.shape[0])
        path = self.rollout_pacejka(state, nominal)[0, :horizon]  # (H, state_dim), the state each action starts from
        pos = path[:, [POSE_X_IDX, POSE_Y_IDX]]
        xy = self._waypoint_xy
        segment = torch.roll(xy, shifts=-1, dims=0) - xy
        length2 = segment.pow(2).sum(dim=-1).clamp_min(1e-12)
        progress = (((pos[:, None, :] - xy[None]) * segment[None]).sum(dim=-1) / length2[None]).clamp(0.0, 1.0)
        projection = xy[None] + progress[..., None] * segment[None]
        start = (pos[:, None, :] - projection).pow(2).sum(dim=-1).argmin(dim=1)  # (H,)
        alpha = progress.gather(1, start[:, None])[:, 0]
        end = (start + 1) % self._waypoints_n
        has_start, has_end = self._wp_action_known[start], self._wp_action_known[end]
        first_start, first_end = self._wp_actions[start, 0], self._wp_actions[end, 0]  # (H, 2)
        mixed = (1.0 - alpha)[:, None] * first_start + alpha[:, None] * first_end
        mixed = torch.where((has_start & ~has_end)[:, None], first_start, mixed)
        mixed = torch.where((has_end & ~has_start)[:, None], first_end, mixed)
        seed = torch.where((has_start | has_end)[:, None], mixed, self.action_buffer[:horizon])
        return seed

    def _ensure_waypoints(self) -> None:
        utils = getattr(self, "waypoint_utils", None)
        waypoints = None if utils is None else getattr(utils, "waypoints", None)
        if waypoints is None or getattr(self, "_waypoints_n", 0) == len(waypoints):
            return
        source = np.asarray(waypoints)
        owned = np.zeros((source.shape[0], WP_TARGET_SLIP_IDX + 1), dtype=np.float32)
        width = min(int(source.shape[1]), WP_TARGET_DRIFT_IDX)
        owned[:, :width] = source[:, :width]
        self._raceline_np = source
        self.waypoints = torch.as_tensor(owned)  # (N, 14). Columns 10 to 13 are the HWM drift, steering, yaw rate and slip targets, starting at 0.
        self._waypoint_xy = self.waypoints[:, 1:3]
        self._waypoints_n = int(self.waypoints.shape[0])
        if not self._wp_speed_cap_applied and Settings.HWM_LIMIT_INITIAL_WP_SPEED:
            self.waypoints[:, WP_TARGET_SPEED_IDX] = torch.minimum(self.waypoints[:, WP_TARGET_SPEED_IDX], torch.tensor(Settings.HWM_INITIAL_SPEED_LIMIT))
            self._wp_speed_cap_applied = True
        # Direction of travel to the next stored point, including the segment that closes the loop.
        # Same angle the reward used to compute on the fly: 0 along +x, matching pose_theta.
        nxt = torch.roll(self._waypoint_xy, shifts=-1, dims=0)
        delta = nxt - self._waypoint_xy
        self.waypoints[:, WP_HEADING_IDX] = torch.atan2(delta[:, 1], delta[:, 0])
        # Fixed at load. Later edits may move a point, but only along this cross-track line.
        heading0 = self.waypoints[:, WP_HEADING_IDX]
        self._wp_anchor_xy = self._waypoint_xy.detach().clone()
        self._wp_tangent = torch.stack((torch.cos(heading0), torch.sin(heading0)), dim=1)
        self._wp_ortho = torch.stack((-torch.sin(heading0), torch.cos(heading0)), dim=1)
        step = (self._waypoint_xy[1:] - self._waypoint_xy[:-1]).norm(dim=-1)
        positive = step[step > 1e-4]
        self._waypoint_spacing = float(positive.median()) if positive.numel() else 0.1

        self.waypoints_update_staged = self.waypoints.clone()
        self._wp_staged = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        horizon = int(Settings.MPC_TRAJECTORY_LENGTH)
        self._wp_actions = torch.zeros((self._waypoints_n, horizon, 2), dtype=torch.float32)
        self._wp_actions_staged = torch.zeros_like(self._wp_actions)
        self._wp_action_staged = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        self._wp_action_known = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        self._wp_action_dist = torch.full((self._waypoints_n,), float("inf"))
        self._wp_initial = self.waypoints.clone()  # as loaded, for a full reset of a waypoint
        # Every committed value, oldest first: each commit pushes what it replaced. A revert pops one.
        self._wp_gamma_divisions = torch.zeros((self._waypoints_n,), dtype=torch.long)  # times each waypoint was reset
        self._wp_hist_state = [[] for _ in range(self._waypoints_n)]  # rows of the waypoint table
        self._wp_hist_actions = [[] for _ in range(self._waypoints_n)]  # (action sequence, was it known)
        self._wp_action_snapped = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        self._wp_frozen_until = torch.zeros((self._waypoints_n,), dtype=torch.long)
        self._wp_off_track_count = torch.zeros((self._waypoints_n,), dtype=torch.float32)
        self._wp_path_states = None
        self._wp_vehicle_state_set = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        self._wp_off_track_marked = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        self._car_waypoint = None
        self._wp_recording = None
        self.last_wp_distance = np.inf
        self._write_raceline()

    def _write_raceline(self) -> None:
        """Copy the shared raceline fields back. The drift column stays on the HWM copy."""
        source = getattr(self, "_raceline_np", None)
        if source is None or not hasattr(self, "waypoints"):
            return
        width = min(int(source.shape[1]), WP_TARGET_DRIFT_IDX)
        source[:, :width] = self.waypoints[:, :width].detach().cpu().numpy()

    def _hwm_policy_drives(self) -> bool:
        """True only while this planner's own policy is commanding the car."""
        if not self.autonomous_driving:
            return False
        if not self._received_weights and Settings.HWM_PP_UNTIL_WEIGHTS:
            return False
        return True

    def _refresh_display_models(self) -> None:
        """Copy the live actor and LLD into the modules the background paths roll out."""
        with self._display_lock:
            self._display_actor.load_state_dict(self.models["actor"].state_dict())
            self._display_lld.load_state_dict(self.models["LLD"].state_dict())

    def _preview_job(self, **kwargs: Any) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._refresh_display_models()
        return self.lld_preview.compute(**kwargs)
    
    def nearest_waypoint_index(self, state: torch.Tensor) -> torch.Tensor:
        """First raceline index ahead of each state along the track. Returns ``(B,)`` long.

        ``state`` is ``(state_dim,)`` or ``(B, state_dim)``. The position is projected
        onto the raceline segments, including the segment that closes the loop. The
        returned index is the waypoint at the end of that segment, so a point the car
        has already passed is not used.
        """
        self._ensure_waypoints()
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        pos = state[:, [POSE_X_IDX, POSE_Y_IDX]]
        xy = self._waypoint_xy
        p0 = xy
        p1 = torch.roll(xy, shifts=-1, dims=0)
        segment = p1 - p0
        length2 = segment.pow(2).sum(dim=-1).clamp_min(1e-12)
        offset = pos.unsqueeze(1) - p0.unsqueeze(0)
        progress = (offset * segment.unsqueeze(0)).sum(dim=-1) / length2
        progress = progress.clamp(0.0, 1.0)
        projection = p0.unsqueeze(0) + progress.unsqueeze(-1) * segment.unsqueeze(0)
        distance2 = (pos.unsqueeze(1) - projection).pow(2).sum(dim=-1)
        segment_index = distance2.argmin(dim=-1)
        along = progress.gather(1, segment_index.unsqueeze(1)).squeeze(1)
        # End of the segment is still ahead. A projection clamped to that end means
        # the car is already on or past it, so the following waypoint is the one in front.
        ahead = torch.where(
            along <= 0,
            segment_index,
            torch.where(along >= 1, segment_index + 2, segment_index + 1),
        )
        # Same skip as jit_update_next_waypoints: the first waypoint it returns is IGNORE_STEPS ahead.
        ahead = ahead + int(Settings.IGNORE_STEPS)
        return ahead.remainder(xy.shape[0])

    def rollout_waypoint_states(self, states: torch.Tensor, closest_index: int | torch.Tensor) -> torch.Tensor:
        """For each rollout, which of its states is closest to each upcoming waypoint.

        ``states`` is ``(B, T, state_dim)``. ``closest_index`` is the first raceline index
        ahead of the start, one integer or ``(B,)``. Waypoint ``j`` of a trajectory is
        ``closest_index + j`` along the raceline, including that first waypoint ahead.

        How many waypoints: the trajectory's average speed over ground, times the rollout
        duration ``(T - 1) * TIMESTEP_CONTROL``, divided by the stored waypoint spacing, plus one.
        Waypoints past that, and any waypoint left after every state has been used once, are NaN.

        Returns ``(B, J)`` float. Entry ``[k, j]`` is the state index in rollout ``k`` closest to
        that waypoint. Each state is used for at most one waypoint, waypoints in order, so the
        earlier waypoint keeps the state when two of them want it.
        """
        self._ensure_waypoints()
        if not hasattr(self, "_waypoint_xy"):
            raise RuntimeError("waypoints are not loaded")
        states = torch.as_tensor(states, dtype=torch.float32)
        if states.ndim != 3:
            raise ValueError(f"states must be (B, T, state_dim), got {tuple(states.shape)}")
        batch, steps, _ = (int(states.shape[0]), int(states.shape[1]), int(states.shape[2]))
        closest = torch.as_tensor(closest_index, dtype=torch.long)
        if closest.numel() == 1:
            closest = closest.reshape(()).expand(batch)
        speed = torch.hypot(states[..., LINEAR_VEL_X_IDX], states[..., LINEAR_VEL_Y_IDX]).mean(dim=1)
        duration = max(steps - 1, 0) * float(Settings.TIMESTEP_CONTROL)
        spacing = max(float(getattr(self, "_waypoint_spacing", 0.1)), 1e-3)
        counts = torch.floor(speed * duration / spacing).to(torch.long) + 1
        counts = counts.clamp(min=1, max=self._waypoints_n)
        look = int(counts.max()) if batch else 0
        out = states.new_full((batch, look), float("nan"))
        if look == 0 or steps == 0:
            return out
        offset = torch.arange(look)
        wp = self._waypoint_xy[(closest[:, None] + offset) % self._waypoints_n]
        pos = states[:, :, [POSE_X_IDX, POSE_Y_IDX]]
        dist = (pos.unsqueeze(2) - wp.unsqueeze(1)).pow(2).sum(dim=-1)
        in_range = offset.unsqueeze(0) < counts.unsqueeze(1)
        dist = dist.masked_fill(~in_range.unsqueeze(1), float("inf"))
        taken = torch.zeros(batch, steps, dtype=torch.bool)
        for j in range(look):
            column = dist[:, :, j].masked_fill(taken, float("inf"))
            best = column.argmin(dim=1)
            reachable = in_range[:, j] & torch.isfinite(column.gather(1, best.unsqueeze(1)).squeeze(1))
            out[reachable, j] = best[reachable].to(out.dtype)
            mark = torch.zeros_like(taken)
            mark.scatter_(1, best.unsqueeze(1), reachable.unsqueeze(1))
            taken |= mark
        return out

    def MPC_reward(
        self,
        states: torch.tensor,
        nearest_waypoint_indices: torch.tensor,
        nearest_waypoint: int,
        ignore_waypoint: int | None = None,
        waypoint_off_track_counts: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Driving and waypoint rewards, each split into the normal part and the crash part.

        The caller adds the two parts. The crash part is left out of the reward EMA.
        """
        batch = int(states.shape[0])
        num_valid_waypoints = nearest_waypoint_indices.shape[1] - nearest_waypoint_indices.isnan().sum(dim=1)
        drive = torch.zeros(batch)
        optimize = torch.zeros(batch)
        start = int(torch.as_tensor(nearest_waypoint).reshape(-1)[0])
        if waypoint_off_track_counts is None:
            waypoint_off_track_counts = torch.zeros(self._waypoints_n)
        for i in range(num_valid_waypoints.shape[0]):
            count = int(num_valid_waypoints[i])
            if count == 0:
                continue
            indices = nearest_waypoint_indices[i, :count].long()
            wp_index = (torch.arange(count) + start) % self._waypoints_n
            if ignore_waypoint is not None:
                keep = wp_index != (int(ignore_waypoint) % self._waypoints_n)
                if not bool(keep.any()):
                    continue
                indices = indices[keep]
                wp_index = wp_index[keep]
                count = int(indices.shape[0])
            local_waypoints = self.waypoints[wp_index]
            # How many times the car was already off track at each of these waypoints, same order as ``local_waypoints``.
            local_off_track_counts = waypoint_off_track_counts[wp_index]
            update_mask = torch.zeros_like(states[i, indices, 0], dtype=torch.bool)
            update_mask[:] = True
            update_mask[Settings.MPC_WAYPOINT_AHEAD : Settings.MPC_WAYPOINT_AHEAD + Settings.MPC_WAYPOINT_AHEAD_SPAN] = False
            position = states[i, indices][:, [POSE_X_IDX, POSE_Y_IDX]] - local_waypoints[:, [WP_X_IDX, WP_Y_IDX]]
            distance_reward = torch.linalg.norm(position, dim=1) #torch.linalg.norm(position * (1 + local_off_track_counts)[:, None], dim=1)
            #distance_reward[-3:] = distance_reward[-3:]*3
            distance_reward = distance_reward.sum()
            heading_error = states[i, indices, POSE_THETA_IDX] - local_waypoints[:, WP_HEADING_IDX]
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
            heading_reward = heading_error.abs()
            speed = states[i, indices, LINEAR_VEL_X_IDX]
            target_speed = local_waypoints[:, WP_TARGET_SPEED_IDX]
            drift = states[i, indices, LINEAR_VEL_Y_IDX]
            target_drift = local_waypoints[:, WP_TARGET_DRIFT_IDX]
            drift_error = (drift - target_drift).abs().sum()
            shortfall = (target_speed - speed).clamp(min=0)
            overspeed = (speed - target_speed).clamp(min=0)

            fast_driving_reward = indices.float().max()
            legal = indices.float()[~update_mask]  # the states matched to the waypoints in the AHEAD window
            fast_driving_legal_reward = legal.max() if legal.numel() else indices.new_zeros(()).float()
            travel_further_reward = -count**0.5 #1 / (1 + count**0.5)
            if self.off_track:
                break
            drive_reward = (10 * distance_reward.sum()  + 2 * heading_reward.sum() + shortfall.sum() * 1 + overspeed.sum() * 2 + 0.25 * drift_error + 0.5 * fast_driving_reward) / count #+ travel_further_reward
            optimize_reward = (10 * distance_reward + 0 * drift_error ) / count # + 0.5 * fast_driving_reward / count
            optimize_reward += (0.5 * (heading_reward * update_mask).sum() + (shortfall * update_mask).sum() * 2 + (overspeed * update_mask).sum() * 2) / update_mask.sum().clamp(min=1)
            optimize_reward += (shortfall * ~update_mask).sum() * 10 / (~update_mask).sum().clamp(min=1)
            #if not self.episode_timesteps % 24: print(optimize_reward - 0.5 * fast_driving_legal_reward / Settings.MPC_WAYPOINT_AHEAD_SPAN)
            optimize_reward += 0.5 * fast_driving_legal_reward / Settings.MPC_WAYPOINT_AHEAD_SPAN
            #optimize_reward = (10 * distance_reward.sum()  + 2 * heading_reward.sum() + 5 * shortfall.sum() + overspeed.sum() * 0 + 0.25 * drift_error + 2 * fast_driving_reward) / count
            

            min_speed = float(speed.min())
            if min_speed < 0:
                reverse = 100 * abs(min_speed)
                drive_reward = drive_reward + reverse
                optimize_reward = optimize_reward + reverse
            drive[i] = drive_reward
            optimize[i] = optimize_reward
       # print(f"distance_reward: {distance_reward/count}, heading_reward: {heading_reward/count}, shortfall: {shortfall/count}, overspeed: {overspeed/count}, drift_error: {drift_error/count}, fast_driving_legal_reward: {fast_driving_legal_reward/count}")

        if self.off_track:
            drive = self._recovery_cost(states, start)
        if self._stall_phase == 1:
            drive = self._backtrack_cost(states)

        poses = states[:, :, [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
        crashed = self.wall_geometry.in_wall(poses)
        has_crash = crashed.any(dim=1)
        first = crashed.float().argmax(dim=1)
        horizon = crashed.shape[1]
        # Steps until impact. A clean rollout survives the whole horizon. An earlier hit costs more,
        # so a plan that follows the corner and clips late beats one that drives into the wall now.
        survived = torch.where(has_crash, first, torch.full_like(first, horizon))
        late = horizon - survived
        # The optimize reward also counts a car position closer to a wall than the waypoints may be.
        close = crashed | (self.wall_geometry.distance_at(poses[..., :2]).to(crashed.device) < float(Settings.MPC_WAYPOINT_MIN_WALL_DISTANCE))
        first_close = close.float().argmax(dim=1)
        survived_opt = torch.where(close.any(dim=1), first_close, torch.full_like(first_close, horizon))
        late_opt = horizon - survived_opt

        def _split(raw: torch.Tensor, late: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            crash = -(torch.max(raw) * 10 * late)
            return -raw, crash

        drive_reward, drive_crash = _split(drive, late)
        optimize_reward, optimize_crash = _split(optimize, late_opt)
        return (drive_reward, optimize_reward), (drive_crash, optimize_crash)

    # Weights of the recovery cost. Meters, radians and m/s.
    RECOVERY = {
        "window": 40,         # raceline waypoints ahead of the car that a rollout state can be matched to
        "distance": 4.0,      # distance to the nearest raceline waypoint
        "heading": 3.0,       # heading error to the direction that leads back onto the line
        "steer_in": 1.5,      # rad of heading offset toward the line per meter of cross-track error
        "steer_in_max": 0.9,  # rad, largest offset toward the line
        "speed_min": 1.5,     # m/s, forward speed wanted while the car points within "gate" rad of the wanted heading
        "gate": 4.0,
        "noise_scale": 3.0,   # sampling noise around the previous plan is this many times larger while off the line
        "speed_min_w": 1.5,
        "speed_max": 3.0,     # m/s, no point going faster than this while recovering
        "speed_max_w": 2.0,
        "reverse": 1.0,
        "slip": 1.0,          # sideways speed above 1 m/s
        "yaw_rate": 0.3,      # yaw rate above 3 rad/s
    }

    def _recovery_cost(self, states: torch.Tensor, start: int) -> torch.Tensor:
        """``(B,)`` cost, lower is better, for rollouts ``(B, T, state_dim)`` of a car that is off the raceline.

        Each state is matched to its nearest raceline waypoint. The wanted heading is the line's
        heading turned toward the line in proportion to the cross-track error, so a car that is
        far off is asked to head back to the line and a car on it is asked to follow it. The end of
        the rollout counts twice, since where the car ends up is what the next step starts from.
        """
        cfg = self.RECOVERY
        n = self._waypoints_n
        window = (torch.arange(-3, int(cfg["window"])) + int(start)) % n
        wp = self.waypoints[window]
        pos = states[..., [POSE_X_IDX, POSE_Y_IDX]]
        distance = torch.linalg.norm(pos[:, :, None, :] - wp[None, None, :, [WP_X_IDX, WP_Y_IDX]], dim=-1)
        near_distance, near = distance.min(dim=-1)
        line = wp[near]
        track_heading = line[..., WP_HEADING_IDX]
        left = torch.stack((-torch.sin(track_heading), torch.cos(track_heading)), dim=-1)
        cross_track = ((pos - line[..., [WP_X_IDX, WP_Y_IDX]]) * left).sum(dim=-1)
        wanted = track_heading - (float(cfg["steer_in"]) * cross_track).clamp(-float(cfg["steer_in_max"]), float(cfg["steer_in_max"]))
        heading_error = states[..., POSE_THETA_IDX] - wanted
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error)).abs()
        speed = states[..., LINEAR_VEL_X_IDX]
        pointing_along = (heading_error < float(cfg["gate"])).float()
        slow = (float(cfg["speed_min"]) - speed).clamp(min=0) * pointing_along
        fast = (speed - float(cfg["speed_max"])).clamp(min=0)
        reverse = (-speed).clamp(min=0)
        slip = (states[..., LINEAR_VEL_Y_IDX].abs() - 1.0).clamp(min=0)
        yaw = (states[..., ANGULAR_VEL_Z_IDX].abs() - 3.0).clamp(min=0)

        def over_time(x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=1) + x[:, -1]

        return (
            float(cfg["distance"]) * over_time(near_distance)
            + float(cfg["heading"]) * over_time(heading_error)
            + float(cfg["speed_min_w"]) * slow.mean(dim=1)
            + float(cfg["speed_max_w"]) * fast.mean(dim=1)
            + float(cfg["reverse"]) * reverse.mean(dim=1)
            + float(cfg["slip"]) * slip.mean(dim=1)
            + float(cfg["yaw_rate"]) * yaw.mean(dim=1)
        )

    def build_observation(self, state: np.ndarray) -> dict[str, torch.Tensor]:
            """Actor input for the current raw state. Shared implementation with the learner."""
            return build_actor_observation(
                state=state,
                memory=self.memory,
                wall_geometry=self.wall_geometry,
                episode_id=self._episode_id,
            )

    def _sample_mpc_actions(self, nominal: torch.Tensor | None = None, count: int | None = None) -> torch.Tensor:
        """Correlated action samples around ``nominal``, clipped like ``select_action``."""
        if nominal is None:
            nominal = self.action_buffer
        rho = min(max(float(Settings.MPC_ACTION_CORRELATION), 0.0), 0.999)
        noise_std = torch.tensor([0.08, 1.5]) / torch.as_tensor(self.action_denormalization_array)
        if self.off_track or self._stall_phase == 1:
            noise_std = noise_std * float(self.RECOVERY["noise_scale"])
        eps = torch.randn(int(count or Settings.MPC_NUM_OF_TRAJECTORIES), Settings.MPC_TRAJECTORY_LENGTH, 2)
        actions = nominal + eps * noise_std
        if rho > 0.0:
            fresh = (1.0 - rho * rho) ** 0.5
            actions[:, 0] = nominal[0] + eps[:, 0] * noise_std
            for t in range(1, eps.shape[1]):
                actions[:, t] = rho * actions[:, t - 1] + (1.0 - rho) * nominal[t] + fresh * noise_std * eps[:, t]
        steer_limit = 1.0
        accel_limit = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL) / float(self.action_denormalization_array[1])
        actions[..., 0].clamp_(-steer_limit, steer_limit)
        actions[..., 1].clamp_(-accel_limit, accel_limit)
        return actions

    def _keep_wall_distance(self, xy: torch.Tensor, idx: int) -> torch.Tensor:
        """Slide ``xy`` along waypoint ``idx``'s cross-track line until it is ``MPC_WAYPOINT_MIN_WALL_DISTANCE`` from the walls.

        A point that already keeps the distance is returned as it is. Otherwise it goes to the
        closest spot on the line, to where it was, that keeps it. If no spot on the line does,
        it goes to the one farthest from the walls.
        """
        minimum = float(Settings.MPC_WAYPOINT_MIN_WALL_DISTANCE)
        if minimum <= 0.0:
            return xy
        anchor = self._wp_anchor_xy[idx]
        ortho = self._wp_ortho[idx]
        slide = float((xy - anchor).dot(ortho))
        offsets = torch.cat((torch.tensor([slide]), torch.linspace(-1.5, 1.5, 301)))
        points = anchor[None, :] + offsets[:, None] * ortho[None, :]
        distance = self.wall_geometry.distance_at(points).to(offsets.dtype)
        if float(distance[0]) >= minimum:
            return xy
        ok = distance[1:] >= minimum
        if bool(ok.any()):
            candidates = offsets[1:][ok]
            best = candidates[(candidates - slide).abs().argmin()]
        else:
            best = offsets[1:][distance[1:].argmax()]
        return anchor + best * ortho

    def _state_on_orthogonal_line(self, path: torch.Tensor, idx: int):
        """Blend the two path states nearest the waypoint's fixed cross-track line.

        The line passes through the original waypoint and is perpendicular to its original
        heading. The result is projected onto that line, so the point cannot leave it.
        Returns ``None`` when the path never comes near the line.
        """
        anchor = self._wp_anchor_xy[idx]
        tangent = self._wp_tangent[idx]
        ortho = self._wp_ortho[idx]
        pos = path[:, [POSE_X_IDX, POSE_Y_IDX]]
        distance = (pos - anchor).matmul(tangent).abs()
        count = int(path.shape[0])
        if count < 2:
            return None
        if float(distance.min()) > 0.5:
            return None
        nearest = torch.topk(distance, k=2, largest=False).indices
        d1 = distance[nearest[0]]
        d2 = distance[nearest[1]]
        if float(d1 + d2) < 1e-8:
            w1 = w2 = 0.5
        else:
            w1 = d2 / (d1 + d2)
            w2 = d1 / (d1 + d2)
        first = path[nearest[0]]
        second = path[nearest[1]]
        xy = w1 * first[[POSE_X_IDX, POSE_Y_IDX]] + w2 * second[[POSE_X_IDX, POSE_Y_IDX]]
        slide = (xy - anchor).dot(ortho)
        xy = anchor + slide * ortho
        heading = torch.atan2(
            w1 * torch.sin(first[POSE_THETA_IDX]) + w2 * torch.sin(second[POSE_THETA_IDX]),
            w1 * torch.cos(first[POSE_THETA_IDX]) + w2 * torch.cos(second[POSE_THETA_IDX]),
        )
        speed = (w1 * first[LINEAR_VEL_X_IDX] + w2 * second[LINEAR_VEL_X_IDX]).clamp(min=0.0)
        drift = w1 * first[LINEAR_VEL_Y_IDX] + w2 * second[LINEAR_VEL_Y_IDX]
        steer = w1 * first[STEERING_ANGLE_IDX] + w2 * second[STEERING_ANGLE_IDX]
        yaw_rate = w1 * first[ANGULAR_VEL_Z_IDX] + w2 * second[ANGULAR_VEL_Z_IDX]
        slip = w1 * first[SLIP_ANGLE_IDX] + w2 * second[SLIP_ANGLE_IDX]
        return xy, heading, speed, drift, steer, yaw_rate, slip

    def _waypoint_resume_steps(self) -> int:
        return max(1, int(round(
            float(Settings.MPC_WAYPOINT_UPDATE_RESUME_DELAY) / float(Settings.TIMESTEP_CONTROL)
        )))

    def _hold_waypoint_updates(self) -> None:
        """Drop every staged waypoint and refuse new ones until the resume delay has elapsed."""
        if hasattr(self, "_wp_staged"):
            self._wp_staged.zero_()
        if hasattr(self, "_wp_action_staged"):
            self._wp_action_staged.zero_()
        self._revert_nearby_waypoints()
        self._waypoint_updates_held = True
        self._waypoint_update_close_since = None

    def _revert_nearby_waypoints(self) -> None:
        """Put the k waypoints ahead and behind one committed update further back, each time this runs.

        Every waypoint keeps all its earlier states and action sequences, so repeated misses
        walk back through them. A waypoint with nothing left to pop stays as it is.
        """
        if not hasattr(self, "_wp_hist_state") or self._car_waypoint is None:
            return
        if not getattr(self, "_seen_on_track", True):
            return  # spawned off the raceline and not on it yet: recovery runs, but no waypoint is reset
        behind = max(0, int(Settings.MPC_WAYPOINT_REVERT_BEHIND))
        ahead = max(0, int(Settings.MPC_WAYPOINT_REVERT_AHEAD))
        steps = max(1, int(Settings.MPC_WAYPOINT_REVERT_STEPS))
        n = self._waypoints_n
        center = int(self._car_waypoint)
        for offset in range(-behind, ahead + 1):
            idx = (center + offset) % n
            was_reset = False
            for _ in range(steps):
                if not self._wp_hist_state[idx]:
                    break
                self.waypoints[idx] = self._wp_hist_state[idx].pop()
                was_reset = True
            for _ in range(steps):
                if not self._wp_hist_actions[idx]:
                    break
                actions, known = self._wp_hist_actions[idx].pop()
                self._wp_actions[idx] = actions
                self._wp_action_known[idx] = bool(known)
                self._wp_action_snapped[idx] = False
                was_reset = True
            if was_reset:
                self._wp_gamma_divisions[idx] += 1  # only reset waypoints get a smaller EMA gamma
                self._wp_action_staged[idx] = False
            hold = max(0, int(Settings.MPC_WAYPOINT_REVERT_HOLD))
            self._wp_frozen_until[idx] = int(getattr(self, "_completed_laps", 0)) + hold
        self._write_raceline()
        self._publish_waypoint_display()

    def _waypoint_frozen(self, idx: int) -> bool:
        until = getattr(self, "_wp_frozen_until", None)
        if until is None:
            return False
        return int(getattr(self, "_completed_laps", 0)) < int(until[int(idx)])

    def _note_reward(self, reward_mean: torch.Tensor) -> None:
        """Track the softmax-weighted reward. A reward far under its EMA holds waypoint updates.

        The value is the negation of a cost, so a smaller reward is a worse plan. This step's reward
        is compared with the EMA from before this step. The first steps are skipped: the EMA is
        still leaving its initial value.
        """
        previous = float(self._reward_EMA)
        keep = float(Settings.MPC_REWARD_EMA_GAMMA)
        self._reward_EMA = keep * self._reward_EMA + (1.0 - keep) * reward_mean
        if self.episode_timesteps < 100:
            return
        drop = previous - float(reward_mean)
        if drop > float(Settings.MPC_WAYPOINT_REWARD_DROP) * abs(previous):
            if not getattr(self, "_reward_drop_warned", False):
                self._reward_drop_warned = True
                self._note_off_track_pass(reward_drop=True)
                self._log_info(
                    f"[HirarchicalPlanner] Warning: reward {float(reward_mean):.4f} fell under its EMA "
                    f"{previous:.4f} by more than {float(Settings.MPC_WAYPOINT_REWARD_DROP):.0%}. Staged waypoint updates discarded."
                )
            self._hold_waypoint_updates()
            self._update_recover_mode(bool(getattr(self, "_waypoint_off_raceline", False)))

    def _update_recover_mode(self, off_track: bool) -> None:
        """True while the car is off the raceline or a reward-EMA drop is still in effect."""
        self.recover_mode = bool(
            off_track or getattr(self, "_reward_drop_warned", False) or getattr(self, "_stall_phase", 0) == 1
        )

    def _ema_gamma(self, idx: int) -> float:
        """EMA gamma of waypoint ``idx``: ``MPC_WAYPOINT_UPDTE_EMA`` divided by ``MPC_WAYPOINT_EMA_RECOVERY_DIVISOR`` once per time it was reset."""
        divisor = max(1.0, float(Settings.MPC_WAYPOINT_EMA_RECOVERY_DIVISOR))
        return float(Settings.MPC_WAYPOINT_UPDTE_EMA) / divisor ** int(self._wp_gamma_divisions[int(idx)])

    def _reset_stall(self) -> None:
        # Also the per-episode reset. A car that spawns off the raceline does not reset any waypoint
        # until it has been on the raceline once; recovery itself still runs.
        self._seen_on_track = False
        self._stall_phase = 0  # 0 driving, 1 reversing (on track and not moving), 2 stalled recovery
        self._reverse_left = 0
        self._stall_anchor = None
        self._backtrack_wp = None
        self._stalled = False
        self._pose_history = collections.deque()

    def _update_stall(self, car_xy: torch.Tensor, car_waypoint: int) -> None:
        """Stall detection. Not moving ``MPC_STALL_DISTANCE`` in ``MPC_STALL_TIME`` s starts it.

        On or off the track it first backtracks (phase 1): the plan is scored on getting to the
        waypoint ``MPC_STALL_BACKTRACK_WAYPOINTS`` behind the car, which replaces the normal
        recovery cost for as long as it runs. That ends on arrival there or after
        ``MPC_STALL_REVERSE_STEPS`` steps, and recovery follows (phase 2). Recovery ends once the
        car is ``MPC_STALL_DISTANCE`` away from where it started, and the history is then started over.
        """
        distance = float(Settings.MPC_STALL_DISTANCE)
        if self._stall_phase == 1:
            target = self._waypoint_xy[self._backtrack_wp]
            arrived = float((car_xy - target).norm()) < 0.5
            if self._reverse_left <= 0 or arrived:
                self._stall_phase = 2
                self._stall_anchor = car_xy.clone()
        elif self._stall_phase == 2:
            if float((car_xy - self._stall_anchor).norm()) >= distance:
                self._stall_phase = 0
                self._pose_history.clear()
        else:
            steps = max(1, int(round(float(Settings.MPC_STALL_TIME) / float(Settings.TIMESTEP_CONTROL))))
            history = self._pose_history
            history.append(car_xy.clone())
            while len(history) > steps + 1:
                history.popleft()
            if len(history) > steps and float((history[-1] - history[0]).norm()) < distance:
                history.clear()
                if int(Settings.MPC_STALL_REVERSE_STEPS) > 0:
                    self._stall_phase = 1
                    self._reverse_left = int(Settings.MPC_STALL_REVERSE_STEPS)
                    self._backtrack_wp = (int(car_waypoint) - int(Settings.MPC_STALL_BACKTRACK_WAYPOINTS)) % self._waypoints_n
                    self._log_info(
                        "[HirarchicalPlanner] Warning: car barely moved. "
                        f"Driving to the waypoint {int(Settings.MPC_STALL_BACKTRACK_WAYPOINTS)} behind it."
                    )
                else:
                    self._stall_phase = 2
                    self._stall_anchor = car_xy.clone()
        self._stalled = self._stall_phase == 2

    def _backtrack_cost(self, states: torch.Tensor) -> torch.Tensor:
        """``(B,)`` cost, lower is better: how far each rollout is from the backtrack waypoint.

        Forward or reverse, whichever gets there. The end of the rollout counts twice. Slip, yaw
        rate and overspeed are penalized like in recovery, so it does not spin to get there.
        """
        cfg = self.RECOVERY
        target = self._waypoint_xy[self._backtrack_wp]
        pos = states[..., [POSE_X_IDX, POSE_Y_IDX]]
        distance = torch.linalg.norm(pos - target, dim=-1)
        slip = (states[..., LINEAR_VEL_Y_IDX].abs() - 1.0).clamp(min=0)
        yaw = (states[..., ANGULAR_VEL_Z_IDX].abs() - 3.0).clamp(min=0)
        fast = (states[..., LINEAR_VEL_X_IDX].abs() - float(cfg["speed_max"])).clamp(min=0)
        return (
            float(cfg["distance"]) * (distance.mean(dim=1) + distance[:, -1])
            + float(cfg["slip"]) * slip.mean(dim=1)
            + float(cfg["yaw_rate"]) * yaw.mean(dim=1)
            + float(cfg["speed_max_w"]) * fast.mean(dim=1)
        )

    def _raceline_distance(self, state: np.ndarray) -> float:
        """Distance from the car to the raceline segment between the waypoint behind and the one ahead."""
        start, progress = self._raceline_segment(state)
        end = (start + 1) % self._waypoints_n
        pos = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        projection = self._waypoint_xy[start] + progress * (self._waypoint_xy[end] - self._waypoint_xy[start])
        return float((pos - projection).norm())

    def _note_off_track_pass(self, reward_drop: bool = False) -> None:
        """Count the waypoint the car is at, once per visit, when the car is off track there or its reward just dropped."""
        counts = getattr(self, "_wp_off_track_count", None)
        if counts is None or self._car_waypoint is None:
            return
        if not getattr(self, "_seen_on_track", True):
            return  # still off the raceline from the spawn: not an event, nothing to count
        n = self._waypoints_n
        car = int(self._car_waypoint)
        gap = (torch.arange(n) - car).remainder(n)
        gap = torch.minimum(gap, n - gap)
        self._wp_off_track_marked &= gap <= 10
        if (self.off_track or reward_drop) and not bool(self._wp_off_track_marked[car]):
            self._wp_off_track_marked[car] = True
            counts[car] += 1.0
            self._publish_waypoint_display()

    def _refresh_recovery(self, state: np.ndarray) -> None:
        """Set recovery from the current pose, before this step chooses a plan."""
        self._ensure_waypoints()
        if not hasattr(self, "_waypoint_xy"):
            return
        car_xy = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        line_distance = self._raceline_distance(state)
        on_track = line_distance <= float(Settings.MPC_WAYPOINT_UPDATE_DISTANCE)
        if on_track:
            self._seen_on_track = True
        self._car_waypoint = int((self._waypoint_xy - car_xy).norm(dim=-1).argmin())
        self._update_stall(car_xy, self._car_waypoint)
        self.off_track = (not on_track) or self._stalled
        self._update_recover_mode(self.off_track)
        self._note_off_track_pass()

    def _waypoint_update_accepted(self, nearest_distance: float) -> bool:
        """Refuse a new waypoint while the car is off the raceline, and for a fixed time after it returns.

        Going farther than ``MPC_WAYPOINT_UPDATE_DISTANCE`` turns the hold on and drops every
        staged update. The resume clock stays unset while the car is away, including during a
        hold that the reward EMA started. It starts once the car is close, and leaving clears
        it, so the full delay has to elapse while the car stays close.
        """
        stalled = bool(getattr(self, "_stalled", False))
        self.off_track = nearest_distance > float(Settings.MPC_WAYPOINT_UPDATE_DISTANCE) or stalled
        if self.off_track:
            if not getattr(self, "_waypoint_off_raceline", False):
                self._waypoint_off_raceline = True
                reason = (
                    f"car barely moved (less than {float(Settings.MPC_STALL_DISTANCE):.2f} m in {float(Settings.MPC_STALL_TIME):.1f} s)"
                    if stalled else f"car left the raceline ({nearest_distance:.3f} m from the line)"
                )
                self._log_info(f"[HirarchicalPlanner] Warning: {reason}. Staged waypoint updates discarded.")
            just_left = (not self._waypoint_updates_held) or self._waypoint_update_close_since is not None
            if just_left:
                self._hold_waypoint_updates()
            self._waypoint_update_close_since = None
            self._update_recover_mode(True)
            return False
        self._waypoint_off_raceline = False
        if not self._waypoint_updates_held:
            self._update_recover_mode(False)
            return True
        if self._waypoint_update_close_since is None:
            self._waypoint_update_close_since = int(self.total_timesteps)
        wait = self._waypoint_resume_steps()
        if int(self.total_timesteps) - int(self._waypoint_update_close_since) < wait:
            self._update_recover_mode(False)
            return False
        self._waypoint_updates_held = False
        self._waypoint_update_close_since = None
        self._reward_drop_warned = False
        self._update_recover_mode(False)
        return True

    def _weighted_state_trajectory(self, trajectories: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Mean state rollout. Same weighted sum as the black plan line, over the whole state.

        Heading is rebuilt from the averaged sin and cos, so the reward does not see a wrapped angle.
        """
        mean = torch.sum(weights[:, None, None] * trajectories, dim=0)
        mean[:, POSE_THETA_IDX] = torch.atan2(mean[:, POSE_THETA_SIN_IDX], mean[:, POSE_THETA_COS_IDX])
        return mean

    def _accept_waypoint_update(
        self,
        rewards: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        averaged: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        """Score the driving and optimization mean trajectories.

        ``rewards`` is the sample split, ``((drive, optimize), (drive_crash, optimize_crash))``.
        ``averaged`` is those two weighted state rollouts. False skips the waypoint target
        computed on this step.
        """
        drive_trajectory, optimize_trajectory = averaged
        states = torch.stack((drive_trajectory, optimize_trajectory), dim=0)
        nearest = int(self.nearest_waypoint_index(states[0, 0]).reshape(-1)[0])
        matching = self.rollout_waypoint_states(states, nearest)
        (drive_reward, optimize_reward), (drive_crash, optimize_crash) = self.MPC_reward(
            states, matching, nearest, waypoint_off_track_counts=self._wp_off_track_count
        )
        #if not self.total_timesteps % 12: print(f"drive reward of drive: {drive_reward[0].item():.4f}, drive reward of optimized: {drive_reward[1].item():.4f}, optimizer reward of drive: {optimize_reward[0].item():.4f}, optimizer reward of optimized: {optimize_reward[1].item():.4f}")
        return drive_reward[0].item() < drive_reward[1].item() / 1.3 or True

    def _update_from_driven_plan(
        self, state: np.ndarray, pacejka: torch.Tensor, apply_update: bool = True
    ) -> None:
        """Stage the car's own plan at the waypoint ``k`` steps ahead, and commit it once passed."""
        if not hasattr(self, "waypoints"):
            return
        path = pacejka[0] if pacejka.ndim == 3 else pacejka
        nearest = int(self.nearest_waypoint_index(state).reshape(-1)[0])
        nearest_distance = self._raceline_distance(state)
        k = max(1, int(Settings.MPC_WAYPOINT_AHEAD))
        target = (nearest + k) % self._waypoints_n
        accepted = self._waypoint_update_accepted(nearest_distance)
        if not bool(torch.isfinite(path).all()):
            accepted = False  # a NaN plan would be staged as garbage (the start of the path, i.e. the car's own slow state)
        reject = (
            self.episode_timesteps >= 100 and accepted
            and bool(getattr(Settings, "MPC_WAYPOINT_REJECT_OPT_OFF_TRACK_UPDATE", False))
            and bool(self._paths_leave_track(path[None, :, [POSE_X_IDX, POSE_Y_IDX]].detach().cpu())[0])
        )  # Only the target-state updates are dropped; the action sequences are untouched.
        span = max(0, int(Settings.MPC_WAYPOINT_AHEAD_SPAN))
        if self.episode_timesteps >= 100 and accepted and not reject and apply_update:
            # Each waypoint from nearest + AHEAD to nearest + AHEAD + SPAN takes the state where the path crosses its own line.
            for offset in range(span + 1):
                target = (nearest + k + offset) % self._waypoints_n
                if self._waypoint_frozen(target):
                    continue
                on_line = self._state_on_orthogonal_line(path, target)
                if on_line is None:
                    continue
                xy, heading, speed, drift, steer, yaw_rate, slip = on_line
                xy = self._keep_wall_distance(xy, target)
                row = self.waypoints[target].clone()
                row[WP_X_IDX] = xy[0]
                row[WP_Y_IDX] = xy[1]
                row[WP_HEADING_IDX] = heading
                row[WP_TARGET_SPEED_IDX] = speed
                row[WP_TARGET_DRIFT_IDX] = drift
                row[WP_TARGET_STEER_IDX] = steer
                row[WP_TARGET_YAW_RATE_IDX] = yaw_rate
                row[WP_TARGET_SLIP_IDX] = slip
                self.waypoints_update_staged[target] = row
                self._wp_staged[target] = True
        self._commit_staged_waypoints(nearest)
        self._draw_driven_plan(path)

    def _commit_staged_waypoints(self, nearest: int) -> None:
        """Copy a staged waypoint onto the raceline once the car has driven past it."""
        if not hasattr(self, "_wp_staged"):
            return
        lag = max(1, int(Settings.MPC_WAYPOINT_COMMIT_LAG))
        n = self._waypoints_n
        for idx in torch.nonzero(self._wp_staged).flatten().tolist():
            passed = (int(nearest) - int(idx)) % n
            # passed > n/2 means the waypoint is still ahead; the difference wrapped the long way.
            if not lag <= passed <= n // 2 or self._waypoint_frozen(idx):
                continue
            gamma = self._ema_gamma(idx)
            current = self.waypoints[idx].clone()
            self._wp_hist_state[idx].append(current)
            staged = self.waypoints_update_staged[idx]
            self.waypoints[idx, WP_X_IDX] = (1.0 - gamma) * current[WP_X_IDX] + gamma * staged[WP_X_IDX]
            self.waypoints[idx, WP_Y_IDX] = (1.0 - gamma) * current[WP_Y_IDX] + gamma * staged[WP_Y_IDX]
            self.waypoints[idx, [WP_X_IDX, WP_Y_IDX]] = self._keep_wall_distance(
                self.waypoints[idx, [WP_X_IDX, WP_Y_IDX]].clone(), idx
            )
            self.waypoints[idx, WP_TARGET_SPEED_IDX] = (
                (1.0 - gamma) * current[WP_TARGET_SPEED_IDX] + gamma * staged[WP_TARGET_SPEED_IDX]
            )
            for column in WP_HWM_COLUMNS:
                self.waypoints[idx, column] = (1.0 - gamma) * current[column] + gamma * staged[column]
            if not bool(self._wp_vehicle_state_set[idx]):
                # Steering, yaw rate and slip start at 0 only because nothing was stored yet. The
                # first commit takes the staged values, so 0 does not pull them down.
                for column in (WP_TARGET_STEER_IDX, WP_TARGET_YAW_RATE_IDX, WP_TARGET_SLIP_IDX):
                    self.waypoints[idx, column] = staged[column]
                self._wp_vehicle_state_set[idx] = True
            self.waypoints[idx, WP_HEADING_IDX] = self._blend_heading(
                current[WP_HEADING_IDX], staged[WP_HEADING_IDX], gamma
            )
            self._wp_staged[idx] = False
        self._commit_staged_actions(nearest)
        self._write_raceline()
        self._publish_waypoint_display()

    def _commit_staged_actions(self, nearest: int) -> None:
        """Blend a staged action sequence once the car has passed that waypoint."""
        if not hasattr(self, "_wp_action_staged"):
            return
        lag = max(1, int(Settings.MPC_WAYPOINT_COMMIT_LAG))
        n = self._waypoints_n
        for idx in torch.nonzero(self._wp_action_staged).flatten().tolist():
            passed = (int(nearest) - int(idx)) % n
            if not lag <= passed <= n // 2 or self._waypoint_frozen(idx):
                continue
            gamma = self._ema_gamma(idx)
            current = self._wp_actions[idx]
            self._wp_hist_actions[idx].append((current.clone(), True))
            staged = self._wp_actions_staged[idx]
            self._wp_actions[idx] = (1.0 - gamma) * current + gamma * staged
            self._wp_action_staged[idx] = False

    def _draw_driven_plan(self, path: torch.Tensor) -> None:
        """The blue line is this plan, the same one the car is driving, drawn from the car."""
        render = getattr(self, "render_utils", None)
        scene = None if render is None else getattr(render, "scene", None)
        if scene is None:
            return
        chosen = path[:, [POSE_X_IDX, POSE_Y_IDX]].detach().cpu().numpy()
        scene.set_trajectories("hwm.opt_samples", None)
        scene.set_trajectories(
            "hwm.opt_plan", chosen[None], color=(25, 70, 190), width=3.0,
            max_trajectories=1, max_points_per_trajectory=int(chosen.shape[0]),
        )
        
    def _publish_waypoint_display(self) -> None:
        """Push the stored raceline, and an arrow of its heading and target speed, to the viewer.

        The web client bakes the original waypoint dots once. These dynamic layers replace them
        so a committed or staged edit shows up on the next frame.
        """
        render = getattr(self, "render_utils", None)
        if render is None or not hasattr(self, "waypoints"):
            return
        source = self.waypoints
        xy = source[:, [WP_X_IDX, WP_Y_IDX]].detach().cpu().numpy()
        heading = source[:, WP_HEADING_IDX].detach().cpu().numpy()
        speed = source[:, WP_TARGET_SPEED_IDX].detach().cpu().numpy()
        scene = getattr(render, "scene", None)
        if scene is None:
            return
        scene.set_dynamic_points("hwm.waypoints", None)
        scene.set_dynamic_points("closest_waypoint", None)
        # Arrow tail is the waypoint position. 0.08 m of length per 1 m/s, along the stored heading.
        length = np.abs(speed) * 0.08
        tip = xy + np.stack((length * np.cos(heading), length * np.sin(heading)), axis=1)
        head = length * 0.35
        wing = 0.5
        left = tip - np.stack((head * np.cos(heading - wing), head * np.sin(heading - wing)), axis=1)
        right = tip - np.stack((head * np.cos(heading + wing), head * np.sin(heading + wing)), axis=1)
        arrows = np.stack((xy, tip, left, tip, right), axis=1)
        if not hasattr(self, "_arrow_speed_max"):
            from utilities.car_files.vehicle_parameters import VehicleParameters
            self._arrow_speed_max = max(float(VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).v_max), 1e-3)
        scale = np.clip(np.abs(speed) / self._arrow_speed_max, 0.0, 1.0)
        stops = ((0, 190, 0), (240, 220, 0), (255, 140, 0), (210, 30, 20))
        colors = []
        for t in scale:
            x = float(t) * (len(stops) - 1)
            i = min(int(x), len(stops) - 2)
            f = x - i
            a, b = stops[i], stops[i + 1]
            colors.append([int(round(a[c] + (b[c] - a[c]) * f)) for c in range(3)])
        counts = getattr(self, "_wp_off_track_count", None)
        if counts is not None:
            scene.set_dynamic_points(
                "hwm.waypoint_off_track", xy, color=(235, 235, 235), radius=0.5,
                texts=[str(int(c)) for c in counts.tolist()],
            )
        paths = self._waypoint_paths() if Settings.HWM_SHOW_WAYPOINT_PATHS else None
        if paths is None:
            scene.set_trajectories("hwm.waypoint_paths", None)
        else:
            scene.set_trajectories(
                "hwm.waypoint_paths", paths, color=(150, 60, 190), width=1.0, stroke=True,
                max_trajectories=int(paths.shape[0]), max_points_per_trajectory=int(paths.shape[1]),
            )
        scene.set_trajectories(
            "hwm.waypoint_arrows", arrows, color=(220, 40, 40), width=1.5, stroke=True,
            colors=colors,
            max_trajectories=int(arrows.shape[0]), max_points_per_trajectory=8,
        )

    def _refresh_waypoint_paths(self) -> None:
        """Roll out, from every waypoint, its stored action sequence. Every ``_WP_PATH_EVERY`` steps.

        The start is the state stored in the waypoint: position, heading, speed, drift, steering
        angle, yaw rate and slip angle. The control delay starts filled with the sequence's first
        action. Sets ``_wp_path_states`` ``(N, T + 1, state_dim)``, ``None`` while no sequence is stored.
        """
        known = getattr(self, "_wp_action_known", None)
        if known is None or not bool(known.any()):
            self._wp_path_states = None
            return
        due = int(self.total_timesteps) - int(getattr(self, "_wp_paths_step", -10**9)) >= self._WP_PATH_EVERY
        if getattr(self, "_wp_path_states", None) is not None and not due:
            return
        self._wp_paths_step = int(self.total_timesteps)
        idx = torch.arange(self._waypoints_n)  # always all of them, so the compiled batch shape never changes
        wp = self.waypoints[idx]
        count = int(idx.shape[0])
        start = torch.zeros((count, int(Settings.HWM_STATE_DIM)), dtype=torch.float32)
        heading = wp[:, WP_HEADING_IDX]
        start[:, POSE_X_IDX] = wp[:, WP_X_IDX]
        start[:, POSE_Y_IDX] = wp[:, WP_Y_IDX]
        start[:, POSE_THETA_IDX] = heading
        start[:, POSE_THETA_SIN_IDX] = torch.sin(heading)
        start[:, POSE_THETA_COS_IDX] = torch.cos(heading)
        start[:, LINEAR_VEL_X_IDX] = wp[:, WP_TARGET_SPEED_IDX]
        start[:, LINEAR_VEL_Y_IDX] = wp[:, WP_TARGET_DRIFT_IDX]
        start[:, STEERING_ANGLE_IDX] = wp[:, WP_TARGET_STEER_IDX]
        start[:, ANGULAR_VEL_Z_IDX] = wp[:, WP_TARGET_YAW_RATE_IDX]
        start[:, SLIP_ANGLE_IDX] = wp[:, WP_TARGET_SLIP_IDX]
        actions = self._wp_actions[idx]
        delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))
        pending = actions[:, :1, :].expand(count, delay, 2)
        states = self.rollout_pacejka(start.numpy(), actions.numpy(), pending_actions=pending.numpy())
        self._wp_path_states = states.detach().cpu()

    def _waypoint_paths(self) -> np.ndarray | None:
        """Path ``(K, T + 1, 2)`` from each waypoint that has a stored action sequence."""
        self._refresh_waypoint_paths()
        states = getattr(self, "_wp_path_states", None)
        if states is None:
            return None
        return states[self._wp_action_known][:, :, [POSE_X_IDX, POSE_Y_IDX]].numpy()

    def _paths_leave_track(self, xy: torch.Tensor) -> torch.Tensor:
        """``(K,)`` bool: some point of path ``k`` is farther than ``MPC_WAYPOINT_UPDATE_DISTANCE`` from the raceline."""
        a = self._waypoint_xy
        seg = torch.roll(a, shifts=-1, dims=0) - a
        length2 = seg.pow(2).sum(dim=-1).clamp_min(1e-12)
        limit = float(Settings.MPC_WAYPOINT_UPDATE_DISTANCE)
        out = torch.zeros(int(xy.shape[0]), dtype=torch.bool)
        for k in range(int(xy.shape[0])):
            p = xy[k]  # (T+1, 2)
            t = (((p[:, None, :] - a[None]) * seg[None]).sum(-1) / length2[None]).clamp(0.0, 1.0)
            proj = a[None] + t[..., None] * seg[None]
            out[k] = bool((p[:, None, :] - proj).norm(dim=-1).min(dim=1).values.max() > limit)
        return out

    def _discard_wall_sequences(self) -> None:
        """Forget every stored action sequence whose path from its waypoint comes within the wall margin.

        The path is the one drawn from the waypoint (see ``_refresh_waypoint_paths``), judged with
        the planner's own crash rule (``HWM_MIN_BORDER_DISTANCE``). A forgotten sequence is written
        again, without averaging, the next time the car passes that waypoint.
        """
        self._refresh_waypoint_paths()
        states = getattr(self, "_wp_path_states", None)
        if states is None or getattr(self, "_wp_paths_checked", None) == self._wp_paths_step:
            return
        self._wp_paths_checked = self._wp_paths_step
        poses = states[:, :, [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
        if Settings.MPC_WAYPOINT_DISCARD_WALL_SEQUENCES:
            bad = self.wall_geometry.in_wall(poses).any(dim=1).cpu()
        else:
            bad = torch.zeros(int(poses.shape[0]), dtype=torch.bool)
        if bool(getattr(Settings, "MPC_WAYPOINT_REJECT_OFF_TRACK", False)):
            bad = bad | self._paths_leave_track(states[:, :, [POSE_X_IDX, POSE_Y_IDX]])
        bad = bad & self._wp_action_known
        if not hasattr(self, "_wp_good_actions"):
            self._wp_good_actions = torch.zeros_like(self._wp_actions)
            self._wp_good_known = torch.zeros_like(self._wp_action_known)
            self._wp_good_rows = self.waypoints.clone()
        changed = False
        if bool(getattr(Settings, "MPC_WAYPOINT_KEEP_ON_BAD_UPDATE", False)):
            fine = self._wp_action_known & ~bad
            self._wp_good_actions[fine] = self._wp_actions[fine]
            self._wp_good_rows[fine] = self.waypoints[fine]
            self._wp_good_known |= fine
            restore = bad & self._wp_good_known
            for idx in torch.nonzero(restore).flatten().tolist():
                # The waypoint state goes back together with its actions: the path was fine with that pair.
                self._wp_actions[idx] = self._wp_good_actions[idx]
                if Settings.MPC_WAYPOINT_STATE_FOLLOWS_ACTIONS:
                    self.waypoints[idx] = self._wp_good_rows[idx]
                    self._wp_staged[idx] = False
                    changed = True
                self._wp_actions_staged[idx].zero_()
                self._wp_action_staged[idx] = False
                # If the restored sequence is still bad at the next check, it gets deleted then.
                self._wp_good_known[idx] = False
            if bool(restore.any()):
                self._wp_rejected_total = int(getattr(self, "_wp_rejected_total", 0)) + int(restore.sum())
                self._wp_rejected_lap = int(getattr(self, "_wp_rejected_lap", 0)) + int(restore.sum())
            bad = bad & ~restore
        for idx in torch.nonzero(bad).flatten().tolist():
            self._wp_good_known[idx] = False
            self._wp_action_known[idx] = False
            self._wp_actions[idx].zero_()
            self._wp_actions_staged[idx].zero_()
            self._wp_action_staged[idx] = False
            self._wp_action_dist[idx] = float("inf")
            self._wp_hist_actions[idx].clear()
            self._wp_action_snapped[idx] = False
            # The waypoint state is reset like the sequence: back to the loaded raceline, nothing averaged.
            if Settings.MPC_WAYPOINT_STATE_FOLLOWS_ACTIONS:
                self.waypoints[idx] = self._wp_initial[idx]
                self._wp_staged[idx] = False
                self._wp_hist_state[idx].clear()
                self._wp_vehicle_state_set[idx] = False
                changed = True
        if changed:
            self._write_raceline()
            self._publish_waypoint_display()
        if bool(bad.any()):
            self._wp_discarded_total = int(getattr(self, "_wp_discarded_total", 0)) + int(bad.sum())
            self._wp_discarded_lap = int(getattr(self, "_wp_discarded_lap", 0)) + int(bad.sum())

    _WP_PATH_EVERY = 25

    def _clear_waypoint_display(self) -> None:
        """Drop the HWM arrows so the normal waypoint dots are drawn again."""
        render = getattr(self, "render_utils", None)
        scene = None if render is None else getattr(render, "scene", None)
        if scene is None:
            return
        scene.set_trajectories("hwm.waypoint_arrows", None)
        scene.set_trajectories("hwm.waypoint_paths", None)
        scene.set_dynamic_points("hwm.waypoint_off_track", None)
        scene.set_trajectories("hwm.opt_samples", None)
        scene.set_trajectories("hwm.opt_plan", None)
        scene.set_dynamic_points("hwm.waypoints", None)

    def select_action(self, obs: dict[str, torch.Tensor], state: np.ndarray) -> np.ndarray:
        """Policy in network output units. ``±1`` is ``±action_denorm`` on the car.

        Default: one sample of the actor (``HWM_SAMPLE_ACTIONS``) or its squashed
        Gaussian mean once weights are available, Pure Pursuit before that when
        ``HWM_PP_UNTIL_WEIGHTS`` is set. Pure Pursuit is not clipped, so it can leave ``[-1, 1]``.
        """
        if not self._received_weights and Settings.HWM_PP_UNTIL_WEIGHTS:
            self._ensure_waypoints()
            self._clear_waypoint_display()
            if not self._warned_no_weights:
                self._log_info("[HirarchicalPlanner] No weights yet; using Pure Pursuit fallback")
                self._warned_no_weights = True
           # return self._fallback_action()
        elif self.episode_timesteps < 8: #start episodes with acceleration only
            return np.array([0.0, 1.0])

        with torch.no_grad():
            self._refresh_recovery(state)
            # Recovery samples around the previous plan. Stored waypoint sequences are only
            # used once the car is back out of recovery.
            nominal = None if self.recover_mode else self._interpolated_waypoint_actions(state)
            if nominal is not None and Settings.HWM_SEED_FROM_ROLLOUT:
                nominal = self._rollout_seeded_actions(state, nominal)
            if Settings.HWM_GRADIENT_MPC and Settings.RUN_MPC_WITH_PACEJKA:
                actions = self._gradient_optimize(state, nominal)
            else:
                actions = self._sample_mpc_actions(nominal)
            if Settings.RUN_MPC_WITH_PACEJKA:
                trajectories = self.rollout_pacejka(state, actions)
            else:
                trajectories, n_soft, n_soft_normalized = self.rollout_lld(state, actions)
            nearest_waypoint_idx = self.nearest_waypoint_index(state)
            trajectry_waypoin_matching = self.rollout_waypoint_states(trajectories, nearest_waypoint_idx)
            (drive_rewards, opt_rewards), (drive_crash, opt_crash) = self.MPC_reward(
                trajectories, trajectry_waypoin_matching, nearest_waypoint_idx,
                waypoint_off_track_counts=self._wp_off_track_count,
            )
            drive_total = drive_rewards + drive_crash
            opt_total = opt_rewards + opt_crash
            weights = torch.softmax(drive_total / Settings.MPC_TEMPERATURE, dim=0)
            reward_mean = torch.sum(drive_rewards * weights, dim=0)
            self._note_reward(reward_mean)
           # if not self.total_timesteps % 12: print(self._reward_EMA)
            action_seq = (weights[:, None, None] * actions).sum(dim=0)
            # Next step samples around this plan with its first command dropped and its last command repeated.
            self.action_buffer = torch.cat((action_seq[1:], action_seq[-1:]), dim=0)
            pacejka = self.rollout_pacejka(state, action_seq)
            self._draw_mpc_trajectories(trajectories, weights, pacejka)
            if self._stall_phase == 1:
                # Backtracking: no waypoint staging or recording from this plan.
                self._reverse_left -= 1
                return action_seq[0].cpu().numpy().astype(np.float32)
            opt_weights = torch.softmax(opt_total / Settings.MPC_TEMPERATURE, dim=0)
            drive_mean = self._weighted_state_trajectory(trajectories, weights)
            opt_mean = self._weighted_state_trajectory(trajectories, opt_weights)
            apply_update = self._accept_waypoint_update(
                ((drive_rewards, opt_rewards), (drive_crash, opt_crash)),
                (drive_mean, opt_mean),
            )
            opt_seq = (opt_weights[:, None, None] * actions).sum(dim=0)
            opt_pacejka = self.rollout_pacejka(state, opt_seq)
            self._update_from_driven_plan(state, opt_pacejka, apply_update)
            self._record_waypoint_action(state, action_seq.detach())
            self._discard_wall_sequences()
            action = action_seq[0].cpu().numpy().astype(np.float32)
            return action

    def _draw_mpc_trajectories(
        self, trajectories: torch.Tensor, weights: torch.Tensor, pacejka: torch.Tensor
    ) -> None:
        """MPC samples, their weighted mean, and Pacejka driven by that whole mean action sequence."""
        if getattr(self, "render_utils", None) is None:
            return
        xy = trajectories[..., [POSE_X_IDX, POSE_Y_IDX]].detach().cpu()
        scene = self.render_utils.scene
        caps = {"max_trajectories": int(xy.shape[0]), "max_points_per_trajectory": int(xy.shape[1])}
        if Settings.MPC_DRAW_SAMPLES:
            scene.set_trajectories("hwm.mpc_samples", xy.numpy(), color=(140, 140, 140), width=1.0, **caps)
        else:
            scene.set_trajectories("hwm.mpc_samples", None)
        mean_xy = (weights.detach().cpu()[:, None, None] * xy).sum(dim=0)
        scene.set_trajectories("hwm.mpc_plan", mean_xy.numpy()[None], color=(20, 20, 20), width=3.0, **caps)
        pace_xy = pacejka[..., [POSE_X_IDX, POSE_Y_IDX]].detach().cpu().numpy()
        scene.set_trajectories(
            "hwm.mpc_pacejka", pace_xy, color=(230, 90, 0), width=3.0,
            max_trajectories=int(pace_xy.shape[0]), max_points_per_trajectory=int(pace_xy.shape[1]),
        )

    @torch.no_grad()
    def rollout_lld(self, state, actions):
        """LLD mean rollout of action sequences.

        ``actions`` is ``(B, N, 2)`` or ``(N, 2)``, the command chosen at each step, in network
        units. ``state`` is the shared start ``(state_dim,)`` or one start per sequence ``(B, state_dim)``.

        Returns ``states`` ``(B, N + 1, state_dim)``, and ``n_soft`` and ``n_soft_norm`` ``(B, N)``.
        Row ``k`` of those two is the attention mass at ``states[:, k]`` under ``actions[:, k]``.
        ``n_soft`` is the sum of ``exp(-distance / temperature)`` over the keys.
        ``n_soft_norm`` divides each key's kernel by that key's stored n_soft before summing.

        The key bank is read once, as views. Imagined rows stay in a small history and are not
        written into memory. The control delay is already in the training pairs, so ``actions``
        is not shifted.
        """
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.memory.device)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        batch, horizon, _ = (int(actions.shape[0]), int(actions.shape[1]), int(actions.shape[2]))
        state_dim = int(Settings.HWM_STATE_DIM)
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.memory.device)
        if s.ndim == 1:
            s = s.reshape(1, -1).expand(batch, -1).contiguous()
        elif int(s.shape[0]) != batch:
            raise ValueError(f"state batch {int(s.shape[0])} does not match actions batch {batch}")

        lld = self.models["LLD"]
        if lld.mean_delta_compiled is None:
            lld._compile_rollout_step(lld._rollout_super_state_dim)
        if lld.extra_dim:
            raise ValueError(
                "rollout_lld assumes the residual net has no retrieved inputs "
                "(HWM_LLD_CORRECTION_INPUT_RESIDUAL and HWM_LLD_CORRECTION_INPUT_COV are off)"
            )
        past, _ = self.memory.recent(self._episode_id, width - 1)
        history = lld_rows_for_super(_normalize_rows(past.unsqueeze(0), state_dim), state_dim)
        history = history.expand(batch, -1, -1).contiguous()
        bank = self.memory.lld_bank()
        if bank is None:
            key_emb = s.new_zeros((0, lld.attention_norm_projector[-1].out_features))
            tokens = s.new_zeros((0, lld.prediction_dim))
            density = s.new_zeros((0,))
            valid = torch.zeros((0,), dtype=torch.bool, device=s.device)
        else:
            supers, deltas, density, rows = bank
            cutoff = int(self.memory.next_step) - (width - 1)
            valid = rows + 1 < cutoff
            key_emb = lld.attention_norm_projector(supers)
            tokens = deltas - lld._correct(supers)

        states = [s]
        n_softs = []
        n_soft_norms = []
        for k in range(horizon):
            a = actions[:, k]
            current = lld_rows_for_super(
                torch.cat((_normalize_states(s), a), dim=-1), state_dim
            )
            query = torch.cat((history.flatten(start_dim=1), current), dim=-1)
            step = getattr(lld, "mean_delta_compiled", lld.mean_delta)
            delta, n_soft, n_soft_norm = step(query, key_emb, tokens, density, valid)
            s = lld.reconstruct_next_state(s, delta)
            history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
            states.append(s)
            n_softs.append(n_soft)
            n_soft_norms.append(n_soft_norm)
        n_soft = torch.stack(n_softs, dim=1) if n_softs else s.new_zeros((batch, 0))
        n_soft_norm = torch.stack(n_soft_norms, dim=1) if n_soft_norms else n_soft
        return torch.stack(states, dim=1), n_soft, n_soft_norm

    def _pacejka_batch_step(self):
        """One Pacejka control step, compiled for whatever batch size it is given."""
        step = getattr(self, "_pacejka_batch", None)
        if step is None:
            import jax
            import jax.numpy as jnp
            from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax_from_settings
            from utilities.car_files.vehicle_parameters import VehicleParameters

            car_params = jnp.asarray(VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_np_array())
            dt = float(Settings.TIMESTEP_CONTROL)
            substeps = int(round(dt / float(Settings.TIMESTEP_SIM)))

            def one(state, control):
                return car_dynamics_pacejka_jax_from_settings(state, control, car_params, dt, substeps)

            step = jax.jit(jax.vmap(one))
            self._pacejka_batch = step
        return self._pacejka_batch

    def rollout_pacejka(self, state, actions, use_control_delay: bool = True, pending_actions=None):
        """states (B, N+1, 10) for the same actions. Applies the command from 2 control steps ago. For testing the Planner with ground truth model in simulation."""
        import jax
        import jax.numpy as jnp
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        batch, horizon, _ = actions.shape
        delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))  # 2
        if pending_actions is not None:
            # Control-delay queue given per sequence (B, delay, 2), instead of the last recorded actions.
            pending = np.asarray(pending_actions, dtype=np.float32).reshape(batch, -1, 2).copy()
        elif use_control_delay:
            rows, _ = self.memory.recent(self._episode_id, delay)
            pending = np.broadcast_to(rows[:, int(Settings.HWM_STATE_DIM):].numpy(), (batch, delay, 2)).copy()
        else:
            pending = np.zeros((batch, 0, 2), dtype=np.float32)
        s = np.broadcast_to(np.asarray(state, dtype=np.float32), (batch, int(Settings.HWM_STATE_DIM))).copy()
        denorm = self.action_denormalization_array
        cap = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
        step = self._pacejka_batch_step()
        states = [s.copy()]
        for k in range(horizon):
            pending = np.concatenate((pending, actions[:, k, None, :]), axis=1)
            applied = pending[:, 0] * denorm
            pending = pending[:, 1:]
            applied[:, 1] = np.clip(applied[:, 1], -cap, cap)
            s = np.array(step(jnp.asarray(s), jnp.asarray(applied)), dtype=np.float32, copy=True)
            s[:, POSE_THETA_IDX] = np.arctan2(s[:, POSE_THETA_SIN_IDX], s[:, POSE_THETA_COS_IDX])
            states.append(s.copy())
        return torch.as_tensor(np.stack(states, axis=1)[:batch])

    def _grad_rollout_fns(self):
        """JAX rollout of the Pacejka model over a horizon and its vector-Jacobian product, both jitted."""
        fns = getattr(self, "_grad_rollout", None)
        if fns is None:
            import jax
            import jax.numpy as jnp
            from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax_from_settings
            from utilities.car_files.vehicle_parameters import VehicleParameters

            car_params = jnp.asarray(VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_np_array())
            dt = float(Settings.TIMESTEP_CONTROL)
            substeps = int(round(dt / float(Settings.TIMESTEP_SIM)))
            step = jax.vmap(lambda s, u: car_dynamics_pacejka_jax_from_settings(s, u, car_params, dt, substeps))

            def forward(s0, applied):  # (B, sd), (B, H, 2) -> (B, H + 1, sd)
                def body(s, u):
                    nxt = step(s, u)
                    nxt = nxt.at[:, POSE_THETA_IDX].set(jnp.arctan2(nxt[:, POSE_THETA_SIN_IDX], nxt[:, POSE_THETA_COS_IDX]))
                    return nxt, nxt

                _, seq = jax.lax.scan(body, s0, jnp.swapaxes(applied, 0, 1))
                return jnp.swapaxes(jnp.concatenate((s0[None], seq), axis=0), 0, 1)

            def backward(s0, applied, cotangent):
                return jax.vjp(lambda a: forward(s0, a), applied)[1](cotangent)[0]

            fns = (jax.jit(forward), jax.jit(backward))
            self._grad_rollout = fns
        return fns

    def _rollout_pacejka_grad(self, state, actions: torch.Tensor) -> torch.Tensor:
        """``rollout_pacejka`` that keeps the graph to ``actions`` ``(B, H, 2)``: states ``(B, H + 1, state_dim)``."""
        batch, horizon, _ = actions.shape
        delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))
        rows, _ = self.memory.recent(self._episode_id, delay)
        pending = rows[:, int(Settings.HWM_STATE_DIM):].to(actions.dtype).expand(batch, delay, 2)
        queue = torch.cat((pending, actions), dim=1)[:, :horizon]
        applied = queue * torch.as_tensor(self.action_denormalization_array, dtype=actions.dtype)
        cap = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
        applied = torch.stack((applied[..., 0], applied[..., 1].clamp(-cap, cap)), dim=-1)
        s0 = np.broadcast_to(np.asarray(state, dtype=np.float32), (batch, int(Settings.HWM_STATE_DIM))).copy()
        forward, backward = self._grad_rollout_fns()
        return _JaxRollout.apply(applied, s0, forward, backward)

    def _gradient_optimize(self, state: np.ndarray, nominal: torch.Tensor | None) -> torch.Tensor:
        """Action sequences ``(K, H, 2)``: noisy copies of the nominal, improved by gradient descent on the MPC cost.

        The cost is the same reward the sampler uses (recovery and backtrack costs included),
        differentiated through the Pacejka rollout. The wall hit is not differentiable, so a soft
        penalty on the corner distance to the walls stands in for it while descending. The caller
        then scores the result like sampled actions.
        """
        count = max(1, int(Settings.HWM_GRAD_SAMPLES))
        base = self.action_buffer if nominal is None else nominal
        u = self._sample_mpc_actions(nominal, count=count)
        u[0] = base  # one noise-free copy
        steer_limit = 1.0
        accel_limit = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL) / float(self.action_denormalization_array[1])
        u = u.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=float(Settings.HWM_GRAD_LR))
        nearest = self.nearest_waypoint_index(state)
        margin = float(Settings.HWM_GRAD_WALL_MARGIN)
        wall_weight = float(Settings.HWM_GRAD_WALL_WEIGHT)
        with torch.enable_grad():
            for _ in range(max(0, int(Settings.HWM_GRAD_STEPS))):
                optimizer.zero_grad()
                states = self._rollout_pacejka_grad(state, u)
                matched = self.rollout_waypoint_states(states.detach(), nearest)
                (reward, _), _ = self.MPC_reward(
                    states, matched, nearest, waypoint_off_track_counts=self._wp_off_track_count,
                )
                poses = states[:, :, [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
                cost = -reward + wall_weight * self.wall_geometry.soft_wall_penalty(poses, margin).mean(dim=1)
                cost.sum().backward()
                if u.grad is not None:
                    u.grad.nan_to_num_(0.0)
                optimizer.step()
                with torch.no_grad():
                    u[..., 0].clamp_(-steer_limit, steer_limit)
                    u[..., 1].clamp_(-accel_limit, accel_limit)
        return u.detach()

    def _log_attention(self) -> None:
        """Log the attention statistics of the last LLD forward pass (main head, and the covariance head if on)."""
        lld = self.models["LLD"]
        tag = " (single encoder)" if lld.standalone else ""
        text = f"[attention]{tag} keys for top 50% weight: {float(lld.last_top_keys):.1f} n_soft: {float(lld.last_n_soft):.2f}"
        text += f" n_soft_norm: {float(lld.last_n_soft_norm):.2f}"
        if lld.cov_head:
            text += f" | cov head: keys50 {float(lld.last_cov_top_keys):.1f} n_soft {float(lld.last_cov_n_soft):.2f}"
        self._log_info(text)

    @torch.no_grad()
    def _log_attention_test_track(self) -> None:
        """Test track only: one LLD pass on the last stored row, to log the attention statistics every step.

        Needs the drive in memory (``TEST_TRACK_STORE_MEMORY``); the output is not used otherwise.
        """
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        state_dim = int(Settings.HWM_STATE_DIM)
        rows, mask = self.memory.recent(self._episode_id, width)
        if not bool(mask[-1]):
            return
        obs = build_LLD_observation_inference(
            rows[-1:, :state_dim],
            rows[-1:, state_dim:],
            self.memory,
            self._episode_id,
            past_states=rows[None, :-1],
            before_steps=torch.tensor([self.memory.next_step - 1]),
        )
        self.models["LLD"](obs)
        self._log_attention()

    @torch.no_grad()
    def lld_dynamics_step(self, state: np.ndarray, physical_control: np.ndarray) -> np.ndarray:
        """Next car state from the LLD, one control step.

        ``state`` is the simulator state before the step. The action is the one just
        stored with it (network units), which is the pair the LLD is trained on.
        ``HWM_SIM_LLD_SAMPLE`` draws one delta; otherwise the mean is used.
        """
        state_dim = int(Settings.HWM_STATE_DIM)
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        rows, mask = self.memory.recent(self._episode_id, width)
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32).reshape(1, -1)
        if bool(mask[-1]):
            action_t = rows[-1:, state_dim:]
            past = rows[None, :-1]
            before = torch.tensor([self.memory.next_step - 1])
        else:
            action_t = torch.as_tensor(
                np.asarray(physical_control, dtype=np.float32) / self.action_denormalization_array,
                dtype=torch.float32,
            ).reshape(1, -1)
            past = torch.zeros(1, width - 1, state_dim + int(Settings.HWM_ACTION_DIM))
            before = None
        obs = build_LLD_observation_inference(
            state_t, action_t, self.memory, self._episode_id, past_states=past, before_steps=before
        )
        prediction = self.models["LLD"](obs)
        delta = prediction.sample() if Settings.HWM_SIM_LLD_SAMPLE else prediction.mean
        nxt = self.models["LLD"].reconstruct_next_state(state_t, delta)
        return nxt[0].cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def process_observation(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        self._controller_observation = controller_observation
        if test_track_enabled():
            self._clear_waypoint_display()
            return self._test_track_step(controller_observation)
        self._maybe_handle_server_terminate()
        self._sync_from_server()

        if not self.autonomous_driving:
            self._clear_waypoint_display()
            action = self._fallback_action(controller_observation)
            self._apply_normalized_action(action)
            state = self.get_car_state(controller_observation)
            self._maybe_update_lld_preview(state, controller_observation)
            self._record_step(state, action)
            return self.angular_control, self.translational_control

        state = self.get_car_state(controller_observation)
        obs = self.build_observation(state)
        action = np.asarray(self.select_action(obs, state), dtype=np.float32).reshape(-1)
        self._apply_normalized_action(action)
        self._maybe_update_lld_preview(state, controller_observation)
        # Network-scale action, unclipped. Pure Pursuit fallback is physical / action_denorm.
        self._record_step(state, action)
        return self.angular_control, self.translational_control

    def _test_track_step(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        """Constant acceleration and a growing sine steer. The LLD weights are left as they are."""
        self._maybe_handle_server_terminate()
        self._sync_from_server()
        state = self.get_car_state(controller_observation)
        steer, accel = test_track_command(self.control_index * float(Settings.TIMESTEP_CONTROL))
        action = np.array([steer, accel], dtype=np.float32) / self.action_denormalization_array
        self.angular_control = float(steer)
        self.translational_control = float(accel)
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control
        self._maybe_update_lld_preview(state, controller_observation)
        if test_track_store_memory():
            self._log_attention_test_track()
            self._record_step(state, action)
        else:
            keep = max(int(Settings.HWM_SUPER_STATE_SIZE), 2 * self.lld_preview.delay + 1) if self.lld_preview else 0
            row = np.concatenate((np.asarray(state, dtype=np.float32), action.astype(np.float32)))
            self._test_rows = (self._test_rows + [row])[-keep:] if keep else []
            self.control_index += 1
            self.total_timesteps += 1
            self.episode_timesteps += 1
        return self.angular_control, self.translational_control

    def _apply_normalized_action(self, action: np.ndarray) -> None:
        steering, accel = np.asarray(action, dtype=np.float32).reshape(-1) * self.action_denormalization_array
        self.angular_control = self.lowpass_alpha * float(steering) + (1 - self.lowpass_alpha) * self.prev_angular_control
        self.translational_control = self.lowpass_alpha * float(accel) + (1 - self.lowpass_alpha) * self.prev_translational_control
        max_translational = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
        self.translational_control = float(np.clip(self.translational_control, -max_translational, max_translational))
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control

    def _maybe_update_lld_preview(self, state: np.ndarray, controller_observation: Dict[str, Any]) -> None:
        """Draw the newest background preview and start the next one; before the step is recorded.

        A new preview starts at most every ``HWM_LLD_PREVIEW_EVERY`` steps, once LLD
        weights exist and the previous one has finished.
        """
        if self.preview_worker is None or not (self._received_weights or test_track_enabled()):
            return
        # MPC draws its own rollouts. The actor preview stays only while Pure Pursuit drives.
        if self.autonomous_driving and not test_track_enabled():
            return
        result = self.preview_worker.pop_result()
        if result is not None and result.generation != self._generation:
            result = None
        if result is not None and self.render_utils is not None:
            age = self.control_index - result.control_index
            plan_states, lld_states, mean_states, residual_states = result.value
            self.lld_preview.render(self.render_utils, plan_states, lld_states, age, mean_states, residual_states)
            self.render_utils.set_label_dict({"LLD preview of": result.source})
        else:
            self.lld_preview.advance(self.render_utils)
        if self.control_index - self._preview_submitted < int(Settings.HWM_LLD_PREVIEW_EVERY):
            return
        on_test_track = test_track_enabled()
        if on_test_track:
            source = "test track"
        elif self.autonomous_driving:
            source = "actor"
        else:
            source = "Pure Pursuit"
        started = self.preview_worker.submit(
            self._generation,
            self.control_index,
            source,
            state=np.array(state, dtype=np.float32),
            past=self.lld_preview.past_rows(
                self._episode_id, self._test_rows if on_test_track and not test_track_store_memory() else None
            ),
            controller_observation=dict(controller_observation),
            waypoint_utils=self.waypoint_utils,
            nearest_index=int(self.waypoint_utils.nearest_waypoint_index or 0),
            correcting_index=self.fallback_planner.correcting_index,
            use_actor=self.autonomous_driving and not on_test_track,
            scripted_time=self.control_index * float(Settings.TIMESTEP_CONTROL) if on_test_track else None,
        )
        if started:
            self._preview_submitted = self.control_index

    def _record_step(self, state: np.ndarray, action: np.ndarray) -> None:
        """Append one ``[state, action]`` row in network output units and stream it."""
        self.memory.add_step(state, np.asarray(action, dtype=np.float32).reshape(-1), self._episode_id)
        self._maybe_bootstrap_send()
        self._maybe_stream_send()
        self.control_index += 1
        self.total_timesteps += 1
        self.episode_timesteps += 1

    def on_step_end(self, driver_obs: Dict[str, Any]) -> None:
        """Called by CarSystem after the physics step; only episode boundaries matter here."""
        self._maybe_handle_server_terminate()

        done = bool(driver_obs.get("done", False))
        if done:
            if hasattr(self, "_wp_staged"):
                self._wp_staged.zero_()
            if hasattr(self, "_wp_action_staged"):
                self._wp_action_staged.zero_()
            self._revert_nearby_waypoints()
            n_rows = self.memory.open_length(self._episode_id)
            self._flush_stream_send(episode_end=True)
            self.memory.end_episode(self._episode_id)
            self._log_debug(f"[HirarchicalPlanner] Episode {self._episode_id} done: {n_rows} rows")
            self._reset_episode_state()

    def on_simulation_end(self, collision: bool = False) -> None:
        if self.terminate_server_after_simulation and self.training_mode and self.client is not None:
            self._terminate_server_with_retry()

    def close(self) -> None:
        if self.preview_worker is not None:
            self.preview_worker.close()
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    def _fallback_action(self, controller_observation: Optional[Dict[str, Any]] = None) -> np.ndarray:
        self.fallback_planner.waypoint_utils = self.waypoint_utils
        self.fallback_planner.lidar_utils = self.lidar_utils
        if controller_observation is None:
            controller_observation = self._controller_observation
        control = np.asarray(self.fallback_planner.process_observation(controller_observation), dtype=np.float32)
        return (control / self.action_denormalization_array).astype(np.float32)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def _init_inference_models(self) -> None:
        model_dir = resolve_model_dir(self.inference_model_name)
        loaded = self.models.load(model_dir, device="cpu")
        self.models.eval()
        if "LLD" in loaded:
            self.memory.refresh_embeddings()
        if "actor" not in loaded:
            raise FileNotFoundError(f"[HirarchicalPlanner] actor.pt not found in {model_dir}")
        self._seed_memory(self.inference_model_name)
        self._log_info(f"[HirarchicalPlanner] Loaded modules {loaded} from {model_dir}")

    def _seed_memory(self, model_name: Optional[str]) -> None:
        """Load the learner's saved neighbor bank once, while this memory is still empty.

        The LLD's prediction is g(q) plus the average of stored residuals. With an empty bank
        that average is zero and g starts at zero, so the predicted step is zero and a car
        that starts at rest never moves. Driving that zero step back into memory keeps it there.
        """
        if not model_name or self.memory.total_steps > 0:
            return
        path = resolve_model_dir(str(model_name)) / "memory.pt"
        if not path.is_file():
            if Settings.HWM_SIM_USE_LLD:
                self._log_info(
                    "[HirarchicalPlanner] LLD sim memory is empty, so the predicted step is zero and the car stays still. "
                    "memory.pt is written when the learner saves a run driven by the real dynamics."
                )
            return
        payload = torch.load(path, weights_only=True)
        n_file = int(payload["rows"].shape[0])
        imported = self.memory.import_rows(payload)
        if imported < n_file:
            snapshot = self.memory.export_rows()
            if snapshot is not None:
                tmp = path.with_name("memory.pt.tmp")
                torch.save(snapshot, tmp)
                tmp.replace(path)
        self.memory.refresh_embeddings()
        self._log_info(f"[HirarchicalPlanner] Loaded {imported} memory rows from {path.name}")

    def _apply_weights(self, sds: dict) -> None:
        # The test track keeps the LLD that was loaded at the start. A later broadcast is a training update.
        if test_track_enabled() and self._received_weights:
            return
        with self._display_lock:
            loaded = self.models.load_state_dicts(sds, strict=True)
            self.models.eval()
            if "LLD" in loaded:
                #self.memory.refresh_embeddings()
                pass
        if loaded:
            self._received_weights = True
            self._log_debug(f"[HirarchicalPlanner] Weights updated: {loaded}")

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------
    def _sync_from_server(self) -> None:
        if not self.training_mode or self.client is None:
            return
        handshake = self.client.pop_handshake()
        if isinstance(handshake, dict):
            self._server_handshake = handshake
            self._seed_memory(handshake.get("model_name"))
            if handshake.get("map_name") and handshake["map_name"] != str(Settings.MAP_NAME):
                self._log_info(
                    f"[HirarchicalPlanner] WARNING: learner map '{handshake['map_name']}' "
                    f"!= client map '{Settings.MAP_NAME}'"
                )
        info = self.client.pop_latest_training_info()
        if isinstance(info, dict):
            self.latest_training_info = info
        sds = self.client.pop_latest_weights()
        if sds:
            self._apply_weights(sds)

    def report_lap_time(self, lap_time: float, mean_distance: float, std_distance: float, max_distance: float) -> None:
        """Forward a finished lap to the learner so it is printed there, not on every train step."""
        self._completed_laps = int(getattr(self, "_completed_laps", 0)) + 1
        discarded = int(getattr(self, "_wp_discarded_lap", 0))
        rejected = int(getattr(self, "_wp_rejected_lap", 0))
        self._wp_discarded_lap = 0
        self._wp_rejected_lap = 0
        if not self._can_stream():
            self._log_info(f"[HirarchicalPlanner] Lap: reset {discarded} waypoint action sequences, rejected {rejected} updates")
            return
        self.client.send_lap_time(
            lap_time, mean_distance, std_distance, max_distance,
            wp_discarded=discarded, wp_rejected=rejected,
        )

    def _can_stream(self) -> bool:
        return self.training_mode and self.client is not None

    def _send_rows(self, start: int, end: int, episode_end: bool = False) -> bool:
        states, actions = self.memory.slice_open(self._episode_id, start, end)
        if states.shape[0] == 0:
            return True
        return self.client.send_raw_batch(states, actions, self._episode_id, episode_end=episode_end)

    def _maybe_bootstrap_send(self) -> None:
        """Send the first rows early (once per planner lifetime) so the learner can init."""
        if not self._can_stream() or self._bootstrap_sent:
            return
        n = self.memory.open_length(self._episode_id)
        if n < self.BOOTSTRAP_ROWS:
            return
        if self._send_rows(0, n):
            self._stream_send_idx = n
            self._bootstrap_sent = True

    def _maybe_stream_send(self) -> None:
        if not self._can_stream():
            return
        n = self.memory.open_length(self._episode_id)
        if n - self._stream_send_idx < self._stream_batch_size:
            return
        end = self._stream_send_idx + self._stream_batch_size
        if self._send_rows(self._stream_send_idx, end):
            self._stream_send_idx = end

    def _flush_stream_send(self, episode_end: bool = False) -> None:
        if not self._can_stream():
            return
        n = self.memory.open_length(self._episode_id)
        min_len = int(Settings.HWM_MIN_EPISODE_END_BATCH_SIZE)
        if episode_end and n < min_len:
            self._log_debug(f"[HirarchicalPlanner] Skipping short episode ({n} rows, min={min_len})")
            self._stream_send_idx = n
            return
        if self._send_rows(self._stream_send_idx, n, episode_end=episode_end):
            self._stream_send_idx = n

    def _reset_episode_state(self) -> None:
        self._episode_id += 1
        self._stream_send_idx = 0
        self.control_index = 0
        self.episode_timesteps = 0
        self._generation += 1
        self._preview_submitted = -int(Settings.HWM_LLD_PREVIEW_EVERY)
        self._test_rows = []
        self.action_buffer = torch.zeros(Settings.MPC_TRAJECTORY_LENGTH, 2)
        self._reset_stall()

    def _terminate_server_with_retry(self) -> None:
        delivered = False
        try:
            for _ in range(2):
                delivered = bool(self.client.send_terminate(wait_for_ack=True, ack_timeout=2.0, urgent=True))
                if delivered:
                    break
            if not delivered:
                self.client.send_terminate(wait_for_ack=False, urgent=True)
            self._log_info("[HirarchicalPlanner] Sent terminate message to server")
        except Exception as e:
            self._log_info(f"[HirarchicalPlanner] Failed to send terminate message: {e}")
        finally:
            self.close()

    def _maybe_handle_server_terminate(self) -> None:
        if self.client is None:
            return
        payload = self.client.pop_server_terminate()
        if payload is None:
            return
        reason = payload.get("reason", "server_requested_terminate")
        self._log_info(f"[HirarchicalPlanner] Received terminate from server: {reason}. Stopping client process.")
        self.close()
        os.kill(os.getpid(), signal.SIGTERM)

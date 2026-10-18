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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:  # keep the sim process single-threaded for torch; may already be fixed by earlier imports
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


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
        if best > float(self._wp_action_dist[idx]):
            return
        seq = torch.as_tensor(action_seq, dtype=torch.float32).reshape(-1, 2)
        horizon = int(self._wp_actions.shape[1])
        if int(seq.shape[0]) < horizon:
            seq = torch.cat((seq, seq[-1:].expand(horizon - int(seq.shape[0]), -1)), dim=0)
        seq = seq[:horizon]
        if not bool(self._wp_action_known[idx]):
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

        Returns ``None`` until both of those waypoints have been passed once, so the first
        lap keeps planning from the previous action sequence.
        """
        if not hasattr(self, "_wp_action_known") or not bool(self._wp_action_known.any()):
            return None
        start, alpha = self._raceline_segment(state)
        end = (start + 1) % self._waypoints_n
        if not (bool(self._wp_action_known[start]) and bool(self._wp_action_known[end])):
            return None
        return (1.0 - alpha) * self._wp_actions[start] + alpha * self._wp_actions[end]

    def _ensure_waypoints(self) -> None:
        utils = getattr(self, "waypoint_utils", None)
        waypoints = None if utils is None else getattr(utils, "waypoints", None)
        if waypoints is None or getattr(self, "_waypoints_n", 0) == len(waypoints):
            return
        source = np.asarray(waypoints)
        owned = np.zeros((source.shape[0], WP_TARGET_DRIFT_IDX + 1), dtype=np.float32)
        width = min(int(source.shape[1]), WP_TARGET_DRIFT_IDX)
        owned[:, :width] = source[:, :width]
        self._raceline_np = source
        self.waypoints = torch.as_tensor(owned)  # (N, 11). Column 10 is the HWM drift target, starting at 0.
        self._waypoint_xy = self.waypoints[:, 1:3]
        self._waypoints_n = int(self.waypoints.shape[0])
        if not self._wp_speed_cap_applied:
            self.waypoints[:, WP_TARGET_SPEED_IDX] = torch.minimum(self.waypoints[:, WP_TARGET_SPEED_IDX], torch.tensor(3.0))
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Driving reward and waypoint-optimization reward for each trajectory.

        Both are higher when better. The crash penalty is applied to each one.
        """
        batch = int(states.shape[0])
        num_valid_waypoints = nearest_waypoint_indices.shape[1] - nearest_waypoint_indices.isnan().sum(dim=1)
        drive = torch.zeros(batch)
        optimize = torch.zeros(batch)
        start = int(torch.as_tensor(nearest_waypoint).reshape(-1)[0])
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
            position = states[i, indices][:, [POSE_X_IDX, POSE_Y_IDX]] - local_waypoints[:, [WP_X_IDX, WP_Y_IDX]]
            distance_reward = torch.linalg.norm(position, dim=1).sum()
            heading_error = states[i, indices, POSE_THETA_IDX] - local_waypoints[:, WP_HEADING_IDX]
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
            heading_reward = heading_error.abs().sum()
            speed = states[i, indices, LINEAR_VEL_X_IDX]
            target_speed = local_waypoints[:, WP_TARGET_SPEED_IDX]
            drift = states[i, indices, LINEAR_VEL_Y_IDX]
            target_drift = local_waypoints[:, WP_TARGET_DRIFT_IDX]
            drift_error = (drift - target_drift).abs().sum()
            shortfall = (target_speed - speed).clamp(min=0).sum()
            overspeed = (speed - target_speed).clamp(min=0).sum()
            if self.off_track:
                drive_reward = (10 * distance_reward + 2 * heading_reward) / count
            else:
                drive_reward = (10 * distance_reward + 2 * heading_reward + shortfall * 2 + overspeed * 2 + 0.5 * drift_error + 0.5 * indices.float().max()) / count
            optimize_reward = (10 * distance_reward + 2 * heading_reward + shortfall * 5 + overspeed + 0.5 * drift_error + 1 * indices.float().max()) / count
            min_speed = float(speed.min())
            if min_speed < 0:
                reverse = 100 * abs(min_speed)
                drive_reward = drive_reward + reverse
                optimize_reward = optimize_reward + reverse
            drive[i] = drive_reward
            optimize[i] = optimize_reward

        poses = states[:, :, [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
        crashed = self.wall_geometry.in_wall(poses)
        has_crash = crashed.any(dim=1)
        first = crashed.float().argmax(dim=1)
        horizon = crashed.shape[1]
        # Steps until impact. A clean rollout survives the whole horizon. An earlier hit costs more,
        # so a plan that follows the corner and clips late beats one that drives into the wall now.
        survived = torch.where(has_crash, first, torch.full_like(first, horizon))
        late = horizon - survived

        def _finish(raw: torch.Tensor) -> torch.Tensor:
            return -1 * (raw + torch.max(raw) * 10 * late)

        return _finish(drive), _finish(optimize)

    def build_observation(self, state: np.ndarray) -> dict[str, torch.Tensor]:
            """Actor input for the current raw state. Shared implementation with the learner."""
            return build_actor_observation(
                state=state,
                memory=self.memory,
                wall_geometry=self.wall_geometry,
                episode_id=self._episode_id,
            )

    def _sample_mpc_actions(self, nominal: torch.Tensor | None = None) -> torch.Tensor:
        """Correlated action samples around ``nominal``, clipped like ``select_action``."""
        if nominal is None:
            nominal = self.action_buffer
        rho = min(max(float(Settings.MPC_ACTION_CORRELATION), 0.0), 0.999)
        noise_std = torch.tensor([0.08, 1.5]) / torch.as_tensor(self.action_denormalization_array)
        eps = torch.randn(Settings.MPC_NUM_OF_TRAJECTORIES, Settings.MPC_TRAJECTORY_LENGTH, 2)
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
        return xy, heading, speed, drift

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
        self._waypoint_updates_held = True
        self._waypoint_update_close_since = None

    def _note_reward(self, reward_mean: torch.Tensor) -> None:
        """Track the softmax-weighted reward. A sharp drop holds waypoint updates.

        The value is the negation of a cost, so a smaller EMA is a worse plan. The first
        steps are skipped: the EMA is still leaving its initial value.
        """
        previous = float(self._reward_EMA)
        self._reward_EMA = 0.8 * self._reward_EMA + 0.2 * reward_mean
        if self.episode_timesteps < 100:
            return
        drop = previous - float(self._reward_EMA)
        if drop > float(Settings.MPC_WAYPOINT_REWARD_DROP) * abs(previous):
            if not getattr(self, "_reward_drop_warned", False):
                self._reward_drop_warned = True
                self._log_info(
                    "[HirarchicalPlanner] Warning: reward EMA fell from "
                    f"{previous:.4f} to {float(self._reward_EMA):.4f}. Staged waypoint updates discarded."
                )
            self._hold_waypoint_updates()
            self._update_recover_mode(bool(getattr(self, "_waypoint_off_raceline", False)))

    def _update_recover_mode(self, off_track: bool) -> None:
        """True while the car is off the raceline or a reward-EMA drop is still in effect."""
        self.recover_mode = bool(off_track or getattr(self, "_reward_drop_warned", False))

    def _refresh_recovery(self, state: np.ndarray) -> None:
        """Set recovery from the current pose, before this step chooses a plan."""
        self._ensure_waypoints()
        if not hasattr(self, "_waypoint_xy"):
            return
        car_xy = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        nearest_distance = float((self._waypoint_xy - car_xy).norm(dim=-1).min())
        self.off_track = nearest_distance > float(Settings.MPC_WAYPOINT_UPDATE_DISTANCE)
        self._update_recover_mode(self.off_track)

    def _waypoint_update_accepted(self, nearest_distance: float) -> bool:
        """Refuse a new waypoint while the car is off the raceline, and for a fixed time after it returns.

        Going farther than ``MPC_WAYPOINT_UPDATE_DISTANCE`` turns the hold on and drops every
        staged update. The resume clock stays unset while the car is away, including during a
        hold that the reward EMA started. It starts once the car is close, and leaving clears
        it, so the full delay has to elapse while the car stays close.
        """
        self.off_track = nearest_distance > float(Settings.MPC_WAYPOINT_UPDATE_DISTANCE)
        if self.off_track:
            if not getattr(self, "_waypoint_off_raceline", False):
                self._waypoint_off_raceline = True
                self._log_info(
                    "[HirarchicalPlanner] Warning: car left the raceline "
                    f"({nearest_distance:.3f} m from the nearest waypoint). Staged waypoint updates discarded."
                )
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

    def _update_from_driven_plan(self, state: np.ndarray, pacejka: torch.Tensor) -> None:
        """Stage the car's own plan at the waypoint ``k`` steps ahead, and commit it once passed."""
        if not hasattr(self, "waypoints"):
            return
        path = pacejka[0] if pacejka.ndim == 3 else pacejka
        nearest = int(self.nearest_waypoint_index(state).reshape(-1)[0])
        car_xy = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        nearest_distance = float((self._waypoint_xy - car_xy).norm(dim=-1).min())
        k = max(1, int(Settings.MPC_WAYPOINT_AHEAD))
        target = (nearest + k) % self._waypoints_n
        accepted = self._waypoint_update_accepted(nearest_distance)
        on_line = None
        if self.episode_timesteps >= 100 and accepted:
            on_line = self._state_on_orthogonal_line(path, target)
        if on_line is not None:
            xy, heading, speed, drift = on_line
            row = self.waypoints[target].clone()
            row[WP_X_IDX] = xy[0]
            row[WP_Y_IDX] = xy[1]
            row[WP_HEADING_IDX] = heading
            row[WP_TARGET_SPEED_IDX] = speed
            row[WP_TARGET_DRIFT_IDX] = drift
            self.waypoints_update_staged[target] = row
            self._wp_staged[target] = True
        self._commit_staged_waypoints(nearest)
        self._draw_driven_plan(path)

    def _commit_staged_waypoints(self, nearest: int) -> None:
        """Copy a staged waypoint onto the raceline once the car has driven past it."""
        if not hasattr(self, "_wp_staged"):
            return
        lag = max(1, int(Settings.MPC_WAYPOINT_COMMIT_LAG))
        gamma = float(Settings.MPC_WAYPOINT_UPDTE_EMA)
        n = self._waypoints_n
        for idx in torch.nonzero(self._wp_staged).flatten().tolist():
            passed = (int(nearest) - int(idx)) % n
            # passed > n/2 means the waypoint is still ahead; the difference wrapped the long way.
            if not lag <= passed <= n // 2:
                continue
            current = self.waypoints[idx].clone()
            staged = self.waypoints_update_staged[idx]
            self.waypoints[idx, WP_X_IDX] = (1.0 - gamma) * current[WP_X_IDX] + gamma * staged[WP_X_IDX]
            self.waypoints[idx, WP_Y_IDX] = (1.0 - gamma) * current[WP_Y_IDX] + gamma * staged[WP_Y_IDX]
            self.waypoints[idx, WP_TARGET_SPEED_IDX] = (
                (1.0 - gamma) * current[WP_TARGET_SPEED_IDX] + gamma * staged[WP_TARGET_SPEED_IDX]
            )
            self.waypoints[idx, WP_TARGET_DRIFT_IDX] = (
                (1.0 - gamma) * current[WP_TARGET_DRIFT_IDX] + gamma * staged[WP_TARGET_DRIFT_IDX]
            )
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
        gamma = float(Settings.MPC_WAYPOINT_UPDTE_EMA)
        n = self._waypoints_n
        for idx in torch.nonzero(self._wp_action_staged).flatten().tolist():
            passed = (int(nearest) - int(idx)) % n
            if not lag <= passed <= n // 2:
                continue
            current = self._wp_actions[idx]
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
        scene.set_trajectories(
            "hwm.waypoint_arrows", arrows, color=(220, 40, 40), width=1.5, stroke=True,
            max_trajectories=int(arrows.shape[0]), max_points_per_trajectory=8,
        )

    def _clear_waypoint_display(self) -> None:
        """Drop the HWM arrows so the normal waypoint dots are drawn again."""
        render = getattr(self, "render_utils", None)
        scene = None if render is None else getattr(render, "scene", None)
        if scene is None:
            return
        scene.set_trajectories("hwm.waypoint_arrows", None)
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

        with torch.no_grad():
            self._refresh_recovery(state)
            # Recovery samples around the previous plan. Stored waypoint sequences are only
            # used once the car is back out of recovery.
            nominal = None if self.recover_mode else self._interpolated_waypoint_actions(state)
            actions = self._sample_mpc_actions(nominal)
            if Settings.RUN_MPC_WITH_PACEJKA:
                trajectories = self.rollout_pacejka(state, actions)
            else:
                trajectories, n_soft, n_soft_normalized = self.rollout_lld(state, actions)
            nearest_waypoint_idx = self.nearest_waypoint_index(state)
            trajectry_waypoin_matching = self.rollout_waypoint_states(trajectories, nearest_waypoint_idx)
            drive_rewards, opt_rewards = self.MPC_reward(
                trajectories, trajectry_waypoin_matching, nearest_waypoint_idx
            )
            weights = torch.softmax(drive_rewards / Settings.MPC_TEMPERATURE, dim=0)
            reward_mean = torch.sum(drive_rewards * weights, dim=0)
            self._note_reward(reward_mean)
           # if not self.total_timesteps % 12: print(self._reward_EMA)
            action_seq = (weights[:, None, None] * actions).sum(dim=0)
            # Next step samples around this plan with its first command dropped and its last command repeated.
            self.action_buffer = torch.cat((action_seq[1:], action_seq[-1:]), dim=0)
            pacejka = self.rollout_pacejka(state, action_seq)
            self._draw_mpc_trajectories(trajectories, weights, pacejka)
            opt_weights = torch.softmax(opt_rewards / Settings.MPC_TEMPERATURE, dim=0)
            opt_seq = (opt_weights[:, None, None] * actions).sum(dim=0)
            opt_pacejka = self.rollout_pacejka(state, opt_seq)
            self._update_from_driven_plan(state, opt_pacejka)
            self._record_waypoint_action(state, opt_seq.detach())
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
        """One Pacejka control step, compiled once for ``MPC_NUM_OF_TRAJECTORIES`` states."""
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
            self._pacejka_batch_size = max(int(Settings.MPC_NUM_OF_TRAJECTORIES), 1)
        return self._pacejka_batch

    def rollout_pacejka(self, state, actions, use_control_delay: bool = True):
        """states (B, N+1, 10) for the same actions. Applies the command from 2 control steps ago. For testing the Planner with ground truth model in simulation."""
        import jax
        import jax.numpy as jnp
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        batch, horizon, _ = actions.shape
        delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))  # 2
        if use_control_delay:
            rows, _ = self.memory.recent(self._episode_id, delay)
            pending = np.broadcast_to(rows[:, int(Settings.HWM_STATE_DIM):].numpy(), (batch, delay, 2)).copy()
        else:
            pending = np.zeros((batch, 0, 2), dtype=np.float32)
        s = np.broadcast_to(np.asarray(state, dtype=np.float32), (batch, int(Settings.HWM_STATE_DIM))).copy()
        denorm = self.action_denormalization_array
        cap = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
        step = self._pacejka_batch_step()
        width = int(self._pacejka_batch_size)
        if batch > width:
            parts = []
            state_arr = np.asarray(state, dtype=np.float32)
            for start in range(0, batch, width):
                state_i = state if state_arr.ndim == 1 else state_arr[start:start + width]
                parts.append(self.rollout_pacejka(state_i, actions[start:start + width]))
            return torch.cat(parts, dim=0)
        if batch < width:
            s = np.concatenate((s, np.repeat(s[:1], width - batch, axis=0)), axis=0)
            actions = np.concatenate((actions, np.repeat(actions[:1], width - batch, axis=0)), axis=0)
            pending = np.concatenate((pending, np.repeat(pending[:1], width - batch, axis=0)), axis=0)
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
        if not self._can_stream():
            return
        self.client.send_lap_time(lap_time, mean_distance, std_distance, max_distance)

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

"""Hierarchical planner: sampling MPC on the Pacejka model, seeded from a waypoint table it improves while driving.

The control step is ``select_action(state)``. The commands go out in network units: ``±1`` is
``±HWM_ACTION_DENORM`` on the car.
"""

from __future__ import annotations

import collections
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from Control_Toolkit_ASF.Controllers import template_planner
from TrainingLite.hwm.profiling import PROF
from TrainingLite.hwm.shared.speed_profile import grip_speed_profile as _grip_speed_profile
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from utilities.test_track import command as test_track_command
from utilities.test_track import enabled as test_track_enabled

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:  # keep the sim process single-threaded for torch; may already be fixed by earlier imports
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class HirarchicalPlanner(template_planner):
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

        self.wall_geometry = WallGeometry()
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
        self.autonomous_driving = True

        self._controller_observation: Optional[Dict[str, Any]] = None

        self.control_index = 0
        self.total_timesteps = 0
        self.episode_timesteps = 0
        self._episode_id = 0
        # The commands issued this episode that the car has not applied yet (CONTROL_DELAY), oldest first.
        self._issued: collections.deque[np.ndarray] = collections.deque(maxlen=self._control_delay_steps())

        self.action_buffer = torch.zeros(Settings.MPC_TRAJECTORY_LENGTH, 2)
        self.former_nearest_waypoint = None
        # Turns on when the reward EMA drops, and keeps the car in recovery.
        self._reward_drop_warned = False
        self.recover_mode = False
        self.off_track = False
        self._reset_stall()
        self._wp_off_track_count = None
        self._wp_off_track_marked = None
        self._completed_laps = 0
        self._wp_speed_cap_applied = False
        self._last_opt_wp = None
        self._reward_EMA = 1
        self._mean_state_seed: torch.Tensor | None = None
        self._grid_crashes = 0
        self._grid_crash_reasons: dict[str, int] = {}
        self._grid_laps: list[float] = []
        self.reset()
        self._setup_profiling()

    def _setup_profiling(self) -> None:
        """``Settings.HWM_PROFILE``: time every part of the control step (see ``profiling.py``)."""
        if not getattr(Settings, "HWM_PROFILE", False):
            return
        PROF.enable(
            output_path=str(getattr(Settings, "HWM_PROFILE_FILE", "hwm_profile")),
            skip_first=int(getattr(Settings, "HWM_PROFILE_SKIP_STEPS", 30)),
            dynamics="pacejka",
            samples=int(Settings.MPC_NUM_OF_TRAJECTORIES),
            horizon=int(Settings.MPC_TRAJECTORY_LENGTH),
        )
        PROF.instrument(self, [
            "_refresh_recovery", "_interpolated_waypoint_actions",
            "_sample_mpc_actions", "rollout_dynamics", "rollout_pacejka",
            "nearest_waypoint_index", "rollout_waypoint_states", "MPC_reward", "_matched_waypoint_costs", "_recovery_cost",
            "_backtrack_cost", "_note_reward", "_draw_mpc_trajectories", "_weighted_state_trajectory",
            "_publish_waypoint_display", "_refresh_waypoint_paths",
            "_seed_from_states", "_ensure_waypoints", "_apply_normalized_action",
            "_record_step", "get_car_state", "_update_stall", "_update_recover_mode",
            "select_action",
        ])
        PROF.instrument(self.wall_geometry, [
            "in_wall", "distance_at", "crash_probability",
        ], prefix="wall.")

    def reset(self):
        self.control_index = 0
        self.episode_timesteps = 0
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        self.action_buffer = torch.zeros(Settings.MPC_TRAJECTORY_LENGTH, 2)
        self._last_opt_wp = None
        self._mean_state_seed = None
        self._action_buffer_valid = False
        self._fill_previous_state = None
        self._driven = None
        self._reset_stall()

    def _matchable_waypoints(self, pos: torch.Tensor) -> torch.Tensor | None:
        """``(B, N)`` bool for positions ``(B, 2)``: the waypoints each one may be matched to, or ``None`` for all.

        The straight nearest waypoint can be on another part of the track behind a wall, and a
        car matched to it heads into that wall. Only the waypoints around the one that the
        shortest path through free space leads to are allowed.
        """
        if not getattr(self, "_track_field", False):
            return None
        _, label, _ = self.wall_geometry.track_field_at(pos)
        n = self._waypoints_n
        gap = (torch.arange(n)[None, :] - label[:, None]).remainder(n)
        allowed = torch.minimum(gap, n - gap) <= Settings.HWM_TRACK_WINDOW
        return allowed | (label < 0)[:, None]

    def _raceline_segment(self, state: np.ndarray) -> tuple[int, float]:
        """``(start index, progress)`` of the raceline segment the car is on."""
        pos = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        xy = self._waypoint_xy
        segment = torch.roll(xy, shifts=-1, dims=0) - xy
        length2 = segment.pow(2).sum(dim=-1).clamp_min(1e-12)
        progress = ((pos - xy) * segment).sum(dim=-1) / length2
        progress = progress.clamp(0.0, 1.0)
        projection = xy + progress[:, None] * segment
        distance2 = (pos - projection).pow(2).sum(dim=-1)
        allowed = self._matchable_waypoints(pos[None])
        if allowed is not None:
            distance2 = distance2.masked_fill(~allowed[0], float("inf"))
        segment_index = int(distance2.argmin())
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

    def _seed_from_states(self, path: torch.Tensor) -> torch.Tensor:
        """Seed ``(H, 2)`` for the states ``path`` ``(H, state_dim)``, the state each action starts from.

        Action ``k`` is the mix, by position between them, of the first stored actions of the two
        raceline waypoints around state ``k``. Where neither waypoint has a stored sequence, it is
        action ``k`` of the running plan (``action_buffer``).
        """
        horizon = int(path.shape[0])
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
        # A stored action fits a car at the waypoint's stored speed. Off that speed it is corrected
        # toward it, or a car that slowed where the stored plans brake would keep braking there.
        speed = self.waypoints[:, WP_TARGET_SPEED_IDX] * float(Settings.MPC_DRIVE_TARGET_SPEED_FACTOR)
        target = (1.0 - alpha) * speed[start] + alpha * speed[end]
        target = torch.where(has_start & ~has_end, speed[start], target)
        target = torch.where(has_end & ~has_start, speed[end], target)
        gain = float(Settings.MPC_SEED_SPEED_GAIN) / float(self.action_denormalization_array[1])
        mixed = mixed.clone()
        mixed[:, 1] = mixed[:, 1] + gain * (target - path[:, LINEAR_VEL_X_IDX])
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
        if not self._wp_speed_cap_applied and Settings.HWM_INITIAL_SPEED_GRIP_SHARE is not None:
            accel = float(Settings.HWM_INITIAL_SPEED_GRIP_SHARE) * self._car_friction_accel()
            profile = _grip_speed_profile(self._waypoint_xy, accel, float(Settings.HWM_INITIAL_SPEED_MAX))
            self.waypoints[:, WP_TARGET_SPEED_IDX] = torch.minimum(self.waypoints[:, WP_TARGET_SPEED_IDX], profile)
            self._wp_speed_cap_applied = True
        elif not self._wp_speed_cap_applied and Settings.HWM_LIMIT_INITIAL_WP_SPEED:
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
        self._track_field = not Settings.BLANK_MAP
        if self._track_field:
            self.wall_geometry.build_track_field(self._wp_anchor_xy, clearance=0.5 * self.wall_geometry.car_width)
        step = (self._waypoint_xy[1:] - self._waypoint_xy[:-1]).norm(dim=-1)
        positive = step[step > 1e-4]
        self._waypoint_spacing = float(positive.median()) if positive.numel() else 0.1

        horizon = int(Settings.MPC_TRAJECTORY_LENGTH)
        self._wp_actions = torch.zeros((self._waypoints_n, horizon, 2), dtype=torch.float32)
        self._wp_action_known = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))
        # Control-delay queue at each waypoint's stored state: the commands issued before it that are still to be applied.
        self._wp_pending = torch.zeros((self._waypoints_n, delay, 2), dtype=torch.float32)
        # What each window update replaced: (control step, table row, action sequence, was it known, delay queue).
        self._wp_window_hist = [[] for _ in range(self._waypoints_n)]
        self._wp_fragile_at = torch.full((self._waypoints_n,), -1, dtype=torch.long)  # last control step marked fragile
        self._wp_off_track_count = torch.zeros((self._waypoints_n,), dtype=torch.float32)
        self._wp_path_states = None
        self._wp_window_accepted = 0
        self._wp_window_tried = 0
        self._wp_vehicle_state_set = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        self._wp_off_track_marked = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        self._car_waypoint = None
        self.last_wp_distance = np.inf
        self._write_raceline()
        self._publish_waypoint_display()

    def _write_raceline(self) -> None:
        """Copy the shared raceline fields back. The drift column stays on the HWM copy."""
        source = getattr(self, "_raceline_np", None)
        if source is None or not hasattr(self, "waypoints"):
            return
        width = min(int(source.shape[1]), WP_TARGET_DRIFT_IDX)
        source[:, :width] = self.waypoints[:, :width].detach().cpu().numpy()

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
        allowed = self._matchable_waypoints(pos)
        if allowed is not None:
            distance2 = distance2.masked_fill(~allowed, float("inf"))
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


    def _matched_waypoint_count(self, closest: int, steps: int) -> int:
        """How many waypoints from ``closest`` on ``rollout_waypoint_states`` matches for rollouts of ``steps`` states."""
        duration = max(steps - 1, 0) * float(Settings.TIMESTEP_CONTROL)
        ahead = (int(closest) + torch.arange(self._waypoints_n)) % self._waypoints_n
        xy = self._waypoint_xy[ahead]
        length = (xy[1:] - xy[:-1]).norm(dim=-1)
        speed = self.waypoints[ahead[:-1], WP_TARGET_SPEED_IDX].clamp(min=0.5)
        covered = int((torch.cumsum(length / speed, dim=0) <= Settings.HWM_MATCH_HORIZON_SHARE * duration).sum())
        return min(max(covered + 1, 1), self._waypoints_n)

    def rollout_waypoint_states(self, states: torch.Tensor, closest_index: int | torch.Tensor) -> torch.Tensor:
        """For each rollout, which of its states is closest to each upcoming waypoint.

        ``states`` is ``(B, T, state_dim)``. ``closest_index`` is the first raceline index
        ahead of the start, one integer or ``(B,)``. Waypoint ``j`` of a trajectory is
        ``closest_index + j`` along the raceline, including that first waypoint ahead.

        How many waypoints: as many as the stored target speeds cover in ``HWM_MATCH_HORIZON_SHARE`` of the
        rollout duration ``(T - 1) * TIMESTEP_CONTROL``, plus one, the same for every rollout (counted
        from the first rollout's start). A rollout that is slower than the targets therefore still has
        to reach them: counted from its own speed, a car that stands still would be scored on the next
        waypoint only. Less than the whole duration, so a rollout at the target speeds reaches the last
        one before the horizon ends and is not pushed to overspeed for it.
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
        covered = self._matched_waypoint_count(int(closest[0]), steps) - 1 if batch else 0
        counts = torch.full((batch,), covered + 1, dtype=torch.long).clamp(min=1, max=self._waypoints_n)
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
        dist, in_range, out_np = dist.numpy(), in_range.numpy(), out.numpy()
        taken = np.zeros((batch, steps), dtype=bool)
        rows = np.arange(batch)
        for j in range(look):
            column = np.where(taken, np.float32(np.inf), dist[:, :, j])
            best = column.argmin(axis=1)
            reachable = in_range[:, j] & np.isfinite(column[rows, best])
            out_np[reachable, j] = best[reachable]
            taken[rows[reachable], best[reachable]] = True
        return out

    def _matched_waypoint_costs(
        self, states: torch.Tensor, nearest_waypoint_indices: torch.Tensor, nearest_waypoint: int,
    ) -> torch.Tensor:
        """``(B,)`` drive cost, lower is better, where each waypoint is matched to its closest rollout state.

        Batched over the rollouts. Rollout ``i`` uses its ``count_i`` matched waypoints (``nearest_waypoint_indices``
        is NaN padded); padded slots are masked out of every sum, max and count.
        """
        batch = int(states.shape[0])
        drive = torch.zeros(batch)
        if self.off_track:
            return drive
        width = int(nearest_waypoint_indices.shape[1])
        if width == 0:
            return drive
        start = int(torch.as_tensor(nearest_waypoint).reshape(-1)[0])
        n = self._waypoints_n
        wp_index = (torch.arange(width) + start) % n  # (J,)
        valid = ~nearest_waypoint_indices.isnan()
        count = valid.sum(dim=1)  # (B,)

        PROF.begin("batched_waypoint_reward")
        indices = torch.where(valid, nearest_waypoint_indices, torch.zeros_like(nearest_waypoint_indices)).long()
        matched = states[torch.arange(batch)[:, None], indices]  # (B, J, D)
        local_waypoints = self.waypoints[wp_index]  # (J, 14)
        zero = matched.new_zeros(())

        position = matched[..., [POSE_X_IDX, POSE_Y_IDX]] - local_waypoints[None, :, [WP_X_IDX, WP_Y_IDX]]
        distance = torch.where(valid, torch.linalg.norm(position, dim=-1), zero).sum(dim=1)
        heading_error = matched[..., POSE_THETA_IDX] - local_waypoints[None, :, WP_HEADING_IDX]
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        heading = torch.where(valid, heading_error.abs(), zero)
        speed = matched[..., LINEAR_VEL_X_IDX]
        target_speed = local_waypoints[None, :, WP_TARGET_SPEED_IDX]
        drift_error = torch.where(
            valid, (matched[..., LINEAR_VEL_Y_IDX] - local_waypoints[None, :, WP_TARGET_DRIFT_IDX]).abs(), zero
        ).sum(dim=1)
        drive_target = target_speed * float(Settings.MPC_DRIVE_TARGET_SPEED_FACTOR)
        drive_shortfall = torch.where(valid, (drive_target - speed).clamp(min=0), zero)
        drive_overspeed = torch.where(valid, (speed - drive_target).clamp(min=0), zero)

        index_f = indices.float()
        fast_driving = torch.where(valid, index_f, torch.full_like(index_f, float("-inf"))).max(dim=1).values
        counts = count.clamp(min=1).to(speed.dtype)
        drive_cost = (
            10 * distance + 2 * heading.sum(dim=1) + drive_shortfall.sum(dim=1)
            + drive_overspeed.sum(dim=1) * float(Settings.MPC_DRIVE_OVERSPEED_WEIGHT)
            + 0.25 * drift_error + float(Settings.MPC_DRIVE_ARRIVAL_WEIGHT) * fast_driving
        ) / counts

        # Every predicted state counts: with only the matched ones, a rollout that backs away from the
        # waypoints is matched by its first, still slow states and its reversing goes unpunished.
        min_speed = states[:, 1:, LINEAR_VEL_X_IDX].min(dim=1).values
        reverse = torch.where(min_speed < 0, 100 * min_speed.abs(), torch.zeros_like(min_speed))
        drive_cost = drive_cost + reverse
        PROF.end()
        return torch.where(count > 0, drive_cost, drive)

    def MPC_reward(
        self, states: torch.Tensor, nearest_waypoint_indices: torch.Tensor, nearest_waypoint: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drive reward ``(B,)``, split into the normal part and the crash part.

        The caller adds the two parts. The crash part is left out of the reward EMA.
        """
        start = int(torch.as_tensor(nearest_waypoint).reshape(-1)[0])
        drive = self._matched_waypoint_costs(states, nearest_waypoint_indices, nearest_waypoint)
        if self.off_track:
            drive = self._recovery_cost(states, start)
        if self._stall_phase == 1:
            drive = self._backtrack_cost(states)

        PROF.begin("crash_and_wall_terms")
        poses = states[:, :, [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
        crashed = self.wall_geometry.in_wall(poses)
        has_crash = crashed.any(dim=1)
        first = crashed.float().argmax(dim=1)
        horizon = crashed.shape[1]
        # Steps until impact. A clean rollout survives the whole horizon. An earlier hit costs more,
        # so a plan that follows the corner and clips late beats one that drives into the wall now.
        survived = torch.where(has_crash, first, torch.full_like(first, horizon))
        late = horizon - survived
        # At least 1 per step of the horizon lost: with every cost near 0 a crash must still cost something.
        drive_crash = -(drive.abs().max().clamp_min(1.0) * 10 * late)
        drive_reward = -drive
        clearance_weight = float(Settings.MPC_DRIVE_WALL_WEIGHT)
        if clearance_weight > 0.0:
            # A plan that passes a wall by a hair is a crash after a few cm of tracking error. Each state
            # whose closest body corner is within MPC_DRIVE_WALL_CLEARANCE of a wall costs up to 1, reached
            # at the crash margin.
            corner = self.wall_geometry.distance_at(self.wall_geometry.corners(poses)).amin(dim=-1).to(drive_reward)
            start = float(Settings.MPC_DRIVE_WALL_CLEARANCE)
            span = max(start - float(Settings.HWM_MIN_BORDER_DISTANCE), 1e-3)
            closeness = ((start - corner) / span).clamp(0.0, 1.0).sum(dim=1)
            drive_reward = drive_reward - clearance_weight * closeness
        PROF.end()
        return drive_reward, drive_crash

    def _jax_rewards_usable(self) -> bool:
        """``MPC_reward_jax`` covers this configuration: matched-waypoint costs on a map, on the CPU."""
        return (
            Settings.HWM_JAX_REWARDS and not Settings.BLANK_MAP
            and self.wall_geometry.device.type == "cpu"
        )

    def _jax_wall_inputs(self) -> tuple[Any, Any, Any]:
        """The wall distance map and the body corners as device arrays, made once per map."""
        geometry = self.wall_geometry
        cached = getattr(self, "_jax_walls", None)
        if cached is None or cached[0] is not geometry.wall_distance:
            import jax

            cached = (
                geometry.wall_distance,
                jax.device_put(geometry.wall_distance.numpy()),
                geometry._corner_x.numpy().astype(np.float64),
                geometry._corner_y.numpy().astype(np.float64),
            )
            self._jax_walls = cached
        return cached[1:]

    def MPC_reward_jax(
        self, states: torch.Tensor, nearest_waypoint: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``MPC_reward`` of ``rollout_waypoint_states`` matches, with the per-rollout part compiled
        (``drive_costs_jax``) and run in chunks on ``HWM_ROLLOUT_THREADS`` threads."""
        from jax.experimental import enable_x64

        from TrainingLite.hwm.drive_costs_jax import drive_costs, squared_distances

        self._ensure_waypoints()
        geometry = self.wall_geometry
        batch, steps = int(states.shape[0]), int(states.shape[1])
        start = int(torch.as_tensor(nearest_waypoint).reshape(-1)[0])
        n = self._waypoints_n
        count = self._matched_waypoint_count(start, steps)
        # Padded to a multiple of 16 waypoints, so only a few widths are ever compiled.
        width = -(-count // 16) * 16
        idx = (start + torch.arange(width)) % n
        match_xy = self._waypoint_xy[idx].numpy()
        rows = self.waypoints[idx][:, [WP_X_IDX, WP_Y_IDX, WP_HEADING_IDX, WP_TARGET_SPEED_IDX, WP_TARGET_DRIFT_IDX]].numpy()
        grid, corner_x, corner_y = self._jax_wall_inputs()
        clearance = float(Settings.MPC_DRIVE_WALL_CLEARANCE)
        consts = (
            float(Settings.MPC_DRIVE_TARGET_SPEED_FACTOR), float(Settings.MPC_DRIVE_OVERSPEED_WEIGHT),
            float(Settings.MPC_DRIVE_ARRIVAL_WEIGHT), float(Settings.HWM_MIN_BORDER_DISTANCE), clearance,
            max(clearance - float(Settings.HWM_MIN_BORDER_DISTANCE), 1e-3),
            geometry.origin_x, geometry.origin_y, geometry.origin_cos, geometry.origin_sin, geometry.resolution,
            float(geometry.map_width), float(geometry.map_height),
        )
        s = np.ascontiguousarray(states.numpy(), dtype=np.float32)

        def costs(lo: int, hi: int, size: int) -> list[np.ndarray]:
            pad = np.minimum(np.arange(lo, lo + size), hi - 1)
            chunk = s[pad]
            dist = squared_distances(chunk, match_xy)
            with enable_x64():
                out = drive_costs(chunk, dist, rows, np.int32(count), grid, corner_x, corner_y, consts)
                return [np.asarray(o)[: hi - lo] for o in out]

        with PROF.section("jax_drive_costs"):
            threads = int(Settings.HWM_ROLLOUT_THREADS)
            if threads <= 1 or batch < 64:
                parts = [costs(0, batch, batch)]
            else:
                per_thread = -(-batch // threads)
                size = -(-per_thread // 8) * 8
                chunks = [(lo, min(lo + size, batch), size) for lo in range(0, batch, size)]
                parts = list(self._rollout_pool(threads).map(lambda chunk: costs(*chunk), chunks))
            drive_m, active, late, closeness = (
                torch.from_numpy(np.concatenate(column)) for column in zip(*parts)
            )

        if not self.off_track:
            drive = torch.where(active, drive_m, torch.zeros(batch))
        else:
            drive = self._recovery_cost(states, start)
        if self._stall_phase == 1:
            drive = self._backtrack_cost(states)
        drive_crash = -(drive.abs().max().clamp_min(1.0) * 10 * late.to(drive.dtype))
        drive_reward = -drive
        clearance_weight = float(Settings.MPC_DRIVE_WALL_WEIGHT)
        if clearance_weight > 0.0:
            drive_reward = drive_reward - clearance_weight * closeness
        return drive_reward, drive_crash


    def _recovery_cost(self, states: torch.Tensor, start: int) -> torch.Tensor:
        """``(B,)`` cost, lower is better, for rollouts ``(B, T, state_dim)`` of a car that is off the raceline.

        Each state is matched to its nearest raceline waypoint. The wanted heading is the line's
        heading turned toward the line in proportion to the cross-track error, so a car that is
        far off is asked to head back to the line and a car on it is asked to follow it. The end of
        the rollout counts twice, since where the car ends up is what the next step starts from.
        """
        cfg = Settings.HWM_RECOVERY
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
        if getattr(self, "_track_field", False):
            # Behind a wall the way back is not toward the nearest waypoint: by as much as the path
            # around the walls is longer, the wanted heading turns to the direction that path takes.
            path, _, direction = self.wall_geometry.track_field_at(pos)
            detour = ((self._detour(pos).to(near_distance) - Settings.HWM_DETOUR_START) / Settings.HWM_DETOUR_SPAN).clamp(0.0, 1.0)
            turn = torch.atan2(direction[..., 1], direction[..., 0]).to(wanted) - wanted
            wanted = wanted + detour * torch.atan2(torch.sin(turn), torch.cos(turn))
            near_distance = torch.where(detour > 0, torch.maximum(near_distance, path.to(near_distance)), near_distance)
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

    def _fit_horizon(self, seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Crop or repeat the last action so ``seq`` ``(H, 2)`` has ``horizon`` steps."""
        seq = seq.reshape(-1, 2)
        n = int(seq.shape[0])
        horizon = int(horizon)
        if n == horizon:
            return seq
        if n > horizon:
            return seq[:horizon].contiguous()
        return torch.cat((seq, seq[-1:].expand(horizon - n, -1)), dim=0)

    def _sample_mpc_actions(
        self, nominal: torch.Tensor | None = None, count: int | None = None, noise_scale: float = 1.0,
        correlation: float | None = None,
    ) -> torch.Tensor:
        """Correlated action samples around ``nominal``, clipped like ``select_action``."""
        if nominal is None:
            nominal = self.action_buffer
        PROF.begin("noise_draw")
        rho = float(Settings.MPC_ACTION_CORRELATION if correlation is None else correlation)
        rho = min(max(rho, 0.0), 0.999)
        noise_std = noise_scale * torch.tensor(Settings.MPC_DRIVE_NOISE, dtype=torch.float32) / torch.as_tensor(self.action_denormalization_array)
        if self.off_track or self._stall_phase == 1:
            noise_std = noise_std * float(Settings.HWM_RECOVERY["noise_scale"])
        eps = torch.randn(int(count or Settings.MPC_NUM_OF_TRAJECTORIES), int(nominal.shape[0]), 2)
        actions = nominal + eps * noise_std
        PROF.end()
        if rho > 0.0:
            with PROF.section("temporal_correlation_loop"):
                fresh = (1.0 - rho * rho) ** 0.5
                drift = ((1.0 - rho) * nominal).numpy()
                kick = (fresh * noise_std * eps).numpy().transpose(1, 0, 2).copy()
                out = actions.numpy()
                steps = out.transpose(1, 0, 2).copy()
                for t in range(1, int(eps.shape[1])):
                    step = steps[t]
                    np.multiply(steps[t - 1], np.float32(rho), out=step)
                    np.add(step, drift[t], out=step)
                    np.add(step, kick[t], out=step)
                out[:] = steps.transpose(1, 0, 2)
        with PROF.section("clamp"):
            brake_limit, accel_limit = self._translational_limits()
            actions[..., 0].clamp_(-1.0, 1.0)
            actions[..., 1].clamp_(brake_limit, accel_limit)
        return actions

    def _translational_limits(self) -> tuple[float, float]:
        """Network-unit range of the translational command. With ``HWM_CLIP_TO_CAR_LIMITS``, the commands the car
        does not saturate: its acceleration and grip limits divided by its brake and accel multipliers."""
        limits = getattr(self, "_translational_range", None)
        if limits is None:
            scale = float(self.action_denormalization_array[1])
            cap = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
            brake, accel = -cap, cap
            if Settings.HWM_CLIP_TO_CAR_LIMITS:
                from utilities.car_files.vehicle_parameters import VehicleParameters
                params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
                grip = self._car_friction_accel()
                brake = max(brake, max(float(params.a_min), -grip) / float(getattr(params, "brake_multiplier", 1.0)))
                accel = min(accel, min(float(params.a_max), grip) / float(getattr(params, "accel_multiplier", 1.0)))
            limits = (brake / scale, accel / scale)
            self._translational_range = limits
        return limits

    def _car_brake_multiplier(self) -> float:
        """Factor the car applies to negative acceleration commands."""
        if getattr(self, "_brake_multiplier", None) is None:
            from utilities.car_files.vehicle_parameters import VehicleParameters
            self._brake_multiplier = float(getattr(VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE), "brake_multiplier", 1.0))
        return self._brake_multiplier

    def _car_friction_accel(self) -> float:
        """mu * g of the car parameters: the most the tires can brake."""
        if getattr(self, "_friction_accel", None) is None:
            from utilities.car_files.vehicle_parameters import VehicleParameters
            params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
            self._friction_accel = float(params.mu) * float(getattr(params, "g", 9.81))
        return self._friction_accel

    def _with_fallback_samples(self, actions: torch.Tensor, nominal: torch.Tensor | None, speed: float) -> torch.Tensor:
        """Replace the last samples with fixed alternatives the noise around ``nominal`` rarely reaches.

        The previous plan as it is and ``nominal`` with its steering scaled down. A car slower than
        ``HWM_CREEP_SPEED`` also pulls away and backs off at fixed steering.
        """
        if nominal is None:
            nominal = self.action_buffer
        horizon = int(nominal.shape[0])
        self.action_buffer = self._fit_horizon(self.action_buffer, horizon)
        fixed = [self.action_buffer]
        # A seed recorded in a different state can steer harder than the car needs now and curl into the inside wall.
        for scale in Settings.HWM_FALLBACK_STEER_SCALES:
            fixed.append(torch.stack((nominal[:, 0] * scale, nominal[:, 1]), dim=1))
        if speed < Settings.HWM_CREEP_SPEED:
            # A slow car may need a sharp pull-away the seed and its noise never try: gentle throttle at fixed steering.
            steps = int(np.ceil((Settings.HWM_CREEP_SPEED - speed) / (Settings.HWM_CREEP_ACCEL * float(Settings.TIMESTEP_CONTROL))))
            throttle = torch.zeros(horizon)
            throttle[: min(steps, horizon)] = Settings.HWM_CREEP_ACCEL / float(self.action_denormalization_array[1])
            for steer in (-1.0, -0.5, 0.5, 1.0):
                fixed.append(torch.stack((torch.full((horizon,), steer), throttle), dim=1))
            # With its nose at a wall, backing off at full lock is the only way out.
            braking = Settings.HWM_CREEP_ACCEL * self._car_brake_multiplier()
            steps = int(np.ceil((max(speed, 0.0) + Settings.HWM_BACK_OFF_SPEED) / (braking * float(Settings.TIMESTEP_CONTROL))))
            reverse = torch.zeros(horizon)
            reverse[: min(steps, horizon)] = -Settings.HWM_CREEP_ACCEL / float(self.action_denormalization_array[1])
            for steer in (-1.0, 1.0):
                fixed.append(torch.stack((torch.full((horizon,), steer), reverse), dim=1))
        fixed = torch.stack(fixed, dim=0)
        if int(fixed.shape[0]) >= int(actions.shape[0]):
            return actions
        actions = actions.clone()
        actions[-int(fixed.shape[0]):] = fixed
        return actions

    def _revert_nearby_waypoints(self) -> None:
        """Put the waypoints around the car back to where the car last got through (``_revert_window_updates``)."""
        if not hasattr(self, "_wp_window_hist") or self._car_waypoint is None:
            return
        if self._permanently_frozen():
            return
        if not getattr(self, "_seen_on_track", True):
            return  # spawned off the raceline and not on it yet: recovery runs, but no waypoint is reset
        self._revert_window_updates()

    def _revert_window_updates(self) -> None:
        """Put the waypoints from ``MPC_WAYPOINT_REVERT_BEHIND`` behind the car to ``MPC_WAYPOINT_REVERT_AHEAD``
        ahead of it back to what they were one lap ago, when the car last got through there.

        All of them go back to the same time, so they still join up. Undoing a fixed number of
        updates per waypoint would mix states of different updates, since some waypoints were
        written more often than others. Nothing happens before the first lap is finished.
        """
        lap_steps = getattr(self, "_last_lap_steps", None)
        if lap_steps is None:
            return
        since = int(self.total_timesteps) - int(lap_steps)
        n = self._waypoints_n
        center = int(self._car_waypoint)
        behind = max(0, int(Settings.MPC_WAYPOINT_REVERT_BEHIND))
        ahead = max(0, int(Settings.MPC_WAYPOINT_REVERT_AHEAD))
        for offset in range(-behind, ahead + 1):
            idx = (center + offset) % n
            history = self._wp_window_hist[idx]
            entry = None
            while history and history[-1][0] > since:
                entry = history.pop()
            if entry is None:
                continue
            _, row, actions, known, pending = entry
            self.waypoints[idx] = row
            self._wp_actions[idx] = actions
            self._wp_action_known[idx] = known
            self._wp_pending[idx] = pending
        self._write_raceline()
        self._publish_waypoint_display()

    def _permanently_frozen(self) -> bool:
        """True once ``HWM_PERMANENT_FREEZE_LAPS`` laps passed without a faster lap: the table never changes again."""
        return bool(getattr(self, "_wp_permanently_frozen", False))

    def _waypoint_snapshot(self) -> dict:
        """Everything a lap-time reset has to put back: the targets, the action sequences and their bookkeeping."""
        return {key: values.clone() for key, values in self._waypoint_snapshot_refs().items()}

    def _capture_driven_table(self) -> None:
        """Keep every waypoint's entry as it was when the car last reached it.

        Window updates rewrite waypoints behind the car, so the last waypoints of a lap change while the lap
        is driven. At the end of a lap this copy is the table that lap drove on.
        """
        car = self._car_waypoint
        if car is None or not hasattr(self, "_wp_actions"):
            return
        n = self._waypoints_n
        table = getattr(self, "_driven_table", None)
        if table is None or int(table["waypoints"].shape[0]) != n:
            self._driven_table, self._driven_last = self._waypoint_snapshot(), car
            return
        ahead = (car - int(self._driven_last)) % n
        if ahead == 0:
            return
        # A jump of more than a quarter lap (reset, backtracking) takes only the waypoint the car is at.
        idx = (int(self._driven_last) + 1 + torch.arange(ahead)) % n if ahead <= n // 4 else torch.tensor([car])
        self._driven_last = car
        live = self._waypoint_snapshot_refs()
        for key, values in live.items():
            table[key][idx] = values[idx]

    def _waypoint_snapshot_refs(self) -> dict:
        return {
            "waypoints": self.waypoints, "actions": self._wp_actions, "known": self._wp_action_known,
            "vehicle_state_set": self._wp_vehicle_state_set, "pending": self._wp_pending,
        }

    def _best_lap_snapshot(self, start: dict) -> dict:
        """The table to keep for a lap that just ended: the one it drove, else (before one was captured) the one it started on."""
        driven = getattr(self, "_driven_table", None)
        if driven is None:
            return start
        return {key: values.clone() for key, values in driven.items()}

    def _restore_waypoint_snapshot(self, snapshot: dict) -> None:
        self.waypoints.copy_(snapshot["waypoints"])
        self._wp_actions.copy_(snapshot["actions"])
        self._wp_action_known.copy_(snapshot["known"])
        self._wp_vehicle_state_set.copy_(snapshot["vehicle_state_set"])
        if "pending" in snapshot:
            self._wp_pending.copy_(snapshot["pending"])
        for history in self._wp_window_hist:
            history.clear()
        self._write_raceline()
        self._publish_waypoint_display()

    def _update_permanent_freeze(self, lap_time: float) -> None:
        """After ``HWM_PERMANENT_FREEZE_LAPS`` finished laps without a faster one, put the table back to the one
        the fastest lap was driven on and freeze it for good: no window updates, staging, filling or
        further resets or reverts. 0: off.
        """
        patience = int(getattr(Settings, "HWM_PERMANENT_FREEZE_LAPS", 0) or 0)
        if self._permanently_frozen():
            self._check_frozen_table()
            return
        if patience <= 0 or not hasattr(self, "_wp_actions"):
            return
        start = getattr(self, "_freeze_lap_start", None)
        self._freeze_lap_start = self._waypoint_snapshot()  # the table the next lap starts on
        best = getattr(self, "_freeze_best_lap", None)
        if start is not None and (best is None or float(lap_time) < best):
            table = self._best_lap_snapshot(start)
            self._freeze_best_lap, self._freeze_best_table, self._freeze_laps_since_best = float(lap_time), table, 0
            self._freeze_best_changed = self._count_changed(start, table)
            return
        if best is None:
            return  # the first lap has no start table: nothing to freeze to yet
        self._freeze_laps_since_best = int(getattr(self, "_freeze_laps_since_best", 0)) + 1
        if self._freeze_laps_since_best < patience:
            return
        self._restore_waypoint_snapshot(self._freeze_best_table)
        self._wp_permanently_frozen = True
        self._frozen_table = self._waypoint_snapshot()
        self._log_info(
            f"[HirarchicalPlanner] No lap faster than {best:.3f}s for {patience} laps: "
            "waypoints put back to that lap's and frozen permanently "
            f"({int(getattr(self, '_freeze_best_changed', 0))} waypoints changed during that lap)"
        )

    def _check_frozen_table(self) -> None:
        """Log any waypoint of the frozen table that changed."""
        frozen = getattr(self, "_frozen_table", None)
        if frozen is None:
            return
        changed = torch.zeros((self._waypoints_n,), dtype=torch.bool)
        for key, live in self._waypoint_snapshot_refs().items():
            diff = live != frozen[key]
            if live.is_floating_point():
                diff &= ~(live.isnan() & frozen[key].isnan())
            changed |= diff.reshape(self._waypoints_n, -1).any(dim=1)
        if bool(changed.any()):
            self._log_info(
                f"[HirarchicalPlanner] Warning: {int(changed.sum())} waypoints of the frozen table changed: "
                f"{changed.nonzero().flatten().tolist()[:20]}"
            )
            self._frozen_table = self._waypoint_snapshot()

    @staticmethod
    def _count_changed(a: dict, b: dict) -> int:
        """Waypoints whose target row or action sequence differ between two snapshots."""
        rows = (a["waypoints"] != b["waypoints"]).any(dim=1)
        actions = (a["actions"] != b["actions"]).flatten(1).any(dim=1)
        return int((rows | actions).sum())

    def _update_best_lap_table(self, lap_time: float) -> None:
        """After ``HWM_BEST_LAP_RESET_LAPS`` laps without a faster one, put the table back to the one the fastest
        lap started with. Window updates go on from there."""
        patience = Settings.HWM_BEST_LAP_RESET_LAPS
        if patience is None or not hasattr(self, "_wp_actions"):
            return
        if self._permanently_frozen():
            return
        start = getattr(self, "_lap_start_table", None)
        self._lap_start_table = self._waypoint_snapshot()
        best = getattr(self, "_best_lap_time", None)
        if start is not None and (best is None or lap_time < best):
            self._best_lap_time, self._best_lap_table, self._laps_since_best = lap_time, self._best_lap_snapshot(start), 0
            return
        if best is None:
            return
        self._laps_since_best = int(getattr(self, "_laps_since_best", 0)) + 1
        if self._laps_since_best < int(patience):
            return
        table = self._best_lap_table
        self.waypoints.copy_(table["waypoints"])
        self._wp_actions.copy_(table["actions"])
        self._wp_action_known.copy_(table["known"])
        self._wp_vehicle_state_set.copy_(table["vehicle_state_set"])
        self._wp_pending.copy_(table["pending"])
        for history in self._wp_window_hist:
            history.clear()
        self._lap_start_table = table
        self._laps_since_best = 0
        self._write_raceline()
        self._publish_waypoint_display()
        self._log_info(f"[HirarchicalPlanner] No lap faster than {best:.3f}s for {patience} laps: table put back to that lap's")

    def _note_reward(self, reward_mean: torch.Tensor) -> None:
        """Track the softmax-weighted reward. A reward far under its EMA reverts the waypoints around the car.

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
                    f"{previous:.4f} by more than {float(Settings.MPC_WAYPOINT_REWARD_DROP):.0%}."
                )
            self._revert_nearby_waypoints()
            self._update_recover_mode(False)

    def _update_recover_mode(self, off_track: bool) -> None:
        """True while the car is off the raceline or a reward-EMA drop is still in effect."""
        self.recover_mode = bool(
            off_track or getattr(self, "_reward_drop_warned", False) or getattr(self, "_stall_phase", 0) == 1
        )

    def _reset_stall(self) -> None:
        # Also the per-episode reset. A car that spawns off the raceline does not reset any waypoint
        # until it has been on the raceline once; recovery itself still runs.
        self._seen_on_track = False
        self._was_off_track = True
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
                self._start_backtrack(car_xy, car_waypoint, "car barely moved")
        self._stalled = self._stall_phase == 2

    def _start_backtrack(self, car_xy: torch.Tensor, car_waypoint: int, reason: str) -> None:
        """Enter phase 1 (backtracking), or phase 2 directly when ``MPC_STALL_REVERSE_STEPS`` is 0."""
        self._pose_history.clear()
        if int(Settings.MPC_STALL_REVERSE_STEPS) > 0:
            self._stall_phase = 1
            self._reverse_left = int(Settings.MPC_STALL_REVERSE_STEPS)
            self._backtrack_wp = (int(car_waypoint) - int(Settings.MPC_STALL_BACKTRACK_WAYPOINTS)) % self._waypoints_n
            self._log_info(
                f"[HirarchicalPlanner] Warning: {reason}. "
                f"Driving to the waypoint {int(Settings.MPC_STALL_BACKTRACK_WAYPOINTS)} behind it."
            )
        else:
            self._stall_phase = 2
            self._stall_anchor = car_xy.clone()
        self._stalled = self._stall_phase == 2


    def _blocked(self, speed: float, trajectories: torch.Tensor, crash: torch.Tensor, best: int) -> bool:
        """A slow car whose best rollout crashes or does not get going.

        Either every way forward hits a wall, or getting around it scores worse than standing
        still. Waiting does not change either, and the mean of the samples creeps into the wall.
        """
        if speed >= Settings.HWM_CREEP_SPEED:
            return False
        if float(crash[best]) < 0:
            return True
        forward = float(trajectories[best, 1:, LINEAR_VEL_X_IDX].clamp(min=0).sum()) * float(Settings.TIMESTEP_CONTROL)
        return forward < Settings.HWM_BLOCKED_DISTANCE

    def _safe_mean(
        self, state: np.ndarray, actions: torch.Tensor, mean: torch.Tensor, weights: torch.Tensor,
        total: torch.Tensor, crash: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The best sample instead of the weighted mean when the mean crashes and the best does not.

        With many samples in the wall, the weight splits between plans that get through in different
        ways, often going on and braking. Their average is neither and can hit the wall.
        """
        best = int(total.argmax())
        if float(crash[best]) < 0 or not bool((crash < 0).any()):
            return mean, weights
        path = self.rollout_dynamics(state, mean)
        if not bool(self.wall_geometry.in_wall(path[..., [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]).any()):
            return mean, weights
        return actions[best].clone(), torch.nn.functional.one_hot(torch.tensor(best), int(weights.shape[0])).to(weights)

    def _backtrack_cost(self, states: torch.Tensor) -> torch.Tensor:
        """``(B,)`` cost, lower is better: how far each rollout is from the backtrack waypoint.

        Forward or reverse, whichever gets there. The end of the rollout counts twice. Slip, yaw
        rate and overspeed are penalized like in recovery, so it does not spin to get there.
        """
        cfg = Settings.HWM_RECOVERY
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
        """Distance from the car to the raceline segment between the waypoint behind and the one ahead.

        The way around a wall when that is longer, so a car next to the line but behind a wall is off it.
        """
        start, progress = self._raceline_segment(state)
        end = (start + 1) % self._waypoints_n
        pos = torch.as_tensor(state, dtype=torch.float32).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]]
        projection = self._waypoint_xy[start] + progress * (self._waypoint_xy[end] - self._waypoint_xy[start])
        distance = float((pos - projection).norm())
        if float(self._detour(pos)) > Settings.HWM_DETOUR_START:
            path, _, _ = self.wall_geometry.track_field_at(pos)
            return max(distance, float(path))
        return distance

    def _detour(self, pos: torch.Tensor) -> torch.Tensor:
        """``(...)`` how much longer the free-space path from positions ``(..., 2)`` to the raceline is than the straight line, in m.

        Both are measured to the waypoint as loaded, which the path ends at: an update may move
        the waypoints sideways, but the paths are those of the loaded line.
        """
        if not getattr(self, "_track_field", False):
            return torch.zeros(pos.shape[:-1])
        path, label, _ = self.wall_geometry.track_field_at(pos)
        straight = (pos - self._wp_anchor_xy[label.clamp(min=0)]).norm(dim=-1)
        return torch.where(label >= 0, path.to(straight) - straight, torch.zeros_like(straight))

    def _off_track_affects_waypoints(self) -> bool:
        """False for ``HWM_OFF_TRACK_WAYPOINT_GRACE`` seconds after an episode starts.

        The car still goes into recovery. The waypoint table does not: no off-track count,
        no revert, and updates are not held for having left the line.
        """
        grace = float(Settings.HWM_OFF_TRACK_WAYPOINT_GRACE)
        if grace <= 0.0:
            return True
        return float(self.episode_timesteps) * float(Settings.TIMESTEP_CONTROL) >= grace

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
        off = self.off_track and self._off_track_affects_waypoints()
        if (off or reward_drop) and not bool(self._wp_off_track_marked[car]):
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
        distance = (self._waypoint_xy - car_xy).norm(dim=-1)
        allowed = self._matchable_waypoints(car_xy[None])
        if allowed is not None:
            distance = distance.masked_fill(~allowed[0], float("inf"))
        self._car_waypoint = int(distance.argmin())
        self._capture_driven_table()
        on_track = line_distance <= float(Settings.MPC_WAYPOINT_UPDATE_DISTANCE)
        limit = float(Settings.MPC_RECOVERY_HEADING_ERROR)
        if on_track and limit > 0.0:
            # On the line but pointing away from it: the waypoint costs alone would drive it into a wall.
            heading = float(np.asarray(state).reshape(-1)[POSE_THETA_IDX])
            error = heading - float(self.waypoints[self._car_waypoint, WP_HEADING_IDX])
            on_track = abs(np.arctan2(np.sin(error), np.cos(error))) <= limit
        if on_track:
            self._seen_on_track = True
        self._update_stall(car_xy, self._car_waypoint)
        self.off_track = (not on_track) or self._stalled
        # Recovery follows the real pose. The waypoint table ignores the opening of the episode.
        affects_waypoints = self.off_track and self._off_track_affects_waypoints()
        if affects_waypoints and not self._was_off_track:
            # The window updates made the waypoints the car could not follow: walk the ones around it back.
            self._revert_nearby_waypoints()
        self._was_off_track = affects_waypoints
        self._update_recover_mode(self.off_track)
        self._note_off_track_pass()

    def _weighted_state_trajectory(self, trajectories: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Mean state rollout. Same weighted sum as the black plan line, over the whole state.

        Heading is rebuilt from the averaged sin and cos, so the reward does not see a wrapped angle.
        """
        mean = torch.sum(weights[:, None, None] * trajectories, dim=0)
        mean[:, POSE_THETA_IDX] = torch.atan2(mean[:, POSE_THETA_SIN_IDX], mean[:, POSE_THETA_COS_IDX])
        return mean

    def _waypoint_known(self, idx: torch.Tensor) -> torch.Tensor:
        return self._wp_action_known[idx] & self._wp_vehicle_state_set[idx]

    def _line_crossings(
        self, states: torch.Tensor, idx: torch.Tensor, reach: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """When, and in which state, each rollout first crosses the fixed cross-track line of each waypoint.

        ``states`` is ``(B, T, state_dim)``, ``idx`` ``(J,)``. Returns ``times`` ``(B, J)`` in control steps
        from the first state, NaN where the rollout does not cross within ``reach`` (default
        ``HWM_WINDOW_LINE_REACH``) of the stored point, and the states ``(B, J, state_dim)`` interpolated at the crossing.
        """
        anchor, tangent, ortho = self._wp_anchor_xy[idx], self._wp_tangent[idx], self._wp_ortho[idx]
        stored = ((self._waypoint_xy[idx] - anchor) * ortho).sum(dim=-1)  # (J,)
        rel = states[:, :, None, :][..., [POSE_X_IDX, POSE_Y_IDX]] - anchor[None, None]  # (B, T, J, 2)
        along = (rel * tangent).sum(dim=-1)
        reach = float(Settings.HWM_WINDOW_LINE_REACH) if reach is None else float(reach)
        near = ((rel * ortho).sum(dim=-1) - stored).abs() < reach
        cross = (along[:, :-1] < 0) & (along[:, 1:] >= 0) & near[:, :-1] & near[:, 1:]  # (B, T - 1, J)
        crossed = cross.any(dim=1)
        k = cross.float().argmax(dim=1)  # (B, J), first crossing
        a0 = along.gather(1, k[:, None]).squeeze(1)
        a1 = along.gather(1, (k + 1)[:, None]).squeeze(1)
        frac = (-a0 / (a1 - a0).clamp_min(1e-9)).clamp(0.0, 1.0)
        times = torch.where(crossed, k.to(frac.dtype) + frac, torch.full_like(frac, float("nan")))
        rows = torch.arange(int(states.shape[0]))[:, None]
        before, after = states[rows, k], states[rows, k + 1]  # (B, J, state_dim)
        at = before + frac[..., None] * (after - before)
        at[..., POSE_THETA_IDX] = torch.atan2(at[..., POSE_THETA_SIN_IDX], at[..., POSE_THETA_COS_IDX])
        return times, at

    def _window_error(self, at: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """``(B, J)`` error (``HWM_WINDOW_ERROR``) of the crossing states ``at`` ``(B, J, state_dim)`` against the states stored at ``idx``."""
        wp = self.waypoints[idx]
        weight = Settings.HWM_WINDOW_ERROR
        offset = at[..., [POSE_X_IDX, POSE_Y_IDX]] - wp[None, :, [WP_X_IDX, WP_Y_IDX]]
        lateral = (offset * self._wp_ortho[idx][None]).sum(dim=-1).abs()
        heading = at[..., POSE_THETA_IDX] - wp[None, :, WP_HEADING_IDX]
        heading = torch.atan2(torch.sin(heading), torch.cos(heading)).abs()
        speed = (at[..., LINEAR_VEL_X_IDX] - wp[None, :, WP_TARGET_SPEED_IDX]).abs()
        drift = (at[..., LINEAR_VEL_Y_IDX] - wp[None, :, WP_TARGET_DRIFT_IDX]).abs()
        yaw_rate = (at[..., ANGULAR_VEL_Z_IDX] - wp[None, :, WP_TARGET_YAW_RATE_IDX]).abs()
        return (
            weight["lateral"] * lateral + weight["heading"] * heading + weight["speed"] * speed
            + weight["drift"] * drift + weight["yaw_rate"] * yaw_rate
        )

    @staticmethod
    def _timeline_at(values: torch.Tensor, start: float, times: torch.Tensor) -> torch.Tensor:
        """``values`` ``(L, 2)`` issued at times ``start, start + 1, ...``, linearly interpolated at ``times`` ``(Q,)``, held at both ends."""
        last = int(values.shape[0]) - 1
        values, times = values.numpy(), times.numpy()
        pos = np.minimum(np.maximum(times - times.dtype.type(start), 0.0), float(last))
        lo = np.minimum(np.floor(pos).astype(np.int64), last)
        hi = np.minimum(lo + 1, last)
        frac = (pos - lo.astype(pos.dtype))[:, None]
        return torch.from_numpy(values[lo] * (1.0 - frac) + values[hi] * frac)

    def _write_waypoint_state(self, idx: int, state: torch.Tensor) -> None:
        """Waypoint ``idx`` takes the car state ``state``, its position put on the waypoint's cross-track line."""
        anchor, ortho = self._wp_anchor_xy[idx], self._wp_ortho[idx]
        xy = anchor + ((state[[POSE_X_IDX, POSE_Y_IDX]] - anchor) * ortho).sum() * ortho
        self.waypoints[idx, WP_X_IDX] = xy[0]
        self.waypoints[idx, WP_Y_IDX] = xy[1]
        self.waypoints[idx, WP_HEADING_IDX] = state[POSE_THETA_IDX]
        self.waypoints[idx, WP_TARGET_SPEED_IDX] = state[LINEAR_VEL_X_IDX].clamp(min=0.0)
        self.waypoints[idx, WP_TARGET_DRIFT_IDX] = state[LINEAR_VEL_Y_IDX]
        self.waypoints[idx, WP_TARGET_STEER_IDX] = state[STEERING_ANGLE_IDX]
        self.waypoints[idx, WP_TARGET_YAW_RATE_IDX] = state[ANGULAR_VEL_Z_IDX]
        self.waypoints[idx, WP_TARGET_SLIP_IDX] = state[SLIP_ANGLE_IDX]
        self._wp_vehicle_state_set[idx] = True

    def _fill_passed_waypoints(self, state: np.ndarray, action_seq: torch.Tensor) -> None:
        """A waypoint with nothing stored takes what the car had when it crossed the waypoint's line since the last step.

        That is the car state at the crossing, the plan it drives now and the commands still in its
        control delay. Stored waypoints are left to the window updates. Below ``HWM_FILL_MIN_SPEED`` nothing
        is stored: the position noise of a car standing on a line crosses it back and forth.
        """
        current = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        previous = getattr(self, "_fill_previous_state", None)
        self._fill_previous_state = current
        if previous is None or self._car_waypoint is None or self._stall_phase == 1:
            return
        if self._permanently_frozen():
            return
        if self.recover_mode:
            return
        if float(current[0, LINEAR_VEL_X_IDX]) < Settings.HWM_FILL_MIN_SPEED:
            return
        idx = (int(self._car_waypoint) + torch.arange(-3, 4)) % self._waypoints_n
        path = torch.cat((previous, current))[None]
        idx = idx[~self._waypoint_known(idx)]
        if not idx.numel():
            return
        times, at = self._line_crossings(path, idx)
        pending = self._issued_actions()
        filled = False
        for j in torch.nonzero(~torch.isnan(times[0])).flatten().tolist():
            k = int(idx[j])
            self._write_waypoint_state(k, at[0, j])
            self._wp_actions[k] = self._fit_horizon(action_seq.reshape(-1, 2), int(self._wp_actions.shape[1]))
            self._wp_pending[k] = pending
            self._wp_action_known[k] = True
            filled = True
        if filled:
            self._write_raceline()
            self._publish_waypoint_display()


    def _record_driven(self, state: np.ndarray, action: torch.Tensor) -> None:
        """Remember the command issued now, and when the car last crossed the lines of the waypoints around it.

        Times are in control steps of this counter; the command of step ``s`` is issued at the state of step ``s``.
        """
        driven = getattr(self, "_driven", None)
        if driven is None or int(driven["cross"].shape[0]) != self._waypoints_n:
            driven = {"step": -1, "first": 0, "actions": [], "previous": None,
                      "cross": torch.full((self._waypoints_n,), float("nan"))}
            self._driven = driven
        driven["step"] += 1
        driven["actions"].append(action.reshape(2).to(torch.float32).cpu())
        if len(driven["actions"]) > Settings.HWM_DRIVEN_MEMORY:
            driven["actions"].pop(0)
            driven["first"] += 1
        current = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        previous, driven["previous"] = driven["previous"], current
        if previous is None or self._car_waypoint is None or self.recover_mode:
            return
        idx = (int(self._car_waypoint) + torch.arange(-3, 4)) % self._waypoints_n
        times, _ = self._line_crossings(torch.cat((previous, current))[None], idx)
        hit = ~torch.isnan(times[0])
        driven["cross"][idx[hit]] = float(driven["step"] - 1) + times[0, hit]

    def _driven_candidate(self, w: int, horizon: int) -> torch.Tensor | None:
        """The commands the car issued from its last crossing of waypoint ``w``'s line on, ``(H, 2)``, or None."""
        driven = getattr(self, "_driven", None)
        if driven is None or int(driven["cross"].shape[0]) != self._waypoints_n:
            return None
        crossed = float(driven["cross"][w])
        if np.isnan(crossed) or crossed < driven["first"] or crossed + horizon - 1 > driven["step"]:
            return None
        timeline = torch.stack(driven["actions"])
        return self._timeline_at(timeline, float(driven["first"]), crossed + torch.arange(horizon, dtype=torch.float32))

    def _window_jobs(self) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """This step's window update, as a list of none or one."""
        job = self._window_job()
        return [] if job is None else [job]

    def _window_job(self) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """This step's window update, or None.

        The window is the ``MPC_WAYPOINT_AHEAD_SPAN`` waypoints after waypoint ``w``, which is
        ``MPC_WAYPOINT_WINDOW_LAG`` behind the car. Returns ``(w, state stored at w, its delay queue,
        samples (K, H, 2))``. Sample 0 is the sequence stored at ``w``, the others are noise around it.
        """
        if self.recover_mode or self._car_waypoint is None:
            return None
        if self._permanently_frozen():
            return None
        n = self._waypoints_n
        span = max(1, int(Settings.MPC_WAYPOINT_AHEAD_SPAN))
        w = (int(self._car_waypoint) - int(Settings.MPC_WAYPOINT_WINDOW_LAG)) % n
        if not bool(self._waypoint_known((w + torch.arange(span + 2)) % n).all()):
            return None
        stored = self._wp_actions[w]
        samples = self._sample_mpc_actions(
            stored, count=int(Settings.MPC_WAYPOINT_WINDOW_SAMPLES),
            noise_scale=float(Settings.MPC_WAYPOINT_WINDOW_NOISE),
            correlation=float(Settings.MPC_WAYPOINT_WINDOW_CORRELATION),
        )
        samples[0] = stored
        if int(samples.shape[0]) > 1:
            driven = self._driven_candidate(w, int(samples.shape[1]))
            if driven is not None:
                samples[1] = driven
        shifted = self._shifted_sequences(stored)
        if shifted is not None and int(samples.shape[0]) > 2 + int(shifted.shape[0]):
            samples[2:2 + int(shifted.shape[0])] = shifted
        return w, self._waypoint_states(torch.tensor([w]))[0], self._wp_pending[w].clone(), samples

    def _shifted_sequences(self, stored: torch.Tensor) -> torch.Tensor | None:
        """``(K, H, 2)`` copies of ``stored`` with one channel played earlier or later by the steps in
        ``HWM_WINDOW_SHIFT_STEPS``: braking, accelerating or turning in sooner or later, which noise
        around the sequence rarely finds."""
        shifts = Settings.HWM_WINDOW_SHIFT_STEPS
        scales = Settings.HWM_WINDOW_STEER_SCALES
        if not shifts and not scales:
            return None
        horizon = int(stored.shape[0])
        out = []
        for channel in (1, 0):
            for shift in shifts:
                index = (torch.arange(horizon) + int(shift)).clamp(0, horizon - 1)
                seq = stored.clone()
                seq[:, channel] = stored[index, channel]
                out.append(seq)
        for scale in scales:
            seq = stored.clone()
            seq[:, 0] = (stored[:, 0] * float(scale)).clamp(-1.0, 1.0)
            out.append(seq)
        return torch.stack(out)

    def _mark_fragile(self, drive_crash: torch.Tensor) -> None:
        if self.off_track or self.recover_mode or self._car_waypoint is None:
            return
        if float((drive_crash < 0).float().mean()) < Settings.HWM_FRAGILE_SHARE:
            return
        idx = (int(self._car_waypoint) + torch.arange(Settings.HWM_FRAGILE_AHEAD + 1)) % self._waypoints_n
        self._wp_fragile_at[idx] = int(self.total_timesteps)

    def _fragile(self, idx: torch.Tensor) -> bool:
        lap_steps = getattr(self, "_last_lap_steps", None)
        if lap_steps is None:
            return False
        marked = self._wp_fragile_at[idx]
        recent = (marked >= 0) & (int(self.total_timesteps) - marked <= Settings.HWM_FRAGILE_MEMORY * float(lap_steps))
        return bool(recent.any())


    def _rollout_with_window(
        self, state: np.ndarray, actions: torch.Tensor, windows: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Driving samples from the car and each window's samples from its waypoint state, in one rollout call."""
        batch = int(actions.shape[0])
        if int(actions.shape[1]) != int(windows[0][3].shape[1]):
            # Free driving uses MPC_FREE_TRAJECTORY_LENGTH. A window sample stays at MPC_TRAJECTORY_LENGTH.
            trajectories = self.rollout_dynamics(state, actions)
            return trajectories, [self._rollout_window(window) for window in windows]
        drive_pending = self._issued_actions()
        starts = [torch.as_tensor(state, dtype=torch.float32).reshape(1, -1).expand(batch, -1)]
        queues = [drive_pending[None].expand(batch, -1, -1)]
        sequences = [actions]
        sizes = []
        for _, start, pending, samples in windows:
            count = int(samples.shape[0])
            starts.append(start[None].expand(count, -1))
            queues.append(pending[None].expand(int(samples.shape[0]), -1, -1))
            sequences.append(samples)
            sizes.append(int(samples.shape[0]))
        out = self.rollout_dynamics(
            torch.cat(starts).numpy(), torch.cat(sequences).numpy(), pending_actions=torch.cat(queues).numpy()
        )
        return out[:batch], list(torch.split(out[batch:], sizes))

    def _rollout_window(self, window: tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """States of one window's samples."""
        _, start, pending, samples = window
        starts = start[None].expand(int(samples.shape[0]), -1)
        queue = pending[None].expand(int(samples.shape[0]), -1, -1)
        return self.rollout_dynamics(starts.numpy(), samples.numpy(), pending_actions=queue.numpy())

    def _window_update(self, window: tuple[int, torch.Tensor, torch.Tensor, torch.Tensor], states: torch.Tensor) -> None:
        """Accept the best window sample that reaches the waypoints after the window sooner and matches them as well.

        Compared are the window and the waypoints after it that the stored sequence (sample 0) crosses
        in a row, at most ``MPC_WAYPOINT_WINDOW_DOWNSTREAM``. A sample qualifies if it crosses all of
        them in order and keeps ``MPC_WAYPOINT_WINDOW_MARGIN`` from the walls until one step past the
        last. The error is the largest one at the waypoints after the window.

        While the stored sequence's error is within ``MPC_WAYPOINT_WINDOW_TOLERANCE``, a qualifying
        sample must also stay within it and cross the last waypoint earlier than the stored sequence
        (which counts as too slow if it hits a wall). Of those, the one with the least arrival time plus
        ``HWM_WINDOW_ERROR_COST`` times its error is taken. Otherwise the stored plan no longer leads to the
        states stored after the window, and the update is a repair: the sample with the least error is
        taken if it is below the stored sequence's, however slow.

        Then the window waypoints take that sample's state at their lines, and ``w`` and the window
        waypoints take its commands from their crossing on, followed by the plan stored at the last
        compared waypoint from its crossing on. The commands between two control steps are interpolated.
        Only the window waypoints are written, the handover is at the first waypoint after the window, and the
        error is taken at every waypoint after the window.
        """
        w, _, pending, actions = window
        n = self._waypoints_n
        span = max(1, int(Settings.MPC_WAYPOINT_AHEAD_SPAN))
        down = max(1, int(Settings.MPC_WAYPOINT_WINDOW_DOWNSTREAM))
        idx = (w + 1 + torch.arange(span + down)) % n
        times, at = self._line_crossings(states, idx)
        usable = (~torch.isnan(times[0])) & self._waypoint_known(idx)
        count = int(usable.long().cumprod(dim=0).sum())
        if count < span + 1:
            return
        idx, times, at = idx[:count], times[:, :count], at[:, :count]
        self._wp_window_tried += 1
        crossed = (~torch.isnan(times)).all(dim=1) & (times[:, 1:] > times[:, :-1]).all(dim=1)
        # The window waypoints take the sample's states, so only the ones after it, which it must rejoin, count.
        error = self._window_error(at[:, span:], idx[span:]).amax(dim=1)
        arrival = times[:, -1]
        horizon = int(states.shape[1])
        poses = states[..., [POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX]]
        corner_distance = self.wall_geometry.distance_at(self.wall_geometry.corners(poses)).amin(dim=-1).cpu().to(times.dtype)
        until = torch.nan_to_num(arrival, nan=float(horizon)).ceil() + 1
        compared = torch.arange(horizon)[None] <= until[:, None]
        clearance = torch.where(compared, corner_distance, torch.full_like(corner_distance, float("inf"))).amin(dim=1)
        # MPC_WAYPOINT_WINDOW_MARGIN from the walls, or as much as the stored sequence keeps where it is
        # closer (a narrow passage), but never under MPC_WAYPOINT_WINDOW_MIN_MARGIN.
        margin = min(float(Settings.MPC_WAYPOINT_WINDOW_MARGIN), float(clearance[0]))
        margin = max(margin, float(Settings.MPC_WAYPOINT_WINDOW_MIN_MARGIN))
        valid = crossed & (clearance >= margin)
        # A slow car pivoting at a large slip angle is a state the driven car does not get into again.
        slip = torch.atan2(states[..., LINEAR_VEL_Y_IDX].abs(), states[..., LINEAR_VEL_X_IDX].clamp(min=Settings.HWM_CREEP_SPEED))
        valid = valid & (torch.where(compared, slip, torch.zeros_like(slip)).amax(dim=1) <= Settings.HWM_WINDOW_MAX_SLIP)
        stored_arrival = float(arrival[0]) if bool(valid[0]) else float("inf")
        tolerance = float(Settings.MPC_WAYPOINT_WINDOW_TOLERANCE)
        repair = float(error[0]) > tolerance
        if repair:
            eligible = valid & (error < float(error[0]))
            score = error
        elif self._fragile(idx):
            eligible = valid & (error <= tolerance) & (clearance >= float(clearance[0]) + Settings.HWM_FRAGILE_CLEARANCE_GAIN)
            score = arrival + float(Settings.HWM_WINDOW_ERROR_COST) * error
        else:
            eligible = valid & (error <= tolerance) & (arrival < stored_arrival)
            score = arrival + float(Settings.HWM_WINDOW_ERROR_COST) * error
        eligible[0] = False
        if not bool(eligible.any()):
            return
        best = int(torch.where(eligible, score, torch.full_like(score, float("inf"))).argmin())

        horizon_steps = int(actions.shape[1])
        delay = int(pending.shape[0])
        sample_timeline = torch.cat((pending, actions[best]))  # issued at -delay .. H - 1
        # The sample's commands are followed until it crosses the handover waypoint, the first one after
        # the window, whose stored sequence (and all stored sequences after it) stays as it is.
        handover = span
        last = int(idx[handover])
        reached = float(times[best, handover])
        stored_timeline = torch.cat((self._wp_pending[last], self._wp_actions[last]))  # issued from reached - delay on

        def commands(at_times: torch.Tensor) -> torch.Tensor:
            own = self._timeline_at(sample_timeline, -delay, at_times)
            downstream = self._timeline_at(stored_timeline, reached - delay, at_times)
            return torch.where((at_times < reached)[:, None], own, downstream)

        steps = torch.arange(horizon_steps, dtype=torch.float32)
        queue = torch.arange(-delay, 0, dtype=torch.float32)
        updates = [(w, commands(steps), None, None)]
        for j in range(span):
            t = float(times[best, j])
            updates.append((int(idx[j]), commands(t + steps), commands(t + queue), at[best, j]))
        for k, seq, queue_k, state_k in updates:
            history = self._wp_window_hist[k]
            history.append((
                int(self.total_timesteps), self.waypoints[k].clone(), self._wp_actions[k].clone(),
                bool(self._wp_action_known[k]), self._wp_pending[k].clone(),
            ))
            del history[: -Settings.HWM_WINDOW_HISTORY]
            self._wp_actions[k] = seq
            if queue_k is not None:
                self._wp_pending[k] = queue_k
                self._write_waypoint_state(k, state_k)
        self._wp_window_accepted += 1
        self._write_raceline()
        self._publish_waypoint_display()

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
        stops = np.array(((0, 190, 0), (240, 220, 0), (255, 140, 0), (210, 30, 20)), dtype=np.float64)
        x = scale.astype(np.float64) * (len(stops) - 1)
        i = np.minimum(x.astype(np.int64), len(stops) - 2)
        f = (x - i)[:, None]
        a, b = stops[i], stops[i + 1]
        colors = np.rint(a + (b - a) * f).astype(np.int64).tolist()
        counts = getattr(self, "_wp_off_track_count", None)
        if counts is not None:
            scene.set_dynamic_points(
                "hwm.waypoint_off_track", xy, color=(235, 235, 235), radius=0.5,
                texts=[str(int(c)) for c in counts.tolist()],
            )
        paths = self._waypoint_paths() if Settings.HWM_SHOW_WAYPOINT_PATHS and not Settings.HWM_LIVE_MODE else None
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
        start = self._waypoint_states(idx)
        actions = self._wp_actions[idx]
        pending = self._wp_pending[idx]
        states = self.rollout_dynamics(start.numpy(), actions.numpy(), pending_actions=pending.numpy())
        self._wp_path_states = states.detach().cpu()

    def _waypoint_states(self, idx: torch.Tensor) -> torch.Tensor:
        """``(K, state_dim)`` car states stored at waypoints ``idx``: position, heading, speed, drift, steering angle, yaw rate and slip angle."""
        wp = self.waypoints[idx]
        start = torch.zeros((int(wp.shape[0]), int(Settings.HWM_STATE_DIM)), dtype=torch.float32)
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
        return start

    def _waypoint_paths(self) -> np.ndarray | None:
        """Path ``(K, T + 1, 2)`` from each waypoint that has a stored action sequence."""
        self._refresh_waypoint_paths()
        states = getattr(self, "_wp_path_states", None)
        if states is None:
            return None
        return states[self._wp_action_known][:, :, [POSE_X_IDX, POSE_Y_IDX]].numpy()

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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """The MPC command for ``state``, in network output units. ``±1`` is ``±action_denorm`` on the car."""
        with torch.no_grad():
            self._refresh_recovery(state)
            # Recovery samples around the previous plan. Stored waypoint sequences are only
            # used once the car is back out of recovery. Without one yet (the first lap) the
            # plan is free too, and rolled out for MPC_FREE_TRAJECTORY_LENGTH. On the priors
            # it is MPC_TRAJECTORY_LENGTH, the same length as a window-update sample.
            nominal = None if self.recover_mode else self._interpolated_waypoint_actions(state)
            horizon = int(Settings.MPC_TRAJECTORY_LENGTH if nominal is not None else Settings.MPC_FREE_TRAJECTORY_LENGTH)
            self.action_buffer = self._fit_horizon(self.action_buffer, horizon)
            if nominal is not None:
                nominal = self._fit_horizon(nominal, horizon)
            if nominal is not None and self._mean_state_seed is not None and int(self._mean_state_seed.shape[0]) == int(nominal.shape[0]):
                # Seed stored at the end of the previous step from its mean states: no rollout now.
                nominal = self._mean_state_seed
            actions = self._sample_mpc_actions(nominal)
            if Settings.MPC_DRIVE_FALLBACK_SAMPLES:
                flat = np.asarray(state).reshape(-1)
                actions = self._with_fallback_samples(actions, nominal, float(flat[LINEAR_VEL_X_IDX]))
            windows = self._window_jobs()
            with PROF.section("rollout_of_samples"):
                if windows:
                    trajectories, window_states = self._rollout_with_window(state, actions, windows)
                else:
                    trajectories = self.rollout_dynamics(state, actions)
            nearest_waypoint_idx = self.nearest_waypoint_index(state)
            if self._jax_rewards_usable():
                drive_rewards, drive_crash = self.MPC_reward_jax(trajectories, nearest_waypoint_idx)
            else:
                matching = self.rollout_waypoint_states(trajectories, nearest_waypoint_idx)
                drive_rewards, drive_crash = self.MPC_reward(trajectories, matching, nearest_waypoint_idx)
            with PROF.section("softmax_weights_and_plan"):
                drive_total = drive_rewards + drive_crash
                weights = torch.softmax(drive_total / Settings.MPC_TEMPERATURE, dim=0)
                reward_mean = torch.sum(drive_rewards * weights, dim=0)
            self._note_reward(reward_mean)
           # if not self.total_timesteps % 12: print(self._reward_EMA)
            with PROF.section("mean_action_sequence"):
                action_seq = (weights[:, None, None] * actions).sum(dim=0)
                if Settings.MPC_SAFE_MEAN:
                    action_seq, weights = self._safe_mean(state, actions, action_seq, weights, drive_total, drive_crash)
                # Next step samples around this plan with its first command dropped and its last command repeated.
                self.action_buffer = torch.cat((action_seq[1:], action_seq[-1:]), dim=0)
                self._action_buffer_valid = True
            if not Settings.HWM_LIVE_MODE:
                with PROF.section("display_rollouts"):
                    with PROF.section("pacejka_of_mean_sequence"):
                        pacejka = self.rollout_pacejka(state, action_seq)
                    self._draw_mpc_trajectories(trajectories, pacejka)
            if self._stall_phase != 1 and Settings.MPC_STALL_ON_BLOCKED and self._blocked(
                float(np.asarray(state).reshape(-1)[LINEAR_VEL_X_IDX]), trajectories, drive_crash,
                int(drive_total.argmax()),
            ):
                car_xy = torch.as_tensor(np.asarray(state).reshape(-1)[[POSE_X_IDX, POSE_Y_IDX]], dtype=torch.float32)
                self._start_backtrack(car_xy, self._car_waypoint, "the best plan does not get going")
            if self._stall_phase == 1:
                # Backtracking: no waypoint staging or recording from this plan.
                self._reverse_left -= 1
                self._mean_state_seed = None
                return action_seq[0].cpu().numpy().astype(np.float32)
            self._mark_fragile(drive_crash)
            with PROF.section("window_update"):
                for window, states_of_window in zip(windows, window_states if windows else []):
                    self._window_update(window, states_of_window)
            self._record_driven(state, action_seq[0].detach())
            self._fill_passed_waypoints(state, action_seq.detach())
            # Next step's seed: the stored first actions at the waypoints around each averaged state
            # (the state the next step's action k starts from is state k + 1 of this plan).
            horizon = int(action_seq.shape[0])
            drive_mean = self._weighted_state_trajectory(trajectories, weights)
            self._mean_state_seed = self._seed_from_states(drive_mean[1 : 1 + horizon].detach())
            return action_seq[0].cpu().numpy().astype(np.float32)

    def _draw_mpc_trajectories(self, trajectories: torch.Tensor, pacejka: torch.Tensor) -> None:
        """MPC samples, and Pacejka driven by the mean action sequence (orange)."""
        if getattr(self, "render_utils", None) is None:
            return
        xy = trajectories[..., [POSE_X_IDX, POSE_Y_IDX]].detach().cpu()
        scene = self.render_utils.scene
        caps = {"max_trajectories": int(xy.shape[0]), "max_points_per_trajectory": int(xy.shape[1])}
        if Settings.MPC_DRAW_SAMPLES:
            scene.set_trajectories("hwm.mpc_samples", xy.numpy(), color=(140, 140, 140), width=1.0, **caps)
        else:
            scene.set_trajectories("hwm.mpc_samples", None)
        pace_xy = pacejka[..., [POSE_X_IDX, POSE_Y_IDX]].detach().cpu().numpy()
        scene.set_trajectories(
            "hwm.mpc_pacejka", pace_xy, color=(230, 90, 0), width=3.0,
            max_trajectories=int(pace_xy.shape[0]), max_points_per_trajectory=int(pace_xy.shape[1]),
        )

    def rollout_dynamics(self, state, actions, pending_actions=None) -> torch.Tensor:
        """States ``(B, N + 1, state_dim)`` from the planner's model. Arguments as ``rollout_pacejka``."""
        return self.rollout_pacejka(state, actions, pending_actions=pending_actions)

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

    def _pacejka_scan_rollout(self):
        """Jitted whole-horizon Pacejka rollout: ``(state (B,D), actions (B,H,2), pending (B,delay,2)) -> (B,H+1,D)``.

        Same per-step math as the former Python loop: push the action into the delay queue, apply the
        oldest entry (denormalized, translational part clipped), one Pacejka control step, then the heading
        angle recomputed from its sin and cos. ``lax.scan`` replaces the loop, so the device is synced once.
        """
        fn = getattr(self, "_pacejka_scan", None)
        if fn is None:
            import jax
            import jax.numpy as jnp

            step = self._pacejka_batch_step()
            denorm = jnp.asarray(np.asarray(self.action_denormalization_array, dtype=np.float32))
            cap = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)

            def run(state, actions, pending):
                def body(carry, action):
                    s, queue = carry
                    queue = jnp.concatenate((queue, action[:, None, :]), axis=1)
                    applied = queue[:, 0] * denorm
                    queue = queue[:, 1:]
                    applied = applied.at[:, 1].set(jnp.clip(applied[:, 1], -cap, cap))
                    s = step(s, applied)
                    s = s.at[:, POSE_THETA_IDX].set(jnp.arctan2(s[:, POSE_THETA_SIN_IDX], s[:, POSE_THETA_COS_IDX]))
                    return (s, queue), s

                _, trajectory = jax.lax.scan(body, (state, pending), jnp.swapaxes(actions, 0, 1))
                return jnp.concatenate((state[:, None], jnp.swapaxes(trajectory, 0, 1)), axis=1)

            fn = jax.jit(run)
            self._pacejka_scan = fn
        return fn

    def rollout_pacejka(self, state, actions, use_control_delay: bool = True, pending_actions=None):
        """states (B, N+1, 10) for the same actions. Applies the command from 2 control steps ago. For testing the Planner with ground truth model in simulation."""
        PROF.begin("setup")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        batch, horizon, _ = actions.shape
        if pending_actions is not None:
            # Control-delay queue given per sequence (B, delay, 2), instead of the car's.
            pending = np.asarray(pending_actions, dtype=np.float32).reshape(batch, -1, 2).copy()
        elif use_control_delay:
            pending = np.broadcast_to(self._issued_actions().numpy(), (batch, self._control_delay_steps(), 2)).copy()
        else:
            pending = np.zeros((batch, 0, 2), dtype=np.float32)
        s = np.broadcast_to(np.asarray(state, dtype=np.float32), (batch, int(Settings.HWM_STATE_DIM))).copy()
        run = self._pacejka_scan_rollout()
        PROF.end()

        def rollout(lo: int, hi: int, rows: int) -> np.ndarray:
            """Sequences ``lo`` to ``hi``, padded with copies of the last one to ``rows`` rows."""
            pad = np.repeat(np.arange(lo, hi), [1] * (hi - lo - 1) + [rows - (hi - lo) + 1])
            out = run(s[pad], actions[pad], pending[pad])
            return np.array(out, dtype=np.float32, copy=True)[: hi - lo]

        with PROF.section("jax_scan_incl_transfer_and_sync"):
            out = self._split_rollout(rollout, batch, (batch, horizon, int(pending.shape[1])))
        with PROF.section("stack_to_torch"):
            return torch.as_tensor(out[:batch])

    def _split_rollout(self, rollout, batch: int, shape: tuple[int, int, int]) -> np.ndarray:
        """``rollout(lo, hi, rows)`` over the whole batch, run in chunks on ``HWM_ROLLOUT_THREADS`` threads.

        XLA's code for a batch computes most samples vectorized and the last few on a scalar path that
        rounds differently, and which samples those are depends on the batch size. So a split is only used
        once it reproduced one call over the whole batch bit for bit; the first rollout of every shape
        makes that check, the shapes no split reproduces stay one call.
        """
        threads = int(Settings.HWM_ROLLOUT_THREADS)
        if threads <= 1 or batch < 32:
            return rollout(0, batch, batch)
        plans = getattr(self, "_rollout_plans", None)
        if plans is None:
            plans = self._rollout_plans = {}
        if shape in plans:
            plan = plans[shape]
            if plan is None:
                return rollout(0, batch, batch)
            return np.concatenate(list(self._rollout_pool(threads).map(lambda chunk: rollout(*chunk), plan)))
        whole = rollout(0, batch, batch)
        plans[shape] = None
        for plan in self._rollout_split_candidates(batch, threads):
            out = np.concatenate(list(self._rollout_pool(threads).map(lambda chunk: rollout(*chunk), plan)))
            if out.tobytes() == whole.tobytes():
                plans[shape] = plan
                break
        return whole

    @staticmethod
    def _rollout_split_candidates(batch: int, threads: int) -> list[list[tuple[int, int, int]]]:
        """Splits ``[(lo, hi, rows)]`` to try: the first ``vectorized`` samples in chunks of a multiple of 8 and
        the rest padded to 9 rows, a batch size XLA computes entirely on the scalar path."""
        candidates = []
        for vectorized in dict.fromkeys((batch - batch % 8, batch - batch % 4, batch, batch - batch % 16)):
            per_thread = -(-vectorized // threads)
            size = -(-per_thread // 8) * 8
            plan = [(lo, min(lo + size, vectorized), min(lo + size, vectorized) - lo) for lo in range(0, vectorized, size)]
            if vectorized < batch:
                plan.append((vectorized, batch, max(batch - vectorized, 9)))
            candidates.append(plan)
        return candidates

    def _rollout_pool(self, threads: int):
        pools = getattr(self, "_rollout_pools", None)
        if pools is None:
            pools = self._rollout_pools = {}
        if threads not in pools:
            from concurrent.futures import ThreadPoolExecutor

            pools[threads] = ThreadPoolExecutor(threads, thread_name_prefix="hwm-rollout")
        return pools[threads]

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def process_observation(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        if not PROF.enabled:
            return self._process_observation(controller_observation)
        PROF.begin_step()
        t0 = time.perf_counter()
        with PROF.section("process_observation"):
            result = self._process_observation(controller_observation)
        PROF.end_step(time.perf_counter() - t0)
        return result

    def _process_observation(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        self._controller_observation = controller_observation
        if test_track_enabled():
            self._clear_waypoint_display()
            return self._test_track_step(controller_observation)

        state = self.get_car_state(controller_observation)
        action = np.asarray(self.select_action(state), dtype=np.float32).reshape(-1)
        self._apply_normalized_action(action)
        # Network-scale action, unclipped.
        self._record_step(action)
        return self.angular_control, self.translational_control

    def _test_track_step(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        """Constant acceleration and a growing sine steer."""
        steer, accel = test_track_command(self.control_index * float(Settings.TIMESTEP_CONTROL))
        self.angular_control = float(steer)
        self.translational_control = float(accel)
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control
        self._record_step(np.array([steer, accel], dtype=np.float32) / self.action_denormalization_array)
        return self.angular_control, self.translational_control

    def _apply_normalized_action(self, action: np.ndarray) -> None:
        steering, accel = np.asarray(action, dtype=np.float32).reshape(-1) * self.action_denormalization_array
        self.angular_control = self.lowpass_alpha * float(steering) + (1 - self.lowpass_alpha) * self.prev_angular_control
        self.translational_control = self.lowpass_alpha * float(accel) + (1 - self.lowpass_alpha) * self.prev_translational_control
        max_translational = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
        self.translational_control = float(np.clip(self.translational_control, -max_translational, max_translational))
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control

    def _record_step(self, action: np.ndarray) -> None:
        """Note the command issued this step, in network output units."""
        self._issued.append(np.array(action, dtype=np.float32).reshape(-1))
        self.control_index += 1
        self.total_timesteps += 1
        self.episode_timesteps += 1

    @staticmethod
    def _control_delay_steps() -> int:
        return int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))

    def _issued_actions(self) -> torch.Tensor:
        """``(delay, 2)`` the last commands of this episode, oldest first, zeros before its first one."""
        pending = torch.zeros((self._control_delay_steps(), 2), dtype=torch.float32)
        if self._issued:
            pending[-len(self._issued):] = torch.from_numpy(np.stack(self._issued))
        return pending

    def on_step_end(self, driver_obs: Dict[str, Any]) -> None:
        """Called by CarSystem after the physics step; only episode boundaries matter here."""
        done = bool(driver_obs.get("done", False))
        if done:
            term = driver_obs.get("episode_termination") or driver_obs
            reasons = [
                name
                for name in ("collision", "stuck", "spinning", "virtual_opponent_collision", "interrupted")
                if term.get(name)
            ]
            if reasons:
                self._grid_crashes = int(getattr(self, "_grid_crashes", 0)) + 1
                counts = getattr(self, "_grid_crash_reasons", None)
                if counts is None:
                    counts = self._grid_crash_reasons = {}
                for name in reasons:
                    counts[name] = int(counts.get(name, 0)) + 1
            self._revert_nearby_waypoints()
            self._log_debug(f"[HirarchicalPlanner] Episode {self._episode_id} done: {self.episode_timesteps} steps")
            self._reset_episode_state()

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    def report_lap_time(self, lap_time: float, mean_distance: float, std_distance: float, max_distance: float) -> None:
        """A finished lap: the freeze and the best-lap table take it, and the lap's window updates are logged."""
        self._completed_laps = int(getattr(self, "_completed_laps", 0)) + 1
        previous = getattr(self, "_lap_end_step", None)
        self._lap_end_step = int(self.total_timesteps)
        if previous is not None:
            self._last_lap_steps = self._lap_end_step - int(previous)
        laps = getattr(self, "_grid_laps", None)
        if laps is None:
            laps = self._grid_laps = []
        laps.append(float(lap_time))
        self._update_permanent_freeze(float(lap_time))
        self._update_best_lap_table(float(lap_time))
        if hasattr(self, "_wp_window_tried"):
            known = self._waypoint_known(torch.arange(self._waypoints_n))
            speed = float(self.waypoints[known, WP_TARGET_SPEED_IDX].mean()) if bool(known.any()) else 0.0
            self._log_info(
                f"[HirarchicalPlanner] Lap: {self._wp_window_accepted}/{self._wp_window_tried} window updates accepted, "
                f"{int(known.sum())}/{self._waypoints_n} waypoints stored, mean stored speed {speed:.2f} m/s"
            )
            self._wp_window_accepted = self._wp_window_tried = 0

    def _reset_episode_state(self) -> None:
        self._episode_id += 1
        self._issued.clear()
        self.control_index = 0
        self.episode_timesteps = 0
        self.action_buffer = torch.zeros(Settings.MPC_TRAJECTORY_LENGTH, 2)
        self._reset_stall()

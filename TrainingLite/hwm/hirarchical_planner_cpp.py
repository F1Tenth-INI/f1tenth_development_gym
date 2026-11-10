"""The HW planner with its control step in C++ (``Settings.HWM_BACKEND = "cpp"``).

Same planner as ``hirarchical_planner.HirarchicalPlanner`` (sources in ``cpp/hw_planner.cpp``): this class
only prepares the waypoint table and the maps once, passes each car state to the C++ step and the lap and
episode events to it, and shows its table. Torch and JAX are not used per step.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch

from Control_Toolkit_ASF.Controllers import template_planner
from TrainingLite.hwm.hirarchical_planner import (
    WP_HEADING_IDX,
    WP_TARGET_DRIFT_IDX,
    WP_TARGET_SLIP_IDX,
    WP_TARGET_SPEED_IDX,
    HirarchicalPlanner,
)
from TrainingLite.hwm.shared.speed_profile import grip_speed_profile
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from utilities.test_track import command as test_track_command
from utilities.test_track import enabled as test_track_enabled


def core_config(denorm: np.ndarray, brake_limit: float, accel_limit: float, brake_multiplier: float) -> dict:
    """The settings the C++ planner reads (``Settings`` names in the comments of ``hw_planner.cpp``)."""
    rec = Settings.HWM_RECOVERY
    ramp_end = Settings.MPC_DRIVE_WALL_RAMP_END
    seed = Settings.RANDOM_SEED
    return {
        "H": int(Settings.MPC_TRAJECTORY_LENGTH), "H_FREE": int(Settings.MPC_FREE_TRAJECTORY_LENGTH),
        "K": int(Settings.MPC_NUM_OF_TRAJECTORIES), "RHO": float(Settings.MPC_ACTION_CORRELATION),
        "TEMPERATURE": float(Settings.MPC_TEMPERATURE), "DRIVE_NOISE": list(Settings.MPC_DRIVE_NOISE),
        "DENORM": [float(x) for x in denorm], "CAP": float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL),
        "BRAKE_LIMIT": brake_limit, "ACCEL_LIMIT": accel_limit, "BRAKE_MULTIPLIER": brake_multiplier,
        "DT": float(Settings.TIMESTEP_CONTROL),
        "DELAY": int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL)),
        "IGNORE_STEPS": int(Settings.IGNORE_STEPS), "TRACK_WINDOW": int(Settings.HWM_TRACK_WINDOW),
        "TRACK_FIELD": int(not Settings.BLANK_MAP), "MATCH_SHARE": float(Settings.HWM_MATCH_HORIZON_SHARE),
        "TARGET_SPEED_FACTOR": float(Settings.MPC_DRIVE_TARGET_SPEED_FACTOR),
        "OVERSPEED_W": float(Settings.MPC_DRIVE_OVERSPEED_WEIGHT), "ARRIVAL_W": float(Settings.MPC_DRIVE_ARRIVAL_WEIGHT),
        "BORDER": float(Settings.HWM_MIN_BORDER_DISTANCE), "CLEARANCE": float(Settings.MPC_DRIVE_WALL_CLEARANCE),
        "RAMP_END": float(Settings.HWM_MIN_BORDER_DISTANCE if ramp_end is None else ramp_end),
        "WALL_W": float(Settings.MPC_DRIVE_WALL_WEIGHT), "DEPTH_W": float(Settings.MPC_DRIVE_CRASH_DEPTH_WEIGHT),
        "CRASH_HORIZON": -1 if Settings.HWM_CRASH_HORIZON is None else int(Settings.HWM_CRASH_HORIZON),
        "SEED_SPEED_GAIN": float(Settings.MPC_SEED_SPEED_GAIN), "FALLBACK": int(bool(Settings.MPC_DRIVE_FALLBACK_SAMPLES)),
        "FALLBACK_STEER_SCALES": list(Settings.HWM_FALLBACK_STEER_SCALES),
        "CREEP_SPEED": float(Settings.HWM_CREEP_SPEED), "CREEP_ACCEL": float(Settings.HWM_CREEP_ACCEL),
        "BACK_OFF_SPEED": float(Settings.HWM_BACK_OFF_SPEED), "BLOCKED_DISTANCE": float(Settings.HWM_BLOCKED_DISTANCE),
        "STALL_ON_BLOCKED": int(bool(Settings.MPC_STALL_ON_BLOCKED)), "SAFE_MEAN": int(bool(Settings.MPC_SAFE_MEAN)),
        "REC_WINDOW": int(rec["window"]), "REC_DISTANCE": rec["distance"], "REC_HEADING": rec["heading"],
        "REC_STEER_IN": rec["steer_in"], "REC_STEER_IN_MAX": rec["steer_in_max"], "REC_SPEED_MIN": rec["speed_min"],
        "REC_GATE": rec["gate"], "REC_NOISE_SCALE": rec["noise_scale"], "REC_SPEED_MIN_W": rec["speed_min_w"],
        "REC_SPEED_MAX": rec["speed_max"], "REC_SPEED_MAX_W": rec["speed_max_w"], "REC_REVERSE": rec["reverse"],
        "REC_SLIP": rec["slip"], "REC_YAW_RATE": rec["yaw_rate"],
        "DETOUR_START": float(Settings.HWM_DETOUR_START), "DETOUR_SPAN": float(Settings.HWM_DETOUR_SPAN),
        "UPDATE_DISTANCE": float(Settings.MPC_WAYPOINT_UPDATE_DISTANCE),
        "RECOVERY_HEADING_ERROR": float(Settings.MPC_RECOVERY_HEADING_ERROR),
        "OFF_TRACK_GRACE": float(Settings.HWM_OFF_TRACK_WAYPOINT_GRACE),
        "STALL_DISTANCE": float(Settings.MPC_STALL_DISTANCE), "STALL_TIME": float(Settings.MPC_STALL_TIME),
        "STALL_REVERSE_STEPS": int(Settings.MPC_STALL_REVERSE_STEPS),
        "STALL_BACKTRACK_WPS": int(Settings.MPC_STALL_BACKTRACK_WAYPOINTS),
        "EMA_GAMMA": float(Settings.MPC_REWARD_EMA_GAMMA), "REWARD_DROP": float(Settings.MPC_WAYPOINT_REWARD_DROP),
        "REVERT_AHEAD": int(Settings.MPC_WAYPOINT_REVERT_AHEAD), "REVERT_BEHIND": int(Settings.MPC_WAYPOINT_REVERT_BEHIND),
        "SPAN": int(Settings.MPC_WAYPOINT_AHEAD_SPAN), "LAG": int(Settings.MPC_WAYPOINT_WINDOW_LAG),
        "DOWNSTREAM": int(Settings.MPC_WAYPOINT_WINDOW_DOWNSTREAM), "W_SAMPLES": int(Settings.MPC_WAYPOINT_WINDOW_SAMPLES),
        "W_NOISE": float(Settings.MPC_WAYPOINT_WINDOW_NOISE), "W_CORR": float(Settings.MPC_WAYPOINT_WINDOW_CORRELATION),
        "W_TOL": float(Settings.MPC_WAYPOINT_WINDOW_TOLERANCE), "W_MARGIN": float(Settings.MPC_WAYPOINT_WINDOW_MARGIN),
        "W_MIN_MARGIN": float(Settings.MPC_WAYPOINT_WINDOW_MIN_MARGIN),
        "SHIFT_STEPS": list(Settings.HWM_WINDOW_SHIFT_STEPS), "W_STEER_SCALES": list(Settings.HWM_WINDOW_STEER_SCALES),
        "W_ERROR": [Settings.HWM_WINDOW_ERROR[k] for k in ("lateral", "heading", "speed", "drift", "yaw_rate")],
        "W_ERROR_COST": float(Settings.HWM_WINDOW_ERROR_COST), "LINE_REACH": float(Settings.HWM_WINDOW_LINE_REACH),
        "FILL_MIN_SPEED": float(Settings.HWM_FILL_MIN_SPEED), "DRIVEN_MEMORY": int(Settings.HWM_DRIVEN_MEMORY),
        "W_HISTORY": int(Settings.HWM_WINDOW_HISTORY), "W_MAX_SLIP": float(Settings.HWM_WINDOW_MAX_SLIP),
        "FRAGILE_SHARE": float(Settings.HWM_FRAGILE_SHARE), "FRAGILE_AHEAD": int(Settings.HWM_FRAGILE_AHEAD),
        "FRAGILE_MEMORY": float(Settings.HWM_FRAGILE_MEMORY), "FRAGILE_GAIN": float(Settings.HWM_FRAGILE_CLEARANCE_GAIN),
        "FREEZE_LAPS": int(getattr(Settings, "HWM_PERMANENT_FREEZE_LAPS", 0) or 0),
        "BEST_LAP_RESET": -1 if Settings.HWM_BEST_LAP_RESET_LAPS is None else int(Settings.HWM_BEST_LAP_RESET_LAPS),
        "SEED": int(np.random.SeedSequence(seed).generate_state(1, np.uint32)[0]) if seed is not None
        else int(np.random.SeedSequence().generate_state(1, np.uint32)[0]),
        "VERBOSE": int(bool(Settings.HWM_AGENT_DEBUG)),
    }


class HirarchicalPlannerCpp(template_planner):
    # Display and limits are shared with the Python planner.
    _publish_waypoint_display = HirarchicalPlanner._publish_waypoint_display
    _clear_waypoint_display = HirarchicalPlanner._clear_waypoint_display
    _translational_limits = HirarchicalPlanner._translational_limits
    _car_friction_accel = HirarchicalPlanner._car_friction_accel
    _car_brake_multiplier = HirarchicalPlanner._car_brake_multiplier

    def __init__(self):
        super().__init__()
        print("Initializing HirarchicalPlanner (C++ control step)")
        self.wall_geometry = WallGeometry()
        if Settings.HWM_SIM_SPEED is not None:
            speed = float(Settings.HWM_SIM_SPEED)
            if speed <= 0:
                raise ValueError(f"HWM_SIM_SPEED must be positive or None, got {speed}")
            Settings.MAX_SIM_FREQUENCY = speed / float(Settings.TIMESTEP_CONTROL)
        self.action_denormalization_array = np.asarray(Settings.HWM_ACTION_DENORM, dtype=np.float32)
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.control_index = 0
        self.core = None
        self._grid_crashes = 0
        self._grid_crash_reasons: dict[str, int] = {}
        self._grid_laps: list[float] = []
        self._completed_laps = 0
        self._step_ms: list[float] = []
        self._profile = bool(getattr(Settings, "HWM_PROFILE", False))

    # ------------------------------------------------------------------ setup
    def _ensure_core(self) -> None:
        if self.core is not None:
            return
        utils = getattr(self, "waypoint_utils", None)
        waypoints = None if utils is None else getattr(utils, "waypoints", None)
        if waypoints is None:
            raise RuntimeError("HirarchicalPlannerCpp needs waypoint_utils.waypoints")
        from TrainingLite.hwm import hw_cpp
        from utilities.car_files.vehicle_parameters import VehicleParameters

        # The table as HirarchicalPlanner._ensure_waypoints builds it.
        source = np.asarray(waypoints)
        n = int(source.shape[0])
        owned = np.zeros((n, WP_TARGET_SLIP_IDX + 1), dtype=np.float32)
        width = min(int(source.shape[1]), WP_TARGET_DRIFT_IDX)
        owned[:, :width] = source[:, :width]
        table = torch.as_tensor(owned)
        xy = table[:, 1:3]
        if Settings.HWM_INITIAL_SPEED_GRIP_SHARE is not None:
            accel = float(Settings.HWM_INITIAL_SPEED_GRIP_SHARE) * self._car_friction_accel()
            profile = grip_speed_profile(xy, accel, float(Settings.HWM_INITIAL_SPEED_MAX))
            table[:, WP_TARGET_SPEED_IDX] = torch.minimum(table[:, WP_TARGET_SPEED_IDX], profile)
        elif Settings.HWM_LIMIT_INITIAL_WP_SPEED:
            table[:, WP_TARGET_SPEED_IDX] = torch.minimum(table[:, WP_TARGET_SPEED_IDX], torch.tensor(Settings.HWM_INITIAL_SPEED_LIMIT))
        delta = torch.roll(xy, shifts=-1, dims=0) - xy
        table[:, WP_HEADING_IDX] = torch.atan2(delta[:, 1], delta[:, 0])
        heading0 = table[:, WP_HEADING_IDX]
        anchor = xy.clone()
        tangent = torch.stack((torch.cos(heading0), torch.sin(heading0)), dim=1)
        ortho = torch.stack((-torch.sin(heading0), torch.cos(heading0)), dim=1)

        geometry = self.wall_geometry
        threads = hw_cpp.set_threads(int(Settings.HWM_ROLLOUT_THREADS))
        track_field = None
        if Settings.BLANK_MAP:
            hw_cpp.set_blank_map()
        else:
            hw_cpp.set_map(geometry)
            geometry.build_track_field(anchor, clearance=0.5 * geometry.car_width)
            track_field = (geometry.track_length.numpy(), geometry.track_label.numpy().astype(np.int32),
                           geometry.track_direction.numpy())
        if Settings.MOTOR_PID_IN_CAR_MODEL:
            raise ValueError("HWM_BACKEND 'cpp' models acceleration control only (MOTOR_PID_IN_CAR_MODEL = False)")
        brake_limit, accel_limit = self._translational_limits()
        dt = float(Settings.TIMESTEP_CONTROL)
        self.core = hw_cpp.HwPlannerCore(
            core_config(self.action_denormalization_array, brake_limit, accel_limit, self._car_brake_multiplier()),
            threads=int(Settings.HWM_ROLLOUT_THREADS),
            car_params=VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_np_array(),
            ode_model=Settings.ODE_MODEL_OF_CAR_DYNAMICS, dt=dt, substeps=int(round(dt / float(Settings.TIMESTEP_SIM))),
            table=table.numpy(), anchor=anchor.numpy(), tangent=tangent.numpy(), ortho=ortho.numpy(),
            track_field=track_field,
        )
        self._raceline_np = source
        self._bind_views()
        self._write_raceline()
        self._publish_waypoint_display()
        print(f"[HirarchicalPlanner] C++ control step on {threads} threads, {n} waypoints")

    def _bind_views(self) -> None:
        self.waypoints = torch.from_numpy(self.core.table)
        self._wp_off_track_count = torch.from_numpy(self.core.off_track_counts)
        self._wp_action_known = torch.from_numpy(self.core.known.view(np.bool_))
        self._wp_actions = torch.from_numpy(self.core.actions)

    def _write_raceline(self) -> None:
        """Copy the shared raceline columns of the table back to waypoint_utils, as the Python planner does."""
        source = getattr(self, "_raceline_np", None)
        if source is None or self.core is None:
            return
        width = min(int(source.shape[1]), WP_TARGET_DRIFT_IDX)
        source[:, :width] = self.core.table[:, :width]

    def _waypoint_paths(self):
        return None

    @property
    def _wp_permanently_frozen(self) -> bool:
        return self.core is not None and self.core.frozen

    # ------------------------------------------------------------------ control loop
    def reset(self):
        super().reset()
        self.control_index = 0
        if self.core is not None:
            self.core.reset()

    def process_observation(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        if test_track_enabled():
            self._clear_waypoint_display()
            steer, accel = test_track_command(self.control_index * float(Settings.TIMESTEP_CONTROL))
            self.angular_control, self.translational_control = float(steer), float(accel)
            self.control_index += 1
            return self.angular_control, self.translational_control
        self._ensure_core()
        state = self.get_car_state(controller_observation)
        t0 = time.perf_counter()
        out = self.core.step(state)
        if self._profile:
            self._step_ms.append(1e3 * (time.perf_counter() - t0))
            if len(self._step_ms) % 500 == 0:
                self.write_profile()
        self.angular_control, self.translational_control = float(out[0]), float(out[1])
        self.control_index += 1
        self._write_raceline()
        if not Settings.HWM_LIVE_MODE and self.control_index % 5 == 0:
            self._publish_waypoint_display()
        return self.angular_control, self.translational_control

    def on_step_end(self, driver_obs: Dict[str, Any]) -> None:
        if not bool(driver_obs.get("done", False)):
            return
        term = driver_obs.get("episode_termination") or driver_obs
        reasons = [name for name in ("collision", "stuck", "spinning", "virtual_opponent_collision", "interrupted")
                   if term.get(name)]
        if reasons:
            self._grid_crashes += 1
            for name in reasons:
                self._grid_crash_reasons[name] = self._grid_crash_reasons.get(name, 0) + 1
        if self.core is not None:
            self.core.episode_done(bool(reasons))

    def report_lap_time(self, lap_time: float, mean_distance: float, std_distance: float, max_distance: float) -> None:
        self._completed_laps += 1
        self._grid_laps.append(float(lap_time))
        if self.core is not None:
            self.core.lap(float(lap_time))
            self._bind_views()
            self._write_raceline()
            self._publish_waypoint_display()

    # ------------------------------------------------------------------ profiling
    def write_profile(self, path: str | None = None) -> str:
        """Step times (wall clock of the whole C++ step, and the mean of its parts) to ``HWM_PROFILE_FILE``_cpp.txt."""
        skip = int(getattr(Settings, "HWM_PROFILE_SKIP_STEPS", 30))
        ms = np.asarray(self._step_ms[skip:]) if len(self._step_ms) > skip else np.asarray(self._step_ms)
        lines = [f"HW planner, C++ control step: {len(ms)} steps, {Settings.MPC_NUM_OF_TRAJECTORIES} samples, "
                 f"horizon {Settings.MPC_TRAJECTORY_LENGTH}, {Settings.HWM_ROLLOUT_THREADS} threads"]
        if ms.size:
            lines.append(f"step: mean {ms.mean():.3f} ms, median {np.median(ms):.3f}, p95 {np.quantile(ms, 0.95):.3f}, "
                         f"p99 {np.quantile(ms, 0.99):.3f}, max {ms.max():.3f}")
        if self.core is not None:
            for name, value in self.core.stats().items():
                lines.append(f"  {name:16s} {value:8.3f}" + ("" if name == "steps" else " ms"))
        text = "\n".join(lines)
        path = path or f"{getattr(Settings, 'HWM_PROFILE_FILE', 'hwm_profile')}_cpp.txt"
        with open(path, "w") as f:
            f.write(text + "\n")
        return text

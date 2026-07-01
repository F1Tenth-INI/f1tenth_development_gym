"""Lightweight lidar-blocking opponents that follow recorded trajectories."""

from __future__ import annotations

import functools
import os
import random
from typing import Optional

import numpy as np
import pandas as pd

from f110_sim.envs.collision_models import collision, get_vertices
from f110_sim.envs.laser_models import ray_cast
from utilities.Settings import Settings
from utilities.map_scale import (
    processed_waypoint_count_at_scale,
    remap_waypoint_indices,
    scale_trajectory_poses,
)
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.lidar_simulator import LidarSimulator
from utilities.state_utilities import POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX


def _resolve_recording_path(recording_name: str) -> str:
    _validate_recording_map(recording_name)
    if os.path.isabs(recording_name):
        return recording_name
    trajectory_folder = getattr(Settings, "VIRTUAL_OPPONENT_TRAJECTORY_FOLDER", None)
    if trajectory_folder:
        committed_path = os.path.join(trajectory_folder, recording_name)
        if os.path.isfile(committed_path):
            return committed_path
    return os.path.join(Settings.RECORDING_FOLDER, recording_name)


def _validate_recording_map(recording_name: str) -> None:
    """Require the active map name to appear in the trajectory CSV filename."""
    map_name = str(getattr(Settings, "MAP_NAME", "") or "")
    if not map_name:
        return
    basename = os.path.basename(recording_name)
    if map_name not in basename:
        raise ValueError(
            f"Virtual opponent recording '{basename}' does not match current map "
            f"'{map_name}'. The map name must appear in the CSV filename."
        )


def load_trajectory_from_recording(
    csv_path: str,
    *,
    trim_to_single_lap: bool = True,
    target_total_waypoints: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (poses[N,3], times[N], waypoint_indices[N]) from a recording CSV."""
    df = pd.read_csv(csv_path, comment="#")
    required = ("time", "pose_x", "pose_y", "pose_theta")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Recording {csv_path} is missing columns: {missing}")

    times = np.asarray(df["time"], dtype=np.float64)
    poses = np.column_stack(
        [
            np.asarray(df["pose_x"], dtype=np.float64),
            np.asarray(df["pose_y"], dtype=np.float64),
            np.asarray(df["pose_theta"], dtype=np.float64),
        ]
    )
    if "nearest_wpt_idx" in df.columns:
        waypoint_indices = np.asarray(df["nearest_wpt_idx"], dtype=np.int64)
    else:
        raise ValueError(
            f"Recording {csv_path} has no nearest_wpt_idx column. "
            "Record with SAVE_RECORDINGS so lap/waypoint indexing is available."
        )

    if len(times) < 2:
        raise ValueError(f"Recording {csv_path} must contain at least two rows.")

    map_scale = float(getattr(Settings, "MAP_SCALE", 1.0))
    unit_total_waypoints = None
    if map_scale != 1.0 and target_total_waypoints is not None:
        unit_total_waypoints = processed_waypoint_count_at_scale(1.0)

    if trim_to_single_lap:
        poses, times, waypoint_indices = trim_trajectory_to_single_lap(
            poses,
            times,
            waypoint_indices,
            total_waypoints=unit_total_waypoints,
        )

    poses = scale_trajectory_poses(poses)

    if map_scale != 1.0 and target_total_waypoints is not None and unit_total_waypoints is not None:
        waypoint_indices = remap_waypoint_indices(
            waypoint_indices,
            unit_total_waypoints,
            target_total_waypoints,
        )

    return poses, times, waypoint_indices


def find_lap_crossing_indices(
    waypoint_indices: np.ndarray,
    total_waypoints: Optional[int] = None,
) -> list[int]:
    """
    Indices where the car crosses the start/finish (high waypoint index -> low).
    Matches ExperimentAnalyzer lap inference.
    """
    if len(waypoint_indices) < 2:
        return []

    indices = np.asarray(waypoint_indices, dtype=np.int64)
    if total_waypoints is None:
        total_waypoints = int(np.max(indices)) + 1
    if total_waypoints <= 1:
        return []

    low_threshold = int(0.25 * total_waypoints)
    high_threshold = int(0.75 * total_waypoints)
    crossings: list[int] = []
    for i in range(1, len(indices)):
        if indices[i - 1] >= high_threshold and indices[i] <= low_threshold:
            crossings.append(i)
    return crossings


def trim_trajectory_to_single_lap(
    poses: np.ndarray,
    times: np.ndarray,
    waypoint_indices: np.ndarray,
    total_waypoints: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep one full lap so looping replays without a pose jump."""
    crossings = find_lap_crossing_indices(waypoint_indices, total_waypoints)
    if len(crossings) >= 2:
        start, end = crossings[0], crossings[1]
        return poses[start:end], times[start:end], waypoint_indices[start:end]
    return poses, times, waypoint_indices


def trim_trajectory_to_n_laps(
    poses: np.ndarray,
    times: np.ndarray,
    waypoint_indices: np.ndarray,
    num_laps: int,
    total_waypoints: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep exactly `num_laps` full laps from the first detected crossing."""
    crossings = find_lap_crossing_indices(waypoint_indices, total_waypoints)
    if len(crossings) > num_laps:
        start, end = crossings[0], crossings[num_laps]
        return poses[start:end], times[start:end], waypoint_indices[start:end]
    return poses, times, waypoint_indices


def interpolate_pose(times: np.ndarray, poses: np.ndarray, t: float) -> np.ndarray:
    """Linear pose interpolation with angle unwrapping."""
    t = float(np.clip(t, times[0], times[-1]))
    x = float(np.interp(t, times, poses[:, 0]))
    y = float(np.interp(t, times, poses[:, 1]))
    theta_unwrapped = np.unwrap(poses[:, 2])
    theta = float(np.interp(t, times, theta_unwrapped))
    theta = float(np.arctan2(np.sin(theta), np.cos(theta)))
    return np.array([x, y, theta], dtype=np.float64)


def _circular_waypoint_distance(a: np.ndarray, b: int, total_waypoints: int) -> np.ndarray:
    diff = np.abs((a - b) % total_waypoints)
    return np.minimum(diff, total_waypoints - diff)


@functools.lru_cache(maxsize=1)
def get_ego_car_dimensions() -> tuple[float, float]:
    """Return (length, width) in meters for the ego vehicle."""
    params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_dict()
    return float(params["length"]), float(params["width"])


@functools.lru_cache(maxsize=1)
def get_virtual_opponent_dimensions() -> tuple[float, float]:
    """Return (length, width) in meters for virtual opponent rectangles."""
    size = getattr(Settings, "VIRTUAL_OPPONENT_SIZE", None)
    if size is not None and len(size) >= 2:
        width, length = float(size[0]), float(size[1])
        return length, width
    params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_dict()
    return float(params["length"]), float(params["width"])


class VirtualOpponent:
    """
    Replay one recording lap.

    Spawn pose is taken N waypoints ahead of ego on track; afterwards the
    opponent advances along the recording timeline at vel_factor * sim_time.
    """

    def __init__(
        self,
        poses: np.ndarray,
        times: np.ndarray,
        waypoint_indices: np.ndarray,
        total_waypoints: int,
        length: float,
        width: float,
        *,
        distance_ahead_waypoints: int = 30,
        distance_ahead_random_max: int = 0,
        vel_factor: float = 1.0,
        loop: bool = True,
        start_offset_s: float = 0.0,
    ):
        self.poses = np.asarray(poses, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.waypoint_indices = np.asarray(waypoint_indices, dtype=np.int64)
        self.total_waypoints = int(total_waypoints)
        self.length = float(length)
        self.width = float(width)
        self._distance_ahead_waypoints_base = int(distance_ahead_waypoints)
        self._distance_ahead_random_max = max(0, int(distance_ahead_random_max))
        self.distance_ahead_waypoints = self._distance_ahead_waypoints_base
        self.vel_factor = float(vel_factor)
        self.loop = bool(loop)
        self.start_offset_s = float(start_offset_s)

        self._anchor_sim_time: Optional[float] = None
        self._anchor_recording_time: Optional[float] = None
        self._current_pose: Optional[np.ndarray] = None # [x, y, theta]

    def _roll_distance_ahead_waypoints(self) -> None:
        if self._distance_ahead_random_max > 0:
            self.distance_ahead_waypoints = self._distance_ahead_waypoints_base + random.randint(
                0, self._distance_ahead_random_max
            )
        else:
            self.distance_ahead_waypoints = self._distance_ahead_waypoints_base

    def clear_anchor(self) -> None:
        self._anchor_sim_time = None
        self._anchor_recording_time = None
        self._current_pose = None

    def target_waypoint_index(self, ego_waypoint_index: int) -> int:
        return (int(ego_waypoint_index) + self.distance_ahead_waypoints) % self.total_waypoints

    def _sample_at_target_waypoint(
        self, ego_waypoint_index: int
    ) -> tuple[np.ndarray, float]:
        """Pose and recording time at ego_wp + distance_ahead (spawn placement)."""
        target_wp = self.target_waypoint_index(ego_waypoint_index)
        ring_wpt = self.waypoint_indices % self.total_waypoints
        distances = _circular_waypoint_distance(ring_wpt, target_wp, self.total_waypoints)

        best_idx = int(np.argmin(distances))
        if distances[best_idx] == 0 or len(self.poses) == 1:
            return self.poses[best_idx].copy(), float(self.times[best_idx])

        prev_idx = (best_idx - 1) % len(self.poses)
        next_idx = (best_idx + 1) % len(self.poses)
        candidates = [prev_idx, best_idx, next_idx]
        local_best = min(candidates, key=lambda idx: distances[idx])
        if distances[local_best] == 0:
            return self.poses[local_best].copy(), float(self.times[local_best])

        order = sorted(candidates, key=lambda idx: ring_wpt[idx])
        idx_a, idx_b = order[0], order[-1]
        wp_a = float(ring_wpt[idx_a])
        wp_b = float(ring_wpt[idx_b])
        if wp_b < wp_a:
            wp_b += self.total_waypoints
        target = float(target_wp)
        if target < wp_a:
            target += self.total_waypoints
        if wp_b == wp_a:
            alpha = 0.0
        else:
            alpha = float(np.clip((target - wp_a) / (wp_b - wp_a), 0.0, 1.0))

        pose_a = self.poses[idx_a]
        pose_b = self.poses[idx_b]
        time_a = float(self.times[idx_a])
        time_b = float(self.times[idx_b])
        theta_a = pose_a[2]
        theta_b = pose_b[2] + 2 * np.pi * np.round((theta_a - pose_b[2]) / (2 * np.pi))
        pose = np.array(
            [
                (1.0 - alpha) * pose_a[0] + alpha * pose_b[0],
                (1.0 - alpha) * pose_a[1] + alpha * pose_b[1],
                np.arctan2(
                    (1.0 - alpha) * np.sin(theta_a) + alpha * np.sin(theta_b),
                    (1.0 - alpha) * np.cos(theta_a) + alpha * np.cos(theta_b),
                ),
            ],
            dtype=np.float64,
        )
        recording_time = (1.0 - alpha) * time_a + alpha * time_b
        return pose, recording_time

    def set_anchor(self, ego_waypoint_index: int, sim_time: float) -> None:
        """Place opponent N waypoints ahead; anchor replay clock for vel_factor."""
        self._roll_distance_ahead_waypoints()
        _pose, recording_time = self._sample_at_target_waypoint(ego_waypoint_index)
        self._anchor_sim_time = float(sim_time)
        self._anchor_recording_time = recording_time + self.start_offset_s
        self._current_pose = self.pose_at_sim_time(sim_time)

    def _wrap_recording_time(self, t: float) -> float:
        if len(self.times) < 2:
            return float(self.times[0])
        t0 = float(self.times[0])
        t1 = float(self.times[-1])
        if not self.loop:
            return float(np.clip(t, t0, t1))
        duration = t1 - t0
        if duration <= 0.0:
            return t0
        return t0 + ((t - t0) % duration)

    def pose_at_sim_time(self, sim_time: float) -> np.ndarray:
        if self._anchor_sim_time is None or self._anchor_recording_time is None:
            raise RuntimeError("Virtual opponent anchor not set.")
        dt_sim = float(sim_time) - self._anchor_sim_time
        recording_t = self._anchor_recording_time + self.vel_factor * dt_sim
        recording_t = self._wrap_recording_time(recording_t)
        if not self.loop:
            recording_t = float(np.clip(recording_t, self.times[0], self.times[-1]))
        pose = interpolate_pose(self.times, self.poses, recording_t)
        self._current_pose = pose
        return pose.copy()

    def current_pose(self) -> np.ndarray:
        if self._current_pose is None:
            raise RuntimeError("Virtual opponent pose not available.")
        return self._current_pose.copy()

    def current_vertices(self) -> np.ndarray:
        return get_vertices(self.current_pose(), self.length, self.width)


def _require_per_opponent_array(attr_name: str, count: int) -> list:
    values = list(getattr(Settings, attr_name, []) or [])
    if len(values) < count:
        raise ValueError(
            f"Settings.{attr_name} must have at least length {count} "
            f"(NUMBER_OF_VIRTUAL_OPPONENTS), got {len(values)}"
        )
    return values[:count]


class VirtualOpponents:
    """Manage multiple virtual opponents and apply them as lidar occluders."""

    def __init__(self, opponents: list[VirtualOpponent]):
        self.opponents = opponents
        self._scan_angles: Optional[np.ndarray] = None
        self._current_poses: list[np.ndarray] = []
        self._initialized = False

    @classmethod
    def from_settings(cls) -> Optional["VirtualOpponents"]:
        count = int(getattr(Settings, "NUMBER_OF_VIRTUAL_OPPONENTS", 0))
        if count <= 0:
            return None

        from utilities.waypoint_utils import WaypointUtils

        waypoint_utils = WaypointUtils()
        total_waypoints = len(waypoint_utils.waypoints)
        length, width = get_virtual_opponent_dimensions()

        recordings = _require_per_opponent_array("VIRTUAL_OPPONENT_RECORDINGS", count)
        distances_ahead = _require_per_opponent_array(
            "VIRTUAL_OPPONENT_DISTANCE_AHEAD_WAYPOINTS", count
        )
        vel_factors = _require_per_opponent_array("VIRTUAL_OPPONENT_VEL_FACTORS", count)
        start_offsets = _require_per_opponent_array("VIRTUAL_OPPONENT_START_OFFSET_S", count)

        trim_to_single_lap = bool(
            getattr(Settings, "VIRTUAL_OPPONENT_TRIM_TO_SINGLE_LAP", True)
        )
        loop = bool(getattr(Settings, "VIRTUAL_OPPONENT_LOOP", True))
        distance_ahead_random_max = int(
            getattr(Settings, "VIRTUAL_OPPONENT_DISTANCE_AHEAD_WAYPOINTS_RANDOM_MAX", 0)
        )

        opponents: list[VirtualOpponent] = []
        for idx in range(count):
            csv_path = _resolve_recording_path(recordings[idx])
            poses, times, waypoint_indices = load_trajectory_from_recording(
                csv_path,
                trim_to_single_lap=trim_to_single_lap,
                target_total_waypoints=total_waypoints,
            )
            opponents.append(
                VirtualOpponent(
                    poses,
                    times,
                    waypoint_indices,
                    total_waypoints,
                    length,
                    width,
                    distance_ahead_waypoints=int(distances_ahead[idx]),
                    distance_ahead_random_max=distance_ahead_random_max,
                    vel_factor=float(vel_factors[idx]),
                    loop=loop,
                    start_offset_s=float(start_offsets[idx]),
                )
            )
        return cls(opponents)

    def reset(self) -> None:
        self._current_poses = []
        self._initialized = False
        for opponent in self.opponents:
            opponent.clear_anchor()

    def set_state(
        self,
        ego_waypoint_index: Optional[int],
        sim_time: float,
    ) -> None:
        if ego_waypoint_index is None:
            return

        sim_time = float(sim_time)
        if not self._initialized:
            for opponent in self.opponents:
                opponent.set_anchor(int(ego_waypoint_index), sim_time)
            self._initialized = True
        else:
            for opponent in self.opponents:
                opponent.pose_at_sim_time(sim_time)

        self._current_poses = [opponent.current_pose() for opponent in self.opponents]

    def _ensure_scan_angles(self) -> np.ndarray:
        if self._scan_angles is None:
            LidarSimulator._ensure_scan_tables_initialized(
                Settings.LIDAR_NUM_SCANS,
                4.7,
            )
            self._scan_angles = LidarSimulator._scan_angles
        return self._scan_angles

    def apply_to_scan(self, ego_car_state: np.ndarray, scan: np.ndarray) -> np.ndarray:
        """Ray-cast virtual opponent bodies into the ego lidar scan."""
        if not self.opponents or not self._initialized:
            return scan

        ego_pose = np.array(
            [
                ego_car_state[POSE_X_IDX],
                ego_car_state[POSE_Y_IDX],
                ego_car_state[POSE_THETA_IDX],
            ],
            dtype=np.float64,
        )
        scan = np.asarray(scan, dtype=np.float64).copy()
        scan_angles = self._ensure_scan_angles()

        for opponent in self.opponents:
            scan = ray_cast(ego_pose, scan, scan_angles, opponent.current_vertices())
        return scan

    def get_poses(self) -> np.ndarray:
        if not self._current_poses:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._current_poses, dtype=np.float32)

    def get_body_polygons(self) -> list[np.ndarray]:
        if not self._initialized:
            return []
        return [opponent.current_vertices() for opponent in self.opponents]

    def collides_with_ego(
        self,
        ego_car_state: np.ndarray,
        ego_length: float,
        ego_width: float,
    ) -> bool:
        if not self._initialized:
            return False
        ego_pose = np.array(
            [
                ego_car_state[POSE_X_IDX],
                ego_car_state[POSE_Y_IDX],
                ego_car_state[POSE_THETA_IDX],
            ],
            dtype=np.float64,
        )
        ego_verts = np.ascontiguousarray(get_vertices(ego_pose, ego_length, ego_width))
        for opponent in self.opponents:
            opp_verts = np.ascontiguousarray(opponent.current_vertices())
            if collision(ego_verts, opp_verts):
                return True
        return False

    def min_clearance_to_ego(
        self,
        ego_car_state: np.ndarray,
        ego_length: float,
        ego_width: float,
    ) -> float:
        """Minimum center-based clearance to any opponent body (0 if overlapping)."""
        if not self._initialized:
            return float("inf")
        ego_pose = np.array(
            [
                ego_car_state[POSE_X_IDX],
                ego_car_state[POSE_Y_IDX],
                ego_car_state[POSE_THETA_IDX],
            ],
            dtype=np.float64,
        )
        ego_verts = get_vertices(ego_pose, ego_length, ego_width)
        ego_radius = 0.5 * float(np.linalg.norm(ego_verts[0] - ego_verts[2]))
        min_clearance = float("inf")
        for opponent in self.opponents:
            opp_verts = opponent.current_vertices()
            if collision(np.ascontiguousarray(ego_verts), np.ascontiguousarray(opp_verts)):
                return 0.0
            opp_radius = 0.5 * float(np.linalg.norm(opp_verts[0] - opp_verts[2]))
            center_dist = float(
                np.linalg.norm(ego_verts.mean(axis=0) - opp_verts.mean(axis=0))
            )
            min_clearance = min(
                min_clearance, center_dist - ego_radius - opp_radius
            )
        return max(0.0, min_clearance)

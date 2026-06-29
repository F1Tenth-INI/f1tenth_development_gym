"""Detect and track opponents (real or virtual) from the ego lidar scan.

Both physically-simulated opponents and the lightweight "virtual opponents"
(see ``utilities/virtual_opponents.py``) appear identically in the ego lidar
scan: a virtual opponent ray-casts its rectangular body into the scan before it
reaches the ego, so from the controller's point of view there is no difference.

This module therefore works purely from the lidar return. It uses the full,
non-decimated scan (every beam) so that small / distant opponents are still
resolved:

1. Keep only lidar hits that fall *inside the track corridor*. Track walls are
   removed using the waypoint border geometry (each waypoint stores the lateral
   distance to the left/right bound), so whatever remains inside the drivable
   corridor is an obstacle/opponent rather than a wall.
2. Cluster the remaining points by angular adjacency. Each cluster is a candidate
   opponent body (typically a rectangle of >= ~0.25 m extent).
3. Associate clusters with the previous frame's tracks to estimate a relative
   position and velocity for every opponent.

The result is a list of :class:`OpponentTrack` objects, each carrying the
opponent's position relative to the ego car (longitudinal / lateral / range /
bearing) as well as a global position and an estimated global velocity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numba import njit

# Per-step feature vector for the single closest opponent (ego frame):
# [present, forward, left, range, bearing, speed, rel_vel_forward, rel_vel_left]
NEAREST_OPPONENT_FEATURE_SIZE = 8

from utilities.lidar_utils import (
    get_points_from_ranges,
    transform_points_from_car_to_global,
)
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_COS_IDX,
    POSE_THETA_SIN_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)
from utilities.waypoint_utils import (
    WP_D_LEFT_IDX,
    WP_D_RIGHT_IDX,
    WP_PSI_IDX,
    WP_X_IDX,
    WP_Y_IDX,
)


@dataclass
class OpponentDetection:
    """A single-frame opponent detection (a cluster of lidar points)."""

    position_global: np.ndarray  # (2,) centroid in map coordinates [x, y]
    position_relative: np.ndarray  # (2,) in ego frame [forward, left]
    distance: float  # range from ego origin to centroid [m]
    bearing: float  # angle to centroid in ego frame [rad], 0 = straight ahead
    size: float  # cluster extent (max pairwise distance) [m]
    num_points: int  # number of lidar points in the cluster


@dataclass
class OpponentTrack:
    """A tracked opponent, persisting across frames with a velocity estimate."""

    track_id: int
    position_global: np.ndarray  # (2,) smoothed map position
    velocity_global: np.ndarray  # (2,) estimated map-frame velocity [m/s]
    position_relative: np.ndarray  # (2,) in ego frame [forward, left]
    distance: float
    bearing: float
    size: float
    num_points: int
    hits: int = 1  # number of frames this track has been confirmed
    missed: int = 0  # consecutive frames without an associated detection
    last_global: np.ndarray = field(default=None, repr=False)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity_global))


def _to_car_frame(points_global: np.ndarray, car_state: np.ndarray) -> np.ndarray:
    """Transform (N, 2) map points into the ego frame [forward, left]."""
    c = float(car_state[POSE_THETA_COS_IDX])
    s = float(car_state[POSE_THETA_SIN_IDX])
    dx = points_global[:, 0] - float(car_state[POSE_X_IDX])
    dy = points_global[:, 1] - float(car_state[POSE_Y_IDX])
    forward = dx * c + dy * s
    left = -dx * s + dy * c
    return np.column_stack((forward, left))


class OpponentTracker:
    """Cluster in-corridor lidar returns into opponents and track them."""

    def __init__(
        self,
        *,
        dt: float = 0.02,
        wall_margin: float = 0.30,
        max_range: float = 8.0,
        min_range: float = 0.05,
        cluster_gap: float = 0.40,
        cluster_angular_gap_deg: float = 3.0,
        min_points: int = 2,
        min_size: float = 0.0,
        max_size: float = 1.5,
        gating_distance: float = 1.5,
        max_missed: int = 5,
        velocity_smoothing: float = 0.5,
        history_len: int = 5,
    ):
        self.dt = float(dt)
        self.wall_margin = float(wall_margin)
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.cluster_gap = float(cluster_gap)
        self.cluster_angular_gap_rad = float(np.deg2rad(cluster_angular_gap_deg))
        self.min_points = int(min_points)
        self.min_size = float(min_size)
        self.max_size = float(max_size)
        self.gating_distance = float(gating_distance)
        self.max_missed = int(max_missed)
        self.velocity_smoothing = float(np.clip(velocity_smoothing, 0.0, 1.0))
        self.history_len = max(1, int(history_len))

        self.tracks: List[OpponentTrack] = []
        self.detections: List[OpponentDetection] = []
        self._next_track_id = 0
        # Rolling history of the closest-opponent feature vector (one entry per
        # update()/control step), oldest -> newest. Lets the policy see how the
        # opponent's position and appearance evolve, which is needed for overtaking.
        self._nearest_history: deque[np.ndarray] = deque(maxlen=self.history_len)

        _warmup_jit()

    @classmethod
    def from_settings(cls) -> "OpponentTracker":
        from utilities.Settings import Settings

        return cls(
            dt=float(getattr(Settings, "TIMESTEP_CONTROL", 0.02)),
            wall_margin=float(getattr(Settings, "OPPONENT_TRACKER_WALL_MARGIN", 0.30)),
            max_range=float(getattr(Settings, "OPPONENT_TRACKER_MAX_RANGE", 8.0)),
            cluster_gap=float(getattr(Settings, "OPPONENT_TRACKER_CLUSTER_GAP", 0.40)),
            cluster_angular_gap_deg=float(
                getattr(Settings, "OPPONENT_TRACKER_CLUSTER_ANGULAR_GAP_DEG", 3.0)
            ),
            min_points=int(getattr(Settings, "OPPONENT_TRACKER_MIN_POINTS", 2)),
            min_size=float(getattr(Settings, "OPPONENT_TRACKER_MIN_SIZE", 0.0)),
            max_size=float(getattr(Settings, "OPPONENT_TRACKER_MAX_SIZE", 1.5)),
            gating_distance=float(
                getattr(Settings, "OPPONENT_TRACKER_GATING_DISTANCE", 1.5)
            ),
            max_missed=int(getattr(Settings, "OPPONENT_TRACKER_MAX_MISSED", 5)),
            history_len=int(getattr(Settings, "OPPONENT_TRACKER_HISTORY_LEN", 5)),
        )

    def reset(self) -> None:
        self.tracks = []
        self.detections = []
        self._next_track_id = 0
        self._nearest_history.clear()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        car_state: Optional[np.ndarray],
        lidar_ranges: Optional[np.ndarray],
        lidar_angles_rad: Optional[np.ndarray],
        waypoints: Optional[np.ndarray],
    ) -> List[OpponentTrack]:
        """Run one detection + tracking step.

        Uses the full (non-decimated) lidar scan: pass every beam's range and
        angle. The global hit points are computed internally with the same lidar
        offset / transform used elsewhere in the stack.

        Args:
            car_state: ego state vector (uses POSE_* indices).
            lidar_ranges: (N,) raw lidar ranges for every beam.
            lidar_angles_rad: (N,) beam angles aligned with the ranges.
            waypoints: (M, 10) local corridor waypoints around the ego (from
                ``WaypointUtils.get_corridor_waypoints``; used for wall removal).

        Returns:
            The current list of confirmed opponent tracks.
        """
        self.detections = self._detect(
            car_state, lidar_ranges, lidar_angles_rad, waypoints
        )
        self._track(car_state)
        # Record the closest opponent once per control step for temporal history.
        self._nearest_history.append(self._nearest_feature_vector(car_state))
        return self.tracks

    def _ego_velocity(self, car_state: Optional[np.ndarray]) -> tuple[float, float]:
        """Ego velocity in the ego frame [forward, left]."""
        if car_state is None:
            return 0.0, 0.0
        vx = float(car_state[LINEAR_VEL_X_IDX])
        vy = float(car_state[LINEAR_VEL_Y_IDX])
        c = float(car_state[POSE_THETA_COS_IDX])
        s = float(car_state[POSE_THETA_SIN_IDX])
        return vx * c + vy * s, -vx * s + vy * c

    def _track_relative_velocity(
        self, track: OpponentTrack, car_state: Optional[np.ndarray]
    ) -> tuple[float, float]:
        """Opponent velocity relative to ego, in the ego frame [forward, left]."""
        opp_vx = float(track.velocity_global[0])
        opp_vy = float(track.velocity_global[1])
        if car_state is None:
            return opp_vx, opp_vy
        c = float(car_state[POSE_THETA_COS_IDX])
        s = float(car_state[POSE_THETA_SIN_IDX])
        ego_v_forward, ego_v_left = self._ego_velocity(car_state)
        rel_v_forward = (opp_vx * c + opp_vy * s) - ego_v_forward
        rel_v_left = (-opp_vx * s + opp_vy * c) - ego_v_left
        return rel_v_forward, rel_v_left

    def _nearest_feature_vector(
        self, car_state: Optional[np.ndarray]
    ) -> np.ndarray:
        """Feature vector for the single closest opponent (or absent default).

        Layout: [present, forward, left, range, bearing, speed,
        rel_vel_forward, rel_vel_left]. ``present`` is 0.0 when no opponent is
        tracked (all other entries are 0.0 in that case).
        """
        if not self.tracks:
            return np.zeros(NEAREST_OPPONENT_FEATURE_SIZE, dtype=np.float32)
        track = min(self.tracks, key=lambda t: t.distance)
        rel_v_forward, rel_v_left = self._track_relative_velocity(track, car_state)
        return np.array(
            [
                1.0,
                float(track.position_relative[0]),
                float(track.position_relative[1]),
                float(track.distance),
                float(track.bearing),
                float(track.speed),
                rel_v_forward,
                rel_v_left,
            ],
            dtype=np.float32,
        )

    def _nearest_history_array(self) -> np.ndarray:
        """(history_len, 8) history of the closest opponent, oldest -> newest.

        Front-padded with absent (all-zero) frames until enough steps accumulate.
        """
        history = np.zeros(
            (self.history_len, NEAREST_OPPONENT_FEATURE_SIZE), dtype=np.float32
        )
        if self._nearest_history:
            stacked = np.asarray(list(self._nearest_history), dtype=np.float32)
            n = min(stacked.shape[0], self.history_len)
            history[-n:] = stacked[-n:]
        return history

    def get_detections(self) -> List[OpponentDetection]:
        return self.detections

    def get_tracks(self) -> List[OpponentTrack]:
        return self.tracks

    def get_render_points(self) -> np.ndarray:
        """Global (x, y) positions for web/pygame rendering; empty when no tracks."""
        if not self.tracks:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(
            [t.position_global for t in self.tracks], dtype=np.float32
        ).reshape(-1, 2)

    def nearest_track(self) -> Optional[OpponentTrack]:
        if not self.tracks:
            return None
        return min(self.tracks, key=lambda t: t.distance)

    def get_relative_positions(self) -> np.ndarray:
        """Return (K, 2) relative [forward, left] positions of all tracks."""
        if not self.tracks:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(
            [t.position_relative for t in self.tracks], dtype=np.float32
        )

    @staticmethod
    def empty_controller_observation(history_len: int = 5) -> dict[str, np.ndarray]:
        """Default opponent fields when the tracker is disabled or has no tracks."""
        return {
            "detected_opponents": np.zeros((0, 7), dtype=np.float32),
            "detected_opponent_positions": np.zeros((0, 2), dtype=np.float32),
            "num_detected_opponents": np.array([0], dtype=np.int32),
            "nearest_detected_opponent": np.zeros(7, dtype=np.float32),
            "nearest_detected_opponent_distance": np.array([np.inf], dtype=np.float32),
            "nearest_detected_opponent_history": np.zeros(
                (max(1, int(history_len)), NEAREST_OPPONENT_FEATURE_SIZE), dtype=np.float32
            ),
        }

    def to_controller_observation(
        self, car_state: Optional[np.ndarray] = None
    ) -> dict[str, np.ndarray]:
        """Export tracked opponents for planner / RL controller observations.

        ``detected_opponents`` rows are:
        [forward, left, range, bearing, speed, rel_vel_forward, rel_vel_left]
        all in the ego frame except ``speed`` (scalar magnitude in map frame).
        """
        if not self.tracks:
            obs = self.empty_controller_observation(self.history_len)
            obs["nearest_detected_opponent_history"] = self._nearest_history_array()
            return obs

        rows = []
        positions = []
        for track in self.tracks:
            rel_v_forward, rel_v_left = self._track_relative_velocity(track, car_state)
            rows.append(
                [
                    float(track.position_relative[0]),
                    float(track.position_relative[1]),
                    float(track.distance),
                    float(track.bearing),
                    float(track.speed),
                    rel_v_forward,
                    rel_v_left,
                ]
            )
            positions.append(track.position_global.astype(np.float32))

        detected = np.asarray(rows, dtype=np.float32).reshape(-1, 7)
        nearest_idx = int(np.argmin(detected[:, 2]))
        return {
            "detected_opponents": detected,
            "detected_opponent_positions": np.asarray(positions, dtype=np.float32).reshape(
                -1, 2
            ),
            "num_detected_opponents": np.array([detected.shape[0]], dtype=np.int32),
            "nearest_detected_opponent": detected[nearest_idx].copy(),
            "nearest_detected_opponent_distance": np.array(
                [float(detected[nearest_idx, 2])], dtype=np.float32
            ),
            "nearest_detected_opponent_history": self._nearest_history_array(),
        }

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def _detect(
        self,
        car_state: Optional[np.ndarray],
        lidar_ranges: Optional[np.ndarray],
        lidar_angles_rad: Optional[np.ndarray],
        waypoints: Optional[np.ndarray],
    ) -> List[OpponentDetection]:
        if (
            car_state is None
            or lidar_ranges is None
            or lidar_angles_rad is None
            or waypoints is None
            or len(lidar_ranges) == 0
            or len(waypoints) == 0
        ):
            return []

        ranges = np.asarray(lidar_ranges, dtype=np.float64)
        angles = np.asarray(lidar_angles_rad, dtype=np.float64)
        if ranges.shape[0] != angles.shape[0]:
            return []

        # 1) keep only valid, in-range returns (also bounds the cost below)
        valid_idx = np.where((ranges > self.min_range) & (ranges < self.max_range))[0]
        if valid_idx.size == 0:
            return []

        # Compute global hit points for the valid beams (same transform/offset
        # as lidar_utils so coordinates match the rest of the stack).
        rel_points = get_points_from_ranges(ranges[valid_idx], angles[valid_idx])
        points = np.asarray(
            transform_points_from_car_to_global(car_state, rel_points),
            dtype=np.float64,
        )

        # 2) keep only returns inside the drivable corridor (drop walls)
        inside = self._inside_track_corridor(points, waypoints)
        if not np.any(inside):
            return []

        keep_local = np.where(inside)[0]
        kept_points = points[keep_local]
        kept_angles = angles[valid_idx[keep_local]]

        # 3) cluster by angular adjacency (beams are ordered by angle)
        clusters = self._cluster(kept_points, kept_angles)

        detections: List[OpponentDetection] = []
        for cluster_points in clusters:
            if cluster_points.shape[0] < self.min_points:
                continue
            size = _cluster_extent(cluster_points)
            if size < self.min_size or size > self.max_size:
                continue
            centroid = cluster_points.mean(axis=0)
            rel = _to_car_frame(centroid[None, :], car_state)[0]
            detections.append(
                OpponentDetection(
                    position_global=centroid.astype(np.float32),
                    position_relative=rel.astype(np.float32),
                    distance=float(np.hypot(rel[0], rel[1])),
                    bearing=float(np.arctan2(rel[1], rel[0])),
                    size=float(size),
                    num_points=int(cluster_points.shape[0]),
                )
            )
        return detections

    def _inside_track_corridor(
        self,
        points: np.ndarray,
        waypoints: np.ndarray,
    ) -> np.ndarray:
        """Boolean mask: True where a lidar point lies inside the track corridor.

        Expects the local corridor waypoint slice from
        ``WaypointUtils.get_corridor_waypoints`` (centred on the car, spanning
        roughly the lidar range in both directions along the track).
        """
        if waypoints.shape[0] == 0:
            return np.zeros(points.shape[0], dtype=bool)

        wp_xy = np.ascontiguousarray(
            waypoints[:, [WP_X_IDX, WP_Y_IDX]].astype(np.float64)
        )
        wp_psi = np.ascontiguousarray(waypoints[:, WP_PSI_IDX].astype(np.float64))
        wp_d_left = np.ascontiguousarray(waypoints[:, WP_D_LEFT_IDX].astype(np.float64))
        wp_d_right = np.ascontiguousarray(waypoints[:, WP_D_RIGHT_IDX].astype(np.float64))

        return _corridor_inside_jit(
            np.ascontiguousarray(points),
            wp_xy,
            wp_psi,
            wp_d_left,
            wp_d_right,
            float(self.wall_margin),
        )

    def _cluster(
        self,
        kept_points: np.ndarray,
        kept_angles: np.ndarray,
    ) -> List[np.ndarray]:
        """Split kept points into clusters by angular and spatial adjacency.

        Resolution-independent: a new cluster starts when the angular gap to the
        previous kept beam exceeds ``cluster_angular_gap_rad`` (i.e. there is a
        blind spot between them) or the Euclidean gap exceeds ``cluster_gap``.
        """
        if kept_points.shape[0] == 0:
            return []
        ids = _cluster_ids_jit(
            np.ascontiguousarray(kept_points),
            np.ascontiguousarray(kept_angles),
            float(self.cluster_angular_gap_rad),
            float(self.cluster_gap),
        )
        # Contiguous runs share an id (points are angle-ordered), so split on change.
        boundaries = np.where(np.diff(ids) != 0)[0] + 1
        return np.split(kept_points, boundaries)

    # ------------------------------------------------------------------ #
    # Tracking
    # ------------------------------------------------------------------ #
    def _track(self, car_state: np.ndarray) -> None:
        detections = self.detections

        # Greedy nearest-neighbour association in the global frame.
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))

        if self.tracks and detections:
            track_pos = np.asarray([t.position_global for t in self.tracks])
            det_pos = np.asarray([d.position_global for d in detections])
            cost = np.linalg.norm(
                track_pos[:, None, :] - det_pos[None, :, :], axis=2
            )
            pairs = sorted(
                (
                    (cost[ti, di], ti, di)
                    for ti in range(len(self.tracks))
                    for di in range(len(detections))
                ),
                key=lambda x: x[0],
            )
            for dist, ti, di in pairs:
                if ti not in unmatched_tracks or di not in unmatched_dets:
                    continue
                if dist > self.gating_distance:
                    continue
                self._update_track(self.tracks[ti], detections[di])
                unmatched_tracks.discard(ti)
                unmatched_dets.discard(di)

        # New tracks for unmatched detections.
        for di in unmatched_dets:
            self._spawn_track(detections[di])

        # Age unmatched tracks; drop the stale ones.
        for ti in unmatched_tracks:
            track = self.tracks[ti]
            track.missed += 1
            # Coast the prediction forward and refresh the ego-relative pose.
            track.position_global = (
                track.position_global + track.velocity_global * self.dt
            )
            self._refresh_relative(track, car_state)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

    def _spawn_track(self, det: OpponentDetection) -> None:
        track = OpponentTrack(
            track_id=self._next_track_id,
            position_global=det.position_global.astype(np.float64).copy(),
            velocity_global=np.zeros(2, dtype=np.float64),
            position_relative=det.position_relative.copy(),
            distance=det.distance,
            bearing=det.bearing,
            size=det.size,
            num_points=det.num_points,
            hits=1,
            missed=0,
            last_global=det.position_global.astype(np.float64).copy(),
        )
        self._next_track_id += 1
        self.tracks.append(track)

    def _update_track(self, track: OpponentTrack, det: OpponentDetection) -> None:
        new_global = det.position_global.astype(np.float64)
        if self.dt > 0.0:
            measured_velocity = (new_global - track.position_global) / self.dt
            alpha = self.velocity_smoothing
            track.velocity_global = (
                alpha * track.velocity_global + (1.0 - alpha) * measured_velocity
            )
        track.last_global = track.position_global.copy()
        track.position_global = new_global.copy()
        track.position_relative = det.position_relative.copy()
        track.distance = det.distance
        track.bearing = det.bearing
        track.size = det.size
        track.num_points = det.num_points
        track.hits += 1
        track.missed = 0

    def _refresh_relative(
        self, track: OpponentTrack, car_state: np.ndarray
    ) -> None:
        rel = _to_car_frame(track.position_global[None, :], car_state)[0]
        track.position_relative = rel.astype(np.float32)
        track.distance = float(np.hypot(rel[0], rel[1]))
        track.bearing = float(np.arctan2(rel[1], rel[0]))


def _cluster_extent(cluster_points: np.ndarray) -> float:
    """Largest pairwise distance within a cluster (a cheap size proxy)."""
    if cluster_points.shape[0] <= 1:
        return 0.0
    mins = cluster_points.min(axis=0)
    maxs = cluster_points.max(axis=0)
    return float(np.hypot(maxs[0] - mins[0], maxs[1] - mins[1]))


@njit(cache=True, fastmath=True)
def _corridor_inside_jit(points, wp_xy, wp_psi, wp_d_left, wp_d_right, wall_margin):
    """Per-point in-corridor mask using a nearest-waypoint loop (no big allocs).

    For each lidar point, find the nearest (windowed) waypoint, project the
    offset onto that waypoint's lateral normal, and keep the point if it lies
    inside the track bounds (minus ``wall_margin`` so wall returns are dropped).
    """
    n = points.shape[0]
    k = wp_xy.shape[0]
    inside = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        px = points[i, 0]
        py = points[i, 1]
        best_d2 = 1.0e18
        best_j = 0
        for j in range(k):
            dx = px - wp_xy[j, 0]
            dy = py - wp_xy[j, 1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        psi = wp_psi[best_j]
        n_x = -np.sin(psi)
        n_y = np.cos(psi)
        dx = px - wp_xy[best_j, 0]
        dy = py - wp_xy[best_j, 1]
        lateral = dx * n_x + dy * n_y
        if lateral < (wp_d_left[best_j] - wall_margin) and lateral > -(
            wp_d_right[best_j] - wall_margin
        ):
            inside[i] = True
    return inside


@njit(cache=True, fastmath=True)
def _cluster_ids_jit(points, angles, angular_gap, spatial_gap):
    """Assign a cluster id per (angle-ordered) point by adjacency gaps."""
    n = points.shape[0]
    ids = np.zeros(n, dtype=np.int64)
    current = 0
    for i in range(1, n):
        ang_gap = angles[i] - angles[i - 1]
        if ang_gap < 0.0:
            ang_gap = -ang_gap
        dx = points[i, 0] - points[i - 1, 0]
        dy = points[i, 1] - points[i - 1, 1]
        spatial = (dx * dx + dy * dy) ** 0.5
        if ang_gap > angular_gap or spatial > spatial_gap:
            current += 1
        ids[i] = current
    return ids


_JIT_WARMED_UP = False


def _warmup_jit() -> None:
    """Trigger numba compilation once so the first real step isn't stalled."""
    global _JIT_WARMED_UP
    if _JIT_WARMED_UP:
        return
    pts = np.zeros((2, 2), dtype=np.float64)
    col = np.zeros(2, dtype=np.float64)
    _corridor_inside_jit(pts, pts, col, col, col, 0.0)
    _cluster_ids_jit(pts, col, 0.1, 0.1)
    _JIT_WARMED_UP = True

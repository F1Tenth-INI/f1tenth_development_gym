"""Opponents: physics cars or trajectory-replay stand-ins with the same observation.

Virtual opponents replay a recording (no physics, no sensors) and ray-cast their
bodies into the ego lidar. Physics opponents are extra CarSystem agents. Both
fill the same privileged observation slots used by reward, termination, and RL.
"""

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
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    NUMBER_OF_STATES,
    POSE_THETA_COS_IDX,
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)

# Privileged ego-frame feature vector per virtual opponent:
# [present, forward, left, heading_rel, vx_body]
# vx_body = opponent forward speed in its own body frame [m/s] (not relative to ego).
VIRTUAL_OPPONENT_STATE_SIZE = 5
_VELOCITY_FD_EPS_S = 0.05


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


def opponent_count() -> int:
    """Configured opponent count (virtual replay or physics cars)."""
    return max(int(getattr(Settings, "NUMBER_OF_OPPONENTS", 0) or 0), 0)


def opponents_are_virtual() -> bool:
    """True when opponents replay recordings instead of running physics agents."""
    return bool(getattr(Settings, "OPPONENTS_VIRTUAL", False))


def solo_episode_fraction() -> float:
    """Probability of a true solo episode (no opponents in lidar / obs / reward)."""
    return float(
        getattr(Settings, "SOLO_EPISODE_FRACTION", None)
        or getattr(Settings, "VO_SOLO_EPISODE_FRACTION", 0.0)
        or 0.0
    )


def is_solo_episode() -> bool:
    """Whether this episode hid all opponents. Slot count is unchanged."""
    return bool(getattr(Settings, "SOLO_EPISODE", False))


def roll_solo_episode(rng: Optional[np.random.Generator] = None) -> bool:
    """Bernoulli roll from SOLO_EPISODE_FRACTION; writes Settings.SOLO_EPISODE."""
    frac = float(np.clip(solo_episode_fraction(), 0.0, 1.0))
    if frac <= 0.0:
        solo = False
    elif frac >= 1.0:
        solo = True
    else:
        draw = rng.random() if rng is not None else float(np.random.random())
        solo = bool(draw < frac)
    Settings.SOLO_EPISODE = solo
    return solo


def physics_opponent_count() -> int:
    """How many extra world-sim / CarSystem agents to spawn."""
    if opponents_are_virtual():
        return 0
    return opponent_count()


def virtual_opponent_slot_count() -> int:
    """Observation slots: at least one so the RL obs dim stays defined when N=0."""
    return max(opponent_count(), 1)


def empty_virtual_opponent_observation() -> dict:
    """Default privileged opponent fields when none are spawned this episode."""
    n_slots = virtual_opponent_slot_count()
    n_opp = opponent_count()
    return {
        "virtual_opponent_poses": np.zeros((0, 3), dtype=np.float32),
        "virtual_opponent_states": np.zeros(
            (n_slots, VIRTUAL_OPPONENT_STATE_SIZE), dtype=np.float32
        ),
        "min_virtual_opponent_distance": float("inf"),
        "virtual_opponent_collision": False,
        "virtual_opponent_waypoint_indices": np.full((n_opp,), -1, dtype=np.int32),
    }


def _pose_from_car_state(car_state: np.ndarray) -> np.ndarray:
    state = np.asarray(car_state, dtype=np.float64)
    return np.array(
        [state[POSE_X_IDX], state[POSE_Y_IDX], state[POSE_THETA_IDX]],
        dtype=np.float64,
    )


def _map_velocity_from_car_state(car_state: np.ndarray) -> np.ndarray:
    """Body-frame (vx, vy) → map-frame [vx, vy]."""
    state = np.asarray(car_state, dtype=np.float64)
    vx = float(state[LINEAR_VEL_X_IDX])
    vy = float(state[LINEAR_VEL_Y_IDX])
    c = float(np.cos(state[POSE_THETA_IDX]))
    s = float(np.sin(state[POSE_THETA_IDX]))
    return np.array([vx * c - vy * s, vx * s + vy * c], dtype=np.float64)


def _waypoint_index_for_pose(pose: np.ndarray, waypoints: Optional[np.ndarray]) -> int:
    if waypoints is None or len(waypoints) == 0:
        return -1
    from utilities.waypoint_utils import get_nearest_waypoint

    dummy = np.zeros(NUMBER_OF_STATES, dtype=np.float64)
    dummy[POSE_X_IDX] = float(pose[0])
    dummy[POSE_Y_IDX] = float(pose[1])
    idx, _ = get_nearest_waypoint(dummy, np.asarray(waypoints))
    return int(idx)


def relative_opponent_states(
    ego_car_state: np.ndarray,
    poses: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """Padded ego-frame feature matrix, shape (slot_count, VIRTUAL_OPPONENT_STATE_SIZE).

    Layout per row: present, forward, left, heading_rel, vx_body.
    """
    n_slots = virtual_opponent_slot_count()
    states = np.zeros((n_slots, VIRTUAL_OPPONENT_STATE_SIZE), dtype=np.float32)
    if ego_car_state is None or poses is None or len(poses) == 0:
        return states

    ego = np.asarray(ego_car_state, dtype=np.float64)
    ego_c = float(ego[POSE_THETA_COS_IDX])
    ego_s = float(ego[POSE_THETA_SIN_IDX])
    ego_theta = float(ego[POSE_THETA_IDX])
    poses = np.asarray(poses, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 2)

    n = min(len(poses), n_slots, len(velocities) if len(velocities) else len(poses))
    for i in range(n):
        pose = poses[i]
        vel = velocities[i] if i < len(velocities) else np.zeros(2, dtype=np.float64)
        dx = float(pose[0]) - float(ego[POSE_X_IDX])
        dy = float(pose[1]) - float(ego[POSE_Y_IDX])
        forward = dx * ego_c + dy * ego_s
        left = -dx * ego_s + dy * ego_c
        heading_rel = float(
            np.arctan2(np.sin(pose[2] - ego_theta), np.cos(pose[2] - ego_theta))
        )
        opp_c = float(np.cos(pose[2]))
        opp_s = float(np.sin(pose[2]))
        vx_body = float(vel[0]) * opp_c + float(vel[1]) * opp_s
        states[i] = np.array(
            [1.0, forward, left, heading_rel, vx_body],
            dtype=np.float32,
        )
    return states


def opponents_collide_with_ego(
    ego_car_state: np.ndarray,
    poses: np.ndarray,
    ego_length: float,
    ego_width: float,
    opp_length: float,
    opp_width: float,
) -> bool:
    if ego_car_state is None or poses is None or len(poses) == 0:
        return False
    ego_pose = _pose_from_car_state(ego_car_state)
    ego_verts = np.ascontiguousarray(get_vertices(ego_pose, ego_length, ego_width))
    for pose in np.asarray(poses, dtype=np.float64):
        opp_verts = np.ascontiguousarray(get_vertices(pose, opp_length, opp_width))
        if collision(ego_verts, opp_verts):
            return True
    return False


def min_clearance_to_ego_from_poses(
    ego_car_state: np.ndarray,
    poses: np.ndarray,
    ego_length: float,
    ego_width: float,
    opp_length: float,
    opp_width: float,
) -> float:
    """Minimum center-based clearance to any opponent body (0 if overlapping)."""
    if ego_car_state is None or poses is None or len(poses) == 0:
        return float("inf")
    ego_pose = _pose_from_car_state(ego_car_state)
    ego_verts = get_vertices(ego_pose, ego_length, ego_width)
    ego_radius = 0.5 * float(np.linalg.norm(ego_verts[0] - ego_verts[2]))
    min_clearance = float("inf")
    for pose in np.asarray(poses, dtype=np.float64):
        opp_verts = get_vertices(pose, opp_length, opp_width)
        if collision(np.ascontiguousarray(ego_verts), np.ascontiguousarray(opp_verts)):
            return 0.0
        opp_radius = 0.5 * float(np.linalg.norm(opp_verts[0] - opp_verts[2]))
        center_dist = float(
            np.linalg.norm(ego_verts.mean(axis=0) - opp_verts.mean(axis=0))
        )
        min_clearance = min(min_clearance, center_dist - ego_radius - opp_radius)
    return max(0.0, min_clearance)


def observation_from_poses(
    ego_car_state: np.ndarray,
    poses: np.ndarray,
    velocities: np.ndarray,
    waypoint_indices: np.ndarray,
    *,
    collision: bool = False,
    length: float,
    width: float,
) -> dict:
    """Privileged opponent observation dict from poses (virtual or physics)."""
    if poses is None or len(poses) == 0 or ego_car_state is None:
        obs = empty_virtual_opponent_observation()
        obs["virtual_opponent_collision"] = bool(collision)
        return obs

    poses = np.asarray(poses, dtype=np.float32)
    ego_length, ego_width = get_ego_car_dimensions()
    return {
        "virtual_opponent_poses": poses,
        "virtual_opponent_states": relative_opponent_states(
            ego_car_state, poses, velocities
        ),
        "min_virtual_opponent_distance": min_clearance_to_ego_from_poses(
            ego_car_state, poses, ego_length, ego_width, length, width
        ),
        "virtual_opponent_collision": bool(collision),
        "virtual_opponent_waypoint_indices": np.asarray(
            waypoint_indices, dtype=np.int32
        ).reshape(-1),
    }


def observation_from_car_states(
    ego_car_state: np.ndarray,
    car_states: list,
    *,
    ego_index: int = 0,
    waypoints: Optional[np.ndarray] = None,
    collision: bool = False,
) -> dict:
    """Build the privileged opponent dict from extra physics agents."""
    if car_states is None or len(car_states) <= 1 or ego_car_state is None:
        return empty_virtual_opponent_observation()

    poses = []
    velocities = []
    waypoint_indices = []
    for i, state in enumerate(car_states):
        if i == int(ego_index):
            continue
        state = np.asarray(state, dtype=np.float64)
        pose = _pose_from_car_state(state)
        poses.append(pose)
        velocities.append(_map_velocity_from_car_state(state))
        waypoint_indices.append(_waypoint_index_for_pose(pose, waypoints))

    if not poses:
        return empty_virtual_opponent_observation()

    length, width = get_ego_car_dimensions()
    return observation_from_poses(
        ego_car_state,
        np.asarray(poses, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
        np.asarray(waypoint_indices, dtype=np.int32),
        collision=collision,
        length=length,
        width=width,
    )


def opponent_poses_from_car_states(car_states: list, ego_index: int = 0) -> np.ndarray:
    """Global [x, y, theta] for every physics agent except ego."""
    poses = []
    for i, state in enumerate(car_states or []):
        if i == int(ego_index):
            continue
        poses.append(_pose_from_car_state(state))
    if not poses:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(poses, dtype=np.float32)


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
        self._current_pose: Optional[np.ndarray] = None  # [x, y, theta]
        self._current_recording_time: Optional[float] = None
        self._current_velocity: Optional[np.ndarray] = None  # [vx, vy] map frame [m/s]

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
        self._current_recording_time = None
        self._current_velocity = None

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
        self._current_recording_time = float(recording_t)
        self._current_velocity = self._velocity_at_recording_time(recording_t)
        return pose.copy()

    def _velocity_at_recording_time(self, recording_t: float) -> np.ndarray:
        """Map-frame [vx, vy] from a recording-time finite difference, scaled by vel_factor."""
        t0 = float(recording_t)
        t_end = float(self.times[-1])
        t_start = float(self.times[0])
        eps = _VELOCITY_FD_EPS_S
        if t0 + eps <= t_end:
            t1 = t0 + eps
            p0 = interpolate_pose(self.times, self.poses, t0)
            p1 = interpolate_pose(self.times, self.poses, t1)
            dt_rec = eps
        elif t0 - eps >= t_start:
            t1 = t0
            t0 = t0 - eps
            p0 = interpolate_pose(self.times, self.poses, t0)
            p1 = interpolate_pose(self.times, self.poses, t1)
            dt_rec = eps
        else:
            return np.zeros(2, dtype=np.float64)
        dxy_drec = (p1[:2] - p0[:2]) / dt_rec
        return dxy_drec * self.vel_factor

    def current_pose(self) -> np.ndarray:
        if self._current_pose is None:
            raise RuntimeError("Virtual opponent pose not available.")
        return self._current_pose.copy()

    def current_velocity(self) -> np.ndarray:
        """Map-frame [vx, vy] in m/s. Zeros before the first pose update."""
        if self._current_velocity is None:
            return np.zeros(2, dtype=np.float64)
        return self._current_velocity.copy()

    def current_waypoint_index(self) -> Optional[int]:
        """Along-track waypoint index at the current replay pose (for overtake detection)."""
        if self._current_recording_time is None:
            return None
        rt = float(self._current_recording_time)
        idx = int(np.searchsorted(self.times, rt, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.waypoint_indices) - 1))
        return int(self.waypoint_indices[idx] % self.total_waypoints)

    def current_vertices(self) -> np.ndarray:
        return get_vertices(self.current_pose(), self.length, self.width)


def _require_per_opponent_array(attr_name: str, count: int) -> list:
    values = list(getattr(Settings, attr_name, []) or [])
    if len(values) < count:
        raise ValueError(
            f"Settings.{attr_name} must have at least length {count} "
            f"(NUMBER_OF_OPPONENTS), got {len(values)}"
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
        count = opponent_count() if opponents_are_virtual() else 0
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
        map_scale = float(getattr(Settings, "MAP_SCALE", 1.0))
        vel_factors = [float(v) / map_scale for v in vel_factors]
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

    def get_velocities(self) -> np.ndarray:
        """Map-frame [vx, vy] per opponent. Empty (0, 2) before the first update."""
        if not self._initialized:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(
            [opponent.current_velocity() for opponent in self.opponents],
            dtype=np.float32,
        )

    def get_relative_states(self, ego_car_state: np.ndarray) -> np.ndarray:
        """Padded ego-frame feature matrix, shape (slot_count, VIRTUAL_OPPONENT_STATE_SIZE).

        Layout per row: present, forward, left, heading_rel, vx_body.
        """
        n_slots = virtual_opponent_slot_count()
        if not self._initialized or ego_car_state is None:
            return np.zeros((n_slots, VIRTUAL_OPPONENT_STATE_SIZE), dtype=np.float32)
        poses = np.asarray(
            [self.opponents[i].current_pose() for i in range(len(self.opponents))],
            dtype=np.float64,
        )
        velocities = np.asarray(
            [self.opponents[i].current_velocity() for i in range(len(self.opponents))],
            dtype=np.float64,
        )
        return relative_opponent_states(ego_car_state, poses, velocities)

    def get_waypoint_indices(self) -> np.ndarray:
        """Per-opponent along-track waypoint index; empty before the first update."""
        if not self._initialized:
            return np.zeros((0,), dtype=np.int32)
        indices = [opponent.current_waypoint_index() for opponent in self.opponents]
        return np.asarray(
            [int(i) if i is not None else -1 for i in indices],
            dtype=np.int32,
        )

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
        if not self._initialized or not self.opponents:
            return False
        poses = np.asarray(
            [opponent.current_pose() for opponent in self.opponents], dtype=np.float64
        )
        return opponents_collide_with_ego(
            ego_car_state,
            poses,
            ego_length,
            ego_width,
            self.opponents[0].length,
            self.opponents[0].width,
        )

    def min_clearance_to_ego(
        self,
        ego_car_state: np.ndarray,
        ego_length: float,
        ego_width: float,
    ) -> float:
        """Minimum center-based clearance to any opponent body (0 if overlapping)."""
        if not self._initialized or not self.opponents:
            return float("inf")
        poses = np.asarray(
            [opponent.current_pose() for opponent in self.opponents], dtype=np.float64
        )
        return min_clearance_to_ego_from_poses(
            ego_car_state,
            poses,
            ego_length,
            ego_width,
            self.opponents[0].length,
            self.opponents[0].width,
        )

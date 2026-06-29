import numpy as np
from scipy.interpolate import CubicSpline

from utilities.Settings import Settings

# Waypoint column indices (mirror waypoint_utils; avoid circular import)
_WP_S, _WP_X, _WP_Y, _WP_PSI, _WP_KAPPA = 0, 1, 2, 3, 4
_WP_VX, _WP_AX, _WP_GID, _WP_DR, _WP_DL = 5, 6, 7, 8, 9


def scale_map_metadata(metadata: dict) -> dict:
    """Scale ROS map origin and resolution; image pixels stay unchanged."""
    s = Settings.MAP_SCALE
    if s == 1.0:
        return metadata
    out = dict(metadata)
    out["resolution"] = metadata["resolution"] * s
    ox, oy, oyaw = metadata["origin"]
    out["origin"] = [ox * s, oy * s, oyaw]
    return out


def scale_positions(positions):
    """Scale [x, y, yaw] starting positions (yaw unchanged)."""
    s = Settings.MAP_SCALE
    if s == 1.0:
        return positions
    return [[p[0] * s, p[1] * s, p[2]] for p in positions]


def median_waypoint_spacing(waypoints: np.ndarray) -> float:
    x = waypoints[:, _WP_X]
    y = waypoints[:, _WP_Y]
    return float(np.median(np.hypot(np.diff(x), np.diff(y))))


def resample_waypoints_uniform(waypoints: np.ndarray, ds: float) -> np.ndarray:
    """Resample raceline at uniform arc-length spacing; recompute psi/kappa from spline."""
    x = waypoints[:, _WP_X]
    y = waypoints[:, _WP_Y]
    seg_len = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = s[-1]
    if total_len < ds or len(s) < 3:
        return waypoints

    s_new = np.arange(0.0, total_len, ds)
    if len(s_new) < 2:
        return waypoints

    cs_x = CubicSpline(s, x, bc_type="natural")
    cs_y = CubicSpline(s, y, bc_type="natural")

    x_new = cs_x(s_new)
    y_new = cs_y(s_new)
    dx = cs_x(s_new, 1)
    dy = cs_y(s_new, 1)
    d2x = cs_x(s_new, 2)
    d2y = cs_y(s_new, 2)

    speed = np.hypot(dx, dy) + 1e-12
    psi = np.arctan2(dy, dx)
    kappa = (dx * d2y - dy * d2x) / (speed ** 3)

    s_norm = s_new / total_len
    s_frac = s / total_len

    out = np.zeros((len(s_new), waypoints.shape[1]), dtype=np.float32)
    out[:, _WP_S] = s_new.astype(np.float32)
    out[:, _WP_X] = x_new.astype(np.float32)
    out[:, _WP_Y] = y_new.astype(np.float32)
    out[:, _WP_PSI] = psi.astype(np.float32)
    out[:, _WP_KAPPA] = kappa.astype(np.float32)
    out[:, _WP_VX] = np.interp(s_norm, s_frac, waypoints[:, _WP_VX]).astype(np.float32)
    out[:, _WP_AX] = np.interp(s_norm, s_frac, waypoints[:, _WP_AX]).astype(np.float32)
    out[:, _WP_DR] = np.interp(s_norm, s_frac, waypoints[:, _WP_DR]).astype(np.float32)
    out[:, _WP_DL] = np.interp(s_norm, s_frac, waypoints[:, _WP_DL]).astype(np.float32)
    out[:, _WP_GID] = np.arange(len(s_new), dtype=np.float32)
    return out


def remap_sector_index(old_idx: int, old_n: int, new_n: int) -> int:
    if old_n <= 1 or new_n <= 1:
        return 0
    return int(round(old_idx * (new_n - 1) / (old_n - 1)))


def remap_waypoint_indices(indices: np.ndarray, old_n: int, new_n: int) -> np.ndarray:
    if old_n == new_n or old_n <= 1 or new_n <= 1:
        return np.asarray(indices, dtype=np.int64)
    mapped = np.round(indices.astype(np.float64) * (new_n - 1) / (old_n - 1))
    return mapped.astype(np.int64)


def scale_trajectory_poses(poses: np.ndarray) -> np.ndarray:
    """Scale recorded [x, y, theta] poses to match MAP_SCALE (theta unchanged)."""
    s = Settings.MAP_SCALE
    if s == 1.0:
        return poses
    scaled = np.asarray(poses, dtype=np.float64).copy()
    scaled[:, 0] *= s
    scaled[:, 1] *= s
    return scaled


def processed_waypoint_count_at_scale(scale: float) -> int:
    """Processed waypoint ring size for the current map at a given MAP_SCALE."""
    saved = Settings.MAP_SCALE
    Settings.MAP_SCALE = scale
    try:
        from utilities.waypoint_utils import WaypointUtils

        return len(WaypointUtils().waypoints)
    finally:
        Settings.MAP_SCALE = saved

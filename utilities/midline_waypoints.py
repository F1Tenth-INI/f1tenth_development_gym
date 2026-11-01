"""Load the map centerline as the waypoint table.

``centerline.csv`` in the map folder is the track midline
(``x_m, y_m, w_tr_right_m, w_tr_left_m``, no header). With
``Settings.WAYPOINT_SOURCE == "midline"`` those points replace the
minimum-curvature raceline. They then go through the same interpolation and
resolution reduction as a normal waypoint file.

Speeds are not read from the raceline. They come from the HWM initial-speed
settings when those are set, and otherwise from ``MIDLINE_DEFAULT_SPEED``.
``GLOBAL_WAYPOINT_VEL_FACTOR`` still scales them. The sector speed-scaling CSV
is indexed to the minimum-curvature waypoints, so it is not applied.
"""

from __future__ import annotations

import os

import numpy as np

from utilities.Settings import Settings

CENTERLINE_FILENAME = "centerline.csv"
_MIN_POINTS = 3


def uses_midline() -> bool:
    source = str(getattr(Settings, "WAYPOINT_SOURCE", "raceline")).strip().lower()
    if source == "midline":
        return True
    if source == "raceline":
        return False
    raise ValueError(
        f"Settings.WAYPOINT_SOURCE must be 'raceline' or 'midline', got {source!r}"
    )


def load_midline_array(map_path: str) -> np.ndarray:
    """``(N, 10)`` waypoints in the internal column layout. Speeds are filled later."""
    path = os.path.join(map_path, CENTERLINE_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Midline file not found: {path}. Expected {CENTERLINE_FILENAME} "
            f"in the map folder (x_m, y_m, w_tr_right_m, w_tr_left_m)."
        )
    rows = _read_centerline(path)
    if Settings.REVERSE_DIRECTION:
        rows = rows[::-1].copy()
        rows[:, [2, 3]] = rows[:, [3, 2]]
    rows = _drop_duplicate_points(rows)
    if len(rows) < _MIN_POINTS:
        raise ValueError(f"Midline file {path} has fewer than {_MIN_POINTS} points.")
    return _as_waypoint_table(rows[:, :2], rows[:, 2], rows[:, 3])


def apply_midline_speeds(waypoints: np.ndarray) -> str:
    """Write target speed and zero acceleration onto an already processed waypoint table.

    Returns a short description of which setting supplied the speed.
    """
    from utilities.waypoint_utils import WP_A_X_IDX, WP_VX_IDX, WP_X_IDX, WP_Y_IDX

    factor = float(np.clip(Settings.GLOBAL_WAYPOINT_VEL_FACTOR, 0, Settings.GLOBAL_SPEED_LIMIT))
    share = getattr(Settings, "HWM_INITIAL_SPEED_GRIP_SHARE", None)
    if share is not None:
        profile = _grip_profile(waypoints[:, [WP_X_IDX, WP_Y_IDX]], float(share))
        speed = np.minimum(profile * factor, profile)
        note = (
            f"HWM_INITIAL_SPEED_GRIP_SHARE={float(share):g} "
            f"(grip profile, cap {float(Settings.HWM_INITIAL_SPEED_MAX):g} m/s)"
        )
    elif Settings.HWM_LIMIT_INITIAL_WP_SPEED:
        limit = float(Settings.HWM_INITIAL_SPEED_LIMIT)
        speed = np.minimum(np.full(len(waypoints), limit, dtype=np.float64) * factor, limit)
        note = f"HWM_INITIAL_SPEED_LIMIT={limit:g} m/s"
    else:
        default = float(Settings.MIDLINE_DEFAULT_SPEED)
        speed = np.full(len(waypoints), default * factor, dtype=np.float64)
        note = f"MIDLINE_DEFAULT_SPEED={default:g} m/s"
    waypoints[:, WP_VX_IDX] = speed.astype(np.float32)
    waypoints[:, WP_A_X_IDX] = 0.0
    return (
        f"{note}, GLOBAL_WAYPOINT_VEL_FACTOR={factor:g} "
        f"-> {float(speed.min()):.2f}–{float(speed.max()):.2f} m/s"
    )


def _read_centerline(path: str) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"Midline file {path} has no numeric rows.")
    return np.asarray(rows, dtype=np.float64)


def _drop_duplicate_points(rows: np.ndarray) -> np.ndarray:
    if len(rows) < 2:
        return rows
    step = np.linalg.norm(np.diff(rows[:, :2], axis=0), axis=1)
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = step > 1e-4
    if np.linalg.norm(rows[0, :2] - rows[-1, :2]) < 1e-3:
        keep[-1] = False
    return rows[keep]


def _as_waypoint_table(xy: np.ndarray, width_right: np.ndarray, width_left: np.ndarray) -> np.ndarray:
    """Internal layout. Heading is 0 along +x, the same frame ``load_waypoints`` stores after adding pi/2."""
    from utilities.waypoint_utils import (
        WP_A_X_IDX,
        WP_D_LEFT_IDX,
        WP_D_RIGHT_IDX,
        WP_GLOBID_IDX,
        WP_KAPPA_IDX,
        WP_PSI_IDX,
        WP_S_IDX,
        WP_VX_IDX,
        WP_X_IDX,
        WP_Y_IDX,
    )

    xy = np.asarray(xy, dtype=np.float64)
    nxt = np.roll(xy, -1, axis=0)
    prev = np.roll(xy, 1, axis=0)
    forward = nxt - xy
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    heading = np.arctan2(forward[:, 1], forward[:, 0])
    a = np.linalg.norm(xy - prev, axis=1)
    b = np.linalg.norm(forward, axis=1)
    c = np.linalg.norm(nxt - prev, axis=1)
    cross = (xy[:, 0] - prev[:, 0]) * forward[:, 1] - (xy[:, 1] - prev[:, 1]) * forward[:, 0]
    kappa = 2.0 * cross / np.maximum(a * b * c, 1e-9)

    table = np.zeros((len(xy), 10), dtype=np.float32)
    table[:, WP_S_IDX] = arc.astype(np.float32)
    table[:, WP_X_IDX] = xy[:, 0].astype(np.float32)
    table[:, WP_Y_IDX] = xy[:, 1].astype(np.float32)
    table[:, WP_PSI_IDX] = heading.astype(np.float32)
    table[:, WP_KAPPA_IDX] = kappa.astype(np.float32)
    table[:, WP_VX_IDX] = 0.0
    table[:, WP_A_X_IDX] = 0.0
    table[:, WP_GLOBID_IDX] = np.arange(len(xy), dtype=np.float32)
    table[:, WP_D_RIGHT_IDX] = np.asarray(width_right, dtype=np.float32)
    table[:, WP_D_LEFT_IDX] = np.asarray(width_left, dtype=np.float32)
    return table


def _grip_profile(xy: np.ndarray, share: float) -> np.ndarray:
    import torch

    from TrainingLite.hwm.shared.speed_profile import grip_speed_profile
    from utilities.car_files.vehicle_parameters import VehicleParameters

    params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
    accel = float(share) * float(params.mu) * float(getattr(params, "g", 9.81))
    points = torch.as_tensor(np.asarray(xy, dtype=np.float32))
    profile = grip_speed_profile(points, accel, float(Settings.HWM_INITIAL_SPEED_MAX))
    return profile.detach().cpu().numpy().astype(np.float64)

"""Rectangular arena and the open-loop drive used when ``Settings.TEST_TRACK`` is on.

The map is a 40 m by 16 m rectangle. Only a border of occupied cells is a wall; the
inside is free. It is written under ``utilities/maps/test_rectangle`` the first time
this module prepares it, and ``Settings`` then points the simulator at that map.

The car starts at the left end, centered between the long walls, heading along the
length, at rest. The command is a constant acceleration and a steering angle

    (TEST_TRACK_STEER_SLOPE * t) * sin(2π t / STEER_PERIOD)

with ``STEER_PERIOD`` = 2 s, clipped to the car's steering limit.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from utilities.Settings import Settings

MAP_NAME = "test_rectangle"
LENGTH = 40.0
WIDTH = 16.0
RESOLUTION = 0.05
WALL = 0.30
STEER_PERIOD = 2.0
STEER_LIMIT = 0.6  # yokomo1 s_max [rad]

_MAP_DIR = Path(__file__).resolve().parent / "maps" / MAP_NAME


def enabled() -> bool:
    return bool(getattr(Settings, "TEST_TRACK", False))


def store_memory() -> bool:
    """``TEST_TRACK_STORE_MEMORY``: keep the drive in memory. The LLD is still not trained."""
    return enabled() and bool(getattr(Settings, "TEST_TRACK_STORE_MEMORY", False))


def prepare() -> None:
    """Write the arena and point ``Settings`` at it. No-op unless ``TEST_TRACK`` is on."""
    if not enabled():
        return
    _write_map()
    Settings.MAP_NAME = MAP_NAME
    Settings.recalculate_paths()
    Settings.START_FROM_RANDOM_POSITION = False
    Settings.REVERSE_DIRECTION = False
    Settings.STARTING_POSITION = [list(start_pose())]
    print(
        f"[test track] {LENGTH:.0f} m x {WIDTH:.0f} m arena, "
        f"accel={float(Settings.TEST_TRACK_ACCEL):.2f} m/s^2, "
        f"steer slope={float(Settings.TEST_TRACK_STEER_SLOPE):.3f} rad/s"
    )


def start_pose() -> tuple[float, float, float]:
    """``(x, y, yaw)`` at the left end, centered, heading along +x."""
    return (WALL + 1.0, 0.5 * WIDTH, 0.0)


def command(time_s: float) -> tuple[float, float]:
    """``(steering rad, acceleration m/s^2)`` at ``time_s`` seconds into the episode."""
    t = max(0.0, float(time_s))
    amplitude = float(Settings.TEST_TRACK_STEER_SLOPE) * t
    steer = amplitude * math.sin(2.0 * math.pi * t / STEER_PERIOD)
    steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))
    return steer, float(Settings.TEST_TRACK_ACCEL)


def side_walls(spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Left and right borders, ``(N, 2)`` world points spaced about ``spacing`` apart.

    Left is the high-y side, which is on the car's left when it heads along +x.
    Each chain runs from the start end to the far end, then along that end wall
    back to the center line.
    """
    step = max(float(spacing), RESOLUTION)
    x0, x1 = WALL, LENGTH - WALL
    y0, y1 = WALL, WIDTH - WALL
    xs = np.arange(x0, x1 + step * 0.5, step, dtype=np.float64)
    mid = 0.5 * WIDTH
    far_left = np.arange(y1, mid - step * 0.5, -step, dtype=np.float64)
    far_right = np.arange(y0, mid + step * 0.5, step, dtype=np.float64)
    left = np.concatenate(
        (
            np.stack((xs, np.full_like(xs, y1)), axis=1),
            np.stack((np.full_like(far_left, x1), far_left), axis=1),
        ),
        axis=0,
    )
    right = np.concatenate(
        (
            np.stack((xs, np.full_like(xs, y0)), axis=1),
            np.stack((np.full_like(far_right, x1), far_right), axis=1),
        ),
        axis=0,
    )
    return left, right


def _write_map() -> None:
    _MAP_DIR.mkdir(parents=True, exist_ok=True)
    image = _occupancy()
    # The loader flips the image vertically and treats row 0 as the lowest y.
    Image.fromarray(image).transpose(Image.FLIP_TOP_BOTTOM).save(_MAP_DIR / f"{MAP_NAME}.png")
    (_MAP_DIR / f"{MAP_NAME}.yaml").write_text(
        "\n".join(
            (
                f"image: {MAP_NAME}.png",
                f"resolution: {RESOLUTION:.6f}",
                "origin: [0.0, 0.0, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            )
        )
    )
    _write_waypoints()


def _occupancy() -> np.ndarray:
    """``(rows, cols)`` uint8, row 0 = low y. 0 is wall, 255 is free."""
    nx = int(round(LENGTH / RESOLUTION))
    ny = int(round(WIDTH / RESOLUTION))
    thick = max(1, int(round(WALL / RESOLUTION)))
    image = np.full((ny, nx), 255, dtype=np.uint8)
    image[:thick, :] = 0
    image[-thick:, :] = 0
    image[:, :thick] = 0
    image[:, -thick:] = 0
    return image


def _write_waypoints() -> None:
    """Center line along +x. The loader adds π/2 to ``psi_rad``, so -π/2 becomes heading 0."""
    step = 0.2
    x0, x1 = WALL + 0.5, LENGTH - WALL - 0.5
    xs = np.arange(x0, x1 + step * 0.5, step, dtype=np.float64)
    half = 0.5 * WIDTH - WALL
    lines = ["s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_right_iqp,d_left_iqp"]
    for i, x in enumerate(xs):
        lines.append(
            f"{i * step:.6f},{x:.6f},{0.5 * WIDTH:.6f},{-0.5 * math.pi:.6f},0.0,0.0,0.0,{half:.6f},{half:.6f}"
        )
    (_MAP_DIR / f"{MAP_NAME}_wp.csv").write_text("\n".join(lines) + "\n")

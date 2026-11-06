"""SVG of a map with a waypoint table drawn as the web viewer draws it.

The walls are the contours of the simulator's collision map (pixels <= 128 of the
map PNG). Each waypoint is an arrow: tail at the waypoint, 0.08 m of length per
m/s of target speed, along the waypoint heading, colored green -> yellow -> orange
-> red from 0 to the car's v_max. Same as ``HirarchicalPlanner._publish_waypoint_display``,
without the off-track counts and without the car.
"""

from __future__ import annotations

import os

import numpy as np
import yaml
from PIL import Image

_STOPS = ((0, 190, 0), (240, 220, 0), (255, 140, 0), (210, 30, 20))


def _speed_color(speed: float, v_max: float) -> str:
    t = float(np.clip(abs(speed) / max(v_max, 1e-3), 0.0, 1.0))
    x = t * (len(_STOPS) - 1)
    i = min(int(x), len(_STOPS) - 2)
    f = x - i
    a, b = _STOPS[i], _STOPS[i + 1]
    r, g, bl = (int(round(a[c] + (b[c] - a[c]) * f)) for c in range(3))
    return f"rgb({r},{g},{bl})"


def _map_walls(map_dir: str, map_name: str):
    """Wall contours in world coordinates and the map extent."""
    from skimage import measure

    with open(os.path.join(map_dir, map_name + ".yaml")) as f:
        meta = yaml.safe_load(f)
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    img = np.asarray(Image.open(os.path.join(map_dir, map_name + ".png")).convert("L"), dtype=np.float64)
    free = (img > 128.0).astype(np.float64)
    h, w = free.shape
    padded = np.pad(free, 1, constant_values=0.0)
    contours = []
    for c in measure.find_contours(padded, 0.5):
        rows, cols = c[:, 0] - 1.0, c[:, 1] - 1.0
        x = ox + (cols + 0.5) * res
        y = oy + (h - rows - 0.5) * res
        if len(x) >= 3:
            contours.append(np.stack((x, y), axis=1))
    extent = (ox, oy, ox + w * res, oy + h * res)
    return contours, extent


def write_waypoint_svg(path: str, map_dir: str, map_name: str, xy, speed, heading=None,
                       v_max: float = 10.0, title: str = "") -> None:
    """Write the map and the waypoint arrows to ``path``.

    ``heading`` defaults to the direction to the next waypoint, closing the loop.
    """
    xy = np.asarray(xy, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    if heading is None:
        d = np.roll(xy, -1, axis=0) - xy
        heading = np.arctan2(d[:, 1], d[:, 0])
    heading = np.asarray(heading, dtype=np.float64)
    contours, (x0, y0, x1, y1) = _map_walls(map_dir, map_name)

    # Crop to the walls and waypoints, with a margin.
    pts = [xy] + contours
    allp = np.concatenate(pts, axis=0)
    margin = 0.5
    cx0, cy0 = allp.min(axis=0) - margin
    cx1, cy1 = allp.max(axis=0) + margin
    cx0, cy0, cx1, cy1 = max(cx0, x0), max(cy0, y0), min(cx1, x1), min(cy1, y1)
    width, height = cx1 - cx0, cy1 - cy0
    px_per_m = 1000.0 / max(width, height)

    def fy(y):
        return cy1 + cy0 - y  # SVG y grows downward

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{cx0:.3f} {cy0:.3f} {width:.3f} {height:.3f}" '
        f'width="{width * px_per_m:.0f}" height="{height * px_per_m:.0f}">',
        f'<title>{title}</title>',
        f'<rect x="{cx0:.3f}" y="{cy0:.3f}" width="{width:.3f}" height="{height:.3f}" fill="rgb(60,60,60)"/>',
    ]
    # Free space is white, walls are what is left of the dark background (even-odd fill of all contours).
    d = []
    for c in contours:
        d.append("M" + " L".join(f"{x:.3f},{fy(y):.3f}" for x, y in c) + " Z")
    out.append(f'<path d="{" ".join(d)}" fill="white" fill-rule="evenodd" stroke="black" stroke-width="0.03"/>')

    length = np.abs(speed) * 0.08
    tip = xy + np.stack((length * np.cos(heading), length * np.sin(heading)), axis=1)
    head = length * 0.35
    wing = 0.5
    left = tip - np.stack((head * np.cos(heading - wing), head * np.sin(heading - wing)), axis=1)
    right = tip - np.stack((head * np.cos(heading + wing), head * np.sin(heading + wing)), axis=1)
    out.append('<g fill="none" stroke-width="0.025" stroke-linecap="round" stroke-linejoin="round">')
    for i in range(len(xy)):
        color = _speed_color(speed[i], v_max)
        p = (xy[i], tip[i], left[i], tip[i], right[i])
        pts_s = " ".join(f"{q[0]:.3f},{fy(q[1]):.3f}" for q in p)
        out.append(f'<polyline points="{pts_s}" stroke="{color}"/>')
    out.append("</g>")

    # Speed legend, bottom left.
    lw = 0.25 * width
    lx, ly = cx0 + 0.03 * width, cy0 + height - 0.05 * height
    out.append('<defs><linearGradient id="spd">' + "".join(
        f'<stop offset="{k / (len(_STOPS) - 1):.3f}" stop-color="rgb{s}"/>' for k, s in enumerate(_STOPS)
    ) + "</linearGradient></defs>")
    bh = 0.015 * height
    fs = 0.025 * height
    out.append(f'<rect x="{lx:.3f}" y="{ly:.3f}" width="{lw:.3f}" height="{bh:.3f}" fill="url(#spd)"/>')
    out.append(f'<text x="{lx:.3f}" y="{ly - 0.4 * bh:.3f}" font-size="{fs:.3f}" font-family="sans-serif" fill="black">0</text>')
    out.append(f'<text x="{lx + lw:.3f}" y="{ly - 0.4 * bh:.3f}" font-size="{fs:.3f}" font-family="sans-serif" '
               f'fill="black" text-anchor="end">{v_max:g} m/s</text>')
    out.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out))

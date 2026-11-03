"""Free-space geometry of the sysid arena, read from the map image.

Everything the SysId planner drives is a circle: a large *main* circle at the
point of largest wall clearance, and two *lobes* of a figure eight that let the
car change from counter-clockwise to clockwise without stopping. All of them are
derived here from the map, so the same code works on the simulated map and on a
freshly recorded map of the real room.

Run as a script to (re)write the circle raceline and the start pose into a map
folder (the raceline is only needed by the simulator's waypoint machinery and by
``Settings.MPC_SAFETY_FILTER``; the planner itself does not use it)::

    python -m Control_Toolkit_ASF.Controllers.SysId.arena --map IPZ51_sysid
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

FREE_PIXEL_MIN = 250  # map pixels >= this are free. 205 (unknown) counts as wall here: conservative


@dataclass
class Circle:
    cx: float
    cy: float
    r: float
    direction: int  # +1 counter-clockwise, -1 clockwise

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    def with_direction(self, direction: int) -> "Circle":
        return Circle(self.cx, self.cy, self.r, int(direction))

    def with_radius(self, r: float) -> "Circle":
        return Circle(self.cx, self.cy, float(r), self.direction)


class Arena:
    """Wall clearance field of a map and the circles that fit in it."""

    def __init__(self, map_dir: str, map_name: str, map_scale: float = 1.0):
        yaml_path = os.path.join(map_dir, map_name + ".yaml")
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        image = np.array(Image.open(os.path.join(map_dir, map_name + ".png")).transpose(Image.FLIP_TOP_BOTTOM))
        if image.ndim == 3:
            image = image[..., 0]
        self.resolution = float(meta["resolution"]) * map_scale
        self.origin = np.array(meta["origin"][:2], dtype=float) * map_scale
        self.free = image >= FREE_PIXEL_MIN  # row 0 = lowest y
        # Keep only the largest free region: specks of free space behind walls are not reachable
        labels, n = ndimage.label(self.free)
        if n > 1:
            sizes = ndimage.sum(self.free, labels, range(1, n + 1))
            self.free = labels == (int(np.argmax(sizes)) + 1)
        self.clearance = self.resolution * ndimage.distance_transform_edt(self.free)

    # ------------------------------------------------------------- queries
    def to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        row = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        return row, col

    def to_world(self, row, col):
        return (self.origin[0] + (np.asarray(col) + 0.5) * self.resolution,
                self.origin[1] + (np.asarray(row) + 0.5) * self.resolution)

    def clearance_at(self, x, y) -> np.ndarray:
        """Distance [m] from (x, y) to the nearest wall. Off the map counts as 0."""
        row, col = self.to_cell(x, y)
        h, w = self.clearance.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        return np.where(inside, self.clearance[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)], 0.0)

    def circle_clearance(self, c: Circle, n: int = 180) -> float:
        """Smallest wall clearance along the circle line."""
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return float(np.min(self.clearance_at(c.cx + c.r * np.cos(a), c.cy + c.r * np.sin(a))))

    # ------------------------------------------------------------- layout
    def largest_circle(self):
        """Center and radius of the largest wall-free disc."""
        row, col = np.unravel_index(np.argmax(self.clearance), self.clearance.shape)
        x, y = self.to_world(row, col)
        return float(x), float(y), float(self.clearance[row, col])

    def main_circle(self, car_half_diagonal: float, margin: float) -> Circle:
        """Largest circle the car can drive with ``margin`` between its body and the walls."""
        cx, cy, r_free = self.largest_circle()
        return Circle(cx, cy, r_free - car_half_diagonal - margin, +1)

    def figure_eight(self, radius: float, car_half_diagonal: float, margin: float):
        """Two lobes of ``radius`` touching each other, both clear of the walls.

        Returns ``(lobe_ccw, lobe_cw, clearance)`` or ``None`` if no figure eight fits.
        The car crosses from the ccw lobe to the cw lobe at their touching point.
        """
        need = radius + car_half_diagonal + margin
        rows, cols = np.where(self.clearance >= need)
        if len(rows) == 0:
            return None
        xs, ys = self.to_world(rows, cols)
        stride = max(1, len(xs) // 4000)
        best = None
        for x1, y1 in zip(xs[::stride], ys[::stride]):
            for th in np.linspace(0, np.pi, 36, endpoint=False):
                x2, y2 = x1 + 2 * radius * math.cos(th), y1 + 2 * radius * math.sin(th)
                c2 = float(self.clearance_at(x2, y2))
                if c2 < need:
                    continue
                score = min(float(self.clearance_at(x1, y1)), c2)
                if best is None or score > best[0]:
                    best = (score, x1, y1, x2, y2)
        if best is None:
            return None
        score, x1, y1, x2, y2 = best
        return Circle(x1, y1, radius, +1), Circle(x2, y2, radius, -1), score - radius - car_half_diagonal


def tangent_segment(a: Circle, b: Circle):
    """Straight segment leaving ``a`` and entering ``b``, both in their driving direction.

    Returns ``(p_a, p_b, t)``: the point where the car leaves ``a``, the point where it
    reaches ``b`` and the unit heading along the segment, or ``None`` (e.g. concentric
    circles). For two touching lobes of a figure eight ``p_a == p_b``.
    A car on a circle with direction d has its center at ``P + d * r * left_normal``.
    """
    D = b.center - a.center
    dist = float(np.hypot(*D))
    k = b.direction * b.r - a.direction * a.r  # needed (C_b - C_a) . left_normal
    if dist < 1e-6 or abs(k) > dist + 1e-9:
        return None
    base = math.atan2(D[1], D[0])
    best = None
    for sign in (+1, -1):
        psi = base + sign * math.acos(float(np.clip(k / dist, -1.0, 1.0)))
        left = np.array([math.cos(psi), math.sin(psi)])
        t = np.array([left[1], -left[0]])  # heading: left normal rotated by -90 deg
        p_a = a.center - a.direction * a.r * left
        p_b = b.center - b.direction * b.r * left
        length = float((p_b - p_a) @ t)
        if length < -1e-6:
            continue
        if best is None or length < best[3]:
            best = (p_a, p_b, t, length)
    return None if best is None else best[:3]


def segment_clearance(arena: "Arena", p_a, p_b, step=0.05) -> float:
    n = max(2, int(np.hypot(*(p_b - p_a)) / step) + 1)
    pts = p_a[None] + np.linspace(0, 1, n)[:, None] * (p_b - p_a)[None]
    return float(np.min(arena.clearance_at(pts[:, 0], pts[:, 1])))


# ----------------------------------------------------------------- raceline file
def circle_waypoints(c: Circle, arena: Arena, speed: float, spacing: float = 0.1) -> np.ndarray:
    """Rows of the ``<map>_wp.csv`` format for one lap of ``c``."""
    n = max(16, int(round(2 * np.pi * c.r / spacing)))
    a = np.arange(n) * 2 * np.pi / n
    if c.direction < 0:
        a = -a
    x = c.cx + c.r * np.cos(a)
    y = c.cy + c.r * np.sin(a)
    heading = a + c.direction * np.pi / 2
    # Distance to the wall along the normals. Left of a ccw circle is the center side
    d_out = np.array([_ray_clearance(arena, xi, yi, ai) for xi, yi, ai in zip(x, y, a)])
    d_in = np.array([_ray_clearance(arena, xi, yi, ai + np.pi) for xi, yi, ai in zip(x, y, a)])
    d_left, d_right = (d_in, d_out) if c.direction > 0 else (d_out, d_in)
    s = np.arange(n) * 2 * np.pi * c.r / n
    return np.column_stack([
        s, x, y,
        heading - np.pi / 2,  # the csv stores the normal angle; waypoint_utils adds pi/2
        np.full(n, c.direction / c.r), np.full(n, speed), np.zeros(n), d_right, d_left,
    ])


def _ray_clearance(arena: Arena, x, y, angle, step=0.02, max_range=10.0) -> float:
    for d in np.arange(0.0, max_range, step):
        if arena.clearance_at(x + d * math.cos(angle), y + d * math.sin(angle)) <= 0.0:
            return float(d)
    return float(max_range)


def write_map_files(map_dir: str, map_name: str, car_half_diagonal: float = 0.23,
                    margin: float = 0.35, speed: float = 1.5) -> Circle:
    """Write ``<map>_wp.csv``, ``<map>_wp_reverse.csv`` and the start pose for the main circle."""
    arena = Arena(map_dir, map_name)
    c = arena.main_circle(car_half_diagonal, margin)
    header = "s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_right_iqp,d_left_iqp"
    for suffix, d in (("_wp.csv", +1), ("_wp_reverse.csv", -1)):
        np.savetxt(os.path.join(map_dir, map_name + suffix), circle_waypoints(c.with_direction(d), arena, speed),
                   delimiter=",", header=header, comments="", fmt="%.6f")
    yaml_path = os.path.join(map_dir, map_name + ".yaml")
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    # Start on the main circle (angle 0 = +x of the center), heading ccw
    meta["starting_positions"] = [[round(c.cx + c.r, 3), round(c.cy, 3), round(math.pi / 2, 4)]]
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=None, sort_keys=False)
    return c


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    maps_root = os.path.normpath(os.path.join(here, "..", "..", "..", "utilities", "maps"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", default="IPZ51_sysid")
    p.add_argument("--car-half-diagonal", type=float, default=0.23, help="[m] yokomo1: hypot(0.38, 0.25) / 2")
    p.add_argument("--margin", type=float, default=0.35, help="[m] body-to-wall margin on the main circle")
    args = p.parse_args()
    d = os.path.join(maps_root, args.map)
    circle = write_map_files(d, args.map, args.car_half_diagonal, args.margin)
    arena = Arena(d, args.map)
    print(f"main circle: center ({circle.cx:.2f}, {circle.cy:.2f}) r={circle.r:.2f} m, "
          f"line clearance {arena.circle_clearance(circle):.2f} m")
    for r in (0.9, 1.0, 1.1, 1.2):
        f8 = arena.figure_eight(r, args.car_half_diagonal, 0.15)
        print(f"figure eight r={r}: " + ("none" if f8 is None else
              f"lobes ({f8[0].cx:.2f},{f8[0].cy:.2f}) ({f8[1].cx:.2f},{f8[1].cy:.2f}), body clearance {f8[2]:.2f} m"))

"""OpenAI CarRacing track generator, writing runnable gym maps.

The loop is the one in ``sim/f110_sim/unittest/random_trackgen.py`` (Hongrui
Zheng, adapted from CarRacing-v0). Each success becomes one map under
``DeepEvolution/maps``, in the same layout as ``utilities/maps``:

* occupancy bitmap, 0 = wall, 255 = free, row 0 = lowest y after the loader's flip
* ``<name>.yaml`` with resolution, origin and a starting pose
* ``<name>_wp.csv`` minimum-curvature raceline, same optimizer and columns as
  ``run/create_min_curve_waypoints.py``

The folder name contains ``OAI_generator``.

Road width is the old generator's wall-to-wall distance: the centerline is
offset by 10 units each side, and that 20-unit band is 3.3 m on the bitmap.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import shapely.geometry as shp
import yaml
from PIL import Image
from shapely.geometry import LinearRing, Point

_GYM_ROOT = Path(__file__).resolve().parents[4]
if str(_GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GYM_ROOT))

from Control_Toolkit_ASF.Controllers.DeepEvolution.map_store import MAPS_DIR
from utilities.Settings import Settings
from utilities.minimum_curvature_optimization.main_globaltraj import main_globaltraj

# Half-width of the corridor in the generator's own coordinates (unchanged).
WIDTH = 10.0
# Border to border, in meters, after that corridor is placed on the bitmap.
ROAD_WIDTH_M = 3.3
RESOLUTION = 0.05  # [m/px], same order as the bundled maps
MARGIN_M = 2.0
_METERS_PER_UNIT = ROAD_WIDTH_M / (2.0 * WIDTH)


def create_track():
    CHECKPOINTS = 16
    SCALE = 6.0
    TRACK_RAD = 900 / SCALE
    TRACK_DETAIL_STEP = 21 / SCALE
    TRACK_TURN_RATE = 0.31

    start_alpha = 0.

    checkpoints = []
    for c in range(CHECKPOINTS):
        alpha = 2 * math.pi * c / CHECKPOINTS + np.random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
        rad = np.random.uniform(TRACK_RAD / 3, TRACK_RAD)
        if c == 0:
            alpha = 0
            rad = 1.5 * TRACK_RAD
        if c == CHECKPOINTS - 1:
            alpha = 2 * math.pi * c / CHECKPOINTS
            start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
            rad = 1.5 * TRACK_RAD
        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

    x, y, beta = 1.5 * TRACK_RAD, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi
        while True:
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2 * math.pi
            continue
        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x
        dest_dy = dest_y - y
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i == 0:
            return False
        pass_through_start = track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break
    print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))

    track = track[i1:i2 - 1]
    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)
    well_glued_together = np.sqrt(
        np.square(first_perp_x * (track[0][2] - track[-1][2]))
        + np.square(first_perp_y * (track[0][3] - track[-1][3]))
    )
    if well_glued_together > TRACK_DETAIL_STEP:
        return False

    track_xy = np.asarray([(px, py) for (_a, _b, px, py) in track], dtype=np.float64)
    track_poly = shp.Polygon(track_xy)
    # buffer(+WIDTH) is the outer wall. buffer(-WIDTH) is the inner wall.
    outer = _wall_coords(track_poly.buffer(WIDTH))
    inner = _wall_coords(track_poly.buffer(-WIDTH))
    if outer is None or inner is None:
        return False
    return track_xy, outer, inner


def _wall_coords(geom):
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        return None
    coords = np.asarray(geom.exterior.coords)
    if coords.ndim != 2 or len(coords) < 3:
        return None
    return coords


def _meters(points):
    return np.asarray(points, dtype=np.float64) * _METERS_PER_UNIT


def _occupancy(outer_m, inner_m, origin, resolution):
    """Row 0 is the lowest y. 0 is wall, 255 is the road."""
    span = np.vstack((outer_m, inner_m))
    max_xy = span.max(axis=0) + MARGIN_M
    width = int(np.ceil((max_xy[0] - origin[0]) / resolution)) + 1
    height = int(np.ceil((max_xy[1] - origin[1]) / resolution)) + 1
    image = np.zeros((height, width), dtype=np.uint8)

    def pixels(points):
        cols = np.rint((points[:, 0] - origin[0]) / resolution).astype(np.int32)
        rows = np.rint((points[:, 1] - origin[1]) / resolution).astype(np.int32)
        return np.column_stack((cols, rows)).reshape(-1, 1, 2)

    cv2.fillPoly(image, [pixels(outer_m)], 255)
    cv2.fillPoly(image, [pixels(inner_m)], 0)
    return image


def _write_bitmap(folder, name, image):
    # The loader flips vertically and then treats row 0 as the lowest y.
    Image.fromarray(image).transpose(Image.FLIP_TOP_BOTTOM).save(folder / f"{name}.png")


def _write_yaml(folder, name, origin, start):
    payload = {
        "image": f"{name}.png",
        "resolution": float(RESOLUTION),
        "origin": [float(origin[0]), float(origin[1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "starting_positions": [[float(start[0]), float(start[1]), float(start[2])]],
    }
    with open(folder / f"{name}.yaml", "w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _reference_centerline(center_m):
    """Evenly spaced centerline with a constant half-width on each side."""
    ring = LinearRing(np.vstack((center_m, center_m[:1])))
    count = max(int(ring.length / 0.2), 32)
    distances = np.linspace(0.0, ring.length, count, endpoint=False)
    points = np.array([ring.interpolate(s).coords[0] for s in distances], dtype=np.float64)
    half = ROAD_WIDTH_M / 2.0
    widths = np.full((len(points), 2), half)
    return np.column_stack((points, widths))


def _wall_clearance(trajectory, outer_m, inner_m):
    """Distance from each raceline point to the wall on its left and its right.

    Frenet ``d`` is positive along the left normal of the path tangent, and the
    waypoint file stores how far that axis can go before each wall.
    """
    xy = trajectory[:, 1:3]
    step = np.roll(xy, -1, axis=0) - xy
    heading = np.arctan2(step[:, 1], step[:, 0])
    left = np.column_stack((-np.sin(heading), np.cos(heading)))
    outer = LinearRing(outer_m)
    inner = LinearRing(inner_m)
    d_left = np.empty(len(xy))
    d_right = np.empty(len(xy))
    for i, point in enumerate(xy):
        here = Point(point)
        for ring in (outer, inner):
            nearest = ring.interpolate(ring.project(here))
            vector = np.array([nearest.x - point[0], nearest.y - point[1]])
            distance = float(np.linalg.norm(vector))
            if np.dot(vector, left[i]) >= 0.0:
                d_left[i] = distance
            else:
                d_right[i] = distance
    # A point sitting on a wall has a zero vector and must not produce a zero bound.
    d_left = np.maximum(d_left, 0.05)
    d_right = np.maximum(d_right, 0.05)
    return d_right, d_left


def generate_map(name):
    """Write one closed random track into ``MAPS_DIR/name`` and return that folder."""
    track = create_track()
    if track is False:
        return None
    center, outer, inner = track
    center_m = _meters(center)
    outer_m = _meters(outer)
    inner_m = _meters(inner)
    origin = np.vstack((outer_m, inner_m)).min(axis=0) - MARGIN_M

    folder = MAPS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    image = _occupancy(outer_m, inner_m, origin, RESOLUTION)
    _write_bitmap(folder, name, image)

    reference = _reference_centerline(center_m)
    centerline_path = folder / "centerline.csv"
    np.savetxt(centerline_path, reference, delimiter=",", fmt="%.6f")

    trajectory, _bound_right, _bound_left, lap_time = main_globaltraj(
        track_name=str(folder / "centerline"),
        curv_opt_type="mincurv_iqp",
        safety_width=float(Settings.MIN_CURV_SAFETY_WIDTH),
        plot=False,
    )
    d_right, d_left = _wall_clearance(trajectory, outer_m, inner_m)
    raceline = np.column_stack((trajectory[:, :7], d_right, d_left))
    np.savetxt(
        folder / f"{name}_wp.csv",
        raceline,
        delimiter=",",
        fmt="%.6f",
        header="s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_right_iqp,d_left_iqp",
        comments="",
    )

    # The loader adds pi/2 to the stored psi, which undoes the optimizer's
    # "psi = 0 is north" shift and leaves the geometric heading.
    x0, y0 = trajectory[0, 1], trajectory[0, 2]
    x1, y1 = trajectory[1, 1], trajectory[1, 2]
    yaw = math.atan2(y1 - y0, x1 - x0)
    _write_yaml(folder, name, origin, (x0, y0, yaw))
    print(f"Wrote {folder}  (estimated lap {lap_time:.2f}s, road {ROAD_WIDTH_M:.1f} m)")
    return folder


def main():
    parser = argparse.ArgumentParser(description="Generate runnable OpenAI-style random maps.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_maps", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=30, help="Tries per map before giving up.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in range(args.num_maps):
        name = f"OAI_generator_{args.seed}_{index}"
        folder = None
        for _try in range(args.attempts):
            folder = generate_map(name)
            if folder is not None:
                break
            print("Random generator failed, retrying")
        if folder is None:
            raise RuntimeError(f"No closed track for {name} after {args.attempts} attempts")
        written += 1
    print(f"{written} map(s) in {MAPS_DIR}")


if __name__ == "__main__":
    main()

"""Track walls from the occupancy map, shared by the hierarchical planner and learner.

The map is the same ``<map>.yaml`` + ``<map>.png`` pair the simulator loads (and the
SLAM map on the real car), read exactly like ``ScanSimulator2D.set_map``: the image is
flipped vertically, pixels above 128 are free, and resolution and origin come from the
yaml after ``scale_map_metadata``.

Two views of the walls:

* Collision: a body corner is in a wall when its map cell has a distance-transform
  value below the collision margin, the rule ``check_body_map_collision`` applies.
  Unlike the simulator, points outside the image also count as collisions.
  ``crash_probability`` applies this rule to a Gaussian over the pose.
* Polylines: left wall, right wall and obstacles as closed point sequences, oriented
  in driving direction, ``point_spacing`` apart. ``get_relative_wall_positions``
  returns upcoming points in the car frame. The points are conservative: for a car
  more than about ``point_spacing`` from every wall, the nearest point is never
  farther away than the nearest blocked cell (see ``_extract_walls``).

The raceline CSV is only used to pick the drivable region and the driving direction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy import ndimage
from skimage import measure

from utilities.map_scale import scale_map_metadata
from utilities.Settings import Settings

ArrayLike = Union[np.ndarray, torch.Tensor]

_CAR_FILES_DIR = Path("utilities") / "car_files"
# Fine cells per map cell along each axis when tracing the wall polylines.
_SUPERSAMPLE = 4


class WallGeometry:
    def __init__(
        self,
        map_name: Optional[str] = None,
        map_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
        reverse_direction: Optional[bool] = None,
        point_spacing: Optional[float] = None,
        car_length: Optional[float] = None,
        car_width: Optional[float] = None,
        collision_margin: float = 0.005,
        crash_samples: int = 64,
    ):
        """
        Args:
            point_spacing: polyline spacing in meters. None uses the median raceline spacing.
            car_length, car_width: body size. None reads ``Settings.ENV_CAR_PARAMETER_FILE``,
                the file the simulator uses.
            collision_margin: the simulator's ``map_collision_margin``.
            crash_samples: default number of shared pose samples in ``crash_probability``.
        """
        self.map_name = str(Settings.MAP_NAME if map_name is None else map_name)
        self.device = torch.device(device)
        self.reverse_direction = (
            bool(Settings.REVERSE_DIRECTION) if reverse_direction is None else bool(reverse_direction)
        )
        map_dir = Path(Settings.MAP_PATH if map_path is None else map_path)

        if car_length is None or car_width is None:
            with open(_CAR_FILES_DIR / Settings.ENV_CAR_PARAMETER_FILE, "r") as f:
                car_params = yaml.safe_load(f)
            car_length = float(car_params["length"]) if car_length is None else car_length
            car_width = float(car_params["width"]) if car_width is None else car_width
        self.car_length = float(car_length)
        self.car_width = float(car_width)
        self.collision_margin = float(collision_margin)
        self.crash_samples = int(crash_samples)
        # Corner offsets in the body frame, in get_vertices order: rl, rr, fr, fl.
        half_l, half_w = 0.5 * self.car_length, 0.5 * self.car_width
        self._corner_x = torch.tensor([-half_l, -half_l, half_l, half_l], dtype=torch.float64, device=self.device)
        self._corner_y = torch.tensor([half_w, -half_w, -half_w, half_w], dtype=torch.float64, device=self.device)
        self._noise: dict[tuple[int, torch.device], torch.Tensor] = {}

        blocked = self._load_map(map_dir / f"{self.map_name}.yaml")
        raceline = self._load_raceline(map_dir, self.map_name, self.reverse_direction)
        if point_spacing is None:
            point_spacing = float(np.median(np.hypot(*np.diff(raceline, axis=0).T)))
        self.point_spacing = float(point_spacing)

        left, right, obstacles = self._extract_walls(blocked, raceline)
        # Raceline waypoints (N, 2) in driving order.
        self.raceline_abs = torch.as_tensor(raceline, dtype=torch.float32, device=self.device)
        self.left_wall_abs = torch.as_tensor(left, dtype=torch.float32, device=self.device)
        self.right_wall_abs = torch.as_tensor(right, dtype=torch.float32, device=self.device)
        self.obstacles_abs = [torch.as_tensor(o, dtype=torch.float32, device=self.device) for o in obstacles]

    # ------------------------------------------------------------------
    # Map loading (mirrors ScanSimulator2D.set_map)
    # ------------------------------------------------------------------
    def _load_map(self, yaml_path: Path) -> np.ndarray:
        """Load the occupancy grid; returns the boolean blocked mask (row 0 = lowest y)."""
        image = np.array(Image.open(yaml_path.with_suffix(".png")).transpose(Image.FLIP_TOP_BOTTOM))
        if image.ndim == 3:
            image = image[..., 0]
        free = image.astype(np.float64) > 128.0
        with open(yaml_path, "r") as f:
            metadata = scale_map_metadata(yaml.safe_load(f))
        self.resolution = float(metadata["resolution"])
        origin = metadata["origin"]
        self.origin_x, self.origin_y = float(origin[0]), float(origin[1])
        self.origin_cos, self.origin_sin = math.cos(float(origin[2])), math.sin(float(origin[2]))
        self.map_height, self.map_width = int(free.shape[0]), int(free.shape[1])

        # Same distance transform and threshold as check_body_map_collision.
        distance = self.resolution * ndimage.distance_transform_edt(free)
        blocked = distance < self.collision_margin
        self.blocked = torch.as_tensor(blocked, device=self.device)
        return blocked

    @staticmethod
    def _load_raceline(map_dir: Path, map_name: str, reverse_direction: bool) -> np.ndarray:
        """Raceline ``(N, 2)`` in driving order, scaled by ``Settings.MAP_SCALE``."""
        stem = f"{map_name}_wp_reverse" if reverse_direction else f"{map_name}_wp"
        csv_path = map_dir / f"{stem}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Waypoint file not found: {csv_path}")
        df = pd.read_csv(csv_path, comment="#")
        try:
            xy = df.loc[:, ["x_m", "y_m"]].to_numpy(dtype=np.float64)
        except KeyError as exc:
            raise KeyError(f"{csv_path} is missing column {exc}; expected x_m and y_m") from exc
        return xy * float(Settings.MAP_SCALE)

    # ------------------------------------------------------------------
    # Wall polylines
    # ------------------------------------------------------------------
    def _map_to_world(self, x_rot: np.ndarray, y_rot: np.ndarray) -> np.ndarray:
        """Map-frame meters (measured from the image corner) to world ``(N, 2)``."""
        x = self.origin_x + x_rot * self.origin_cos - y_rot * self.origin_sin
        y = self.origin_y + x_rot * self.origin_sin + y_rot * self.origin_cos
        return np.stack((x, y), axis=-1)

    def _world_to_cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``xy_2_rc`` for an array of points. Returns ``(rows, cols, inside)``."""
        x_t = xy[:, 0] - self.origin_x
        y_t = xy[:, 1] - self.origin_y
        x_rot = x_t * self.origin_cos + y_t * self.origin_sin
        y_rot = -x_t * self.origin_sin + y_t * self.origin_cos
        inside = (
            (x_rot >= 0) & (x_rot < self.map_width * self.resolution)
            & (y_rot >= 0) & (y_rot < self.map_height * self.resolution)
        )
        cols = np.clip((x_rot / self.resolution).astype(np.int64), 0, self.map_width - 1)
        rows = np.clip((y_rot / self.resolution).astype(np.int64), 0, self.map_height - 1)
        return rows, cols, inside

    def _extract_walls(
        self, blocked: np.ndarray, raceline: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Conservative wall polylines around the drivable region.

        On a grid with ``a = resolution / _SUPERSAMPLE`` (every map cell split into
        fine cells, so blocked cells keep their exact squares), ``D`` is the distance
        between fine-cell centers and the nearest blocked one. The true distance to the
        blocked squares is at least ``D - a/√2`` at a center, so the traced boundary of
        ``{D > L}`` lies at least ``L - a(1 + 1/√2)`` from every blocked cell. With
        ``L = s/2 + a(1 + 1/√2)`` that is ``s/2``, ``s = point_spacing``.

        A car inside that boundary reaches any blocked cell only through it, at a point
        ``q`` at least ``s/2`` from the wall, and some sampled point lies within ``s/2``
        of ``q`` (arc spacing ``<= s``). So the nearest sampled point is at most as far
        as the nearest blocked cell.
        """
        # Drivable region: the free component holding most raceline points.
        labels, n_labels = ndimage.label(~blocked)
        rows, cols, inside = self._world_to_cells(raceline)
        hits = labels[rows[inside], cols[inside]]
        hits = hits[hits > 0]
        if n_labels == 0 or hits.size == 0:
            raise RuntimeError(f"raceline of map '{self.map_name}' does not lie in free space")
        region = labels == np.bincount(hits).argmax()

        # Everything outside the region counts as wall, including a margin around the image.
        a = self.resolution / _SUPERSAMPLE
        fine = np.pad(np.kron(region, np.ones((_SUPERSAMPLE, _SUPERSAMPLE), dtype=bool)), _SUPERSAMPLE)
        distance = a * ndimage.distance_transform_edt(fine)
        level = 0.5 * self.point_spacing + a * (1.0 + math.sqrt(0.5))
        clear_labels, _ = ndimage.label(distance > level)
        rows_f = (rows[inside] + 1) * _SUPERSAMPLE + _SUPERSAMPLE // 2
        cols_f = (cols[inside] + 1) * _SUPERSAMPLE + _SUPERSAMPLE // 2
        hits = clear_labels[rows_f, cols_f]
        hits = hits[hits > 0]
        if hits.size == 0:
            raise RuntimeError(
                f"map '{self.map_name}': no raceline point is {level:.3f} m from the walls; "
                f"reduce point_spacing (now {self.point_spacing:.3f} m)"
            )
        clear = clear_labels == np.bincount(hits).argmax()

        # Fine cells are 4-connected (ndimage.label), so the rest must be 8-connected: fully_connected="low".
        contours = measure.find_contours(clear.astype(np.float64), 0.5, fully_connected="low")
        polygons = []
        for contour in contours:
            # Fine index k is the center of [k, k + 1) fine cells; the pad shifts by one map cell.
            y_rot = (contour[:, 0] + 0.5) * a - self.resolution
            x_rot = (contour[:, 1] + 0.5) * a - self.resolution
            poly = self._map_to_world(x_rot, y_rot)
            if np.allclose(poly[0], poly[-1]):
                poly = poly[:-1]
            if poly.shape[0] >= 3:
                polygons.append(poly)
        if len(polygons) < 2:
            raise RuntimeError(
                f"map '{self.map_name}': drivable region has {len(polygons)} boundary, need an inner and outer wall"
            )

        # The outer wall encloses the most area, the inner wall is the largest hole.
        polygons.sort(key=lambda p: abs(_signed_area(p)), reverse=True)
        outer, inner, obstacles = polygons[0], polygons[1], polygons[2:]
        # Driving counterclockwise puts the inner wall on the left.
        driving_sign = math.copysign(1.0, _signed_area(raceline))
        left, right = (inner, outer) if driving_sign > 0 else (outer, inner)

        def oriented(poly: np.ndarray) -> np.ndarray:
            return poly if math.copysign(1.0, _signed_area(poly)) == driving_sign else poly[::-1]

        left = _resample_closed(oriented(left), self.point_spacing)
        right = _resample_closed(oriented(right), self.point_spacing)
        obstacles = [_resample_closed(o, self.point_spacing) for o in obstacles]
        return left, right, obstacles

    # ------------------------------------------------------------------
    # Relative wall points
    # ------------------------------------------------------------------
    def get_relative_wall_positions(
        self,
        current_state: torch.Tensor,
        number_of_points: int,
    ) -> dict[str, torch.Tensor]:
        """Return the next ``number_of_points`` wall points relative to each car.

        Args:
            current_state: ``(B, 3)`` poses ``[x, y, theta]``.
            number_of_points: upcoming points along each wall, starting at the nearest one.

        Returns:
            Dict with
            - ``left``:  ``(B, number_of_points, 2)`` car-frame points
            - ``right``: ``(B, number_of_points, 2)`` car-frame points

            Car-frame convention: x forward, y left. Points are ``point_spacing`` apart.
        """
        if number_of_points < 1:
            raise ValueError(f"number_of_points must be >= 1, got {number_of_points}")
        if not torch.is_tensor(current_state):
            raise TypeError("current_state must be a torch.Tensor of shape (B, 3)")
        if current_state.ndim != 2 or current_state.shape[1] != 3:
            raise ValueError(
                f"current_state must have shape (B, 3) [x, y, theta], got {tuple(current_state.shape)}"
            )

        poses = current_state.to(dtype=torch.float32)
        out = {}
        for side, wall in (("left", self.left_wall_abs), ("right", self.right_wall_abs)):
            wall = wall.to(poses.device)
            nearest = self._nearest_indices(poses[:, :2], wall)
            offsets = torch.arange(number_of_points, device=poses.device)
            out[side] = self._to_car_frame(wall[(nearest[:, None] + offsets) % wall.shape[0]], poses)
        return out

    @staticmethod
    def _nearest_indices(pos: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
        """Exact nearest wall vertex, using a centroid ball when the batch is clustered.

        If every position lies within radius R of centroid C, the globally nearest wall
        point W* of any position satisfies ||C - W*|| <= 2R + ||C - W_C||, where W_C is
        the wall point nearest C. Searching only that ball is therefore exact.
        """
        if pos.shape[0] > 1:
            centroid = pos.mean(dim=0)
            radius = (pos - centroid).norm(dim=1).max()
            dist_c_sq = ((wall - centroid) ** 2).sum(dim=1)
            limit = 2.0 * radius + dist_c_sq.min().sqrt() + 1.0e-3
            candidates = torch.nonzero(dist_c_sq <= limit * limit, as_tuple=True)[0]
            if 0 < candidates.numel() < wall.shape[0]:
                local = _argmin_to_points(pos, wall.index_select(0, candidates))
                return candidates[local]
        return _argmin_to_points(pos, wall)

    @staticmethod
    def _to_car_frame(points_abs: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        cos_t = torch.cos(poses[:, 2])[:, None]
        sin_t = torch.sin(poses[:, 2])[:, None]
        dx = points_abs[..., 0] - poses[:, 0, None]
        dy = points_abs[..., 1] - poses[:, 1, None]
        return torch.stack((dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t), dim=-1)

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------
    def corners(self, poses: ArrayLike) -> torch.Tensor:
        """Body corners ``(..., 4, 2)`` for poses ``(..., 3)``, order rl, rr, fr, fl (float64)."""
        p = torch.as_tensor(poses, device=self.device).to(torch.float64)
        cos_t = torch.cos(p[..., 2:3])
        sin_t = torch.sin(p[..., 2:3])
        x = p[..., 0:1] + cos_t * self._corner_x - sin_t * self._corner_y
        y = p[..., 1:2] + sin_t * self._corner_x + cos_t * self._corner_y
        return torch.stack((x, y), dim=-1)

    def corners_in_wall(self, poses: ArrayLike) -> torch.Tensor:
        """``(..., 4)`` bool: body corners in a blocked cell or outside the map (or NaN)."""
        corners = self.corners(poses)
        if Settings.BLANK_MAP:
            return torch.zeros(corners.shape[:-1], dtype=torch.bool, device=self.device)
        x_t = corners[..., 0] - self.origin_x
        y_t = corners[..., 1] - self.origin_y
        x_rot = x_t * self.origin_cos + y_t * self.origin_sin
        y_rot = -x_t * self.origin_sin + y_t * self.origin_cos
        # NaN coordinates fail every comparison, so they count as outside.
        inside = (
            (x_rot >= 0) & (x_rot < self.map_width * self.resolution)
            & (y_rot >= 0) & (y_rot < self.map_height * self.resolution)
        )
        cols = (x_rot / self.resolution).floor().nan_to_num(0.0).clamp(0, self.map_width - 1).long()
        rows = (y_rot / self.resolution).floor().nan_to_num(0.0).clamp(0, self.map_height - 1).long()
        return ~inside | self.blocked[rows, cols]

    def in_wall(self, poses: ArrayLike) -> torch.Tensor:
        """``(...)`` bool: any body corner in a wall or outside the map."""
        return self.corners_in_wall(poses).any(dim=-1)

    def crash_probability(
        self,
        mean: ArrayLike,
        cov: ArrayLike,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Probability that any body corner is in a wall, for Gaussian poses.

        ``mean`` is ``(..., 3)`` ``[x, y, theta]`` and ``cov`` ``(..., 3, 3)``. Every pose
        distribution is evaluated at the same standard-normal draws ``noise`` ``(K, 3)``,
        so results are deterministic and differences between batch entries come from
        their distributions, not from sampling. Default draws: ``crash_samples`` scrambled
        Sobol points mapped to normal, fixed per object. Pass fresh ``noise`` to redraw.

        Returns ``(...)``: the fraction of samples whose pose ``in_wall`` flags.
        """
        mean = torch.as_tensor(mean, device=self.device).to(torch.float64)
        cov = torch.as_tensor(cov, device=self.device).to(torch.float64)
        batch_shape = mean.shape[:-1]
        mean = mean.reshape(-1, 3)
        cov = cov.reshape(-1, 3, 3)
        if noise is None:
            noise = self.shared_noise(self.crash_samples)
        noise = noise.to(device=self.device, dtype=torch.float64)

        scale = _matrix_sqrt(cov)
        poses = mean[:, None, :] + torch.einsum("kj,bij->bki", noise, scale)
        hit = self.in_wall(poses)
        return hit.to(torch.float32).mean(dim=-1).reshape(batch_shape)

    def shared_noise(self, num_samples: int) -> torch.Tensor:
        """Fixed ``(num_samples, 3)`` standard-normal draws (scrambled Sobol, seed 0)."""
        key = (int(num_samples), self.device)
        if key not in self._noise:
            sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=0)
            uniform = sobol.draw(int(num_samples), dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
            normal = torch.distributions.Normal(0.0, 1.0).icdf(uniform)
            self._noise[key] = normal.to(self.device)
        return self._noise[key]


def _argmin_to_points(pos: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    delta = points.unsqueeze(0) - pos.unsqueeze(1)
    return (delta * delta).sum(dim=-1).argmin(dim=1)


def _matrix_sqrt(cov: torch.Tensor) -> torch.Tensor:
    """``S`` with ``S S^T = cov``: Cholesky, or an eigen square root where that fails (singular)."""
    scale, info = torch.linalg.cholesky_ex(cov)
    bad = info != 0
    if bool(bad.any()):
        w, v = torch.linalg.eigh(0.5 * (cov[bad] + cov[bad].transpose(-1, -2)))
        scale = scale.clone()
        scale[bad] = v * w.clamp_min(0.0).sqrt()[..., None, :]
    return scale


def _signed_area(poly: np.ndarray) -> float:
    """Shoelace area of a closed polygon ``(N, 2)``; positive when counterclockwise."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _resample_closed(poly: np.ndarray, spacing: float) -> np.ndarray:
    """Points on a closed polyline at uniform arc length no larger than ``spacing``."""
    closed = np.vstack((poly, poly[:1]))
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))))
    n = max(3, int(math.ceil(s[-1] / spacing)))
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack((np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])), axis=-1)

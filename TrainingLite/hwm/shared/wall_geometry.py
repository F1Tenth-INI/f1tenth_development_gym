"""
Return relative wall positions (batched) for a given car states (batched) as torch tensors or dicts thereof.
Map is taken from settings by default.

Static track-wall geometry shared by the hierarchical planner and learner.

Absolute left/right wall polylines are built once from a map waypoint CSV (or
from an already-loaded waypoint array). Queries transform a sliding window of
upcoming wall points into the car frame for a batch of states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from utilities.Settings import Settings
from utilities.waypoint_utils import (
    WP_D_LEFT_IDX,
    WP_D_RIGHT_IDX,
    WP_KAPPA_IDX,
    WP_PSI_IDX,
    WP_S_IDX,
    WP_X_IDX,
    WP_Y_IDX,
)

ArrayLike = Union[np.ndarray, torch.Tensor]


class WallGeometry:
    def __init__(
        self,
        map_name: Optional[str] = None,
        map_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
        reverse_direction: Optional[bool] = None,
    ):
        self.map_name = str(Settings.MAP_NAME if map_name is None else map_name)
        self.device = torch.device(device)
        self.reverse_direction = (
            bool(Settings.REVERSE_DIRECTION)
            if reverse_direction is None
            else bool(reverse_direction)
        )
        map_dir = Path(Settings.MAP_PATH if map_path is None else map_path)

        waypoints = self._load_waypoints_csv(map_dir, self.map_name, self.reverse_direction)
        left_abs, right_abs = self._walls_from_waypoints(waypoints)
        self._set_walls(left_abs, right_abs)

    @classmethod
    def from_waypoints(
        cls,
        waypoints: ArrayLike,
        device: Union[str, torch.device] = "cpu",
        map_name: Optional[str] = None,
    ) -> "WallGeometry":
        """Build from a (N, 10) waypoint array already loaded by CarSystem."""
        obj = cls.__new__(cls)
        obj.map_name = str(Settings.MAP_NAME if map_name is None else map_name)
        obj.device = torch.device(device)
        obj.reverse_direction = bool(Settings.REVERSE_DIRECTION)
        waypoints_np = np.asarray(waypoints, dtype=np.float32)
        if waypoints_np.ndim != 2 or waypoints_np.shape[1] < 10:
            raise ValueError(
                f"waypoints must have shape (N, 10), got {waypoints_np.shape}"
            )
        left_abs, right_abs = cls._walls_from_waypoints(waypoints_np)
        obj._set_walls(left_abs, right_abs)
        return obj

    def _set_walls(self, left_abs: np.ndarray, right_abs: np.ndarray) -> None:
        if left_abs.shape != right_abs.shape or left_abs.ndim != 2 or left_abs.shape[1] != 2:
            raise ValueError(
                f"wall arrays must be (N, 2); got left {left_abs.shape}, right {right_abs.shape}"
            )
        if left_abs.shape[0] == 0:
            raise ValueError("wall polylines are empty")
        self.left_wall_abs = torch.as_tensor(left_abs, dtype=torch.float32, device=self.device)
        self.right_wall_abs = torch.as_tensor(right_abs, dtype=torch.float32, device=self.device)
        self.num_points = int(self.left_wall_abs.shape[0])
        self._query_device: Optional[torch.device] = None
        self._left_q: Optional[torch.Tensor] = None
        self._right_q: Optional[torch.Tensor] = None
        self._offsets_n = 0
        self._offsets: Optional[torch.Tensor] = None

    def _walls_on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._query_device != device:
            self._left_q = self.left_wall_abs.to(device=device, copy=False)
            self._right_q = self.right_wall_abs.to(device=device, copy=False)
            self._query_device = device
            self._offsets = None
        assert self._left_q is not None and self._right_q is not None
        return self._left_q, self._right_q

    def _point_offsets(self, number_of_points: int, device: torch.device) -> torch.Tensor:
        if self._offsets is None or self._offsets_n != number_of_points or self._offsets.device != device:
            self._offsets = torch.arange(number_of_points, device=device)
            self._offsets_n = number_of_points
        return self._offsets

    def get_relative_wall_positions(
        self,
        current_state: torch.Tensor,
        number_of_points: int,
    ) -> dict[str, torch.Tensor]:
        """Return the next ``number_of_points`` wall points relative to each car.

        Args:
            current_state: ``(B, 3)`` poses ``[x, y, theta]``.
            number_of_points: upcoming points along each wall polyline.

        Returns:
            Dict with
            - ``left``:  ``(B, number_of_points, 2)`` car-frame points
            - ``right``: ``(B, number_of_points, 2)`` car-frame points

            Car-frame convention: x forward, y left.
        """
        if number_of_points < 1:
            raise ValueError(f"number_of_points must be >= 1, got {number_of_points}")
        if not torch.is_tensor(current_state):
            raise TypeError("current_state must be a torch.Tensor of shape (B, 3)")
        if current_state.ndim != 2 or current_state.shape[1] != 3:
            raise ValueError(
                f"current_state must have shape (B, 3) [x, y, theta], got {tuple(current_state.shape)}"
            )

        poses = current_state if current_state.dtype == torch.float32 else current_state.float()
        left_wall, right_wall = self._walls_on(poses.device)
        nearest = self._nearest_indices(poses, left_wall)
        gather_idx = (nearest[:, None] + self._point_offsets(number_of_points, poses.device)) % self.num_points
        return self._both_to_car_frame(
            left_wall[gather_idx],
            right_wall[gather_idx],
            poses,
        )

    def _nearest_indices(self, poses: torch.Tensor, left_wall: torch.Tensor) -> torch.Tensor:
        """Exact nearest left-wall vertex, using a centroid ball when the batch is clustered.

        If every pose lies within radius R of centroid C, the globally nearest wall
        point W* of any pose satisfies ||C - W*|| <= 2R + ||C - W_C||, where W_C is
        the wall point nearest C. Searching only that ball is therefore exact.
        """
        pos = poses[:, :2]
        if pos.shape[0] == 1:
            return self._argmin_to_points(pos, left_wall)

        centroid = pos.mean(dim=0)
        offset = pos - centroid
        radius = (offset * offset).sum(dim=1).max().sqrt()
        to_centroid = left_wall - centroid
        dist_c_sq = (to_centroid * to_centroid).sum(dim=1)
        d_c = dist_c_sq.min().sqrt()
        limit = 2.0 * radius + d_c + 1.0e-3
        candidates = torch.nonzero(dist_c_sq <= limit * limit, as_tuple=True)[0]
        if candidates.numel() == 0 or candidates.numel() >= self.num_points:
            return self._argmin_to_points(pos, left_wall)

        local = self._argmin_to_points(pos, left_wall.index_select(0, candidates))
        return candidates[local]

    @staticmethod
    def _argmin_to_points(pos: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        delta = points.unsqueeze(0) - pos.unsqueeze(1)
        return (delta * delta).sum(dim=-1).argmin(dim=1)

    @staticmethod
    def _both_to_car_frame(
        left_abs: torch.Tensor,
        right_abs: torch.Tensor,
        poses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = poses[:, 0, None]
        y = poses[:, 1, None]
        theta = poses[:, 2]
        cos_t = torch.cos(theta)[:, None]
        sin_t = torch.sin(theta)[:, None]

        def _transform(points_abs: torch.Tensor) -> torch.Tensor:
            dx = points_abs[..., 0] - x
            dy = points_abs[..., 1] - y
            return torch.stack((dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t), dim=-1)

        return {"left": _transform(left_abs), "right": _transform(right_abs)}

    @staticmethod
    def _load_waypoints_csv(
        map_dir: Path,
        map_name: str,
        reverse_direction: bool,
    ) -> np.ndarray:
        stem = f"{map_name}_wp"
        if reverse_direction:
            stem = f"{stem}_reverse"
        csv_path = map_dir / f"{stem}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Waypoint file not found: {csv_path}")

        waypoints_df = pd.read_csv(csv_path, comment="#")
        try:
            waypoints = np.zeros((waypoints_df.shape[0], 10), dtype=np.float32)
            waypoints[:, WP_S_IDX] = waypoints_df.loc[:, "s_m"].to_numpy()
            waypoints[:, WP_X_IDX] = waypoints_df.loc[:, "x_m"].to_numpy()
            waypoints[:, WP_Y_IDX] = waypoints_df.loc[:, "y_m"].to_numpy()
            waypoints[:, WP_PSI_IDX] = waypoints_df.loc[:, "psi_rad"].to_numpy()
            waypoints[:, WP_KAPPA_IDX] = waypoints_df.loc[:, "kappa_radpm"].to_numpy()
            waypoints[:, WP_D_RIGHT_IDX] = waypoints_df.loc[:, "d_right_iqp"].to_numpy()
            waypoints[:, WP_D_LEFT_IDX] = waypoints_df.loc[:, "d_left_iqp"].to_numpy()
        except KeyError as exc:
            raise KeyError(
                f"{csv_path} is missing a required column: {exc}. "
                "Expected s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2, "
                "d_right_iqp, d_left_iqp."
            ) from exc

        # CSV psi is the track normal; translational heading is psi + pi/2.
        waypoints[:, WP_PSI_IDX] += 0.5 * np.pi

        map_scale = float(Settings.MAP_SCALE)
        if map_scale != 1.0:
            from utilities.map_scale import median_waypoint_spacing, resample_waypoints_uniform

            spacing = Settings.WAYPOINT_SPACING or median_waypoint_spacing(waypoints)
            waypoints[:, WP_X_IDX] *= map_scale
            waypoints[:, WP_Y_IDX] *= map_scale
            waypoints[:, WP_S_IDX] *= map_scale
            waypoints[:, WP_KAPPA_IDX] /= map_scale
            waypoints[:, WP_D_RIGHT_IDX] *= map_scale
            waypoints[:, WP_D_LEFT_IDX] *= map_scale
            waypoints = resample_waypoints_uniform(waypoints, spacing)

        return waypoints

    @staticmethod
    def _walls_from_waypoints(waypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        track_left = waypoints[:, WP_D_LEFT_IDX]
        track_right = waypoints[:, WP_D_RIGHT_IDX]
        psi = waypoints[:, WP_PSI_IDX]
        border_left_x = waypoints[:, WP_X_IDX] + track_left * np.cos(psi + np.pi / 2)
        border_left_y = waypoints[:, WP_Y_IDX] + track_left * np.sin(psi + np.pi / 2)
        border_right_x = waypoints[:, WP_X_IDX] + track_right * np.cos(psi - np.pi / 2)
        border_right_y = waypoints[:, WP_Y_IDX] + track_right * np.sin(psi - np.pi / 2)
        left_abs = np.column_stack((border_left_x, border_left_y)).astype(np.float32)
        right_abs = np.column_stack((border_right_x, border_right_y)).astype(np.float32)
        return left_abs, right_abs

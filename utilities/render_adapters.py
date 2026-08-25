"""Adapters from RenderScene snapshots to web / pygame / ROS backends."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from utilities.render_scene import RenderScene
from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    STEERING_ANGLE_IDX,
)

# Canonical scene layer names used when mirroring RenderUtils.
LAYER_WAYPOINTS = "waypoints"
LAYER_WAYPOINTS_ALT = "waypoints_alternative"
LAYER_NEXT_WAYPOINTS = "next_waypoints"
LAYER_NEXT_WAYPOINTS_POLY = "next_waypoints_polynomial"
LAYER_NEXT_WAYPOINTS_ALT = "next_waypoints_alternative"
LAYER_LIDAR = "lidar"
LAYER_TRACK_BORDER = "track_border"
LAYER_TRACK_BORDER_LINES = "track_border_lines"
LAYER_GAP = "gap"
LAYER_TARGET = "target"
LAYER_OBSTACLES = "obstacles"
LAYER_VIRTUAL_OPPONENTS = "virtual_opponents"
LAYER_DETECTED_OPPONENTS = "detected_opponents"
LAYER_HISTORY_ALT = "history_alt"
LAYER_HISTORY_GT = "history_gt"
LAYER_HISTORY_PRIOR = "history_prior"
LAYER_HISTORY_PRIOR_FULL = "history_prior_full"
LAYER_MPC_ROLLOUTS = "mpc.rollouts"
LAYER_MPC_OPTIMAL = "mpc.optimal"
LAYER_STEERING_ARROW = "steering_arrow"
LAYER_EMERGENCY_SLOWDOWN = "emergency_slowdown"

# Legacy web_overlay keys that index.html / pygame already draw specially.
LEGACY_LAYER_KEYS: Dict[str, str] = {
    LAYER_WAYPOINTS: "waypoints",
    LAYER_WAYPOINTS_ALT: "waypoints_alternative",
    LAYER_NEXT_WAYPOINTS: "next_waypoints",
    LAYER_NEXT_WAYPOINTS_POLY: "next_waypoints_polynomial",
    LAYER_NEXT_WAYPOINTS_ALT: "next_waypoints_alternative",
    LAYER_LIDAR: "lidar_border_points",
    LAYER_TRACK_BORDER: "track_border_points",
    LAYER_TRACK_BORDER_LINES: "track_border_lines",
    LAYER_GAP: "largest_gap_middle_point",
    LAYER_TARGET: "target_point",
    LAYER_OBSTACLES: "obstacles",
    LAYER_VIRTUAL_OPPONENTS: "virtual_opponents",
    LAYER_DETECTED_OPPONENTS: "detected_opponents",
    LAYER_HISTORY_ALT: "past_car_states_alternative",
    LAYER_HISTORY_GT: "past_car_states_gt",
    LAYER_HISTORY_PRIOR: "past_car_states_prior",
    LAYER_HISTORY_PRIOR_FULL: "past_car_states_prior_full",
    LAYER_MPC_ROLLOUTS: "rollout_trajectory",
    LAYER_MPC_OPTIMAL: "optimal_trajectory",
}

COLOR_LEGACY_KEYS: Dict[str, str] = {
    LAYER_WAYPOINTS: "waypoints",
    LAYER_NEXT_WAYPOINTS: "next_waypoints",
    LAYER_NEXT_WAYPOINTS_POLY: "next_waypoints_polynomial",
    LAYER_NEXT_WAYPOINTS_ALT: "next_waypoints_alternative",
    LAYER_LIDAR: "lidar",
    LAYER_GAP: "gap",
    LAYER_MPC_ROLLOUTS: "mppi",
    LAYER_MPC_OPTIMAL: "optimal",
    LAYER_TARGET: "target",
    LAYER_OBSTACLES: "obstacles",
    LAYER_VIRTUAL_OPPONENTS: "virtual_opponents",
    LAYER_DETECTED_OPPONENTS: "detected_opponents",
    LAYER_TRACK_BORDER: "track_border",
    LAYER_TRACK_BORDER_LINES: "track_border",
    LAYER_HISTORY_ALT: "history_alt",
    LAYER_HISTORY_GT: "history_gt",
    LAYER_HISTORY_PRIOR: "history_prior",
    LAYER_HISTORY_PRIOR_FULL: "history_prior_full",
}

KNOWN_GENERIC_SKIP: Set[str] = set(LEGACY_LAYER_KEYS.keys()) | {
    LAYER_STEERING_ARROW,
    LAYER_EMERGENCY_SLOWDOWN,
}


def to_pose_points(data: Any) -> Optional[List[List[float]]]:
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size < 3:
            return None
        return [[float(arr[0]), float(arr[1]), float(arr[2])]]
    flat = arr.reshape(-1, arr.shape[-1])
    if flat.shape[1] < 3:
        return None
    return [[float(p[0]), float(p[1]), float(p[2])] for p in flat]


def to_xy_points(data: Any, state_like: bool = False) -> Optional[List[List[float]]]:
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size < 2:
            return None
        if state_like and arr.size > max(POSE_X_IDX, POSE_Y_IDX):
            return [[float(arr[POSE_X_IDX]), float(arr[POSE_Y_IDX])]]
        if arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
            return [[float(p[0]), float(p[1])] for p in arr]
        return [[float(arr[0]), float(arr[1])]]

    flat = arr.reshape(-1, arr.shape[-1])
    if state_like and flat.shape[1] > max(POSE_X_IDX, POSE_Y_IDX):
        return [[float(p[POSE_X_IDX]), float(p[POSE_Y_IDX])] for p in flat]
    if flat.shape[1] >= 2:
        return [[float(p[0]), float(p[1])] for p in flat]
    return None


def to_trajectory_list(data: Any) -> Optional[List[List[List[float]]]]:
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        pts = to_xy_points(arr, state_like=True)
        return [pts] if pts else None
    if arr.ndim >= 3:
        trajectories = []
        for traj in arr:
            pts = to_xy_points(traj, state_like=True)
            if pts:
                trajectories.append(pts)
        return trajectories if trajectories else None
    return None


def to_track_border_lines(data: Any) -> Optional[List[List[List[float]]]]:
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    if arr.ndim >= 3 and arr.shape[0] >= 2:
        left = to_xy_points(arr[0])
        right = to_xy_points(arr[1])
        lines = []
        if left:
            lines.append(left)
        if right:
            lines.append(right)
        return lines if lines else None
    return to_trajectory_list(arr)


def _serialize_layer_data(kind: str, data: Any, style: Dict[str, Any]) -> Any:
    state_like = bool(style.get("state_like", False))
    if kind == "points":
        return to_xy_points(data, state_like=state_like)
    if kind == "poses":
        return to_pose_points(data)
    if kind == "trajectories":
        return to_trajectory_list(data)
    if kind == "polyline":
        # Either a single polyline (Nx2) or list of polylines / (2,N,2) borders.
        lines = to_track_border_lines(data)
        if lines is not None:
            return lines
        pts = to_xy_points(data, state_like=state_like)
        return [pts] if pts else None
    if kind == "sprite":
        if isinstance(data, dict):
            out = {}
            for key, value in data.items():
                if key.endswith("_line") or key == "display_position":
                    out[key] = to_xy_points(value)
                else:
                    out[key] = value
            return out
        return data
    return data


class WebOverlayAdapter:
    """Map a RenderScene to the existing web_overlay dict (+ generic layers)."""

    @staticmethod
    def to_overlay(
        scene: RenderScene,
        *,
        force_plot_publish: bool = False,
        default_colors: Optional[Dict[str, Iterable[int]]] = None,
        virtual_opponent_size: Optional[List[float]] = None,
        car_state: Any = None,
    ) -> Dict[str, Any]:
        colors: Dict[str, List[int]] = {}
        if default_colors:
            for key, value in default_colors.items():
                colors[str(key)] = [int(v) for v in list(value)[:3]]

        overlay: Dict[str, Any] = {
            "label_dict": dict(scene.labels),
            "force_plot_publish": bool(force_plot_publish),
            "colors": colors,
            "layers": {},
        }

        for layer in scene.layers():
            serialized = _serialize_layer_data(layer.kind, layer.data, layer.style)
            if layer.color is not None:
                color_key = COLOR_LEGACY_KEYS.get(layer.name, layer.name)
                colors[color_key] = list(layer.color)

            legacy_key = LEGACY_LAYER_KEYS.get(layer.name)
            if legacy_key is not None:
                if layer.name == LAYER_TRACK_BORDER:
                    # Keep both dotted points and structured polylines for backends.
                    overlay["track_border_points"] = to_xy_points(layer.data)
                    overlay["track_border_lines"] = to_track_border_lines(layer.data)
                else:
                    overlay[legacy_key] = serialized
                continue

            if layer.name == LAYER_STEERING_ARROW and isinstance(serialized, dict):
                overlay["steering_arrow"] = serialized
                continue
            if layer.name == LAYER_EMERGENCY_SLOWDOWN and isinstance(serialized, dict):
                overlay["emergency_slowdown"] = serialized
                continue

            # Unknown / custom layers for generic frontend/pygame drawers.
            overlay["layers"][layer.name] = {
                "kind": layer.kind,
                "data": serialized,
                "lifetime": layer.lifetime,
                "color": list(layer.color) if layer.color is not None else None,
                "style": dict(layer.style),
            }

        # Derive track_border_lines from points if only points were set.
        if overlay.get("track_border_lines") is None and overlay.get("track_border_points") is not None:
            border_layer = scene.get(LAYER_TRACK_BORDER)
            if border_layer is not None:
                overlay["track_border_lines"] = to_track_border_lines(border_layer.data)

        if virtual_opponent_size and overlay.get("virtual_opponents"):
            overlay["virtual_opponent_size"] = list(virtual_opponent_size)

        # Steering arrow from car_state when not already set on the scene.
        if "steering_arrow" not in overlay and car_state is not None:
            car_state_arr = np.asarray(car_state)
            if car_state_arr.size > max(POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, STEERING_ANGLE_IDX):
                start_x = float(car_state_arr[POSE_X_IDX])
                start_y = float(car_state_arr[POSE_Y_IDX])
                heading = float(car_state_arr[POSE_THETA_IDX] + car_state_arr[STEERING_ANGLE_IDX])
                overlay["steering_arrow"] = {
                    "start": [start_x, start_y],
                    "end": [start_x + np.cos(heading), start_y + np.sin(heading)],
                }

        if not overlay["layers"]:
            overlay.pop("layers", None)
        return overlay


class PygameLayerDrawer:
    """Draw unknown ``overlay['layers']`` entries with existing pygame helpers."""

    def __init__(self, renderer: Any):
        self.renderer = renderer

    def draw(self, overlay: Dict[str, Any]) -> None:
        layers = overlay.get("layers")
        if not isinstance(layers, dict) or not layers:
            return
        draw_points = getattr(self.renderer, "_draw_points", None)
        draw_line = getattr(self.renderer, "_draw_line", None)
        draw_trajectories = getattr(self.renderer, "_draw_trajectories", None)
        draw_car = getattr(self.renderer, "draw_car", None)

        for _name, spec in layers.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("kind")
            data = spec.get("data")
            color = spec.get("color") or (200, 200, 200)
            if isinstance(color, list):
                color = tuple(int(v) for v in color[:3])
            style = spec.get("style") or {}
            radius = float(style.get("radius", 2.0))
            width = float(style.get("width", 1.5))
            max_points = style.get("max_points")

            if kind == "points" and draw_points is not None:
                if max_points is not None:
                    draw_points(data, color, radius, max_points=int(max_points))
                else:
                    draw_points(data, color, radius)
            elif kind == "polyline" and draw_line is not None:
                if (
                    isinstance(data, list)
                    and data
                    and isinstance(data[0], list)
                    and data[0]
                    and isinstance(data[0][0], (list, tuple))
                ):
                    for line in data:
                        draw_line(line, color, width)
                else:
                    draw_line(data, color, width)
            elif kind == "trajectories" and draw_trajectories is not None:
                draw_trajectories(data, color, width)
            elif kind == "poses" and draw_car is not None and isinstance(data, list):
                for pose in data:
                    if not isinstance(pose, (list, tuple)) or len(pose) < 3:
                        continue
                    draw_car((float(pose[0]), float(pose[1]), float(pose[2])), color)


class RosAdapter:
    """Future ROS visualization_msgs bridge.

    Consumes the same ``RenderScene.snapshot()`` as web/pygame. MarkerArray
    publishing is not implemented in-tree yet; wire this when the physical
    bridge package is available.
    """

    @staticmethod
    def snapshot_to_markers(snapshot: Dict[str, Any]) -> None:
        raise NotImplementedError(
            "RosAdapter.snapshot_to_markers is a stub. Publish visualization_msgs "
            "MarkerArray from snapshot['layers'] in the ROS bridge package."
        )

    @staticmethod
    def publish_scene(scene: RenderScene) -> None:
        RosAdapter.snapshot_to_markers(scene.snapshot())

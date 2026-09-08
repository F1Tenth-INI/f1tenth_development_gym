"""Convert an experiment CSV into a video that matches the live sim overlays."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import trange

from utilities.map_scale import scale_map_metadata
from utilities.recording_replay import (
    lidar_columns_and_angles,
    lidar_points_from_ranges,
    load_virtual_opponent_replay_poses,
    next_waypoints_array_from_dataframe,
    resolve_map_for_recording,
)


def _bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _car_vertices(x, y, theta, length, width):
    """Axis-aligned body rectangle rotated into world coordinates (same as sim)."""
    c = math.cos(theta)
    s = math.sin(theta)
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[-hl, hw], [-hl, -hw], [hl, -hw], [hl, hw]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([x, y], dtype=np.float64)


# Match pygame / web renderer colors (stored here as RGB, drawn as BGR).
COLOR_BACKGROUND = _bgr((9, 32, 87))
COLOR_MAP = _bgr((183, 193, 222))
COLOR_EGO = _bgr((172, 97, 185))
COLOR_VIRTUAL_OPPONENT = _bgr((255, 140, 0))
COLOR_WAYPOINTS = _bgr((64, 190, 255))
COLOR_NEXT_WAYPOINTS = _bgr((0, 127, 0))
COLOR_LIDAR = _bgr((255, 0, 255))
COLOR_TRACK_BORDER = _bgr((255, 0, 0))
COLOR_STEERING = _bgr((0, 204, 0))
COLOR_TEXT = (255, 255, 255)
COLOR_CAR_OUTLINE = (255, 255, 255)


class RecordingToVideoConverter:
    """Render a recording CSV with the same overlays as the live sim renderer."""

    def __init__(
        self,
        csv_file_path,
        recording_name,
        map_name=None,
        *,
        follow_camera=True,
        width=1280,
        height=720,
        zoom=60.0,
    ):
        self.csv_file = os.path.join(csv_file_path, recording_name + ".csv")
        self.video_output_path = os.path.join(csv_file_path, recording_name + "_data")
        self.video_output_file = os.path.join(self.video_output_path, "recording.mp4")
        os.makedirs(self.video_output_path, exist_ok=True)

        self.follow_camera = bool(follow_camera)
        self.zoom = float(zoom)
        self.width = int(width)
        self.height = int(height)

        self.df = pd.read_csv(self.csv_file, comment="#")
        required = ("pose_x", "pose_y", "pose_theta")
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise KeyError(f"Recording is missing columns required for video: {missing}")

        self.pose_x = self.df["pose_x"].to_numpy(dtype=np.float64)
        self.pose_y = self.df["pose_y"].to_numpy(dtype=np.float64)
        self.pose_theta = self.df["pose_theta"].to_numpy(dtype=np.float64)
        self.n_frames = int(len(self.df))

        if "linear_vel_x" in self.df.columns:
            self.vel_x = self.df["linear_vel_x"].to_numpy(dtype=np.float64)
        else:
            self.vel_x = np.zeros(self.n_frames, dtype=np.float64)
        if "steering_angle" in self.df.columns:
            self.steering = self.df["steering_angle"].to_numpy(dtype=np.float64)
        else:
            self.steering = np.zeros(self.n_frames, dtype=np.float64)
        if "time" in self.df.columns:
            self.times = self.df["time"].to_numpy(dtype=np.float64)
        else:
            self.times = np.arange(self.n_frames, dtype=np.float64) * 0.04

        self._load_map(map_name)
        self._load_vehicle_sizes()
        self._load_static_overlays()
        self._load_dynamic_arrays()

        self.fps = self._infer_fps()
        if not self.follow_camera and self.map_img is not None:
            self.width = int(self.map_img.shape[1])
            self.height = int(self.map_img.shape[0])
            self.zoom = 1.0 / max(self.map_resolution, 1e-9)

        self.cam_x = float(self.pose_x[0]) if self.n_frames else 0.0
        self.cam_y = float(self.pose_y[0]) if self.n_frames else 0.0

    def _infer_fps(self) -> float:
        if self.n_frames >= 2:
            dt = np.median(np.diff(self.times))
            if np.isfinite(dt) and dt > 1e-4:
                return float(np.clip(round(1.0 / dt), 5, 60))
        try:
            from utilities.Settings import Settings

            return float(np.clip(round(1.0 / max(Settings.TIMESTEP_CONTROL, 1e-4)), 5, 60))
        except Exception:
            return 25.0

    def _load_map(self, map_name):
        self.map_img = None
        self._map_scaled = None
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_resolution = 0.05
        self.map_render_path = None
        self.map_name = map_name

        try:
            map_render_path, resolved_name = resolve_map_for_recording(
                self.csv_file, map_override=map_name
            )
            self.map_render_path = map_render_path
            self.map_name = resolved_name or map_name
        except Exception:
            if map_name:
                self.map_render_path = os.path.join("utilities", "maps", map_name)
                self.map_name = map_name

        if not self.map_render_path or not self.map_name:
            return

        yaml_path = self.map_render_path + ".yaml"
        if not os.path.isfile(yaml_path):
            yaml_path = os.path.join(os.path.dirname(self.map_render_path), f"{self.map_name}.yaml")
        if not os.path.isfile(yaml_path):
            print(f"Video renderer: map yaml not found for {self.map_name}")
            return

        with open(yaml_path, "r") as f:
            map_config = scale_map_metadata(yaml.safe_load(f))
        origin = map_config.get("origin", [0.0, 0.0, 0.0])
        self.map_origin_x = float(origin[0])
        self.map_origin_y = float(origin[1])
        self.map_resolution = float(map_config.get("resolution", 0.05))

        image_ref = str(map_config.get("image") or f"{self.map_name}.png")
        image_path = (
            image_ref if os.path.isabs(image_ref) else os.path.join(os.path.dirname(yaml_path), image_ref)
        )
        occupancy = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if occupancy is None:
            print(f"Video renderer: map image not found: {image_path}")
            return
        if occupancy.ndim == 3:
            occupancy = occupancy[:, :, 0]

        # Pygame occupancy style: zero pixels are obstacles on the dark sim background.
        map_h, map_w = occupancy.shape[:2]
        map_bgr = np.full((map_h, map_w, 3), COLOR_BACKGROUND, dtype=np.uint8)
        map_bgr[occupancy == 0] = COLOR_MAP
        self.map_img = map_bgr
        self._map_scaled = None
        self._map_draw_w = 0
        self._map_draw_h = 0
        self._map_height_m = map_h * self.map_resolution

    def _load_vehicle_sizes(self):
        self.car_length = 0.58
        self.car_width = 0.31
        self.opponent_length = 0.58
        self.opponent_width = 0.31
        try:
            from utilities.Settings import Settings
            from utilities.car_files.vehicle_parameters import VehicleParameters

            params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
            self.car_length = float(params.length)
            self.car_width = float(params.width)
        except Exception:
            pass
        try:
            from utilities.Settings import Settings

            size = getattr(Settings, "VIRTUAL_OPPONENT_SIZE", None)
            if size is not None and len(size) >= 2:
                self.opponent_width = float(size[0])
                self.opponent_length = float(size[1])
            else:
                self.opponent_length, self.opponent_width = self.car_length, self.car_width
        except Exception:
            self.opponent_length, self.opponent_width = self.car_length, self.car_width

    def _load_static_overlays(self):
        self.global_waypoints = None
        self.track_border_left = None
        self.track_border_right = None
        if not self.map_render_path or not self.map_name:
            return
        map_dir = os.path.dirname(self.map_render_path + ".yaml")
        wp_path = os.path.join(map_dir, f"{self.map_name}_wp.csv")
        try:
            from utilities.Settings import Settings

            if bool(getattr(Settings, "REVERSE_DIRECTION", False)):
                reverse_path = os.path.join(map_dir, f"{self.map_name}_wp_reverse.csv")
                if os.path.isfile(reverse_path):
                    wp_path = reverse_path
            scale = float(getattr(Settings, "MAP_SCALE", 1.0) or 1.0)
        except Exception:
            scale = 1.0
        if not os.path.isfile(wp_path):
            return
        try:
            wp_df = pd.read_csv(wp_path, comment="#")
            if not {"x_m", "y_m"}.issubset(wp_df.columns):
                return
            xy = wp_df[["x_m", "y_m"]].to_numpy(dtype=np.float32) * scale
            self.global_waypoints = xy
            left_col = "d_left_iqp" if "d_left_iqp" in wp_df.columns else "d_left_m"
            right_col = "d_right_iqp" if "d_right_iqp" in wp_df.columns else "d_right_m"
            if "psi_rad" in wp_df.columns and left_col in wp_df.columns and right_col in wp_df.columns:
                # WaypointUtils stores translational heading = psi_rad + pi/2.
                psi = wp_df["psi_rad"].to_numpy(dtype=np.float64) + 0.5 * np.pi
                d_left = wp_df[left_col].to_numpy(dtype=np.float64) * scale
                d_right = wp_df[right_col].to_numpy(dtype=np.float64) * scale
                x = xy[:, 0].astype(np.float64)
                y = xy[:, 1].astype(np.float64)
                self.track_border_left = np.column_stack(
                    (x + d_left * np.cos(psi + np.pi / 2), y + d_left * np.sin(psi + np.pi / 2))
                ).astype(np.float32)
                self.track_border_right = np.column_stack(
                    (x + d_right * np.cos(psi - np.pi / 2), y + d_right * np.sin(psi - np.pi / 2))
                ).astype(np.float32)
        except Exception as exc:
            print(f"Video renderer: waypoints/track borders skipped: {exc}")

    def _load_dynamic_arrays(self):
        self.next_waypoints = next_waypoints_array_from_dataframe(self.df)
        self.vo_poses = None
        try:
            self.vo_poses = load_virtual_opponent_replay_poses(self.csv_file)
        except Exception:
            self.vo_poses = None

        self.lidar_cols, self.lidar_angles = lidar_columns_and_angles(self.df.columns)
        if self.lidar_cols:
            self.lidar_ranges = self.df[self.lidar_cols].to_numpy(dtype=np.float32)
        else:
            self.lidar_ranges = None

    def _set_camera(self, i: int):
        if self.follow_camera:
            self.cam_x = float(self.pose_x[i])
            self.cam_y = float(self.pose_y[i])
            return
        if self.map_img is None:
            self.cam_x = float(self.pose_x[i])
            self.cam_y = float(self.pose_y[i])
            return
        map_h, map_w = self.map_img.shape[:2]
        self.cam_x = self.map_origin_x + 0.5 * map_w * self.map_resolution
        self.cam_y = self.map_origin_y + 0.5 * map_h * self.map_resolution

    def world_to_screen_array(self, xy) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        pts = pts.reshape(-1, pts.shape[-1])[:, :2]
        sx = (pts[:, 0] - self.cam_x) * self.zoom + self.width / 2.0
        sy = self.height / 2.0 - (pts[:, 1] - self.cam_y) * self.zoom
        return np.stack([sx, sy], axis=1).astype(np.int32)

    def _ensure_scaled_map(self):
        """Cache the occupancy image at the current zoom (pixels per meter)."""
        if self.map_img is None:
            self._map_scaled = None
            return
        map_h, map_w = self.map_img.shape[:2]
        draw_w = max(1, int(round(map_w * self.map_resolution * self.zoom)))
        draw_h = max(1, int(round(map_h * self.map_resolution * self.zoom)))
        if (
            getattr(self, "_map_scaled", None) is not None
            and self._map_scaled.shape[1] == draw_w
            and self._map_scaled.shape[0] == draw_h
        ):
            return
        self._map_scaled = cv2.resize(
            self.map_img, (draw_w, draw_h), interpolation=cv2.INTER_NEAREST
        )
        self._map_draw_w = draw_w
        self._map_draw_h = draw_h
        self._map_height_m = map_h * self.map_resolution

    def _draw_map(self, frame):
        """Place the occupancy image like WebRenderer: NW corner at origin+(0, height)."""
        frame[:] = COLOR_BACKGROUND
        self._ensure_scaled_map()
        if self._map_scaled is None:
            return
        # Same placement as sim/f110_sim/envs/WebRenderer/index.html drawMapLayer().
        x0 = (self.map_origin_x - self.cam_x) * self.zoom + 0.5 * self.width
        y0 = self.height - (
            (self.map_origin_y + self._map_height_m - self.cam_y) * self.zoom
            + 0.5 * self.height
        )
        dst_x = int(round(x0))
        dst_y = int(round(y0))
        ix0 = max(0, dst_x)
        iy0 = max(0, dst_y)
        ix1 = min(self.width, dst_x + self._map_draw_w)
        iy1 = min(self.height, dst_y + self._map_draw_h)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        sx0 = ix0 - dst_x
        sy0 = iy0 - dst_y
        frame[iy0:iy1, ix0:ix1] = self._map_scaled[
            sy0 : sy0 + (iy1 - iy0),
            sx0 : sx0 + (ix1 - ix0),
        ]

    def _in_view_mask(self, screen_xy, margin=8):
        return (
            (screen_xy[:, 0] >= -margin)
            & (screen_xy[:, 0] < self.width + margin)
            & (screen_xy[:, 1] >= -margin)
            & (screen_xy[:, 1] < self.height + margin)
        )

    def _draw_points(self, frame, xy, color, radius=2):
        if xy is None:
            return
        pts = self.world_to_screen_array(xy)
        if pts.shape[0] == 0:
            return
        pts = pts[self._in_view_mask(pts, margin=radius + 2)]
        radius = max(1, int(radius))
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)

    def _draw_polyline(self, frame, xy, color, thickness=1, closed=False):
        if xy is None:
            return
        pts = np.asarray(xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return
        screen = self.world_to_screen_array(pts)
        if closed:
            screen = np.vstack([screen, screen[:1]])
        cv2.polylines(frame, [screen], False, color, int(max(1, thickness)), lineType=cv2.LINE_AA)

    def _draw_car(self, frame, x, y, theta, length, width, color):
        vertices = _car_vertices(x, y, theta, length, width)
        pts = self.world_to_screen_array(vertices)
        cv2.fillConvexPoly(frame, pts, color, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, COLOR_CAR_OUTLINE, 1, lineType=cv2.LINE_AA)
        # Nose mark so heading is obvious at a glance.
        front_x = x + 0.5 * length * math.cos(theta)
        front_y = y + 0.5 * length * math.sin(theta)
        nose = self.world_to_screen_array([[front_x, front_y], [x, y]])
        cv2.line(frame, tuple(nose[1]), tuple(nose[0]), COLOR_CAR_OUTLINE, 1, lineType=cv2.LINE_AA)

    def _velocity_to_color(self, v, v_min=0.0, v_max=10.0):
        t = float(np.clip((v - v_min) / max(v_max - v_min, 1e-6), 0.0, 1.0))
        return _bgr((int(255 * t), int(255 * (1.0 - t)), 50))

    def _draw_trail(self, frame, i, max_points=200):
        start = max(0, i - max_points)
        if i - start < 2:
            return
        xs = self.pose_x[start : i + 1]
        ys = self.pose_y[start : i + 1]
        vs = self.vel_x[start : i + 1]
        screen = self.world_to_screen_array(np.column_stack([xs, ys]))
        for k in range(1, len(screen)):
            color = self._velocity_to_color(vs[k])
            cv2.line(frame, tuple(screen[k - 1]), tuple(screen[k]), color, 2, lineType=cv2.LINE_AA)

    def _draw_hud(self, frame, i, n_opponents):
        v = float(self.vel_x[i])
        t = float(self.times[i])
        text = (
            f"t={t:6.2f}s   x={self.pose_x[i]:6.2f} y={self.pose_y[i]:6.2f}   "
            f"yaw={self.pose_theta[i]:5.2f}   v={v:5.2f} m/s"
        )
        if n_opponents:
            text += f"   opponents={n_opponents}"
        cv2.putText(
            frame,
            text,
            (12, self.height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

    def render_video(self):
        if self.n_frames == 0:
            print("Video renderer: recording is empty, nothing to write")
            return

        lidar_max_points = 220
        try:
            from utilities.Settings import Settings

            lidar_max_points = int(getattr(Settings, "WEB_RENDER_LIDAR_POINTS", lidar_max_points))
        except Exception:
            pass

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            self.video_output_file, fourcc, self.fps, (self.width, self.height)
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.video_output_file}")

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in trange(self.n_frames, desc="Rendering frames"):
            self._set_camera(i)
            self._draw_map(frame)

            self._draw_polyline(frame, self.global_waypoints, COLOR_WAYPOINTS, thickness=1, closed=True)
            self._draw_points(frame, self.global_waypoints, COLOR_WAYPOINTS, radius=2)
            self._draw_polyline(frame, self.track_border_left, COLOR_TRACK_BORDER, thickness=1, closed=True)
            self._draw_polyline(frame, self.track_border_right, COLOR_TRACK_BORDER, thickness=1, closed=True)

            self._draw_trail(frame, i)

            if self.next_waypoints is not None:
                wypt = self.next_waypoints[i]
                finite = np.isfinite(wypt).all(axis=1)
                self._draw_points(frame, wypt[finite], COLOR_NEXT_WAYPOINTS, radius=3)

            if self.lidar_ranges is not None and self.lidar_angles is not None:
                lidar_xy = lidar_points_from_ranges(
                    self.lidar_ranges[i],
                    self.lidar_angles,
                    self.pose_x[i],
                    self.pose_y[i],
                    self.pose_theta[i],
                    max_points=lidar_max_points,
                )
                self._draw_points(frame, lidar_xy, COLOR_LIDAR, radius=3)

            n_opponents = 0
            if self.vo_poses is not None and i < len(self.vo_poses):
                for slot_pose in self.vo_poses[i]:
                    if not np.isfinite(slot_pose[0]):
                        continue
                    n_opponents += 1
                    self._draw_car(
                        frame,
                        float(slot_pose[0]),
                        float(slot_pose[1]),
                        float(slot_pose[2]),
                        self.opponent_length,
                        self.opponent_width,
                        COLOR_VIRTUAL_OPPONENT,
                    )

            x, y, theta = float(self.pose_x[i]), float(self.pose_y[i]), float(self.pose_theta[i])
            heading = theta + float(self.steering[i])
            arrow = np.array([[x, y], [x + math.cos(heading), y + math.sin(heading)]], dtype=np.float64)
            self._draw_polyline(frame, arrow, COLOR_STEERING, thickness=2)
            self._draw_car(frame, x, y, theta, self.car_length, self.car_width, COLOR_EGO)
            self._draw_hud(frame, i, n_opponents)
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved to {self.video_output_file}")


if __name__ == "__main__":
    converter = RecordingToVideoConverter("ExperimentRecordings", "recording", "RCA1")
    converter.render_video()

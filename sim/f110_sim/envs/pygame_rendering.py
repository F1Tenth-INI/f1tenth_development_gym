"""
Real-time pygame renderer.

Mirrors the visualization features of `WebEnvRenderer` but updates the screen
synchronously, frame-for-frame with the simulator. The web renderer trades
latency for smooth playback (it interpolates a buffered history); this backend
is meant for live debugging where the lowest possible visual delay matters.

The simulator publishes the same `render_obs` payload as for the web client,
including a flattened `web_overlay` dict. We consume that overlay directly
and draw the same set of visual elements:

- map background image
- global / alternative / next waypoints
- LiDAR border points
- track border lines (with a points fallback)
- obstacle markers
- per-history past car states (gt, prior, prior_full, alternative)
- MPPI/RPGD rollout cloud and optimal trajectory
- largest-gap middle point and target point
- steering direction arrow
- emergency slowdown left/right/stop lines + factor display
- live angular/translational control time-series plots

Performance design notes
------------------------
Pygame is software-rasterized, so a naive `for pt in pts: draw.circle(...)`
loop becomes the dominant cost when a frame contains thousands of waypoints
or rollout samples. The hot loops below use three batching tricks:

1. Cached dot sprites + `Surface.blits([...])`: per-color/per-radius dots are
   rendered once into a small SRCALPHA surface and replicated across the screen
   using a single C-level batch call.
2. Vectorized world->screen with numpy and an early off-screen frustum cull,
   so we only generate blit destinations for points that are actually visible.
3. A "static-layer" surface that bakes the map plus genuinely-static overlay
   layers (global waypoints, alt waypoints, track border points) into one
   world-space image; we re-scale it only when the zoom level changes.

The renderer also caps redraw to roughly 60 FPS based on wall time. State is
always updated through `update_obs`, so when the sim runs faster than 60 Hz we
simply skip the `pygame.display.flip()` step rather than spending time on
draws the user cannot perceive.
"""

import time
from collections import deque

import numpy as np

from utilities.sdl_env import configure_macos_bluetooth_gamepad

configure_macos_bluetooth_gamepad()
import pygame  # noqa: E402
import yaml
from PIL import Image

from f110_sim.envs.collision_models import get_vertices
from utilities.Settings import Settings
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    MOTOR_ANGULAR_VEL_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)
from sim.f110_sim.envs.motor_speed_audio import MotorSpeedAudio, motor_erpm_from_omega


CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR
ZOOM_MIN = 2.0
ZOOM_MAX = 500.0

CONTROL_PLOT_WINDOW_S = 10.0
CONTROL_PLOT_MAX_SAMPLES = 900

# Cap how many points we ever try to draw for high-cardinality dynamic layers.
# These caps mirror what the web renderer already enforces on the publisher
# side, but we re-apply them defensively so a slow consumer cannot get pinned
# by a dense scan.
LIDAR_MAX_POINTS = 200
HISTORY_MAX_POINTS = 200

# Wall-clock-driven render throttle. State always updates, so the next draw
# is always fresh; we just skip the GPU/display flip when frames arrive faster
# than the user can perceive. The default mirrors typical 60 Hz monitors and
# is overridden at construction time from `Settings.PYGAME_RENDER_FPS`.
DEFAULT_RENDER_FPS = 60.0

# When rasterizing the static-overlay surface we anchor it at a chosen world
# resolution (meters per pixel). Smaller -> sharper at high zoom but bigger
# surface; this mirrors how the map surface is built.
STATIC_OVERLAY_PIXELS_PER_METER = 100.0
STATIC_OVERLAY_MAX_DIM = 4000  # safety clamp on either dimension

DEFAULT_COLORS = {
    "background": (9, 32, 87),
    "map_dot": (183, 193, 222),
    "ego_car": (172, 97, 185),
    "opponent_car": (99, 52, 94),
    "waypoints": (64, 190, 255),
    "waypoints_alternative": (170, 170, 170),
    "next_waypoints": (0, 127, 0),
    "next_waypoints_alternative": (127, 0, 127),
    "lidar": (255, 0, 255),
    "track_border": (255, 0, 0),
    "obstacles": (255, 0, 0),
    "history_alt": (255, 255, 0),
    "history_gt": (0, 128, 255),
    "history_prior": (255, 255, 255),
    "history_prior_full": (0, 255, 255),
    "mppi": (250, 25, 30),
    "optimal": (255, 165, 0),
    "gap": (0, 255, 0),
    "target": (255, 204, 0),
    "steering_arrow": (0, 204, 0),
    "emergency_stop": (255, 0, 0),
    "panel_bg": (4, 12, 38),
    "panel_border": (255, 255, 255),
    "plot_angular": (255, 179, 71),
    "plot_translational": (118, 215, 196),
    "text": (255, 255, 255),
}


def _as_color(value, fallback):
    """Coerce overlay color arrays (list/tuple of 3 numbers) to a pygame RGB tuple."""
    if value is None:
        return fallback
    try:
        if len(value) >= 3:
            return (
                int(np.clip(value[0], 0, 255)),
                int(np.clip(value[1], 0, 255)),
                int(np.clip(value[2], 0, 255)),
            )
    except TypeError:
        pass
    return fallback


def _to_xy_array(points):
    """
    Coerce arbitrary 2D-point inputs to an (N, 2) float32 numpy array, or
    None if the input has no usable points.
    """
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size < 2 or arr.size % 2 != 0:
            return None
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def _parse_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if np.isfinite(parsed):
            return parsed
    return None


def _normalize_label_name(key):
    text = str(key or "").strip().lower()
    if ":" in text:
        # Strip leading numeric prefix like "0: angular_control".
        text = text.split(":", 1)[1].strip()
    return text


def _resolve_control_key(labels, contains_text):
    if not isinstance(labels, dict):
        return None
    for key, value in labels.items():
        if _parse_numeric(value) is None:
            continue
        if contains_text in _normalize_label_name(key):
            return key
    return None


def _layer_fingerprint(arr, color):
    """
    Cheap signature of a static layer: shape, color and a few sample rows.
    Used to detect when the underlying point set has actually changed so we
    can rebuild the cached static-overlay surface, which is otherwise reused
    across frames.
    """
    if arr is None:
        return None
    n = arr.shape[0]
    if n == 0:
        return (0, color)
    sample_idx = (0, n // 2, n - 1)
    samples = tuple(tuple(arr[i].tolist()) for i in sample_idx)
    return (n, color, samples)


class EnvRenderer:
    """
    Live pygame visualization that consumes the same `web_overlay` payload as
    the browser client. Suitable for low-latency, in-loop debugging.
    """

    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("F1TENTH Sim (pygame)")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)
        self.small_font = pygame.font.SysFont(None, 16)

        self.width = width
        self.height = height
        self.zoom_level = 60.0
        self.camera_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.camera_follow_ego = bool(getattr(Settings, "CAMERA_AUTO_FOLLOW", True))

        # Map state.
        self.map_surface = None
        self.scaled_map_surface = None
        self.last_map_scale = None
        self.map_origin = np.array([0.0, 0.0], dtype=np.float32)
        self.map_resolution = 1.0
        self.map_image_shape = (0, 0)  # (width_px, height_px)

        # Static overlay surface (waypoints + track borders pre-rasterized).
        self._static_layers_surface = None
        self._static_layers_origin = np.array([0.0, 0.0], dtype=np.float32)
        self._static_layers_max_y = 0.0
        self._static_layers_shape = (0, 0)  # (width_px, height_px)
        self._static_layers_resolution = 1.0 / STATIC_OVERLAY_PIXELS_PER_METER
        self._static_layers_scaled = None
        self._static_layers_last_scale = None
        self._static_layers_signature = None

        # Sim state.
        self.poses = None
        self.vels = None
        self.car_states = None
        self.ego_idx = 0
        self.sim_time = 0.0
        self.lap_count = 0
        self.overlay = {}

        # Mouse / keyboard.
        self.dragging = False
        self.last_mouse_pos = None

        # Toggles match the default state of the web renderer UI.
        self.show_map = True
        self.show_position_history = False
        self.show_track_borders = True
        self.show_car_info = False
        self.show_control_plots = True

        # Sprite cache for dot rendering (color, radius) -> SRCALPHA surface.
        self._dot_sprite_cache = {}

        # Live time-series plot state for angular / translational controls.
        self._control_series = {
            "angular_control": deque(),
            "translational_control": deque(),
        }
        self._plot_panel_size = (440, 280)

        # Throttle bookkeeping. `PYGAME_RENDER_FPS = 0` (or None) disables the
        # cap entirely and lets the renderer flip on every sim step.
        configured_fps = float(getattr(Settings, "PYGAME_RENDER_FPS", DEFAULT_RENDER_FPS) or 0.0)
        self._target_fps = configured_fps if configured_fps > 0.0 else 0.0
        self._min_frame_dt = (1.0 / self._target_fps) if self._target_fps > 0.0 else 0.0
        self._last_draw_wall_s = 0.0
        self._fps_ema = 0.0

        self._debug_controls = None

        self._motor_audio = None
        if getattr(Settings, "PYGAME_MOTOR_AUDIO", True):
            self._motor_audio = MotorSpeedAudio.from_settings()
            if self._motor_audio.available:
                self._motor_audio.start()

    # ------------------------------------------------------------------
    # Map and observation handling
    # ------------------------------------------------------------------

    def update_map(self, map_path, map_ext):
        with open(map_path + ".yaml", "r") as f:
            metadata = yaml.safe_load(f)
        res = float(metadata["resolution"])
        origin_x, origin_y = (float(metadata["origin"][0]), float(metadata["origin"][1]))

        img = Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)
        img_arr = np.array(img)
        if img_arr.ndim == 3:
            # Convert RGB(A) maps to single-channel grayscale for the obstacle test.
            img_arr = img_arr[..., 0]
        map_height, map_width = img_arr.shape[:2]

        # Vectorized rasterization: zero pixels are obstacles, everything else
        # transparent. This replaces a tens-of-thousands-deep `pygame.draw.circle`
        # loop with a single numpy assignment + `pygame.image.frombuffer`.
        rgba = np.zeros((map_height, map_width, 4), dtype=np.uint8)
        obstacle_mask = img_arr == 0
        rgba[obstacle_mask, 0:3] = DEFAULT_COLORS["map_dot"]
        rgba[obstacle_mask, 3] = 255
        # `pygame.image.frombuffer` expects bytes in row-major order matching
        # surface coordinates (width, height).
        self.map_surface = pygame.image.frombuffer(
            rgba.tobytes(), (map_width, map_height), "RGBA"
        ).convert_alpha()
        # Row 0 is world min-Y after FLIP_TOP_BOTTOM; flip so blit top-left = north edge.
        self.map_surface = pygame.transform.flip(self.map_surface, False, True)

        self.map_origin = np.array([origin_x, origin_y], dtype=np.float32)
        self.map_resolution = res
        self.map_image_shape = (map_width, map_height)
        self.last_map_scale = None
        self.scaled_map_surface = None

    def update_obs(self, obs):
        self.ego_idx = int(obs.get("ego_idx", 0))
        self.sim_time = float(obs.get("simulation_time", 0.0))

        # Support both legacy renderer obs (poses_x/poses_y/poses_theta) and
        # current simulator obs (car_states).
        if all(k in obs for k in ("poses_x", "poses_y", "poses_theta")):
            self.poses = np.stack(
                (obs["poses_x"], obs["poses_y"], obs["poses_theta"]), axis=-1
            )
            self.vels = np.asarray(
                obs.get("linear_vels_x", np.zeros(len(self.poses), dtype=np.float32))
            )
            self.car_states = None
        elif "car_states" in obs:
            car_states = np.asarray(obs["car_states"])
            self.car_states = car_states
            self.poses = np.stack(
                (
                    car_states[:, POSE_X_IDX],
                    car_states[:, POSE_Y_IDX],
                    car_states[:, POSE_THETA_IDX],
                ),
                axis=-1,
            )
            self.vels = car_states[:, LINEAR_VEL_X_IDX]
        else:
            self.poses = None
            self.vels = np.zeros(1, dtype=np.float32)
            self.car_states = None

        lap_counts = obs.get("lap_counts")
        if lap_counts is not None and len(lap_counts) > self.ego_idx:
            self.lap_count = int(lap_counts[self.ego_idx])
        else:
            self.lap_count = 0

        self.overlay = obs.get("web_overlay", {}) or {}
        self._debug_controls = obs.get("debug_controls")
        self._update_control_series()
        self._update_motor_audio()

    def _update_motor_audio(self) -> None:
        """Feed latest motor speed to the async audio stream (non-blocking)."""
        if self._motor_audio is None or not self._motor_audio.available:
            return
        if self.car_states is None or self.car_states.shape[1] <= MOTOR_ANGULAR_VEL_IDX:
            return
        idx = min(self.ego_idx, len(self.car_states) - 1)
        omega = float(self.car_states[idx, MOTOR_ANGULAR_VEL_IDX])
        if np.isfinite(omega):
            self._motor_audio.set_target_omega(omega)

    # ------------------------------------------------------------------
    # Camera and input
    # ------------------------------------------------------------------

    @staticmethod
    def _screen_int(x, y):
        if not (np.isfinite(x) and np.isfinite(y)):
            return 0, 0
        return int(x), int(y)

    def world_to_screen(self, point):
        x, y = float(point[0]), float(point[1])
        sx = (x - self.camera_offset[0]) * self.zoom_level + self.width / 2
        # Match web renderer: world +Y is up, pygame screen +Y is down.
        sy = self.height / 2 - (y - self.camera_offset[1]) * self.zoom_level
        return self._screen_int(sx, sy)

    def _world_to_screen_array(self, pts, offset_xy=(0.0, 0.0)):
        """
        Vectorized world->screen for an (N, 2) array. `offset_xy` is subtracted
        from the screen-space result and is used to position pre-baked sprites
        at their top-left corner instead of their center.
        """
        sx = (pts[:, 0] - self.camera_offset[0]) * self.zoom_level + (
            self.width / 2 - offset_xy[0]
        )
        sy = self.height / 2 - (pts[:, 1] - self.camera_offset[1]) * self.zoom_level - (
            offset_xy[1]
        )
        return sx, sy

    def handle_events(self):
        pygame.event.pump()
        # Do not drain JOY* events — manual control reads axes via pygame.joystick.
        ui_events = (
            pygame.QUIT,
            pygame.VIDEORESIZE,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
            pygame.MOUSEMOTION,
            pygame.KEYDOWN,
        )
        for event in pygame.event.get(ui_events):
            if event.type == pygame.QUIT:
                pygame.quit()
                raise Exception("Rendering window was closed.")
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode(
                    (self.width, self.height), pygame.RESIZABLE
                )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                    # Manual pan disables auto-follow until the user re-enables it.
                    self.camera_follow_ego = False
                elif event.button == 4:
                    self.zoom(1)
                elif event.button == 5:
                    self.zoom(-1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                current_mouse_pos = pygame.mouse.get_pos()
                dx, dy = np.subtract(current_mouse_pos, self.last_mouse_pos)
                self.pan(dx, dy)
                self.last_mouse_pos = current_mouse_pos
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key):
        if key == pygame.K_SPACE or key == pygame.K_f:
            self.camera_follow_ego = not self.camera_follow_ego
        elif key == pygame.K_m:
            self.show_map = not self.show_map
        elif key == pygame.K_h:
            self.show_position_history = not self.show_position_history
        elif key == pygame.K_b:
            self.show_track_borders = not self.show_track_borders
        elif key == pygame.K_i:
            self.show_car_info = not self.show_car_info
        elif key == pygame.K_p:
            self.show_control_plots = not self.show_control_plots
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.zoom(1)
        elif key == pygame.K_MINUS:
            self.zoom(-1)

    def zoom(self, direction):
        factor = ZOOM_IN_FACTOR if direction > 0 else ZOOM_OUT_FACTOR
        new_zoom = self.zoom_level * factor
        if not (ZOOM_MIN < new_zoom < ZOOM_MAX):
            return
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen_center = np.array([self.width / 2, self.height / 2])
        world_mouse_before = np.array(
            [
                (mouse_x - self.width / 2) / self.zoom_level + self.camera_offset[0],
                self.camera_offset[1]
                - (mouse_y - self.height / 2) / self.zoom_level,
            ]
        )
        self.zoom_level = new_zoom
        world_mouse_after = np.array(
            [
                (mouse_x - self.width / 2) / self.zoom_level + self.camera_offset[0],
                self.camera_offset[1]
                - (mouse_y - self.height / 2) / self.zoom_level,
            ]
        )
        self.camera_offset += world_mouse_before - world_mouse_after
        # Re-scaling of cached surfaces happens lazily in their draw paths.
        self.last_map_scale = None
        self._static_layers_last_scale = None

    def pan(self, dx, dy):
        self.camera_offset -= np.array([dx, dy]) / self.zoom_level

    def _update_camera_follow(self):
        if not self.camera_follow_ego or self.poses is None or len(self.poses) == 0:
            return
        ego = self.poses[min(self.ego_idx, len(self.poses) - 1)]
        if np.all(np.isfinite(ego[:2])):
            self.camera_offset = np.array([float(ego[0]), float(ego[1])], dtype=np.float32)

    # ------------------------------------------------------------------
    # Sprite cache + batched dot drawing
    # ------------------------------------------------------------------

    def _get_dot_sprite(self, color, radius):
        radius = max(1, int(radius))
        key = (color, radius)
        sprite = self._dot_sprite_cache.get(key)
        if sprite is None:
            size = radius * 2 + 1
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(sprite, color, (radius, radius), radius)
            sprite = sprite.convert_alpha()
            self._dot_sprite_cache[key] = sprite
        return sprite

    def _draw_points_array(self, pts, color, radius, target_surface=None,
                           offset_world=(0.0, 0.0), pixels_per_meter=None):
        """
        Batch-blit a sprite at every visible point.

        `pts` is an (N, 2) world-coord numpy array. When `target_surface` is
        None we draw onto the live screen with the current camera; otherwise
        we draw onto a static cached surface, in which case `offset_world` and
        `pixels_per_meter` describe the surface's world-space transform.
        """
        if pts is None or pts.shape[0] == 0:
            return
        sprite = self._get_dot_sprite(color, radius)
        sw, sh = sprite.get_size()
        half_w = sw / 2
        half_h = sh / 2

        if target_surface is None:
            sx, sy = self._world_to_screen_array(pts, offset_xy=(half_w, half_h))
            surf = self.screen
            surf_w = self.width
            surf_h = self.height
        else:
            ppm = pixels_per_meter or 1.0
            sx = (pts[:, 0] - offset_world[0]) * ppm - half_w
            sy = (pts[:, 1] - offset_world[1]) * ppm - half_h
            surf = target_surface
            surf_w, surf_h = target_surface.get_size()

        # Frustum cull off-screen points before generating blit destinations.
        mask = (sx > -sw) & (sx < surf_w) & (sy > -sh) & (sy < surf_h)
        if not np.any(mask):
            return
        sx = sx[mask].astype(np.int32, copy=False)
        sy = sy[mask].astype(np.int32, copy=False)

        # `Surface.blits([(src, dest), ...])` is a single C-level batch call.
        coords = np.stack((sx, sy), axis=1)
        surf.blits(
            [(sprite, (int(c[0]), int(c[1]))) for c in coords],
            doreturn=False,
        )

    def _draw_points(self, points, color, radius, max_points=None):
        arr = _to_xy_array(points)
        if arr is None:
            return
        if max_points is not None and arr.shape[0] > max_points:
            stride = max(1, int(np.ceil(arr.shape[0] / max_points)))
            arr = arr[::stride]
        self._draw_points_array(arr, color, radius)

    def _draw_line(self, points, color, width):
        arr = _to_xy_array(points)
        if arr is None or arr.shape[0] < 2:
            return
        sx, sy = self._world_to_screen_array(arr)
        # Cull whole line if its bounding box is fully off-screen.
        if (
            sx.max() < 0
            or sx.min() > self.width
            or sy.max() < 0
            or sy.min() > self.height
        ):
            return
        screen_points = np.stack((sx, sy), axis=1).astype(np.int32, copy=False)
        if screen_points.shape[0] >= 2:
            pygame.draw.lines(
                self.screen,
                color,
                False,
                [tuple(p) for p in screen_points],
                max(1, int(round(width))),
            )

    def _draw_trajectories(self, trajectories, color, radius):
        if not isinstance(trajectories, list):
            return
        # Concatenate all sub-trajectories into a single array so we get one
        # vectorized cull + one batched blit instead of one per trajectory.
        chunks = []
        for traj in trajectories:
            arr = _to_xy_array(traj)
            if arr is not None and arr.shape[0] > 0:
                chunks.append(arr)
        if not chunks:
            return
        combined = np.concatenate(chunks, axis=0)
        self._draw_points_array(combined, color, radius)

    def _draw_single_point(self, point_list, color, radius):
        arr = _to_xy_array(point_list)
        if arr is None or arr.shape[0] == 0:
            return
        # Use only the first point for "single point" overlays.
        self._draw_points_array(arr[:1], color, radius)

    def draw_car(self, pose, color):
        vertices = get_vertices(pose, CAR_LENGTH, CAR_WIDTH)
        points = [self.world_to_screen(v) for v in vertices]
        pygame.draw.polygon(self.screen, color, points)

    # ------------------------------------------------------------------
    # Static overlay layer
    #
    # Pre-rasterizes data that does not change between frames (global
    # waypoints, alt waypoints, track border points) into a single
    # world-space surface that gets re-scaled only when the user zooms.
    # ------------------------------------------------------------------

    def _build_static_layer_signature(self):
        overlay = self.overlay or {}
        colors = overlay.get("colors", {}) or {}

        global_waypoints = (
            overlay.get("global_waypoints")
            or overlay.get("waypoints_global")
            or overlay.get("waypoints")
        )
        wp_arr = _to_xy_array(global_waypoints)
        wp_alt_arr = _to_xy_array(overlay.get("waypoints_alternative"))
        tb_arr = _to_xy_array(overlay.get("track_border_points"))
        wp_color = _as_color(colors.get("waypoints"), DEFAULT_COLORS["waypoints"])
        wp_alt_color = DEFAULT_COLORS["waypoints_alternative"]
        tb_color = _as_color(
            colors.get("track_border"), DEFAULT_COLORS["track_border"]
        )

        signature = (
            _layer_fingerprint(wp_arr, wp_color),
            _layer_fingerprint(wp_alt_arr, wp_alt_color),
            _layer_fingerprint(tb_arr, tb_color),
        )
        layers = [
            (wp_arr, wp_color, 2),
            (wp_alt_arr, wp_alt_color, 2),
            (tb_arr, tb_color, 2),
        ]
        return signature, layers

    def _maybe_rebuild_static_layers(self):
        signature, layers = self._build_static_layer_signature()
        if signature == self._static_layers_signature and self._static_layers_surface is not None:
            return
        self._static_layers_signature = signature

        all_pts = [arr for arr, _, _ in layers if arr is not None and arr.shape[0] > 0]
        if not all_pts:
            self._static_layers_surface = None
            self._static_layers_scaled = None
            self._static_layers_last_scale = None
            self._static_layers_max_y = 0.0
            return

        combined = np.concatenate(all_pts, axis=0)
        min_xy = combined.min(axis=0) - 0.5
        max_xy = combined.max(axis=0) + 0.5
        ppm = STATIC_OVERLAY_PIXELS_PER_METER
        size_px = np.maximum(
            (max_xy - min_xy) * ppm,
            np.array([1.0, 1.0]),
        )
        # Adapt resolution if we'd otherwise blow past the per-axis cap.
        scale_factor = max(1.0, float(np.max(size_px) / STATIC_OVERLAY_MAX_DIM))
        if scale_factor > 1.0:
            ppm = ppm / scale_factor
            size_px = np.maximum(
                (max_xy - min_xy) * ppm,
                np.array([1.0, 1.0]),
            )
        width_px = int(np.ceil(size_px[0]))
        height_px = int(np.ceil(size_px[1]))
        surface = pygame.Surface((width_px, height_px), pygame.SRCALPHA)

        for arr, color, radius in layers:
            if arr is None or arr.shape[0] == 0:
                continue
            self._draw_points_array(
                arr,
                color,
                radius,
                target_surface=surface,
                offset_world=(float(min_xy[0]), float(min_xy[1])),
                pixels_per_meter=ppm,
            )

        self._static_layers_surface = pygame.transform.flip(
            surface.convert_alpha(), False, True
        )
        self._static_layers_origin = min_xy.astype(np.float32)
        self._static_layers_max_y = float(max_xy[1])
        self._static_layers_shape = (width_px, height_px)
        self._static_layers_resolution = 1.0 / ppm
        self._static_layers_scaled = None
        self._static_layers_last_scale = None

    def _draw_static_layers(self):
        self._maybe_rebuild_static_layers()
        if self._static_layers_surface is None:
            return
        scale = self.zoom_level * self._static_layers_resolution
        if scale != self._static_layers_last_scale:
            sw, sh = self._static_layers_shape
            scaled_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
            self._static_layers_scaled = pygame.transform.smoothscale(
                self._static_layers_surface, scaled_size
            )
            self._static_layers_last_scale = scale
        if self._static_layers_scaled is not None:
            nw = (
                float(self._static_layers_origin[0]),
                float(self._static_layers_max_y),
            )
            sx, sy = self.world_to_screen(nw)
            self.screen.blit(self._static_layers_scaled, (sx, sy))

    # ------------------------------------------------------------------
    # Map drawing
    # ------------------------------------------------------------------

    def _draw_map(self):
        if not self.show_map or self.map_surface is None:
            return
        scale = self.zoom_level * self.map_resolution
        if scale != self.last_map_scale:
            scaled_size = [max(1, int(s * scale)) for s in self.map_image_shape]
            self.scaled_map_surface = pygame.transform.smoothscale(
                self.map_surface, scaled_size
            )
            self.last_map_scale = scale

        if self.scaled_map_surface is not None:
            map_h_m = self.map_image_shape[1] * self.map_resolution
            map_nw = (
                float(self.map_origin[0]),
                float(self.map_origin[1] + map_h_m),
            )
            map_pos_on_screen = self.world_to_screen(map_nw)
            self.screen.blit(self.scaled_map_surface, map_pos_on_screen)

    # ------------------------------------------------------------------
    # Overlay drawing (mirrors WebRenderer/index.html `draw()`)
    # ------------------------------------------------------------------

    def _draw_overlay(self):
        overlay = self.overlay or {}
        colors = overlay.get("colors", {}) or {}

        # `track_border_lines` (a list of polylines, when available) is rendered
        # via `pygame.draw.lines` — much cheaper than dot-rasterizing each point.
        # We still pre-bake the dotted `track_border_points` layer in the static
        # surface as a fallback for older payloads that lack the line form.
        if self.show_track_borders:
            border_color = _as_color(
                colors.get("track_border"), DEFAULT_COLORS["track_border"]
            )
            border_lines = overlay.get("track_border_lines")
            if isinstance(border_lines, list) and len(border_lines) > 0:
                for border_line in border_lines:
                    self._draw_line(border_line, border_color, 2)

        # Highly dynamic overlays (per-frame cost dominated by these).
        self._draw_points(
            overlay.get("next_waypoints_alternative"),
            _as_color(
                colors.get("next_waypoints_alternative"),
                DEFAULT_COLORS["next_waypoints_alternative"],
            ),
            2,
        )
        self._draw_points(
            overlay.get("next_waypoints"),
            _as_color(colors.get("next_waypoints"), DEFAULT_COLORS["next_waypoints"]),
            3,
        )
        self._draw_points(
            overlay.get("lidar_border_points"),
            _as_color(colors.get("lidar"), DEFAULT_COLORS["lidar"]),
            3,
            max_points=LIDAR_MAX_POINTS,
        )
        self._draw_points(
            overlay.get("obstacles"),
            _as_color(colors.get("obstacles"), DEFAULT_COLORS["obstacles"]),
            3,
        )

        if self.show_position_history:
            self._draw_points(
                overlay.get("past_car_states_alternative"),
                _as_color(colors.get("history_alt"), DEFAULT_COLORS["history_alt"]),
                2,
                max_points=HISTORY_MAX_POINTS,
            )
            self._draw_points(
                overlay.get("past_car_states_gt"),
                _as_color(colors.get("history_gt"), DEFAULT_COLORS["history_gt"]),
                2,
                max_points=HISTORY_MAX_POINTS,
            )
            self._draw_points(
                overlay.get("past_car_states_prior"),
                _as_color(colors.get("history_prior"), DEFAULT_COLORS["history_prior"]),
                2,
                max_points=HISTORY_MAX_POINTS,
            )
            self._draw_points(
                overlay.get("past_car_states_prior_full"),
                _as_color(
                    colors.get("history_prior_full"),
                    DEFAULT_COLORS["history_prior_full"],
                ),
                2,
                max_points=HISTORY_MAX_POINTS,
            )

        # MPC rollouts and optimal trajectory.
        self._draw_trajectories(
            overlay.get("rollout_trajectory"),
            _as_color(colors.get("mppi"), DEFAULT_COLORS["mppi"]),
            2,
        )
        self._draw_trajectories(
            overlay.get("optimal_trajectory"),
            _as_color(colors.get("optimal"), DEFAULT_COLORS["optimal"]),
            2,
        )

        # Gap / target highlights.
        self._draw_single_point(
            overlay.get("largest_gap_middle_point"),
            _as_color(colors.get("gap"), DEFAULT_COLORS["gap"]),
            5,
        )
        self._draw_single_point(
            overlay.get("target_point"),
            _as_color(colors.get("target"), DEFAULT_COLORS["target"]),
            7,
        )

        # Steering direction arrow.
        steering = overlay.get("steering_arrow")
        if isinstance(steering, dict):
            start = steering.get("start")
            end = steering.get("end")
            if (
                isinstance(start, (list, tuple))
                and isinstance(end, (list, tuple))
                and len(start) >= 2
                and len(end) >= 2
            ):
                self._draw_line([start, end], DEFAULT_COLORS["steering_arrow"], 2)

        # Emergency slowdown lines + factor display.
        emergency = overlay.get("emergency_slowdown")
        if isinstance(emergency, dict):
            try:
                factor = float(emergency.get("speed_reduction_factor", 1.0))
            except (TypeError, ValueError):
                factor = 1.0
            factor = max(0.0, min(1.0, factor))
            red = int((1.0 - factor) * 255)
            green = int(factor * 255)
            dyn_color = (red, green, 0)
            self._draw_line(emergency.get("left_line"), dyn_color, 2)
            self._draw_line(emergency.get("right_line"), dyn_color, 2)
            self._draw_line(
                emergency.get("stop_line"), DEFAULT_COLORS["emergency_stop"], 2
            )
            display_pos = emergency.get("display_position")
            if (
                isinstance(display_pos, list)
                and len(display_pos) > 0
                and factor < 1.0
            ):
                p = display_pos[0]
                try:
                    sx, sy = self.world_to_screen((float(p[0]), float(p[1])))
                    text = self.small_font.render(
                        f"{factor:.2f}", True, dyn_color
                    )
                    self.screen.blit(text, (sx, sy))
                except (TypeError, ValueError, IndexError):
                    pass

    def _draw_cars(self):
        if self.poses is None:
            return
        for i, pose in enumerate(self.poses):
            color = (
                DEFAULT_COLORS["ego_car"]
                if i == self.ego_idx
                else DEFAULT_COLORS["opponent_car"]
            )
            self.draw_car(pose, color)

    # ------------------------------------------------------------------
    # Status bar and HUD
    # ------------------------------------------------------------------

    def _draw_status_bar(self):
        if self.poses is None or len(self.poses) == 0:
            return
        ego_pose = self.poses[min(self.ego_idx, len(self.poses) - 1)]
        ego_v = (
            float(self.vels[min(self.ego_idx, len(self.vels) - 1)])
            if self.vels is not None and len(self.vels) > 0
            else 0.0
        )
        motor_omega = None
        if self.car_states is not None and self.car_states.shape[1] > MOTOR_ANGULAR_VEL_IDX:
            motor_omega = float(
                self.car_states[min(self.ego_idx, len(self.car_states) - 1), MOTOR_ANGULAR_VEL_IDX]
            )
        cam_mode = "follow-car" if self.camera_follow_ego else "free"
        base_text = (
            f"Sim: {self.sim_time:.2f}s | Lap: {self.lap_count} | "
            f"x: {ego_pose[0]:.2f} y: {ego_pose[1]:.2f} yaw: {ego_pose[2]:.2f} v: {ego_v:.2f} | "
            f"cam: {cam_mode} | zoom: {self.zoom_level:.1f} | fps: {self._fps_ema:.0f}"
        )
        if motor_omega is not None and np.isfinite(motor_omega):
            motor_rpm = motor_omega * 60.0 / (2.0 * np.pi)
            motor_erpm = motor_erpm_from_omega(motor_omega)
            base_text += (
                f" | motor: {motor_omega:.0f} rad/s ({motor_rpm:.0f} RPM, {motor_erpm:.0f} ERPM)"
            )
        text_surface = self.font.render(base_text, True, DEFAULT_COLORS["text"])
        self.screen.blit(text_surface, (10, self.height - 26))

        if self.show_car_info:
            label_dict = (self.overlay or {}).get("label_dict") or {}
            if label_dict:
                items = list(label_dict.items())[:8]
                line = " | ".join(f"{k}: {v}" for k, v in items)
                surf = self.small_font.render(line, True, DEFAULT_COLORS["text"])
                self.screen.blit(surf, (10, self.height - 46))

        hint = (
            "M:map  H:history  B:borders  I:car-info  P:plots  "
            "SPACE/F:follow  scroll:zoom  drag:pan"
        )
        if self._debug_controls is not None and len(self._debug_controls) > self.ego_idx:
            c = self._debug_controls[self.ego_idx]
            hint += f"  |  steer: {float(c[0]):+.3f}  thr: {float(c[1]):+.3f}"
        hint_surf = self.small_font.render(hint, True, (200, 200, 200))
        self.screen.blit(hint_surf, (10, 6))

    # ------------------------------------------------------------------
    # Live control plots (angular / translational)
    # ------------------------------------------------------------------

    def _update_control_series(self):
        labels = (self.overlay or {}).get("label_dict") or {}
        if not labels:
            return
        angular_key = _resolve_control_key(labels, "angular_control")
        translational_key = _resolve_control_key(labels, "translational_control")
        t = float(self.sim_time)

        if angular_key is not None:
            v = _parse_numeric(labels[angular_key])
            if v is not None:
                self._control_series["angular_control"].append((t, v))
        if translational_key is not None:
            v = _parse_numeric(labels[translational_key])
            if v is not None:
                self._control_series["translational_control"].append((t, v))

        # Trim series to (window_s, max_samples) for both axes.
        min_t = max(0.0, t - CONTROL_PLOT_WINDOW_S)
        for series in self._control_series.values():
            while series and series[0][0] < min_t:
                series.popleft()
            while len(series) > CONTROL_PLOT_MAX_SAMPLES:
                series.popleft()

    def _draw_control_plots(self):
        if not self.show_control_plots:
            return
        if all(len(s) == 0 for s in self._control_series.values()):
            return

        panel_w, panel_h = self._plot_panel_size
        panel_w = min(panel_w, self.width - 24)
        panel_h = min(panel_h, max(120, self.height - 80))
        panel_x = self.width - panel_w - 12
        panel_y = self.height - panel_h - 36

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((*DEFAULT_COLORS["panel_bg"], 235))
        pygame.draw.rect(
            panel,
            (*DEFAULT_COLORS["panel_border"], 90),
            panel.get_rect(),
            width=1,
            border_radius=6,
        )

        pad_left, pad_right, pad_top, pad_bottom = 54, 14, 14, 22
        plot_w = max(1, panel_w - pad_left - pad_right)
        plot_h_total = max(1, panel_h - pad_top - pad_bottom)
        subplot_gap = 12
        keys = [
            ("angular_control", DEFAULT_COLORS["plot_angular"], "angular_control"),
            (
                "translational_control",
                DEFAULT_COLORS["plot_translational"],
                "translational_control",
            ),
        ]
        active_keys = [k for k in keys if len(self._control_series[k[0]]) > 0]
        if not active_keys:
            return
        n = len(active_keys)
        sub_h = (plot_h_total - subplot_gap * (n - 1)) / n

        now_t = float(self.sim_time)
        min_t = max(0.0, now_t - CONTROL_PLOT_WINDOW_S)

        for idx, (key, color, title) in enumerate(active_keys):
            y0 = pad_top + idx * (sub_h + subplot_gap)
            series = self._control_series[key]
            window_pts = [(t, v) for t, v in series if t >= min_t]
            if not window_pts:
                continue
            ys = [v for _, v in window_pts]
            y_min = min(ys)
            y_max = max(ys)
            span = max(1e-3, y_max - y_min)
            y_pad = 0.14 * span
            plot_min = y_min - y_pad
            plot_max = y_max + y_pad
            plot_span = max(1e-3, plot_max - plot_min)

            # Gridlines + axis frame.
            for j in range(4):
                gy = y0 + (j / 3) * sub_h
                pygame.draw.line(
                    panel,
                    (255, 255, 255, 50),
                    (pad_left, gy),
                    (pad_left + plot_w, gy),
                    1,
                )
            pygame.draw.line(
                panel,
                (255, 255, 255, 90),
                (pad_left, y0),
                (pad_left, y0 + sub_h),
                1,
            )
            pygame.draw.line(
                panel,
                (255, 255, 255, 90),
                (pad_left, y0 + sub_h),
                (pad_left + plot_w, y0 + sub_h),
                1,
            )

            # Plot trace.
            screen_pts = []
            for t, v in window_pts:
                tx = (t - min_t) / CONTROL_PLOT_WINDOW_S
                ty = (v - plot_min) / plot_span
                px = pad_left + tx * plot_w
                py = y0 + (1.0 - ty) * sub_h
                screen_pts.append((px, py))
            if len(screen_pts) >= 2:
                pygame.draw.lines(panel, color, False, screen_pts, 2)

            # Y-axis labels and series title.
            max_label = self.small_font.render(
                f"{plot_max:.3f}", True, (220, 220, 220)
            )
            min_label = self.small_font.render(
                f"{plot_min:.3f}", True, (220, 220, 220)
            )
            panel.blit(max_label, (4, int(y0) + 2))
            panel.blit(min_label, (4, int(y0 + sub_h) - 14))
            title_surf = self.small_font.render(title, True, color)
            panel.blit(title_surf, (pad_left + 6, int(y0) + 2))

        # X-axis time labels.
        time_left = self.small_font.render(f"{min_t:.1f}s", True, (220, 220, 220))
        time_right = self.small_font.render(f"{now_t:.1f}s", True, (220, 220, 220))
        panel.blit(time_left, (pad_left, panel_h - 18))
        panel.blit(
            time_right,
            (pad_left + plot_w - time_right.get_width(), panel_h - 18),
        )

        self.screen.blit(panel, (panel_x, panel_y))

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def render(self):
        self.handle_events()
        self._update_camera_follow()

        # Wall-time throttle: skip the actual draw if we already drew a frame
        # within the last 1/PYGAME_RENDER_FPS seconds. State has already been
        # applied via `update_obs`, so the next draw is automatically fresh.
        now_s = time.monotonic()
        dt = now_s - self._last_draw_wall_s if self._last_draw_wall_s else None
        if (
            self._min_frame_dt > 0.0
            and dt is not None
            and dt < self._min_frame_dt
        ):
            return
        if dt is not None and dt > 0.0:
            inst_fps = 1.0 / dt
            self._fps_ema = (
                inst_fps if self._fps_ema == 0.0 else 0.85 * self._fps_ema + 0.15 * inst_fps
            )
        self._last_draw_wall_s = now_s

        self.screen.fill(DEFAULT_COLORS["background"])
        self._draw_map()
        self._draw_static_layers()
        self._draw_overlay()
        self._draw_cars()
        self._draw_status_bar()
        self._draw_control_plots()

        pygame.display.flip()

    def close(self):
        if self._motor_audio is not None:
            self._motor_audio.stop()
            self._motor_audio = None
        pygame.quit()

    def flip(self):
        self.render()

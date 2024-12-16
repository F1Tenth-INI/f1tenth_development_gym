# WaypointsEditor.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
from matplotlib.widgets import Slider
import datetime
import matplotlib
import threading

from RaceTuner.FileSynchronizer import FileSynchronizer, upload_to_remote_via_sftp, download_map_files_via_sftp
from RaceTuner.HoverMarker import HoverMarker
from RaceTuner.MapHelper import MapConfig
from RaceTuner.SocketWaypointsEditor import SocketWatpointEditor
from RaceTuner.WaypointsHistoryManager import WaypointHistoryManager
from utilities.Settings import Settings

from TunerSettings import (
    MAP_NAME,
    LOCAL_MAP_DIR,
    REMOTE_MAP_DIR,
    REMOTE_AT_LOCAL_DIR,
    USE_REMOTE_FILES,
    MAP_LIMITS_X,
    MAP_LIMITS_Y,
)

from utilities.waypoint_utils import get_speed_scaling

from matplotlib.gridspec import GridSpec

class WaypointsModifier:
    def __init__(self, waypoint_manager):
        self.waypoint_manager = waypoint_manager
        self.cs_x = None
        self.cs_y = None
        self.dense_t = None
        self.dense_x = None
        self.dense_y = None

    def recalculate_splines(self, full_init=False):
        wm = self.waypoint_manager
        self.cs_x = CubicSpline(wm.t, wm.x)
        self.cs_y = CubicSpline(wm.t, wm.y)
        self.dense_t = np.linspace(wm.t[0], wm.t[-1], 500)
        self.dense_x = self.cs_x(self.dense_t)
        self.dense_y = self.cs_y(self.dense_t)
        if full_init:
            wm.history_manager.save_waypoint_state(wm.x, wm.y, wm.vx)

    def apply_weighted_adjustment_2d(self, dx, dy, drag_index, scale):
        wm = self.waypoint_manager
        n = len(wm.x)
        for i in range(n):
            d = min(abs(i - drag_index), abs(i - drag_index + n), abs(i - drag_index - n))
            w = np.exp(-0.5 * (d / scale) ** 2)
            wm.x[i] += dx * w
            wm.y[i] += dy * w

    def apply_weighted_adjustment_1d(self, dv, drag_index, scale):
        wm = self.waypoint_manager
        if wm.vx is None:
            return
        n = len(wm.vx)
        for i in range(n):
            d = min(abs(i - drag_index), abs(i - drag_index + n), abs(i - drag_index - n))
            w = np.exp(-0.5 * (d / scale) ** 2)
            wm.vx[i] += dv * w


class WaypointDataManager:
    def __init__(self, map_name, path_to_maps, waypoints_new_file_name=None):
        self.map_name = map_name
        self.path_to_maps = path_to_maps
        self.waypoints_new_file_name = waypoints_new_file_name
        self.path_to_waypoints = os.path.join(path_to_maps, map_name, map_name + "_wp.csv")
        self.path_to_waypoints_reverse = os.path.join(path_to_maps, map_name, map_name + "_wp_reverse.csv")
        self.original_data = None
        self.x = None
        self.y = None
        self.vx = None
        self.vx_original = None  # Store original speeds
        self.scale = 1.0  # Scaling factor
        self.initial_x = None
        self.initial_y = None
        self.initial_vx = None
        self.t = None
        self.dense_t = None
        self.dense_x = None
        self.dense_y = None
        self.initial_load_done = False  # Tracks if initial load has occurred

        # A lock to ensure thread-safe updates if needed
        self.lock = threading.Lock()

        # Initialize a separate manager to handle history and undo operations
        self.history_manager = WaypointHistoryManager()

        # Initialize WaypointsModifier
        self.modifier = WaypointsModifier(self)

    def load_waypoints_from_file(self):
        """Load waypoints from the waypoint CSV file."""
        with self.lock:
            # Load waypoints from local (or updated) file
            self.original_data = pd.read_csv(self.path_to_waypoints, comment="#")
            self.x = self.original_data['x_m'].to_numpy()
            self.y = self.original_data['y_m'].to_numpy()

            # If this is the initial load, store initial values
            # so we can reference them later. Avoid overwriting them on subsequent loads.
            if not self.initial_load_done:
                self.initial_x = self.x.copy()
                self.initial_y = self.y.copy()

            if 'vx_mps' in self.original_data.columns:
                self.vx_original = self.original_data['vx_mps'].to_numpy()
                # get_speed_scaling relies on files that must be present locally.
                # If remote mode is on, they have been downloaded above.
                self.scale = get_speed_scaling(len(self.x), os.path.join(self.path_to_maps, self.map_name), self.map_name, Settings)
                self.vx = self.vx_original * self.scale  # Apply scaling

                if not self.initial_load_done:
                    self.initial_vx = self.vx.copy()

            self.t = np.arange(len(self.x))
            self.modifier.recalculate_splines(full_init=(not self.initial_load_done))
            self.initial_load_done = True  # Mark that initial load is complete now

    def save_waypoints_to_file(self, message_box_update_callback):
        file_path = self.waypoints_new_file_name if self.waypoints_new_file_name else self.path_to_waypoints
        data = pd.DataFrame({"x_m": self.x, "y_m": self.y})
        for col in self.original_data.columns:
            if col not in data.columns:
                if col == 'vx_mps' and self.vx is not None:
                    data[col] = self.vx / self.scale  # Unscale before saving
                else:
                    data[col] = self.original_data[col]
        data = data[self.original_data.columns]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_comment = f"# Updated waypoints saved on {timestamp}\n"
        with open(file_path, "w") as f:
            f.write(header_comment)
            data.to_csv(f, index=False, float_format="%.6f")
        message_box_update_callback(f"Waypoints saved to {file_path} at {timestamp}")

        if USE_REMOTE_FILES:
            upload_to_remote_via_sftp(file_path, os.path.join(REMOTE_MAP_DIR, MAP_NAME, MAP_NAME + "_wp.csv"))
            upload_to_remote_via_sftp(file_path, os.path.join(REMOTE_MAP_DIR, MAP_NAME, MAP_NAME + "_wp_reverse.csv"))  # FIXME: IT should be either or

    def create_backup_if_needed(self):
        backup_path = self.path_to_waypoints.replace(".csv", "_backup.csv")
        if not os.path.exists(backup_path):
            with open(self.path_to_waypoints, 'r') as original_file:
                with open(backup_path, 'w') as backup_file:
                    backup_file.write(original_file.read())
            upload_to_remote_via_sftp(backup_path, os.path.join(REMOTE_MAP_DIR, MAP_NAME, MAP_NAME + "_wp_backup.csv"))


        backup_path = self.path_to_waypoints_reverse.replace(".csv", "_backup_reverse.csv")
        if not os.path.exists(backup_path):
            with open(self.path_to_waypoints_reverse, 'r') as original_file:
                with open(backup_path, 'w') as backup_file:
                    backup_file.write(original_file.read())
            upload_to_remote_via_sftp(backup_path, os.path.join(REMOTE_MAP_DIR, MAP_NAME, MAP_NAME + "_wp_backup_reverse.csv"))

    def undo(self):
        state = self.history_manager.undo()
        if state is not None:
            # State returned can be (x, y) or (x, y, vx)
            self.x = state[0].copy()
            self.y = state[1].copy()
            if len(state) > 2:
                self.vx = state[2].copy()
            self.modifier.recalculate_splines()
            return True
        return False


class WaypointEditorUI:
    def __init__(self, waypoint_manager, map_config, socket_client, initial_scale=20.0, update_frequency=5.0, reload_event=None):
        self.waypoint_manager = waypoint_manager
        self.map_config = map_config
        self.scale = initial_scale
        self.socket_client = socket_client
        self.update_frequency = update_frequency
        self.reload_event = reload_event  # Added to handle reload signaling

        if sys.platform == 'darwin':
            matplotlib.use('MacOSX')
        plt.rcParams.update({'font.size': 15})

        self.fig = plt.figure(figsize=(16, 10))
        self.divider_line_dragging = False
        self.divider_y = 0.5  # Initial position of the divider
        if self.waypoint_manager.vx is not None:
            self.gs = GridSpec(2, 1, height_ratios=[1, 1], figure=self.fig)
            self.ax = self.fig.add_subplot(self.gs[0, 0])
            self.ax2 = self.fig.add_subplot(self.gs[1, 0])
            plt.subplots_adjust(hspace=0.3)

            # Add draggable line
            self.divider_line = plt.Line2D(
                [0, 1], [self.divider_y, self.divider_y], transform=self.fig.transFigure, color="gray", lw=2, picker=True
            )
            self.fig.add_artist(self.divider_line)

        else:
            self.fig, self.ax = plt.subplots(figsize=(16, 10))
            self.ax2 = None

            self.divider_line = None


        self.fig.canvas.manager.set_window_title("INIvincible Waypoints Editor")
        self.text_box = None
        self.sigma_slider = None
        self.dragging = False
        self.drag_index = None
        self.dragging_vx = False
        self.drag_index_vx = None
        self.image_loaded = False

        # Dynamic elements
        self.car_marker = None
        self.car_speed_marker = None
        self.background_main = None
        self.background_speed = None

        self.update_interval = 1000 / self.update_frequency  # in milliseconds

        # Initialize car state
        self.car_x = None
        self.car_y = None
        self.car_v = None
        self.car_wpt_idx = None

        self.hover_marker = HoverMarker(self.ax, self.ax2, self.waypoint_manager)



    def load_image_background(self, grayscale=True):
        img = self.map_config.load_map_image(grayscale=grayscale)
        yaml_data = self.map_config.load_map_config()
        resolution = yaml_data['resolution']
        origin = yaml_data['origin']
        extent = [
            origin[0],
            origin[0] + img.shape[1] * resolution,
            origin[1],
            origin[1] + img.shape[0] * resolution
        ]
        self.ax.imshow(img, extent=extent, aspect='equal', cmap='gray' if grayscale else None)  # Set aspect to 'equal'
        self.image_loaded = True

    def setup_static_plot(self):
        # Plot static elements
        if self.image_loaded:
            self.load_image_background()
        wm = self.waypoint_manager
        self.ax.plot(wm.initial_x, wm.initial_y, color="blue", linestyle="--", label="Initial Raceline")
        self.ax.plot(wm.x, wm.y, color="green", linestyle="-")
        self.ax.scatter(wm.x, wm.y, color="red", label="Target Raceline")

        # Set labels and grid for main axis
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid()

        if self.ax2 and wm.vx is not None:
            self.ax2.plot(wm.t, wm.initial_vx, color="blue", linestyle="--", label="Initial Speed")
            self.ax2.plot(wm.t, wm.vx, color="green", linestyle="-")
            self.ax2.scatter(wm.t, wm.vx, color="red", label="Target Speed")
            self.ax2.set_xlabel("Waypoint Index")
            self.ax2.set_ylabel("Speed vx (m/s)")
            self.ax2.grid()

            if self.divider_line:
                # Update the y-coordinates of the line
                self.divider_line.set_ydata([self.divider_y, self.divider_y])

        # Combine legends from both subplots

        _, labels = self.ax.get_legend_handles_labels()
        if self.ax2:
            labels[0] = 'Initial Raceline & Speed'
            labels[1] = "Target Raceline & Speed"

        self.ax.legend(labels, loc="upper right", bbox_to_anchor=(2.0, 1), frameon=False)

        self.ax.set_xlim(MAP_LIMITS_X)
        self.ax.set_ylim(MAP_LIMITS_Y)

    def setup_dynamic_artists(self):
        # Initialize dynamic artists for car position
        if self.car_x is not None and self.car_y is not None:
            self.car_marker, = self.ax.plot(self.car_x, self.car_y, 'o', color='orange', markersize=12,
                                            label="Car Position")
            if not any(label.get_text() == "Car Position" for label in self.ax.get_legend().get_texts()):
                self.ax.legend(loc='upper right')
        else:
            self.car_marker, = self.ax.plot([], [], 'o', color='orange', markersize=12, label="Car Position")

        # Initialize dynamic artists for car speed
        if self.ax2 and self.car_v is not None and self.car_wpt_idx is not None:
            self.car_speed_marker, = self.ax2.plot(self.car_wpt_idx, self.car_v, 'o', color='orange', markersize=12,
                                                   label="Car Speed")
            if not any(label.get_text() == "Car Speed" for label in self.ax2.get_legend().get_texts()):
                self.ax2.legend(loc='upper right')
        elif self.ax2:
            self.car_speed_marker, = self.ax2.plot([], [], 'o', color='orange', markersize=12, label="Car Speed")

    def capture_background(self):
        # Capture background for main axis
        self.fig.canvas.draw()
        self.background_main = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # Capture background for speed axis
        if self.ax2:
            self.background_speed = self.fig.canvas.copy_from_bbox(self.ax2.bbox)

    def redraw_static_elements(self):
        # Clear the axes
        self.ax.clear()
        if self.ax2:
            self.ax2.clear()

        # Redraw static and dynamic plot elements
        self.setup_static_plot()
        self.setup_dynamic_artists()

        # Recreate hover markers after clearing the axes
        self.hover_marker.reconnect_markers()

        # Redraw the canvas and capture background
        self.fig.canvas.draw()
        self.capture_background()

    def redraw_plot(self):
        # Redraw everything and recapture backgrounds
        self.redraw_static_elements()
        # Update text box
        self.update_text_box(f"Waypoints updated. Current scale: {self.scale:.1f}")

    def update_text_box(self, message):
        if self.text_box:
            self.text_box.clear()
            self.text_box.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
            self.text_box.set_xticks([])
            self.text_box.set_yticks([])
            self.text_box.set_frame_on(False)
        else:
            # Create text box if it doesn't exist
            self.text_box = self.fig.add_axes([0.1, 0.05, 0.8, 0.05])
            self.text_box.axis('off')
            self.text_box.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)

    def on_press(self, event):

        if self.divider_line.contains(event)[0]:
            self.divider_line_dragging = True

        # Detect if Ctrl or Cmd is pressed
        if event.key in ['control', 'ctrl', 'command', 'cmd']:
            self.hover_marker.plant_marker()
            return  # Early exit to prevent other click handling

        if event.inaxes == self.ax:
            wm = self.waypoint_manager
            for i, (px, py) in enumerate(zip(wm.x, wm.y)):
                if np.hypot(event.xdata - px, event.ydata - py) < 0.3:
                    self.dragging = True
                    self.drag_index = i
                    break
        elif self.ax2 and event.inaxes == self.ax2 and self.waypoint_manager.vx is not None:
            wm = self.waypoint_manager
            for i, (tx, ty) in enumerate(zip(wm.t, wm.vx)):
                if abs(event.xdata - tx) < 0.5 and abs(event.ydata - ty) < 0.3:
                    self.dragging_vx = True
                    self.drag_index_vx = i
                    break

    def on_release(self, event):

        self.divider_line_dragging = False

        if self.dragging:
            self.dragging = False
            self.drag_index = None
            self.waypoint_manager.modifier.recalculate_splines()
            # After position changes are finalized, record the new state
            self.waypoint_manager.history_manager.save_waypoint_state(
                self.waypoint_manager.x,
                self.waypoint_manager.y,
                self.waypoint_manager.vx
            )
            self.redraw_plot()
        if self.dragging_vx:
            self.dragging_vx = False
            self.drag_index_vx = None
            self.waypoint_manager.history_manager.save_waypoint_state(
                self.waypoint_manager.x,
                self.waypoint_manager.y,
                self.waypoint_manager.vx
            )
            self.redraw_plot()

    def on_motion(self, event):

        if self.divider_line_dragging and event.y is not None:
            # Normalize y position to figure coordinates
            self.divider_y = event.y / self.fig.get_size_inches()[1] / self.fig.dpi
            self.resize_subplots()

        if self.dragging and self.drag_index is not None and event.inaxes == self.ax:
            wm = self.waypoint_manager
            dx = event.xdata - wm.x[self.drag_index]
            dy = event.ydata - wm.y[self.drag_index]
            wm.modifier.apply_weighted_adjustment_2d(dx, dy, self.drag_index, self.scale)
            # Update static plot and recapture background
            self.redraw_static_elements()
        if self.dragging_vx and self.drag_index_vx is not None and event.inaxes == self.ax2:
            wm = self.waypoint_manager
            dv = event.ydata - wm.vx[self.drag_index_vx]
            wm.modifier.apply_weighted_adjustment_1d(dv, self.drag_index_vx, self.scale)
            # Update static plot and recapture background
            self.redraw_static_elements()

    def resize_subplots(self):
        # Update subplot heights based on the divider's position
        bottom_ratio = self.divider_y
        top_ratio = 1 - bottom_ratio
        self.gs.set_height_ratios([top_ratio, bottom_ratio])
        self.fig.subplots_adjust(hspace=0.3)  # Adjust spacing if needed
        self.fig.canvas.draw_idle()
        self.redraw_plot()
        self.capture_background()

    def update_sigma(self, val):
        self.scale = val
        self.update_text_box(f"Scale updated to: {self.scale:.1f}")
        self.redraw_plot()

    def key_press_handler(self, event):
        key = event.key.lower() if event.key else ''
        wm = self.waypoint_manager
        if key in ["ctrl+s", "cmd+s"]:
            wm.save_waypoints_to_file(self.update_text_box)
            # After saving, refresh the plot to reflect any changes
            self.redraw_plot()
        elif key in ["ctrl+z", "cmd+z"]:
            if wm.undo():
                self.update_text_box("Undo successful.")
                self.redraw_plot()
            else:
                self.update_text_box("No more undo steps available.")
        # Check for "Ctrl+C" or "Cmd+C" to erase marker
        elif key in ["ctrl+c", "cmd+c"]:
            self.hover_marker.erase_marker()
        elif key == 'up':
            self.adjust_all_speeds(delta=0.1)
        elif key == 'down':
            self.adjust_all_speeds(delta=-0.1)
        elif key == 'r':
            # Check if Shift modifier is held
            if event.key == 'r' and event.key not in ["ctrl+r", "cmd+r"]:
                self.reset_xy()
        elif key in ["ctrl+r", "cmd+r"]:
            self.reset_vx()

    def adjust_all_speeds(self, delta):
        wm = self.waypoint_manager
        with wm.lock:
            # Save current state for undo
            wm.history_manager.save_waypoint_state(wm.x, wm.y, wm.vx)

            # Adjust speeds
            if wm.vx is not None:
                wm.vx += delta
                # Optional: Clamp speeds to a minimum value (e.g., 0 m/s)
                wm.vx = np.maximum(wm.vx, 0.0)

                # Recalculate any dependent variables if necessary
                wm.modifier.recalculate_splines()

                # Update the speed plot
                self.redraw_plot()

                # Provide feedback to the user
                action = "Increased" if delta > 0 else "Decreased"
                self.update_text_box(f"{action} all waypoint speeds by {abs(delta):.1f} m/s.")

    def reset_xy(self):
        wm = self.waypoint_manager
        with wm.lock:
            # Save current state for undo
            wm.history_manager.save_waypoint_state(wm.x, wm.y, wm.vx)

            # Reset x and y to initial values
            wm.x = wm.initial_x.copy()
            wm.y = wm.initial_y.copy()

            # Recalculate splines
            wm.modifier.recalculate_splines()

            # Update the plot
            self.redraw_plot()

            # Provide feedback to the user
            self.update_text_box("Reset all waypoint X and Y coordinates to initial values.")


    def reset_vx(self):
        wm = self.waypoint_manager
        with wm.lock:
            # Check if vx exists
            if wm.vx_original is None:
                self.update_text_box("No speed data (vx) to reset.")
                return

            # Save current state for undo
            wm.history_manager.save_waypoint_state(wm.x, wm.y, wm.vx)

            # Reset vx to initial values and apply scaling
            wm.vx = wm.vx_original.copy() * wm.scale

            # Recalculate splines if necessary
            wm.modifier.recalculate_splines()

            # Update the plot
            self.redraw_plot()

            # Provide feedback to the user
            self.update_text_box("Reset all waypoint speeds (vx) to initial values.")


    def setup_ui_elements(self):
        plt.subplots_adjust(bottom=0.25)
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.sigma_slider = Slider(ax_slider, "Scale", 1.0, 50.0, valinit=self.scale, valstep=0.1)
        self.sigma_slider.on_changed(self.update_sigma)
        self.text_box = plt.axes([0.1, 0.05, 0.8, 0.05])
        self.update_text_box("Drag waypoints to change, CMD+S or Ctrl+S to save, CMD+Z or Ctrl+Z to undo.")

    def connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_handler)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

    def on_resize(self, event):
        # On resize, redraw static elements and recapture background
        self.redraw_plot()

    def start_periodic_update(self):
        # Start a timer to update the car position based on the update_frequency
        self.timer = self.fig.canvas.new_timer(interval=self.update_interval)
        self.timer.add_callback(self.periodic_update)
        self.timer.start()

    def periodic_update(self):
        # Fetch the latest car state from the socket server
        # car_state = self.socket_client.get_car_state()
        # if car_state:
        #     self.car_x = car_state.get('car_x')
        #     self.car_y = car_state.get('car_y')
        #     self.car_v = car_state.get('car_v')
        #     self.car_wpt_idx = car_state.get('idx_global') * Settings.DECREASE_RESOLUTION_FACTOR

        # Update dynamic artists if they exist, else create them
        if self.car_marker is None:
            self.car_marker, = self.ax.plot([], [], 'o', color='orange', markersize=12, label="Car Position")
            # Avoid adding duplicate legend entries
            if not any(label.get_text() == "Car Position" for label in self.ax.get_legend().get_texts()):
                self.ax.legend(loc='upper right')
        if self.ax2 and self.car_speed_marker is None:
            self.car_speed_marker, = self.ax2.plot([], [], 'o', color='orange', markersize=12, label="Car Speed")
            if not any(label.get_text() == "Car Speed" for label in self.ax2.get_legend().get_texts()):
                self.ax2.legend(loc='upper right')

        # Update car position
        if self.car_marker is not None and self.car_x is not None and self.car_y is not None:
            self.car_marker.set_data([self.car_x], [self.car_y])

        # Update car speed
        if self.ax2 and self.car_speed_marker is not None and self.car_v is not None and self.car_wpt_idx is not None:
            self.car_speed_marker.set_data([self.car_wpt_idx], [self.car_v])

        # Efficiently redraw only the dynamic elements
        self.fig.canvas.restore_region(self.background_main)
        self.ax.draw_artist(self.car_marker)
        # Draw hover marker if visible
        self.hover_marker.draw_markers()
        self.fig.canvas.blit(self.ax.bbox)

        if self.ax2:
            self.fig.canvas.restore_region(self.background_speed)
            self.ax2.draw_artist(self.car_speed_marker)
            # Draw hover speed marker if visible
            self.hover_marker.draw_markers()
            self.fig.canvas.blit(self.ax2.bbox)

        # Check if a reload has been signaled
        if self.reload_event and self.reload_event.is_set():
            print("Cleared_event")
            self.redraw_plot()
            self.reload_event.clear()  # Reset the event

    def run_blitting(self):
        # Initial draw and capture background
        self.redraw_plot()
        self.capture_background()

    def run(self):
        self.setup_ui_elements()
        self.setup_static_plot()
        self.setup_dynamic_artists()
        self.run_blitting()
        self.connect_events()
        self.hover_marker.connect(self.fig.canvas)
        self.start_periodic_update()
        plt.show()


class WaypointsEditorApp:
    def __init__(self, waypoints_new_file_name=None, scale_initial=20.0, update_frequency=5.0):
        if USE_REMOTE_FILES:
            path_to_maps = REMOTE_AT_LOCAL_DIR
            download_map_files_via_sftp(
                map_name=MAP_NAME,
                remote_dir=REMOTE_MAP_DIR,
                local_dir=path_to_maps,
                mode='initial'
            )
        else:
            path_to_maps = LOCAL_MAP_DIR

        self.map_config = MapConfig(MAP_NAME, path_to_maps)
        self.waypoint_manager = WaypointDataManager(MAP_NAME, path_to_maps, waypoints_new_file_name)
        self.socket_client = SocketWatpointEditor()
        self.running = True  # If needed to stop threads gracefully later

        # Event to signal UI to reload
        self.reload_event = threading.Event()

        # Initialize FileSynchronizer
        self.file_synchronizer = FileSynchronizer(self.waypoint_manager, self.reload_event, interval=5)

    def run(self):
        try:
            # Continue with the GUI
            self.waypoint_manager.create_backup_if_needed()
            self.waypoint_manager.load_waypoints_from_file()

            # Start FileSynchronizer thread
            if USE_REMOTE_FILES:
                self.file_synchronizer.start()

            # Start the UI
            self.ui = WaypointEditorUI(
                self.waypoint_manager,
                self.map_config,
                self.socket_client,
                initial_scale=20.0,  # You can set this to waypoint_manager.scale if desired
                update_frequency=20.0,  # Default frequency
                reload_event=self.reload_event  # Pass the event to the UI
            )
            self.ui.load_image_background()
            self.ui.run()
        except KeyboardInterrupt:
            print("Shutting down WaypointsEditorApp.")
        finally:
            self.running = False
            self.file_synchronizer.stop()
            self.socket_client.close()

if __name__ == "__main__":
    app = WaypointsEditorApp()
    app.run()

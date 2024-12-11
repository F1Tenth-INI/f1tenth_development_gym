import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
from matplotlib.image import imread
from matplotlib.widgets import Slider
from copy import deepcopy
import datetime
import matplotlib
import yaml


class MapConfig:
    """Handles reading map image, YAML configuration, and ensuring coordinate alignment."""
    def __init__(self, map_name, path_to_maps):
        self.map_name = map_name
        self.path_to_maps = path_to_maps
        self.path_to_map_png = os.path.join(path_to_maps, map_name, map_name + ".png")
        self.path_to_map_config = os.path.join(path_to_maps, map_name, map_name + ".yaml")

    def load_map_image(self, grayscale=True):
        # By converting to grayscale when needed, we can get consistent visuals.
        img = imread(self.path_to_map_png)
        if grayscale and img.ndim == 3:
            img = np.dot(img[..., :3], [0.2989, 0.587, 0.114])
        return img

    def load_map_config(self):
        # The map config provides resolution and origin, crucial for correct scaling.
        with open(self.path_to_map_config, 'r') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data


class WaypointDataManager:
    """Manages loading, storing, and manipulating waypoints and their splines."""
    def __init__(self, map_name, path_to_maps, waypoints_new_file_name=None):
        self.map_name = map_name
        self.path_to_maps = path_to_maps
        self.waypoints_new_file_name = waypoints_new_file_name

        self.path_to_waypoints = os.path.join(path_to_maps, map_name, map_name + "_wp.csv")
        self.original_data = None

        # Arrays to store waypoint coordinates
        self.x = None
        self.y = None
        self.initial_x = None
        self.initial_y = None

        # Parameters for spline interpolation
        self.t = None
        self.cs_x = None
        self.cs_y = None
        self.dense_t = None
        self.dense_x = None
        self.dense_y = None

        # Undo/Redo stack for safe editing
        self.waypoint_history = []

    def load_waypoints(self):
        # After load, we prepare splines for smooth editing.
        self.original_data = pd.read_csv(self.path_to_waypoints, comment="#")
        self.x = self.original_data['x_m'].to_numpy()
        self.y = self.original_data['y_m'].to_numpy()
        self.initial_x = self.x.copy()
        self.initial_y = self.y.copy()

        self.t = np.arange(len(self.x))
        self._recalculate_splines(full_init=True)

    def _recalculate_splines(self, full_init=False):
        # Recomputing splines after modifications ensures updated curve continuity.
        self.cs_x = CubicSpline(self.t, self.x)
        self.cs_y = CubicSpline(self.t, self.y)
        self.dense_t = np.linspace(self.t[0], self.t[-1], 500)
        self.dense_x = self.cs_x(self.dense_t)
        self.dense_y = self.cs_y(self.dense_t)
        if full_init:
            self._save_waypoint_state()

    def apply_weighted_adjustment(self, dx, dy, drag_index, scale):
        # Gaussian-weighted adjustments propagate changes smoothly to neighbors.
        n = len(self.x)
        for i in range(n):
            d = min(abs(i - drag_index), abs(i - drag_index + n), abs(i - drag_index - n))
            weight = np.exp(-0.5 * (d / scale) ** 2)
            self.x[i] += dx * weight
            self.y[i] += dy * weight

    def _save_waypoint_state(self):
        # Storing states for undo operations.
        if self.waypoint_history:
            last_state = self.waypoint_history[-1]
            if np.array_equal(last_state[0], self.x) and np.array_equal(last_state[1], self.y):
                return
        self.waypoint_history.append((self.x.copy(), self.y.copy()))
        if len(self.waypoint_history) > 10:
            self.waypoint_history.pop(0)

    def undo(self):
        # Reverts to a previous waypoint state if available.
        if len(self.waypoint_history) > 1:
            self.waypoint_history.pop()
            self.x, self.y = deepcopy(self.waypoint_history[-1])
            self._recalculate_splines()
            return True
        return False

    def save_waypoints_to_file(self, message_box_update_callback):
        # Saving preserves the edits. Retains original columns and logs a timestamp.
        file_path = self.waypoints_new_file_name if self.waypoints_new_file_name else self.path_to_waypoints
        data = pd.DataFrame({"x_m": self.x, "y_m": self.y})
        # Retain original columns order
        for col in self.original_data.columns:
            if col not in data.columns:
                data[col] = self.original_data[col]
        data = data[self.original_data.columns]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_comment = f"# Updated waypoints saved on {timestamp}\n"
        with open(file_path, "w") as f:
            f.write(header_comment)
            f.write("# " + ", ".join(data.columns) + "\n")
            data.to_csv(f, index=False, float_format="%.6f", header=False)

        message_box_update_callback(f"Waypoints saved to {file_path} at {timestamp}")

    def create_backup_if_needed(self):
        # Create a backup once for safety before modifications.
        backup_path = self.path_to_waypoints.replace(".csv", "_backup.csv")
        if not os.path.exists(backup_path):
            with open(self.path_to_waypoints, 'r') as original_file:
                with open(backup_path, 'w') as backup_file:
                    backup_file.write(original_file.read())


class WaypointEditorUI:
    """Manages all interactive UI elements: figure, axes, sliders, event bindings, and displays."""
    def __init__(self, waypoint_manager, map_config, initial_scale=20.0):
        self.waypoint_manager = waypoint_manager
        self.map_config = map_config
        self.scale = initial_scale

        # Figure and Axes Setup
        if sys.platform == 'darwin':
            matplotlib.use('MacOSX')
        plt.rcParams.update({'font.size': 15})
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title("INIvincible Waypoints Editor")

        # Slider and text box for feedback
        self.text_box = None
        self.sigma_slider = None

        # Interaction states
        self.dragging = False
        self.drag_index = None
        self.image_loaded = False

    def load_image_background(self, grayscale=True):
        # Map image is rescaled to true-world coordinates for accurate waypoint placement.
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
        self.ax.imshow(img, extent=extent, aspect='auto', cmap='gray' if grayscale else None)
        self.image_loaded = True

    def redraw_plot(self):
        # Redraw keeps UI up-to-date: initial line in blue, edited line in green, points in red.
        self.ax.clear()
        if self.image_loaded:
            self.load_image_background()

        wm = self.waypoint_manager
        self.ax.plot(wm.initial_x, wm.initial_y, color="blue", label="Initial Waypoints Line", linestyle="--")
        self.ax.plot(wm.x, wm.y, color="green", label="Adjusted Waypoints Line", linestyle="-")
        self.ax.scatter(wm.x, wm.y, color="red", label="Waypoints")
        self.ax.legend()
        plt.draw()
        self.update_text_box(f"Waypoints updated. Current scale: {self.scale:.1f}")

    def update_text_box(self, message):
        # Central feedback mechanism guiding the user, showing status and shortcuts.
        if self.text_box:
            self.text_box.clear()
            self.text_box.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
            self.text_box.set_xticks([])
            self.text_box.set_yticks([])
            plt.draw()

    def on_press(self, event):
        # Initiates dragging if click is close to a waypoint. Low threshold for ease-of-use.
        if event.inaxes != self.ax:
            return
        wm = self.waypoint_manager
        for i, (px, py) in enumerate(zip(wm.x, wm.y)):
            if np.hypot(event.xdata - px, event.ydata - py) < 0.3:
                self.dragging = True
                self.drag_index = i
                break

    def on_release(self, event):
        # Stops dragging, finalizes adjustments, updates spline and saves new state.
        if self.dragging:
            self.dragging = False
            self.drag_index = None
            self.waypoint_manager._recalculate_splines()
            self.waypoint_manager._save_waypoint_state()
            self.redraw_plot()

    def on_motion(self, event):
        # While dragging, apply weighted changes around the grabbed waypoint for smooth edits.
        if self.dragging and self.drag_index is not None:
            wm = self.waypoint_manager
            dx = event.xdata - wm.x[self.drag_index]
            dy = event.ydata - wm.y[self.drag_index]
            wm.apply_weighted_adjustment(dx, dy, self.drag_index, self.scale)
            self.redraw_plot()

    def update_sigma(self, val):
        # Adjusts the "spread" of influence during dragging.
        self.scale = val
        self.update_text_box(f"Scale updated to: {self.scale}")
        self.redraw_plot()

    def key_press_handler(self, event):
        # Handles keyboard shortcuts: save with CMD/CTRL+S and undo with CMD/CTRL+Z.
        key = event.key.lower() if event.key else ''
        wm = self.waypoint_manager
        if key in ["ctrl+s", "cmd+s"]:
            wm.save_waypoints_to_file(self.update_text_box)
        elif key in ["ctrl+z", "cmd+z"]:
            if wm.undo():
                self.redraw_plot()
                self.update_text_box("Undo successful.")
            else:
                self.update_text_box("No more undo steps available.")

    def setup_ui_elements(self):
        # Slider and text box setup after plot is created.
        plt.subplots_adjust(bottom=0.2)
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.sigma_slider = Slider(ax_slider, "Scale", 1.0, 50.0, valinit=self.scale, valstep=0.1)
        self.sigma_slider.on_changed(self.update_sigma)

        # A simple text box area at the bottom for live feedback.
        self.text_box = plt.axes([0.1, 0.05, 0.8, 0.05])
        self.update_text_box("Drag waypoints to change, CMD+S or Ctrl+S to save, CMD+Z or Ctrl+Z to undo.")

    def connect_events(self):
        # Connect mouse and keyboard event handlers to the matplotlib figure.
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_handler)


class WaypointsEditorApp:
    """High-level application orchestrating data loading, UI, and user interactions."""
    def __init__(self, map_name="RCA1", path_to_maps="./maps/", waypoints_new_file_name=None, scale_initial=20.0):
        self.map_config = MapConfig(map_name, path_to_maps)
        self.waypoint_manager = WaypointDataManager(map_name, path_to_maps, waypoints_new_file_name)
        self.ui = WaypointEditorUI(self.waypoint_manager, self.map_config, initial_scale=scale_initial)

    def run(self):
        # Initialization and execution of the interactive editing session.
        self.waypoint_manager.create_backup_if_needed()
        self.waypoint_manager.load_waypoints()

        self.ui.load_image_background()
        self.ui.redraw_plot()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid()

        self.ui.setup_ui_elements()
        self.ui.connect_events()

        plt.show()


if __name__ == "__main__":
    app = WaypointsEditorApp()
    app.run()

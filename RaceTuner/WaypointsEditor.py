# WaypointsEditor.py

import os
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import datetime
import threading

from RaceTuner.FileSynchronizer import FileSynchronizer, upload_to_remote_via_sftp, download_map_files_via_sftp
from RaceTuner.MapHelper import MapConfig
from RaceTuner.SocketWaypointsEditor import SocketWatpointEditor
from RaceTuner.WaypointsEditorUI import WaypointEditorUI
from RaceTuner.WaypointsHistoryManager import WaypointHistoryManager
from utilities.Settings import Settings

from TunerSettings import (
    MAP_NAME,
    LOCAL_MAP_DIR,
    REMOTE_MAP_DIR,
    REMOTE_AT_LOCAL_DIR,
    USE_REMOTE_FILES,
    # MAP_LIMITS_PLZ_WORK
)

from utilities.waypoint_utils import get_speed_scaling


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
        self.initial_sector_load_done = False

        self.path_to_sectors = os.path.join(path_to_maps, map_name, map_name + "_speed_scaling.csv")
        self.original_sector_data = None
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

    def load_sectors_from_file(self):
        with self.lock:
            # Load waypoints from local (or updated) file
            self.original_sector_data = pd.read_csv(self.path_to_sectors)
            self.sector_idxs = self.original_sector_data['#Start'].to_numpy().astype(int)
            self.sector_speeds = self.original_sector_data[' Sector'].to_numpy()

            # If this is the initial load, store initial values
            # so we can reference them later. Avoid overwriting them on subsequent loads.
            if not self.initial_sector_load_done:
                self.initial_sector_idxs = self.x.copy()
                self.initial_sector_speeds = self.y.copy()

            self.initial_sector_load_done = True

    def local_reload(self):
        self.load_waypoints_from_file()
        self.load_sectors_from_file()


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
            self.waypoint_manager.load_sectors_from_file()

            # Start FileSynchronizer thread
            # TODO: only synchronizes when this is remote -> added workaround for now, press '-' to reload from local files
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

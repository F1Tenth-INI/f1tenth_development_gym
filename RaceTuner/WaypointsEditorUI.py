import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from RaceTuner.HoverMarker import HoverMarker


class DraggableDivider:
    def __init__(self, fig, gs, initial_y=0.5, color="gray", lw=2, picker=5, on_move=None):
        self.fig = fig
        self.gs = gs
        self.y = initial_y
        self.on_move = on_move
        self.divider_line = plt.Line2D(
            [0, 1], [self.y, self.y], transform=self.fig.transFigure,
            color=color, lw=lw, picker=picker
        )
        self.fig.add_artist(self.divider_line)
        self.dragging = False
        self.connect_events()

    def connect_events(self):
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

    def on_pick(self, event):
        if event.artist == self.divider_line:
            self.dragging = True

    def on_motion(self, event):
        if self.dragging and event.y is not None:
            fig_height = self.fig.get_size_inches()[1] * self.fig.dpi
            new_y = event.y / fig_height
            if abs(new_y - self.y) > 0.001:  # Threshold to prevent excessive redraws
                self.y = new_y
                self.gs.set_height_ratios([1 - self.y, self.y])
                self.fig.subplots_adjust(hspace=0.3)
                self.divider_line.set_ydata([self.y, self.y])
                self.fig.canvas.draw_idle()
                if self.on_move:
                    self.on_move()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False

    def get_position(self) -> float:
        return self.y

    def set_position(self, y: float):
        self.y = y
        self.divider_line.set_ydata([self.y, self.y])
        self.gs.set_height_ratios([1 - self.y, self.y])
        self.fig.subplots_adjust(hspace=0.3)
        self.fig.canvas.draw_idle()
        if self.on_move:
            self.on_move()


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
        self.divider = None  # Placeholder for DraggableDivider instance

        if self.waypoint_manager.vx is not None:
            self.gs = GridSpec(2, 1, height_ratios=[1, 1], figure=self.fig)
            self.ax = self.fig.add_subplot(self.gs[0, 0])
            self.ax2 = self.fig.add_subplot(self.gs[1, 0])
            plt.subplots_adjust(hspace=0.3)

            # Initialize DraggableDivider with callback
            self.divider = DraggableDivider(
                self.fig,
                self.gs,
                initial_y=0.5,
                color="gray",
                lw=2,
                picker=5,
                on_move=self.redraw_plot
            )

        else:
            self.fig, self.ax = plt.subplots(figsize=(16, 10))
            self.ax2 = None

        self.fig.canvas.manager.set_window_title("INIvincible Waypoints Editor")
        self.text_box = None
        self.sigma_slider = None
        self.dragging = False
        self.drag_index = None
        self.dragging_vx = False
        self.drag_index_vx = None
        self.image_loaded = False
        self.panning = False
        self.pan_start = None

        # Dynamic elements
        self.car_marker = None
        self.car_speed_marker = None
        self.background_main = None
        self.background_speed = None
        self.x_limit = None
        self.y_limit = None

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

        # Highlight sector start points
        if hasattr(wm, 'sector_idxs'):
            sector_starts_x = wm.x[wm.sector_idxs]
            sector_starts_y = wm.y[wm.sector_idxs]
            for i, (x, y) in enumerate(zip(sector_starts_x, sector_starts_y)):
                self.ax.text(x, y, f"S{i}", color="black", fontsize=12, ha="center", va="center")

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

            # Highlight sector start points in the second plot (ax2)
            if hasattr(wm, 'sector_idxs'):
                sector_start_times = wm.t[wm.sector_idxs]
                for i, t in enumerate(sector_start_times):
                    self.ax2.axvline(x=t, color="black", linestyle="--", label=f"Sector Start {i}")
                    self.ax2.text(t, self.ax2.get_ylim()[1] * (1.0), f"S{i}", color="black", fontsize=12, ha="center",
                                  va="bottom")


        # Combine legends from both subplots

        _, labels = self.ax.get_legend_handles_labels()
        if self.ax2:
            labels[0] = 'Initial Raceline & Speed'
            labels[1] = "Target Raceline & Speed"

        self.ax.legend(labels, loc="upper right", bbox_to_anchor=(2.0, 1), frameon=False)

        # if AUTO_SCALE_MAP:
        if self.x_limit == None or self.y_limit == None:
            self.x_limit = [min(self.waypoint_manager.x) - 4, max(self.waypoint_manager.x) + 4]
            self.y_limit = [min(self.waypoint_manager.y) - 4, max(self.waypoint_manager.y) + 4]

        self.ax.set_xlim(self.x_limit)
        self.ax.set_ylim(self.y_limit)

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

        #on right click
        if event.button == 3:
            self.pan_start = (event.xdata, event.ydata)
            self.panning = True
            return

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
        if event.button == 3:
            if self.panning:
                self.panning = False
            return

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

        # does not properly handle when starting drag inside plot an ending outside
        if self.panning and all(v is not None for v in [event.xdata, event.ydata, self.pan_start]):

            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata

            # print("x before: ", self.ax.get_xlim())
            self.x_limit = ([x + dx for x in self.ax.get_xlim()])
            self.y_limit = ([y + dy for y in self.ax.get_ylim()])
            # print("x after: ", self.x_limit)

            self.ax.set_xlim(self.x_limit)
            self.ax.set_ylim(self.y_limit)
            self.fig.canvas.draw()

            self.redraw_plot()


    def on_scroll(self, event):
        """Handle zoom using scroll wheel"""
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        # Define the zoom scale (how much zoom happens per scroll step)
        zoom_factor = 1.2
        if event.button == 'up':  # Zoom in
            scale = 1 / zoom_factor
        elif event.button == 'down':  # Zoom out
            scale = zoom_factor
        else:
            return

        wm = self.waypoint_manager

        # Get current axis limits
        x_center = (current_xlim[0] + current_xlim[1]) / 2
        y_center = (current_ylim[0] + current_ylim[1]) / 2

        # Adjust the limits based on the zoom factor
        new_xlim = [(x - x_center) * scale + x_center for x in current_xlim]
        new_ylim = [(y - y_center) * scale + y_center for y in current_ylim]

        self.x_limit = new_xlim
        self.y_limit = new_ylim

        # # Apply new limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        self.redraw_plot()

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
        elif key == 'v':
            self.reset_view()
        elif key == '-':
            wm.local_reload()
            self.redraw_plot()

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

    def reset_view(self):
        self.x_limit = None
        self.y_limit = None
        self.redraw_plot()
        return



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
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

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
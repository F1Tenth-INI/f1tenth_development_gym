import numpy as np


class HoverMarker:
    def __init__(self, ax, ax2, waypoint_manager, threshold_main=1.0, threshold_speed=1.0, threshold_wpt_idx=10.0):
        self.ax = ax
        self.ax2 = ax2
        self.wm = waypoint_manager
        self.threshold_main = threshold_main
        self.threshold_speed = threshold_speed
        self.threshold_wpt_idx = threshold_wpt_idx
        self.create_markers()
        self.last_hovered_index = None

    def create_markers(self):
        # Create markers for visual feedback when hovering over a waypoint.
        self.hover_marker_main, = self.ax.plot([], [], 'o', color='yellow', markersize=11, alpha=0.8)
        self.hover_marker_main.set_visible(False)

        if self.ax2 and self.wm.vx is not None:
            self.hover_marker_speed, = self.ax2.plot([], [], 'o', color='yellow', markersize=11, alpha=0.8)
            self.hover_marker_speed.set_visible(False)
        else:
            self.hover_marker_speed = None

    def reconnect_markers(self):
        # Remove existing markers
        if hasattr(self, 'hover_marker_main'):
            self.hover_marker_main.remove()
        if self.hover_marker_speed:
            self.hover_marker_speed.remove()

        # Recreate markers
        self.create_markers()

    def connect(self, canvas):
        self.cid = canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def on_mouse_move(self, event):
        if event.inaxes not in [self.ax, self.ax2]:
            self.hide_markers()
            return

        hovered_index = None
        threshold = self.threshold_main if event.inaxes == self.ax else self.threshold_speed

        if event.inaxes == self.ax:
            distances_squared = (self.wm.x - event.xdata) ** 2 + (self.wm.y - event.ydata) ** 2
            min_dist_squared = np.min(distances_squared)
            if min_dist_squared <  self.threshold_main ** 2:
                hovered_index = np.argmin(distances_squared)
        elif event.inaxes == self.ax2 and self.wm.vx is not None:
            d_idx = np.abs(self.wm.t - event.xdata)
            d_v = np.abs(self.wm.vx - event.ydata)
            hovered_index = np.argmin(d_idx)
            min_dist_v = d_v[hovered_index]
            if min_dist_v > self.threshold_speed:
                hovered_index = None

        if hovered_index is not None:
            self.show_markers(hovered_index)
        else:
            self.hide_markers()

    def show_markers(self, index):
        if index == self.last_hovered_index:
            return
        self.last_hovered_index = index

        # Update main hover marker
        self.hover_marker_main.set_data([self.wm.x[index]], [self.wm.y[index]])
        self.hover_marker_main.set_visible(True)

        # Update speed hover marker if applicable
        if self.hover_marker_speed and self.wm.vx is not None:
            self.hover_marker_speed.set_data([self.wm.t[index]], [self.wm.vx[index]])
            self.hover_marker_speed.set_visible(True)

        # Remove the following line to prevent draw_idle conflicts
        # self.ax.figure.canvas.draw_idle()

    def hide_markers(self):
        if self.last_hovered_index is not None:
            self.hover_marker_main.set_visible(False)
            if self.hover_marker_speed:
                self.hover_marker_speed.set_visible(False)
            self.last_hovered_index = None

            # Remove the following line to prevent draw_idle conflicts
            # self.ax.figure.canvas.draw_idle()

    def draw_markers(self):
        # Method to draw hover markers if they are visible
        if self.last_hovered_index is not None:
            self.ax.draw_artist(self.hover_marker_main)
            if self.ax2 and self.hover_marker_speed:
                self.ax2.draw_artist(self.hover_marker_speed)

import numpy as np


class HoverMarker:
    def __init__(self, ax, ax2, waypoint_manager, threshold_main=1.0, threshold_speed=1.0, threshold_wpt_idx=10.0):
        self.ax = ax
        self.ax2 = ax2
        self.wm = waypoint_manager
        self.threshold_main = threshold_main
        self.threshold_speed = threshold_speed
        self.threshold_wpt_idx = threshold_wpt_idx
        self.last_hovered_index = None
        self.planted_markers = []  # List to store tuples of (index, marker_artist, color)

        # Define a list of colors to cycle through
        self.color_cycle = ['yellow', 'green', 'blue', 'red', 'cyan', 'magenta', 'orange', 'purple', 'brown']
        self.next_color_idx = 0  # Pointer to the next color in the cycle

        # Initialize hover_color_idx to manage hover_marker_main's color independently
        self.hover_color_idx = 0

        self.create_markers()

    def create_markers(self):
        # Create markers for visual feedback when hovering over a waypoint.
        self.hover_marker_main, = self.ax.plot([], [], 'o', color=self.color_cycle[self.hover_color_idx], markersize=11, alpha=0.8)
        self.hover_marker_main.set_visible(False)

        if self.ax2 and self.wm.vx is not None:
            self.hover_marker_speed, = self.ax2.plot([], [], 'o', color=self.color_cycle[self.hover_color_idx], markersize=11, alpha=0.8)
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


    def hide_markers(self):
        if self.last_hovered_index is not None:
            self.hover_marker_main.set_visible(False)
            if self.hover_marker_speed:
                self.hover_marker_speed.set_visible(False)
            self.last_hovered_index = None

            self.hover_marker_main.set_color(self.color_cycle[self.hover_color_idx])
            if self.hover_marker_speed:
                self.hover_marker_speed.set_color(self.color_cycle[self.hover_color_idx])

    def draw_markers(self):
        # Method to draw hover markers if they are visible
        if self.last_hovered_index is not None:
            self.ax.draw_artist(self.hover_marker_main)
            if self.ax2 and self.hover_marker_speed:
                self.ax2.draw_artist(self.hover_marker_speed)

    def plant_marker(self):
        """
        Adds the currently hovered marker to the list of planted markers.
        Changes the hover marker's color to the next color from the color cycle.
        """
        if self.last_hovered_index is not None:
            index = self.last_hovered_index

            # Get the next color from the cycle
            color = self.color_cycle[self.next_color_idx]
            self.next_color_idx = (self.next_color_idx + 1) % len(self.color_cycle)

            # Plot the planted marker with the selected color
            x = self.wm.x[index]
            y = self.wm.y[index]
            vx = self.wm.vx[index]

            # Store the planted marker information
            self.planted_markers.append((x, y, vx, index, color))

            # Change hover marker color to the next color in the cycle
            self.hover_color_idx = (self.hover_color_idx + 1) % len(self.color_cycle)
            self.hover_marker_main.set_color(self.color_cycle[self.hover_color_idx])
            if self.hover_marker_speed:
                self.hover_marker_speed.set_color(self.color_cycle[self.hover_color_idx])

        print(self.planted_markers)

    def erase_marker(self):
        """
        Removes the most recently planted marker from the list.
        Reverts the hover marker's color to the previous color in the color cycle.
        """
        if self.planted_markers:
            self.planted_markers.pop()

            # Decrement the hover_color_idx to go back to the previous color
            self.hover_color_idx = (self.hover_color_idx - 1) % len(self.color_cycle)

            # Change hover marker color to the previous color in the cycle
            self.hover_marker_main.set_color(self.color_cycle[self.hover_color_idx])
            if self.hover_marker_speed:
                self.hover_marker_speed.set_color(self.color_cycle[self.hover_color_idx])


    def get_latest_color_for_index(self, index):
        """
        Returns the most recently planted color for the given index.
        If no markers are planted at this index, returns None.
        """
        for planted_index, _, color in reversed(self.planted_markers):
            if planted_index == index:
                return color
        return None

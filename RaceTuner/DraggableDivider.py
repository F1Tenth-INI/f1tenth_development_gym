from matplotlib import pyplot as plt


class DraggableDivider:
    def __init__(self, fig, gs, axs, color="gray", lw=2, picker=5, on_move=None):
        self.fig = fig
        self.gs = gs
        ax1, ax2 = axs
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()

        y1_min = pos1.y0
        y1_max = pos1.y1
        y2_min = pos2.y0
        y2_max = pos2.y1

        h1 = y1_max - y1_min
        h2 = y2_max - y2_min

        y_inbetween = (h1*y2_min + h2*y1_max)/(h1+h2)

        self.y1_max = lambda: ax1.get_position().y1
        self.y2_min = lambda: ax2.get_position().y0

        self.y = y_inbetween
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
                self.gs.set_height_ratios([self.y1_max()-self.y, self.y-self.y2_min()])
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
        self.gs.set_height_ratios([self.y1_max()-self.y, self.y-self.y2_min()])
        self.fig.canvas.draw_idle()
        if self.on_move:
            self.on_move()

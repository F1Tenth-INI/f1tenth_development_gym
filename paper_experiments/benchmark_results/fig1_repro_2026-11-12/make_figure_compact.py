#!/usr/bin/env python3
"""Compact variant of the paper's Fig. 1 (``method_overview/make_figure.py``), one control step of Auto-Tuning MPPI.

    python make_figure_compact.py [step]   ->  $FIG1_OUT/method_overview_compact.{pdf,png}

Left: the driving step and the optimizer rollouts (window candidates) behind the car. Right: a zoom on the window
update, linked to the rectangle on the left by magnifier lines (no panel letters). The stored targets move
along their cross-track lines to the accepted candidate. Without the waypoint-range panel (c) of the paper
figure, and with the window names written out ("update window", "rejoin section") instead of W_win / W_down.
Same env overrides as make_figure.py: FIG1_CODE, FIG1_STEPS, FIG1_OUT.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Polygon, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# Env overrides (defaults: the original layout): FIG1_CODE = code snapshot with utilities/ and the maps (the
# tune_noise_2026-11-08 snapshot, e.g. ../snapshot_aac2064a), FIG1_STEPS = folder of step_<n>.npz,
# FIG1_OUT = output folder.
CODE = os.environ.get("FIG1_CODE", os.path.join(HERE, "..", "..", "..", "tune_noise_2026-11-08", "code"))
STEPS = os.environ.get("FIG1_STEPS", os.path.join(HERE, "steps"))
sys.path.insert(0, CODE)
from utilities.state_utilities import POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX  # noqa: E402

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2604  # lap 6
MAP = "icra2022"
OUT = os.environ.get("FIG1_OUT", os.path.join(HERE, "..", "figures"))
# Settings of the run (utilities/Settings.py)
LAG, SPAN, DOWN, BEHIND, AHEAD = 20, 5, 10, 50, 5
H, DT, ETA = 35, 0.04, 0.8
CAR_LENGTH, CAR_WIDTH = 0.50, 0.28
ZOOM_DOWN = int(os.environ.get("FIG1_ZOOM_DOWN", "2"))  # downstream waypoints shown in the zoom (rejoin section)

C_ROLL = "#a8a8a8"
C_PLAN = "black"
C_RETR = "#1b9e77"
C_MEM = "#3b6fb6"
C_CAND = "#f5c08a"
C_BEST = "#d95f02"
C_UPD = "#7b3294"
C_CORRIDOR = "#3b6fb6"  # rejoin corridor, drawn very light
# Rejoin corridor: a candidate is accepted only if its error at each of the MPC_WAYPOINT_WINDOW_DOWNSTREAM = 10
# waypoints after the window is <= MPC_WAYPOINT_WINDOW_TOLERANCE = 0.2, with error = 0.5 * cross-track offset + heading,
# speed, drift and yaw-rate terms (HWM_WINDOW_ERROR). In position alone that is +-0.2 / 0.5 = 0.4 m around the
# stored targets (less if the other terms are not zero).
REJOIN_HALF_WIDTH = float(os.environ.get("FIG1_CORRIDOR_HALF", 0.2 / 0.5))  # [m], drawn with FIG1_CORRIDOR=1
REJOIN_DOWNSTREAM = 10
CORRIDOR_ALPHA = 0.22
SHOW_CORRIDOR = os.environ.get("FIG1_CORRIDOR", "0") == "1"  # off: described in the caption instead  # updated targets in the zoom (purple: apart from the accepted candidate)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",  # editable text (TrueType, <text>)
                     "font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5, "font.family": "serif",
                     "mathtext.fontset": "cm"})


def load_map():
    with open(os.path.join(CODE, "utilities", "maps", MAP, f"{MAP}.yaml")) as f:
        y = yaml.safe_load(f)
    img = np.array(Image.open(os.path.join(CODE, "utilities", "maps", MAP, y["image"])).convert("L"))
    res, (ox, oy) = y["resolution"], y["origin"][:2]
    h, w = img.shape
    return img, (ox, ox + w * res, oy, oy + h * res)


def crossing(path, anchor, ortho):
    """Where the polyline ``path`` ``(T, 2)`` first crosses the line through ``anchor`` with normal
    ``ortho``, within 1 m of the anchor, or None."""
    tangent = np.array([ortho[1], -ortho[0]])
    rel = path - anchor
    along, side = rel @ tangent, rel @ ortho
    hit = np.nonzero((along[:-1] < 0) & (along[1:] >= 0) & (np.abs(side[:-1]) < 1.0))[0]
    if not len(hit):
        return None
    k = hit[0]
    f = -along[k] / (along[k + 1] - along[k])
    return path[k] + f * (path[k + 1] - path[k])


def car_patch(state, **kw):
    x, y, th = state[POSE_X_IDX], state[POSE_Y_IDX], state[POSE_THETA_IDX]
    c, s = np.cos(th), np.sin(th)
    corners = np.array([[CAR_LENGTH / 2, CAR_WIDTH / 2], [CAR_LENGTH / 2, -CAR_WIDTH / 2],
                        [-CAR_LENGTH / 2, -CAR_WIDTH / 2], [-CAR_LENGTH / 2, CAR_WIDTH / 2]])
    return Polygon(corners @ np.array([[c, s], [-s, c]]) + [x, y], closed=True, **kw)


def draw_track(ax, img, extent):
    ax.imshow(img > 250, cmap=matplotlib.colors.ListedColormap(["#6a6a6a", "white"]), extent=extent,
              origin="upper", interpolation="nearest", zorder=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


class Labels:
    """Places labels in ``ax`` where they cover the fewest drawn points and no other label.

    ``obstacles``: data points (lines sampled densely) that a label should not hide. Each ``place`` tries its
    candidate positions in order and keeps the first one with the lowest cost, so the preferred position comes
    first.
    """

    def __init__(self, ax, obstacles):
        self.ax, self.taken = ax, []
        self.fig = ax.figure
        self.renderer = self.fig.canvas.get_renderer()
        self.pts = ax.transData.transform(np.asarray(obstacles, float))

    def cost(self, bb):
        pad = 2.0
        inside = ((self.pts[:, 0] > bb.x0 - pad) & (self.pts[:, 0] < bb.x1 + pad) &
                  (self.pts[:, 1] > bb.y0 - pad) & (self.pts[:, 1] < bb.y1 + pad)).sum()
        overlap = sum(max(0, min(bb.x1, t.x1) - max(bb.x0, t.x0) + 3) * max(0, min(bb.y1, t.y1) - max(bb.y0, t.y0) + 3)
                      for t in self.taken)
        ab = self.ax.get_window_extent(self.renderer)
        out = max(0, ab.x0 + 2 - bb.x0) + max(0, bb.x1 - ab.x1 + 2) + max(0, ab.y0 + 2 - bb.y0) + max(0, bb.y1 - ab.y1 + 2)
        return 1000 * out + 100 * overlap + inside

    def place(self, make, candidates):
        """``make(c)`` draws the label for candidate ``c`` and returns its artists (text first)."""
        best = None
        for c in candidates:
            arts = make(c)
            bb = arts[0].get_window_extent(self.renderer)
            cost = self.cost(bb)
            for a in arts:
                a.remove()
            if best is None or cost < best[0]:
                best = (cost, c)
        arts = make(best[1])
        self.taken.append(arts[0].get_window_extent(self.renderer))
        return best[1]

    def block(self, pts):
        """Reserve the box around display-space ``pts`` (e.g. the scale bar line)."""
        from matplotlib.transforms import Bbox
        p = self.ax.transData.transform(np.asarray(pts, float))
        self.taken.append(Bbox([p.min(0) - 3, p.max(0) + 3]))


def scalebar(ax, length, labels=None):
    """Scale bar in the corner of ``ax`` with the fewest drawn points (top left without ``labels``)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = [(x0 + 0.05 * (x1 - x0), y0 + 0.92 * (y1 - y0)), (x1 - 0.05 * (x1 - x0) - length, y0 + 0.92 * (y1 - y0)),
               (x0 + 0.05 * (x1 - x0), y0 + 0.06 * (y1 - y0)), (x1 - 0.05 * (x1 - x0) - length, y0 + 0.06 * (y1 - y0))]
    if labels is None:
        corners = corners[:1]

    def make(c):
        x, y = c
        line = ax.plot([x, x + length], [y, y], color="black", lw=1.5, solid_capstyle="butt", zorder=9)[0]
        txt = ax.text(x + length / 2, y + 0.02 * (y1 - y0), f"{length:g} m", ha="center", va="bottom", fontsize=7,
                      zorder=9, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=1.0))
        return [txt, line]

    if labels is None:
        make(corners[0])
        return
    x, y = labels.place(make, corners)
    labels.block([[x, y], [x + length, y]])


def square_box(points, pad, aspect):
    """``(lo, hi)`` around ``points``, widened to the width/height ratio ``aspect``."""
    lo, hi = points.min(0) - pad, points.max(0) + pad
    size, mid = hi - lo, (hi + lo) / 2
    if size[0] / size[1] < aspect:
        size[0] = size[1] * aspect
    else:
        size[1] = size[0] / aspect
    return mid - size / 2, mid + size / 2


def main():
    d = np.load(os.path.join(STEPS, f"step_{STEP}.npz"))
    img, extent = load_map()
    n = len(d["anchor"])
    state, w, it = d["state"], int(d["w"]), int(d["car_waypoint"])
    before, after, anchor, ortho = d["stored_before"], d["stored_after"], d["anchor"], d["ortho"]
    window = (w + 1 + np.arange(SPAN)) % n
    shown_down = (w + 1 + SPAN + np.arange(ZOOM_DOWN)) % n
    cand = d["candidates"]
    # The accepted candidate: the one crossing the first window line where the stored target moved to.
    gaps = []
    for c in cand:
        p = crossing(c, anchor[window[0]], ortho[window[0]])
        gaps.append(np.inf if p is None else np.linalg.norm(p - after[window[0]]))
    best = int(np.argmin(gaps))

    fig = plt.figure(figsize=(7.0, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.16)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # (a) driving step
    ax = ax_a
    draw_track(ax, img, extent)
    loop = np.vstack((before, before[:1]))
    ax.plot(loop[:, 0], loop[:, 1], color=C_MEM, lw=0.8, marker="o", ms=1.8, zorder=2)
    weights = d["weights"] / d["weights"].max()
    for k in np.argsort(weights):
        r = d["rollouts"][k]
        ax.plot(r[:, 0], r[:, 1], color=C_ROLL, lw=0.5, alpha=0.25 + 0.6 * weights[k], zorder=3)
    for c in cand:
        ax.plot(c[:, 0], c[:, 1], color=C_CAND, lw=0.5, alpha=0.8, zorder=3)
    ax.plot(cand[best, :, 0], cand[best, :, 1], color=C_BEST, lw=1.3, zorder=5)
    mean = d["mean"]
    ax.plot(mean[:, 0], mean[:, 1], color=C_PLAN, lw=1.5, zorder=5)
    retr = d["retrieval"][::3]
    # retrieval points are not drawn (they only stay in the axis limits and label placement, so the layout is unchanged)
    ax.add_patch(car_patch(state, facecolor="#ffd23f", edgecolor="black", lw=0.7, zorder=8))
    ax.scatter(*before[w], s=26, marker="s", color=C_BEST, edgecolors="black", linewidths=0.6, zorder=8)
    # zoom region: the stored state r_w, the window and the first downstream waypoints
    zoom_pts = np.vstack((before[np.r_[w, window, shown_down]], after[window]))
    zlo, zhi = square_box(zoom_pts, 0.1, 1.0)
    # the left panel always contains the whole zoom rectangle, so the magnifier lines start at its corners
    pts = np.vstack((mean, cand[best], retr, before[[w, it]], d["rollouts"].reshape(-1, 2),
                     zlo - 0.15, zhi + 0.15))
    lo, hi = square_box(pts, 0.5, 1.0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    scalebar(ax, 1.0)
    # car label: left of / above the car, wherever it hides the fewest lines
    fig.canvas.draw()
    obst = np.vstack([loop, mean, cand[best], retr] + list(d["rollouts"]) + list(cand))
    car_xy = (state[POSE_X_IDX], state[POSE_Y_IDX])
    Labels(ax, obst).place(
        lambda c: [ax.annotate("car", car_xy, xytext=c, textcoords="offset points", ha="center", va="center",
                               fontsize=7.5, zorder=9, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))],
        [(-4, 11), (-14, 9), (-18, 0), (-14, -9), (6, 11), (14, -9), (0, -12), (18, 0)])

    ax.add_patch(Rectangle(zlo, *(zhi - zlo), fill=False, lw=0.9, color="black", zorder=9))
    # magnifier: the right corners of the small rectangle to the left corners of the zoom panel
    for y_data, y_axes in ((zlo[1], 0.0), (zhi[1], 1.0)):
        fig.add_artist(ConnectionPatch(xyA=(zhi[0], y_data), coordsA=ax_a.transData, xyB=(0.0, y_axes),
                                       coordsB=ax_b.transAxes, color="black", lw=0.7, zorder=10))

    # (b) zoom: the window update
    ax = ax_b
    draw_track(ax, img, extent)
    if SHOW_CORRIDOR:
        corridor = (w + 1 + SPAN + np.arange(REJOIN_DOWNSTREAM)) % n
        left = before[corridor] + REJOIN_HALF_WIDTH * ortho[corridor]
        right = before[corridor] - REJOIN_HALF_WIDTH * ortho[corridor]
        ax.add_patch(Polygon(np.vstack((left, right[::-1])), closed=True, facecolor=C_CORRIDOR, alpha=CORRIDOR_ALPHA,
                             edgecolor="none", zorder=0.5))
    for j in np.r_[window, shown_down]:
        a, o = anchor[j], ortho[j]
        ax.plot([a[0] - 0.45 * o[0], a[0] + 0.45 * o[0]], [a[1] - 0.45 * o[1], a[1] + 0.45 * o[1]],
                color="#b5b5b5", lw=0.8, zorder=1)
    for c in cand:
        ax.plot(c[:, 0], c[:, 1], color=C_CAND, lw=0.6, alpha=0.8, zorder=2)
    ax.plot(cand[best, :, 0], cand[best, :, 1], color=C_BEST, lw=1.5, zorder=3)
    old = before[np.r_[w, window, shown_down]]
    ax.plot(old[:, 0], old[:, 1], color=C_MEM, lw=0.9, zorder=4)
    moved = window[np.linalg.norm(after[window] - before[window], axis=1) > 0.01]
    kept = np.setdiff1d(np.r_[window, shown_down], moved)
    ax.scatter(before[kept, 0], before[kept, 1], s=10, color=C_MEM, edgecolors="white", linewidths=0.4, zorder=5)
    ax.scatter(before[moved, 0], before[moved, 1], s=10, facecolors="white", edgecolors=C_MEM, linewidths=0.9,
               zorder=5)
    ax.scatter(after[moved, 0], after[moved, 1], s=10, color=C_UPD, edgecolors="white", linewidths=0.4, zorder=6)
    for j in moved:
        ax.add_patch(FancyArrowPatch(before[j], after[j], arrowstyle="-|>,head_length=1.6,head_width=0.9", lw=0.8,
                                     color="black", shrinkA=2.2, shrinkB=2.0, zorder=7))
    ax.scatter(*before[w], s=40, marker="s", color=C_BEST, edgecolors="black", linewidths=0.7, zorder=8)
    ax.set_xlim(zlo[0], zhi[0])
    ax.set_ylim(zlo[1], zhi[1])

    # labels: placed where they hide the fewest drawn points and do not overlap each other or the scale bar
    def sample(p, k=8):
        p = np.asarray(p, float)
        return np.vstack([p[i] + t * (p[i + 1] - p[i]) for i in range(len(p) - 1) for t in np.linspace(0, 1, k)])

    obst = [sample(c) for c in cand] + [sample(old), after[moved], before[[w]]]
    obst += [sample([anchor[j] - 0.45 * ortho[j], anchor[j] + 0.45 * ortho[j]]) for j in np.r_[window, shown_down]]
    obst += [sample([before[j], after[j]]) for j in moved]
    ax.figure.canvas.draw()  # fixes the axes positions (equal aspect) before measuring in pixels
    labels = Labels(ax, np.vstack(obst))
    scalebar(ax, 0.25, labels)

    def side_label(points, text):
        mid = points.mean(0)
        o = ortho[int(points_idx[text][len(points_idx[text]) // 2])]
        cands = [mid + k * s_ * o for k in (0.33, 0.45, 0.25, 0.55) for s_ in (+1, -1)]
        labels.place(lambda c: [ax.text(*c, text, ha="center", va="center", fontsize=7.5, zorder=9,
                                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=1.0))], cands)

    points_idx = {"update window": window, "rejoin section": shown_down}
    side_label(before[window], "update window")
    side_label(before[shown_down], "rejoin section")
    labels.place(lambda c: [ax.annotate(r"$\mathbf{r}_w$", before[w], xytext=c, textcoords="offset points",
                                        ha="center", va="center", fontsize=8.5, zorder=9,
                                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=1.0))],
                 [(-12, -2), (12, -2), (0, 11), (0, -11), (-10, 9), (10, 9), (-10, -9), (10, -9)])

    # cross-track line: pointed to at the end of one of the grey lines, on whichever line and side is free
    ct = [(j, sgn, off) for j in np.r_[window[1:], shown_down, window[:1]] for sgn in (-1, +1)
          for off in ((-62, -8), (8, -8), (-62, 6), (8, 6), (-30, -18), (-30, 14))]

    def make_ct(c):
        j, sgn, off = c
        tip = anchor[j] + sgn * 0.4 * ortho[j]
        a = ax.annotate("cross-track line", tip, xytext=off, textcoords="offset points", fontsize=7, zorder=9,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"),
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=1.0))
        return [a]

    labels.place(make_ct, ct)

    handles = [
        Line2D([], [], color=C_ROLL, lw=1.2, label="MPPI rollouts"),
        Line2D([], [], color=C_PLAN, lw=1.5, label=r"executed plan $\bar{\mathbf{X}}_t$"),
        Line2D([], [], color=C_MEM, marker="o", ms=3, lw=0.9, label="stored targets"),
        Line2D([], [], color=C_CAND, lw=1.2, label="optimizer rollouts"),
        Line2D([], [], color=C_BEST, lw=1.5, label="accepted candidate"),
        Line2D([], [], color=C_UPD, marker="o", ms=4, lw=0, mec="white", label="updated targets"),
    ] + ([Patch(facecolor=C_CORRIDOR, alpha=CORRIDOR_ALPHA, edgecolor="none", label=f"rejoin corridor (±{REJOIN_HALF_WIDTH:g} m)")]
         if SHOW_CORRIDOR else [])
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.1),
               handlelength=1.6, columnspacing=1.2)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"method_overview_compact.{ext}"), bbox_inches="tight", dpi=300)
    print("wrote", os.path.join(OUT, "method_overview_compact.pdf"), "step", STEP, "best candidate", best)


if __name__ == "__main__":
    main()

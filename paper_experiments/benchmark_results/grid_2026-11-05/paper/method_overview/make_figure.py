#!/usr/bin/env python3
"""``method_overview.pdf``: one control step of Auto-Tuning MPPI, from a step saved by ``capture.py``.

    python make_figure.py [step]

(a) the driving step and the window candidates rolled out behind the car. (b) a zoom on the window:
the stored targets move along their cross-track lines to the accepted candidate. (c) the waypoint
ranges each part of the method acts on, relative to the car.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "..", "tune_noise_2026-11-08", "code")
sys.path.insert(0, CODE)
from utilities.state_utilities import POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX  # noqa: E402

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2604  # lap 6
MAP = "icra2022"
OUT = os.path.join(HERE, "..", "figures")
# Settings of the run (utilities/Settings.py)
LAG, SPAN, DOWN, BEHIND, AHEAD = 20, 5, 10, 50, 5
H, DT, ETA = 35, 0.04, 0.8
CAR_LENGTH, CAR_WIDTH = 0.50, 0.28
ZOOM_DOWN = 2  # downstream waypoints shown in the zoom

C_ROLL = "#a8a8a8"
C_PLAN = "black"
C_RETR = "#1b9e77"
C_MEM = "#3b6fb6"
C_CAND = "#f5c08a"
C_BEST = "#d95f02"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5, "font.family": "serif",
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


def scalebar(ax, length, where=(0.05, 0.92)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x, y = x0 + where[0] * (x1 - x0), y0 + where[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], color="black", lw=1.5, solid_capstyle="butt", zorder=9)
    ax.text(x + length / 2, y + 0.02 * (y1 - y0), f"{length:g} m", ha="center", va="bottom", fontsize=7, zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))


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
    d = np.load(os.path.join(HERE, "steps", f"step_{STEP}.npz"))
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

    fig = plt.figure(figsize=(7.0, 4.25))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.0), height_ratios=(3.2, 0.85), hspace=0.3, wspace=0.08)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

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
    ax.scatter(retr[:, 0], retr[:, 1], s=8, color=C_RETR, zorder=7, linewidths=0)
    ax.add_patch(car_patch(state, facecolor="#ffd23f", edgecolor="black", lw=0.7, zorder=8))
    ax.annotate("car", (state[POSE_X_IDX], state[POSE_Y_IDX]), xytext=(10, 4), textcoords="offset points",
                fontsize=7.5, zorder=9)
    ax.scatter(*before[w], s=26, marker="s", color=C_BEST, edgecolors="black", linewidths=0.6, zorder=8)
    pts = np.vstack((mean, cand[best], retr, before[[w, it]], d["rollouts"].reshape(-1, 2)))
    lo, hi = square_box(pts, 0.5, 1.0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    scalebar(ax, 1.0)
    ax.set_title("(a) Driving step and window candidates", loc="left")

    # zoom region: the stored state r_w, the window and the first downstream waypoints
    zoom_pts = np.vstack((before[np.r_[w, window, shown_down]], after[window]))
    zlo, zhi = square_box(zoom_pts, 0.1, 1.0)
    ax.add_patch(Rectangle(zlo, *(zhi - zlo), fill=False, lw=0.9, color="black", zorder=9))
    ax.text(zhi[0], zhi[1], "(b)", ha="right", va="bottom", fontsize=7.5, zorder=9)

    # (b) zoom: the window update
    ax = ax_b
    draw_track(ax, img, extent)
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
    ax.scatter(after[moved, 0], after[moved, 1], s=10, color=C_BEST, edgecolors="white", linewidths=0.4, zorder=6)
    for j in moved:
        ax.add_patch(FancyArrowPatch(before[j], after[j], arrowstyle="-|>,head_length=1.6,head_width=0.9", lw=0.8,
                                     color="black", shrinkA=2.2, shrinkB=2.0, zorder=7))
    ax.scatter(*before[w], s=40, marker="s", color=C_BEST, edgecolors="black", linewidths=0.7, zorder=8)
    ax.set_xlim(zlo[0], zhi[0])
    ax.set_ylim(zlo[1], zhi[1])

    def label(points, text, side):
        """``text`` next to the middle of ``points``, offset along their cross-track direction."""
        mid = points.mean(0)
        o = ortho[int(np.r_[window, shown_down][0])]
        o = o if side > 0 else -o
        ax.annotate(text, mid + 0.28 * o, ha="center", va="center", fontsize=8, zorder=9,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))

    label(before[window], r"$\mathcal{W}_{\mathrm{win}}$", +1)
    label(before[shown_down], r"$\mathcal{W}_{\mathrm{down}}$", +1)
    ax.annotate(r"$\mathbf{r}_w$", before[w], xytext=(-8, -2), textcoords="offset points", ha="right", va="center",
                fontsize=8.5, zorder=9, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
    j0 = int(window[1])
    tip = anchor[j0] - 0.4 * ortho[j0]
    ax.annotate("cross-track line", tip, xytext=(-62, -8), textcoords="offset points", fontsize=7, zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"),
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
    scalebar(ax, 0.25)
    ax.set_title("(b) Window update (zoom)", loc="left")

    # (c) waypoint ranges relative to the car
    ax = ax_c
    speed = np.maximum(d["speed"], 0.1)
    spacing = np.linalg.norm(np.roll(before, -1, 0) - before, axis=1)
    ahead = (it + np.arange(n)) % n
    m_t = int(np.searchsorted(np.cumsum(spacing[ahead] / speed[ahead]), ETA * H * DT)) + 1

    def bar(y, a, b, **kw):
        ax.add_patch(Rectangle((a - 0.45, y - 0.3), b - a + 0.9, 0.6, lw=0, **kw))

    bar(3, 1, m_t, color=C_PLAN)
    bar(2, -LAG + 1, -LAG + SPAN, color=C_BEST)
    bar(2, -LAG + SPAN + 1, -LAG + SPAN + DOWN, color=C_CAND)
    bar(1, -BEHIND, AHEAD, color="#bdbdbd")
    ax.scatter([-LAG], [2], s=26, marker="s", color=C_BEST, edgecolors="black", linewidths=0.6, zorder=5)
    ax.text(m_t + 1.2, 3, "driving cost", va="center", fontsize=7.5)
    ax.text(-LAG - 1.8, 2, "window update", va="center", ha="right", fontsize=7.5)
    ax.text(-LAG + (SPAN + 1) / 2, 2.42, r"$\mathcal{W}_{\mathrm{win}}$", ha="center", va="bottom", fontsize=7.5)
    ax.text(-LAG + SPAN + (DOWN + 1) / 2, 2.42, r"$\mathcal{W}_{\mathrm{down}}$", ha="center", va="bottom",
            fontsize=7.5)
    ax.text(AHEAD + 1.2, 1, "rollback after a gating event", va="center", fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-BEHIND - 3, AHEAD + 30)
    ax.set_ylim(0.45, 3.75)
    ax.set_xticks([-BEHIND, -LAG, 0, AHEAD])
    ax.set_xticklabels([r"$i_t-B$", r"$w=i_t-L$", r"$i_t$", r"$i_t+F$"])
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title("(c) Waypoints each part acts on (to scale)", loc="left")

    handles = [
        Line2D([], [], color=C_ROLL, lw=1.2, label="MPPI rollouts"),
        Line2D([], [], color=C_PLAN, lw=1.5, label=r"executed plan $\bar{\mathbf{X}}_t$"),
        Line2D([], [], color=C_RETR, marker="o", ms=3.5, lw=0, label="retrieval points"),
        Line2D([], [], color=C_MEM, marker="o", ms=3, lw=0.9, label="stored targets"),
        Line2D([], [], color=C_CAND, lw=1.2, label="window candidates"),
        Line2D([], [], color=C_BEST, lw=1.5, label="accepted candidate"),
        Line2D([], [], color=C_BEST, marker="o", ms=4, lw=0, mec="white", label="updated targets"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.05),
               handlelength=1.6, columnspacing=1.2)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"method_overview.{ext}"), bbox_inches="tight", dpi=300)
    print("wrote", os.path.join(OUT, "method_overview.pdf"), "step", STEP, "best candidate", best, "M_t", m_t)


if __name__ == "__main__":
    main()

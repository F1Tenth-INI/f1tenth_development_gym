#!/usr/bin/env python3
"""Thesis figures: HW (50 rollouts) vs MPPI (2000 rollouts) on icra2022, raceline start, mu 0.5.

    python make_plots.py --noise on --seed 4 --hw-rollouts 100

1. Lap times per lap with collisions and the HW waypoint freeze.
2. HW waypoint table (raceline arrows) at the start and at the end of training.
3. One representative lap per controller after training, colored by speed.

Lap times come from the grid run (``../runs``). Trajectories come from the re-runs in
``runs/`` (record_traj.py, same code, same seed), which are checked against the grid run.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.dirname(HERE)
_BENCH = os.path.join(GRID, "code", "scripts", "benchmark")
if not os.path.isfile(os.path.join(_BENCH, "bench_svg.py")):  # no code snapshot next to the results: the repository's
    _BENCH = os.path.join(GRID, "..", "..", "..", "scripts", "benchmark")
sys.path.insert(0, _BENCH)
from bench_svg import _map_walls  # noqa: E402

MAP_DIR = os.path.join(GRID, "code", "utilities", "maps", "icra2022")
HW_RUN = MPPI_RUN = HW_TRAJ = MPPI_TRAJ = OUT = NOISE = HW_N = VARIANT = None


def configure(noise: str, seed: int, hw_n: int, frozen_revert: bool = False) -> None:
    """The grid has HW with 50 and 200 rollouts; other counts exist only as re-runs here."""
    global HW_RUN, MPPI_RUN, HW_TRAJ, MPPI_TRAJ, OUT, NOISE, HW_N, VARIANT
    cfg = f"icra2022_raceline_mu0.5_noise{noise}_seed{seed}"
    tag = "_frozenrevert" if frozen_revert else ""
    HW_TRAJ = os.path.join(HERE, "runs", f"hw{hw_n}_noise{noise}_s{seed}{tag}")
    HW_RUN = os.path.join(GRID, "runs", f"hw{hw_n}_{cfg}")
    if frozen_revert or not os.path.exists(os.path.join(HW_RUN, "result.json")):
        HW_RUN = HW_TRAJ
    MPPI_RUN = os.path.join(GRID, "runs", f"mppi2000_{cfg}")
    MPPI_TRAJ = os.path.join(HERE, "runs", f"mppi2000_noise{noise}_s{seed}")
    OUT = os.path.join(HERE, "figures", f"noise{noise}_seed{seed}_hw{hw_n}{tag}")
    NOISE, HW_N = noise, hw_n
    VARIANT = ", waypoints revert after the freeze" if frozen_revert else ""

AT = "Auto-Tuning MPPI"  # display name of the HirarchicalPlanner (HW)
HW_COLOR = "#1f77b4"
MPPI_COLOR = "#444444"
COLLISION_COLOR = "#d62728"
SPEED_CMAP = LinearSegmentedColormap.from_list("speed", plt.cm.plasma(np.linspace(0.0, 0.88, 256)))

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


def load(path):
    with open(os.path.join(path, "result.json")) as f:
        return json.load(f)


def collision_laps(result):
    """1-based index of the lap each collision cluster falls in (the next lap to complete)."""
    lap_t = [l["t"] for l in result["laps"]]
    out = []
    for cluster in result["collision_clusters"]:
        t = result["collisions_raw"][cluster[0]]["t"]
        idx = next((i for i, lt in enumerate(lap_t) if lt >= t), None)
        if idx is not None:
            out.append(idx + 1)
    return out


# ---------------------------------------------------------------------------
# 1. Lap times
# ---------------------------------------------------------------------------

def lap_time_plot(hw, mppi, y_max=19.4, show_collisions=True, title=True, figsize=(7.2, 4.0), legend=True,
                  compact=False):
    fig, ax = plt.subplots(figsize=figsize)
    freeze_lap = hw["freeze_after_lap"]
    n_max = max(len(hw["laps"]), len(mppi["laps"]))
    ax.axvspan(freeze_lap + 0.5, n_max + 1, color=HW_COLOR, alpha=0.07, lw=0)
    ax.axvline(freeze_lap + 0.5, color=HW_COLOR, ls="--", lw=1.2)
    note = (f"frozen after lap {freeze_lap}" if compact
            else f"{AT} waypoints frozen\n(after lap {freeze_lap}, t = {hw['freeze_t']:.0f} s)")
    ax.text(freeze_lap + 1.0, y_max - 0.15, note, color=HW_COLOR, va="top", ha="left",
            fontsize=7 if compact else 8.5)

    for res, color, label, z in ((mppi, MPPI_COLOR, "MPPI, 2000 rollouts", 2), (hw, HW_COLOR, f"{AT}, {HW_N} rollouts", 3)):
        lt = np.array([l["lap_time"] for l in res["laps"]])
        n = np.arange(1, len(lt) + 1)
        shown = np.minimum(lt, y_max)
        ax.plot(n, shown, "-", color=color, lw=1.0, alpha=0.6, zorder=z)
        inside = lt <= y_max
        ax.plot(n[inside], lt[inside], "o", color=color, ms=3.2, zorder=z, label=label)
        for i in np.flatnonzero(~inside):
            ax.plot(n[i], y_max, marker="^", color=color, ms=6, zorder=z, clip_on=False)
            ax.annotate(f"{lt[i]:.1f} s", (n[i], y_max), xytext=(0, 6), textcoords="offset points",
                        color=color, fontsize=7.5, va="bottom", ha="center", annotation_clip=False)
        for c in collision_laps(res) if show_collisions else []:
            ax.plot(c, min(lt[c - 1], y_max), marker="X", color=COLLISION_COLOR, ms=9, mec=color, mew=1.2,
                    zorder=5, ls="none")

    hw_after = np.array(hw["after_freeze"]["lap_times"])
    mppi_lt = np.array([l["lap_time"] for l in mppi["laps"]])[1:]
    ax.axhline(np.median(hw_after), xmin=0, xmax=1, color=HW_COLOR, lw=0.8, ls=":")
    ax.axhline(np.median(mppi_lt), color=MPPI_COLOR, lw=0.8, ls=":")
    ax.text(n_max + 1.3, np.median(hw_after), f"{np.median(hw_after):.2f} s", color=HW_COLOR, va="top", fontsize=8)
    ax.text(n_max + 1.3, np.median(mppi_lt), f"{np.median(mppi_lt):.2f} s", color=MPPI_COLOR, va="bottom", fontsize=8)

    n_coll_hw, n_coll_mppi = hw["n_collisions"], mppi["n_collisions"]
    if show_collisions:
        ax.plot([], [], "X", color=COLLISION_COLOR, ms=8, mec="white", label=f"collision, edge = controller ({AT} {n_coll_hw}, MPPI {n_coll_mppi})")
    ax.plot([], [], ":", color="k", lw=0.8, label=f"median ({AT} after freeze, MPPI from lap 2)")
    ax.set_xlim(0, n_max + 1)
    ax.set_ylim(13.6, y_max)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2 if show_collisions else 3, fontsize=8, frameon=False)
    if title:
        ax.set_title(f"icra2022, μ = 0.5, 1000 s simulated{VARIANT}", fontsize=10, pad=16)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

_WALLS = None


def walls():
    global _WALLS
    if _WALLS is None:
        _WALLS = _map_walls(MAP_DIR, "icra2022")
    return _WALLS


def draw_track(ax, pad=0.4):
    contours, _ = walls()
    # The outer contour is the map border; keep the ones that bound the track.
    for c in contours:
        ax.plot(c[:, 0], c[:, 1], color="k", lw=0.9, solid_capstyle="round", zorder=1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def set_track_limits(ax, xy, pad=0.5):
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


# ---------------------------------------------------------------------------
# 2. Waypoint table at start and end of training
# ---------------------------------------------------------------------------

VIEWER_STOPS = np.array(((0, 190, 0), (240, 220, 0), (255, 140, 0), (210, 30, 20))) / 255.0
VIEWER_CMAP = LinearSegmentedColormap.from_list("viewer_speed", VIEWER_STOPS)
VIEWER_V_MAX = 10.0  # v_max of yokomo1_car_parameters.yml, the viewer's top color


def viewer_arrows(xy, heading, speed):
    """Arrow polylines (N, 5, 2) as ``HirarchicalPlanner._publish_waypoint_display`` draws them:
    tail at the waypoint, 0.08 m per m/s along the heading, head 0.35 of the length at 0.5 rad."""
    length = np.abs(speed) * 0.08
    tip = xy + np.stack((length * np.cos(heading), length * np.sin(heading)), axis=1)
    head = length * 0.35
    left = tip - np.stack((head * np.cos(heading - 0.5), head * np.sin(heading - 0.5)), axis=1)
    right = tip - np.stack((head * np.cos(heading + 0.5), head * np.sin(heading + 0.5)), axis=1)
    return np.stack((xy, tip, left, tip, right), axis=1)


def raceline_plot(norm=None):
    tables = [("Start of training", np.load(os.path.join(HW_RUN, "initial_raceline.npz"))),
              ("End of training", np.load(os.path.join(HW_RUN, "final_raceline.npz")))]
    vnorm = Normalize(0.0, VIEWER_V_MAX)
    fig, axes = plt.subplots(2, 1, figsize=(9, 9.6))
    init_xy = tables[0][1]["xy"]
    for k, (ax, (title, t)) in enumerate(zip(axes, tables)):
        draw_track(ax)
        xy, hd, sp = t["xy"], t["heading"], t["speed"]
        if k == 1:
            ax.plot(np.r_[init_xy[:, 0], init_xy[0, 0]], np.r_[init_xy[:, 1], init_xy[0, 1]], color="0.6",
                    lw=0.8, ls="--", zorder=2, label="initial raceline")
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        lc = LineCollection(viewer_arrows(xy, hd, sp), colors=VIEWER_CMAP(vnorm(np.abs(sp))), linewidths=1.1,
                            capstyle="round", joinstyle="round", zorder=3)
        ax.add_collection(lc)
        rng = f"{sp.min():.1f} m/s (uniform)" if np.ptp(sp) < 1e-3 else f"{sp.min():.1f}–{sp.max():.1f} m/s"
        ax.set_title(f"{title}: target speed {rng}", fontsize=10)
        set_track_limits(ax, np.concatenate(walls()[0]), pad=0.1)
    fig.subplots_adjust(left=0.01, right=0.88, hspace=0.08, top=0.96, bottom=0.04)
    cax = fig.add_axes([0.9, 0.3, 0.015, 0.4])
    fig.colorbar(plt.cm.ScalarMappable(norm=vnorm, cmap=VIEWER_CMAP), cax=cax,
                 label="Target speed (m/s), arrow length 0.08 m per m/s")
    return fig


def finish_layout(fig, norm, cbar_label):
    fig.subplots_adjust(left=0.01, right=0.89, wspace=0.03, top=0.86, bottom=0.02)
    cax = fig.add_axes([0.905, 0.12, 0.013, 0.66])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SPEED_CMAP), cax=cax, label=cbar_label)


# ---------------------------------------------------------------------------
# 3. Representative laps after training
# ---------------------------------------------------------------------------

def representative_lap(rerun, first_lap):
    """(t0, t1, lap_time) of the collision-free lap from ``first_lap`` on closest to the median lap time."""
    laps = rerun["laps"]
    coll_t = [c["t"] for c in rerun["collisions_raw"]]
    cands = []
    for i in range(max(first_lap, 1), len(laps)):
        t0, t1 = laps[i - 1]["t"], laps[i]["t"]
        if any(t0 <= c <= t1 for c in coll_t):
            continue
        cands.append((t0, t1, laps[i]["lap_time"], i + 1))
    med = np.median([c[2] for c in cands])
    return min(cands, key=lambda c: abs(c[2] - med))


def lap_segment(traj_dir, rerun, first_lap):
    d = np.load(os.path.join(traj_dir, "trajectory.npz"))
    t0, t1, lap_time, lap_no = representative_lap(rerun, first_lap)
    dt = float(d["t"][1] - d["t"][0])
    m = (d["t"] >= t0 - dt) & (d["t"] <= t1 + dt)
    return d["x"][m], d["y"][m], d["v"][m], lap_time, lap_no


def draw_lap(ax, x, y, v, norm, lw=2.4):
    pts = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=SPEED_CMAP, norm=norm, linewidths=lw, capstyle="round", zorder=3)
    lc.set_array(0.5 * (v[:-1] + v[1:]))
    ax.add_collection(lc)
    ax.plot(x[0], y[0], marker="o", ms=6, mfc="white", mec="k", zorder=4)
    k = 12
    ax.annotate("", xy=(x[k], y[k]), xytext=(x[0], y[0]),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, shrinkA=4, shrinkB=0), zorder=4)


def trajectory_plot(laps, norm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, (x, y, v, lap_time, lap_no)) in zip(axes, laps):
        draw_track(ax)
        draw_lap(ax, x, y, v, norm)
        ax.set_title(f"{name}\nlap {lap_no}: {lap_time:.2f} s, v̄ = {v.mean():.2f} m/s, max {v.max():.2f} m/s",
                     fontsize=10)
        set_track_limits(ax, np.concatenate(walls()[0]), pad=0.1)
    finish_layout(fig, norm, "Speed (m/s)")
    return fig


HW_SPEED_CMAP = LinearSegmentedColormap.from_list("hw_speed", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
MPPI_SPEED_CMAP = LinearSegmentedColormap.from_list("mppi_speed", plt.cm.Oranges(np.linspace(0.3, 1.0, 256)))


def speed_line(ax, x, y, v, norm, cmap, lw, z):
    pts = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw, capstyle="round", zorder=z)
    lc.set_array(0.5 * (v[:-1] + v[1:]))
    ax.add_collection(lc)


def overlay_plot(laps, compact=False):
    """Both laps on one track: hue is the controller, lightness the speed.

    MPPI is drawn wide underneath, HW narrow on top with a white edge, so where the
    paths coincide HW shows as a blue core inside an orange band.
    """
    (hw_name, hw), (mppi_name, mppi) = laps
    v_all = np.concatenate([hw[2], mppi[2]])
    norm = Normalize(np.floor(v_all.min()), np.ceil(v_all.max()))
    fig, ax = plt.subplots(figsize=(3.7, 2.5) if compact else (8.8, 4.6))
    draw_track(ax)
    wide, narrow = (3.2, 1.3) if compact else (6.0, 2.2)
    speed_line(ax, *mppi[:3], norm, MPPI_SPEED_CMAP, lw=wide, z=3)
    ax.plot(hw[0], hw[1], color="white", lw=narrow + 0.7, solid_capstyle="round", zorder=4)
    speed_line(ax, *hw[:3], norm, HW_SPEED_CMAP, lw=narrow, z=5)
    x, y = hw[0], hw[1]
    ax.plot(x[0], y[0], marker="o", ms=3.5 if compact else 6, mfc="white", mec="k", zorder=6)
    k = 14
    ax.annotate("", xy=(x[k], y[k]), xytext=(x[0], y[0]),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, shrinkA=4, shrinkB=0), zorder=6)
    set_track_limits(ax, np.concatenate(walls()[0]), pad=0.1)
    handles = [
        plt.Line2D([], [], color=HW_SPEED_CMAP(0.7), lw=narrow,
                   label=f"{'AT-MPPI' if compact else hw_name}: lap {hw[4]}, {hw[3]:.2f} s"),
        plt.Line2D([], [], color=MPPI_SPEED_CMAP(0.7), lw=wide,
                   label=f"{'MPPI 2000' if compact else mppi_name}: lap {mppi[4]}, {mppi[3]:.2f} s"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6 if compact else 8.5, frameon=False)
    right = 0.80 if compact else 0.84
    fig.subplots_adjust(left=0.01, right=right, top=0.98, bottom=0.02)
    for i, (cmap, label) in enumerate(((HW_SPEED_CMAP, "AT" if compact else "Auto-\nTuning\nMPPI"),
                                       (MPPI_SPEED_CMAP, "MPPI"))):
        cax = fig.add_axes([(0.825 if compact else 0.855) + i * (0.07 if compact else 0.045),
                            0.18, 0.02 if compact else 0.014, 0.55])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cax.set_title(label, fontsize=6 if compact else 9)
        if i == 0:
            cb.ax.tick_params(labelleft=False, labelright=False)
        else:
            cb.set_label("Speed (m/s)")
            if compact:
                cb.ax.tick_params(labelsize=6)
                cb.ax.yaxis.label.set_size(6)
    return fig


def check_rerun(orig, rerun, name):
    a = [round(l["lap_time"], 2) for l in orig["laps"]]
    b = [round(l["lap_time"], 2) for l in rerun["laps"]]
    same = a == b
    n_same = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), min(len(a), len(b)))
    print(f"{name}: re-run reproduces grid run: {same} (first {n_same} of {len(a)} laps identical)")
    return same


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--noise", choices=["on", "off"], default="on")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--hw-rollouts", type=int, default=50)
    p.add_argument("--frozen-revert", action="store_true",
                   help="the run was made with HWM_FROZEN_REVERT (waypoints around the car go back after the freeze)")
    args = p.parse_args()
    configure(args.noise, args.seed, args.hw_rollouts, args.frozen_revert)
    os.makedirs(OUT, exist_ok=True)
    hw, mppi = load(HW_RUN), load(MPPI_RUN)

    fig = lap_time_plot(hw, mppi)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"laptimes_hw{HW_N}_vs_mppi2000.{ext}"), bbox_inches="tight")

    have_traj = all(os.path.exists(os.path.join(p, "trajectory.npz")) for p in (HW_TRAJ, MPPI_TRAJ))
    hw_final = np.load(os.path.join(HW_RUN, "final_raceline.npz"))["speed"]
    v_top = float(hw_final.max())
    if have_traj:
        hw_re, mppi_re = load(HW_TRAJ), load(MPPI_TRAJ)
        if HW_RUN != HW_TRAJ:
            check_rerun(hw, hw_re, "HW")
        check_rerun(mppi, mppi_re, "MPPI")
        laps = [(f"{AT}, {HW_N} rollouts (waypoints frozen)", lap_segment(HW_TRAJ, hw_re, hw_re["freeze_after_lap"] + 1)),
                ("MPPI, 2000 rollouts", lap_segment(MPPI_TRAJ, mppi_re, hw_re["freeze_after_lap"] + 1))]
        v_top = max(v_top, *(l[1][2].max() for l in laps))
    norm = Normalize(0.0, np.ceil(v_top))

    fig = raceline_plot(norm)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"hw{HW_N}_raceline_start_end.{ext}"), bbox_inches="tight")

    if have_traj:
        fig = trajectory_plot(laps, norm)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"trajectories_after_training.{ext}"), bbox_inches="tight")
        fig = overlay_plot(laps)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"trajectories_after_training_overlay.{ext}"), bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

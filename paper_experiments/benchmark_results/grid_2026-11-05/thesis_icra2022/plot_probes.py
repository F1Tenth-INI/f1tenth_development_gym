#!/usr/bin/env python3
"""Command fields of the frozen HW planner and of MPPI from probe_policies.py.

One arrow per probe state (car at rest, heading along the nearest frozen waypoint):
direction = heading + front-wheel angle, length = longitudinal acceleration (backwards
when braking/reversing), color = front-wheel angle. The third figure shows the arrow
difference HW - MPPI, colored by the difference of the wheel angles.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GRID, "code", "scripts", "benchmark"))
from bench_svg import _map_walls  # noqa: E402

PROBES = os.path.join(HERE, "probes")
OUT = os.path.join(HERE, "figures", "command_fields")
MAP_DIR = os.path.join(GRID, "code", "utilities", "maps", "icra2022")
FROZEN = os.path.join(GRID, "runs", "hw50_icra2022_raceline_mu0.5_noiseon_seed4", "final_raceline.npz")

# yokomo1_car_parameters.yml: the car applies these factors to the commands.
STEERING_MULTIPLIER = 1.6
ACCEL_MULTIPLIER = 2.0
BRAKE_MULTIPLIER = 3.0
A_MAX = 0.5 * 9.81  # mu g: both controllers' acceleration limit
DELTA_MAX = 0.6

AT = "Auto-Tuning MPPI"  # display name of the HirarchicalPlanner (HW)
ARROW_PER_ACCEL = 0.27 / A_MAX  # [m per m/s^2]: full acceleration is just under the grid spacing (0.3 m)
DIFF_SCALE = 0.5  # differences reach twice the full acceleration
TRACKING_BAND = 0.6  # [m] MPC_WAYPOINT_UPDATE_DISTANCE: farther from its waypoints HW is in recovery mode
STEER_CMAP = plt.get_cmap("berlin")

plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "pdf.fonttype": 42})


def physical(d):
    delta = STEERING_MULTIPLIER * d["steer"]
    a = np.where(d["accel"] > 0, ACCEL_MULTIPLIER * d["accel"], BRAKE_MULTIPLIER * d["accel"])
    return delta, a


def arrow_vectors(theta, delta, a):
    """xy of the arrow: along the front wheels, length by acceleration, reversed when negative."""
    ang = theta + delta
    return ARROW_PER_ACCEL * a * np.cos(ang), ARROW_PER_ACCEL * a * np.sin(ang)


def draw_track(ax):
    contours, _ = _map_walls(MAP_DIR, "icra2022")
    for c in contours:
        ax.plot(c[:, 0], c[:, 1], color="k", lw=0.8, zorder=1)
    allc = np.concatenate(contours)
    lo, hi = allc.min(axis=0) - 0.1, allc.max(axis=0) + 0.1
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    t = np.load(FROZEN)
    xy = np.vstack([t["xy"], t["xy"][:1]])
    ax.plot(xy[:, 0], xy[:, 1], color="0.6", lw=0.8, ls="--", zorder=1.5)
    # Band around the frozen waypoints in which HW tracks them (outside: recovery mode).
    from scipy.spatial import cKDTree

    seg = np.linspace(0, 1, 20, endpoint=False)[:, None, None]
    dense = (xy[:-1][None] + seg * (xy[1:] - xy[:-1])[None]).reshape(-1, 2)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 900), np.linspace(lo[1], hi[1], 600))
    dist, _ = cKDTree(dense).query(np.c_[gx.ravel(), gy.ravel()])
    from probe_policies import free_space

    free, res, ox, oy = free_space()
    h, w = free.shape
    col = np.clip(np.floor((gx - ox) / res).astype(int), 0, w - 1)
    row = np.clip(h - 1 - np.floor((gy - oy) / res).astype(int), 0, h - 1)
    band = ((dist.reshape(gx.shape) <= TRACKING_BAND) & free[row, col]).astype(float)
    ax.contourf(gx, gy, band, levels=[0.5, 1.5], colors=["#e8eef7"], zorder=0.5)


def field(ax, x, y, u, v, c, norm, cmap):
    return ax.quiver(x, y, u, v, c, cmap=cmap, norm=norm, angles="xy", scale_units="xy", scale=1,
                     width=0.0022, headwidth=3.2, headlength=3.5, headaxislength=3.0, minlength=0.3,
                     pivot="tail", zorder=3)


def finish(fig, ax, mappable, label, title, note):
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label)
    ax.text(0.0, -0.005, note, transform=ax.transAxes, fontsize=8, color="0.3", va="top")
    fig.tight_layout()


def scale_bar(ax, length, label):
    x0, y0 = ax.get_xlim()[0] + 0.6, ax.get_ylim()[1] - 0.6
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(x0 + length + 0.15, y0, label, va="center", fontsize=8)
    ax.fill_between([], [], color="#e8eef7", label=f"≤ {TRACKING_BAND} m from the frozen {AT} waypoints "
                                                       f"({AT} tracks them; outside it is in recovery mode)")
    ax.plot([], [], color="0.6", lw=0.8, ls="--", label=f"frozen {AT} waypoints")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.035), fontsize=8, frameon=False, ncol=2)


def main():
    os.makedirs(OUT, exist_ok=True)
    st = np.load(os.path.join(PROBES, "states.npz"))
    hw, mppi = np.load(os.path.join(PROBES, "hw50.npz")), np.load(os.path.join(PROBES, "mppi2000.npz"))
    x, y, th = st["x"], st["y"], st["theta"]
    d_hw, a_hw = physical(hw)
    d_mp, a_mp = physical(mppi)
    deg = np.degrees
    steer_norm = Normalize(-deg(DELTA_MAX), deg(DELTA_MAX))
    note = f"car at rest, heading along the nearest frozen {AT} waypoint; arrow: front-wheel direction, " \
           "length ∝ acceleration (pointing back = braking/reverse)"

    for name, label, delta, a, extra in (("hw50", f"{AT}, 50 rollouts, frozen waypoints", d_hw, a_hw, hw),
                                         ("mppi2000", "MPPI, 2000 rollouts", d_mp, a_mp, None)):
        fig, ax = plt.subplots(figsize=(11, 6.6))
        draw_track(ax)
        u, v = arrow_vectors(th, delta, a)
        q = field(ax, x, y, u, v, deg(delta), steer_norm, STEER_CMAP)
        scale_bar(ax, ARROW_PER_ACCEL * A_MAX, f"{A_MAX:.1f} m/s² (full acceleration)")
        stats = f"mean |δ| {np.mean(np.abs(deg(delta))):.1f}°, mean a {np.mean(a):.2f} m/s², " \
                f"braking/reverse at {np.mean(a < 0) * 100:.0f}% of states"
        if extra is not None and "recovery" in extra:
            stats += f", recovery mode at {np.mean(extra['recovery']) * 100:.0f}%"
        finish(fig, ax, q, "Front-wheel angle δ (°), + = left", f"{label}: command at rest\n{stats}", note)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"command_field_{name}.{ext}"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    draw_track(ax)
    u_hw, v_hw = arrow_vectors(th, d_hw, a_hw)
    u_mp, v_mp = arrow_vectors(th, d_mp, a_mp)
    dd = deg(d_hw - d_mp)
    lim = float(np.ceil(np.percentile(np.abs(dd), 98) / 5) * 5)
    q = field(ax, x, y, DIFF_SCALE * (u_hw - u_mp), DIFF_SCALE * (v_hw - v_mp), dd, Normalize(-lim, lim),
              plt.get_cmap("berlin"))
    scale_bar(ax, DIFF_SCALE * ARROW_PER_ACCEL * A_MAX, f"{A_MAX:.1f} m/s² (half the scale of the other two plots)")
    da = a_hw - a_mp
    stats = f"mean |Δδ| {np.mean(np.abs(dd)):.1f}°, mean Δa {np.mean(da):+.2f} m/s², " \
            f"|arrow difference| median {np.median(np.hypot(u_hw - u_mp, v_hw - v_mp)) / ARROW_PER_ACCEL:.2f} m/s²"
    finish(fig, ax, q, f"Δ front-wheel angle, {AT} − MPPI (°)", f"Difference {AT} − MPPI\n{stats}",
           f"arrow: {AT} arrow minus MPPI arrow (vector difference), at half the length scale; "
           f"color: {AT} wheel angle minus MPPI wheel angle")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"command_field_difference.{ext}"), bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

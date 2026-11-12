#!/usr/bin/env python3
"""Lap-time development on icra2022 (raceline, mu 0.5, no noise, seed 1): tuned Auto-Tuning MPPI with
50 and 1000 rollouts (one run each) and MPPI with 2000 rollouts (steering noise 0.2).

    python make_icra_laptimes_rollouts.py   ->  figures/icra_laptimes_rollouts.{pdf,png}

Each Auto-Tuning MPPI curve has its freeze lap marked (open circle); the legend gives the median lap time
after the freeze (MPPI: from lap 2). Configs per rollout count: RUNS below.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import make_sim_figures as sf

mp = sf.mp
SEED = 1
BR = os.path.join(sf.GRID, "..")
TUNE = os.path.join(BR, "hw50_tune_2026-11-10", "runs")
RUNS = {  # rollouts: run folder (set from the r3 sweeps)
    50: os.path.join(TUNE, "r2", "wm10_tsf103", f"seed{SEED}"),
    1000: os.path.join(TUNE, "r3_n1000", os.environ.get("CFG1000", "wm10_tsf103"), f"seed{SEED}"),
}
MPPI_RUN = os.path.join(BR, "mppi_tune_2026-11-10", "runs", "plotrun", "ns_s02", f"seed{SEED}")
BLUES = {50: mp.HW_COLOR, 1000: "#08306b"}
Y_MIN, Y_MAX = 13.8, 16.0


def curve(ax, laps, color, label, z, ms=3.0):
    lt = np.array(laps)
    n = np.arange(1, len(lt) + 1)
    ax.plot(n, np.minimum(lt, Y_MAX), "-", color=color, lw=0.9, alpha=0.6, zorder=z)
    inside = lt <= Y_MAX
    ax.plot(n[inside], lt[inside], "o", color=color, ms=ms, zorder=z, label=label)
    for i in np.flatnonzero(~inside):
        ax.plot(n[i], Y_MAX, marker="^", color=color, ms=5, zorder=z, clip_on=False)
    return lt, n


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    mppi = mp.load(MPPI_RUN)
    assert mppi["n_collisions"] == 0
    mppi_lt = [l["lap_time"] for l in mppi["laps"]]
    curve(ax, mppi_lt, mp.MPPI_COLOR, f"MPPI, 2000 rollouts ({np.median(mppi_lt[1:]):.2f} s)", 2)
    n_max = len(mppi_lt)
    first_laps = []
    for z, (k, path) in enumerate(RUNS.items(), start=3):
        hw = mp.load(path)
        lt = [l["lap_time"] for l in hw["laps"]]
        n_max = max(n_max, len(lt))
        med = np.median(hw["after_freeze"]["lap_times"])
        label = f"{mp.AT}, {k} rollouts ({med:.2f} s)" + (f", {hw['n_collisions']} coll." if hw["n_collisions"] else "")
        curve(ax, lt, BLUES[k], label, z)
        f = hw["freeze_after_lap"]
        ax.plot(f, min(lt[f - 1], Y_MAX), "o", ms=8, mfc="none", mec=BLUES[k], mew=1.4, zorder=z + 10)
        first_laps.append((k, lt[0]))
        print(f"HW {k}: {path}\n   first lap {lt[0]:.1f} s, freeze after lap {f} ({hw['freeze_t']:.0f} s), "
              f"median after freeze {med:.2f} s, mean {np.mean(hw['after_freeze']['lap_times']):.2f}, "
              f"collisions {hw['n_collisions']} (after freeze {hw['after_freeze']['n_collisions']})")
    print(f"MPPI 2000: median from lap 2 {np.median(mppi_lt[1:]):.2f} s")
    above = ([f"MPPI {mppi_lt[0]:.1f} s"] if mppi_lt[0] > Y_MAX else []) + \
        [f"{k} rollouts {t:.1f} s" for k, t in first_laps if t > Y_MAX]
    if above:
        ax.text(6.0, Y_MAX - 0.05, "first lap (▲): " + ", ".join(above), fontsize=7, va="top", color="0.3")
    ax.plot([], [], "o", ms=7, mfc="none", mec="0.3", mew=1.2, label="memory frozen after this lap")
    ax.set_xlim(0, n_max + 1)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.5, frameon=False,
              title="median after freeze", title_fontsize=7.5)
    fig.tight_layout()
    sf.save(fig, "icra_laptimes_rollouts")


if __name__ == "__main__":
    main()

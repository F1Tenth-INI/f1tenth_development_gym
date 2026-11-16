#!/usr/bin/env python3
"""Variant of make_icra_laptimes_rollouts.py with the freeze drawn into each curve.

    python make_icra_laptimes_rollouts_v2.py   ->  figures/icra_laptimes_rollouts_v2.{pdf,png}

Auto-Tuning MPPI laps are open markers while the memory still learns and filled once it is frozen;
a dotted line in the curve colour marks the freeze lap. A thin dashed line from the freeze lap on is the
median lap time after the freeze (MPPI: from lap 2), the number in the legend.
"""
import numpy as np
import matplotlib.pyplot as plt

import make_sim_figures as sf
from make_icra_laptimes_rollouts import BLUES, MPPI_RUN, RUNS, Y_MAX, Y_MIN, mp


def main(runs=RUNS, mppi_run=MPPI_RUN, name="icra_laptimes_rollouts_v2", y_max=Y_MAX, mppi_rollouts=2000):
    Y_MAX = y_max

    def clipped(lt):
        return np.minimum(lt, Y_MAX), lt > Y_MAX

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    mppi = mp.load(mppi_run)
    mppi_lt = np.array([l["lap_time"] for l in mppi["laps"]])
    n_max = len(mppi_lt)
    first_above = []
    label_laps = []  # freeze laps already labelled, to stack labels that would overlap

    # MPPI: no memory, so no freeze; all markers filled
    n = np.arange(1, len(mppi_lt) + 1)
    y, above = clipped(mppi_lt)
    med = np.median(mppi_lt[1:])
    ax.plot(n, y, "-", color=mp.MPPI_COLOR, lw=0.9, alpha=0.6, zorder=2)
    ax.plot(n[~above], y[~above], "o", color=mp.MPPI_COLOR, ms=3, zorder=2,
            label=f"MPPI, {mppi_rollouts} rollouts ({med:.2f} s)" + (f", {mppi['n_collisions']} coll." if mppi["n_collisions"] else ""))
    ax.hlines(med, 2, len(mppi_lt), color=mp.MPPI_COLOR, lw=0.8, ls="--", alpha=0.8, zorder=1)
    if above.any():
        first_above.append("MPPI " + ", ".join(f"{t:.1f}" for t in mppi_lt[above]))

    for z, (k, path) in enumerate(runs.items(), start=3):
        hw = mp.load(path)
        lt = np.array([l["lap_time"] for l in hw["laps"]])
        n = np.arange(1, len(lt) + 1)
        n_max = max(n_max, len(lt))
        c = BLUES[k]
        f = hw["freeze_after_lap"]  # laps 1..f learn, laps after f drive the frozen memory
        y, above = clipped(lt)
        learn, frozen = n <= f, n > f
        med = np.median(hw["after_freeze"]["lap_times"])
        ax.plot(n, y, "-", color=c, lw=0.9, alpha=0.6, zorder=z)
        ax.plot(n[learn & ~above], y[learn & ~above], "o", ms=3.5, mfc="white", mec=c, mew=0.9, zorder=z)
        coll = f", {hw['n_collisions']} coll." if hw["n_collisions"] else ""
        ax.plot(n[frozen & ~above], y[frozen & ~above], "o", color=c, ms=3, zorder=z,
                label=f"{mp.AT}, {k} rollouts ({med:.2f} s){coll}")
        for i in np.flatnonzero(above):
            ax.plot(n[i], Y_MAX, marker="^", color=c, ms=5, zorder=z, clip_on=False)
        if above.any():
            first_above.append(f"{k} rollouts " + ", ".join(f"{t:.1f}" for t in lt[above]))
        ax.axvline(f + 0.5, color=c, lw=1.0, ls=":", zorder=1)
        dy = 0.22 * sum(abs(f - g) < 4 for g in label_laps)
        label_laps.append(f)
        ax.text(f + 0.8, Y_MAX - 0.04 - dy, f"frozen\n({k})", color=c, fontsize=6.5, va="top", ha="left")
        ax.hlines(med, f + 0.5, len(lt), color=c, lw=0.8, ls="--", alpha=0.8, zorder=1)

    if first_above:
        ax.text(n_max, Y_MAX - 0.05, f"laps above {Y_MAX:.0f} s (▲), in s:\n" + "\n".join(first_above), fontsize=7, va="top", ha="right",
                color="0.3")
    ax.plot([], [], "o", ms=3.5, mfc="white", mec="0.3", mew=0.9, label="learning (before freeze)")
    ax.plot([], [], "--", color="0.3", lw=0.8, label="median after freeze")
    ax.set_xlim(0, n_max + 1)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout()
    sf.save(fig, name)


if __name__ == "__main__":
    main()

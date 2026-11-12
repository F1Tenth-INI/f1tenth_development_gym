#!/usr/bin/env python3
"""Lap time per lap on icra2022 (raceline, mu 0.5) with observation noise, in the style of make_icra_laptimes_rollouts_v2.py.

    python make_icra_laptimes_noise.py   ->  figures/icra_laptimes_noise{half,on}.{pdf,png}

Runs are those of the all_tracks figures: Auto-Tuning MPPI 50 = soft25w02, MPPI 50 / 2000 with the default
steering noise 0.1. Half noise (0.025) from tune_noise_2026-11-08/runs, full noise (0.05) MPPI from the grid.
One seed each: the median collision-free Auto-Tuning MPPI seed of that setting.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import make_sim_figures as sf

mp = sf.mp
TUNE = sf.TUNE_RUNS
GRID_RUNS = os.path.join(sf.GRID, "runs")
NAME = "icra2022_raceline_mu0.5_noise{}_seed{}"
SEEDS = {"half": 4, "on": 2}
Y_MIN, Y_MAX = 13.8, 17.0
MPPI50_COLOR = "#b0b0b0"
AT1000_COLOR = "#08306b"
# Half noise: curves and freezes labelled in the plot instead of a legend.
# Positions (lap, lap time) of each label text and of the point its arrow points at.
INLINE = {
    "half": {
        "mppi": ((42.2, 15.0), (41, 14.74)),
        "50": ((47.5, 13.75), (49, 14.53)),
        "1000": ((28, 13.75), (29, 14.06)),
        "frozen 50": ((16.8, 16.72), (17.5, 16.72)),
        "frozen 1000": ((19.5, 16.72), (18.5, 16.72)),
    },
}
ARROW = dict(arrowstyle="-|>", lw=0.8, mutation_scale=7, shrinkA=1, shrinkB=2)
HW_TUNE = os.path.join(sf.GRID, "..", "hw50_tune_2026-11-10", "runs", "noise_n1000")


def runs(noise):
    s = SEEDS[noise]
    at = os.path.join(TUNE, "soft25w02", NAME.format(noise, s))
    if noise == "half":
        m = {n: os.path.join(TUNE, f"mppi{n}", NAME.format(noise, s)) for n in ("50", "2000")}
    else:
        m = {n: os.path.join(GRID_RUNS, f"mppi{n}_" + NAME.format(noise, s)) for n in ("50", "2000")}
    at1000 = os.path.join(HW_TUNE, f"{noise}_wm10_tsf103", f"seed{s}")
    return at, at1000 if os.path.exists(os.path.join(at1000, "result.json")) else None, m


def plot(noise):
    at_path, at1000_path, mppi_paths = runs(noise)
    inline = INLINE.get(noise)
    fig, ax = plt.subplots(figsize=(7.2, 3.2 if inline else 3.6))
    n_max, above_txt = 0, []

    def note(key, text, color, **kw):
        xy_text, xy = inline[key]
        ax.annotate(text, xy=xy, xytext=xy_text, color=color, fontsize=7, arrowprops=dict(ARROW, color=color),
                    zorder=20, **kw)

    def draw(lt, result, color, label, z, learn_until=None):
        nonlocal n_max
        n = np.arange(1, len(lt) + 1)
        n_max = max(n_max, len(lt))
        y, above = np.minimum(lt, Y_MAX), lt > Y_MAX
        ax.plot(n, y, "-", color=color, lw=0.9, alpha=0.6, zorder=z)
        frozen = n > learn_until if learn_until else np.ones_like(n, bool)
        ax.plot(n[~frozen & ~above], y[~frozen & ~above], "o", ms=3.5, mfc="white", mec=color, mew=0.9, zorder=z)
        ax.plot(n[frozen & ~above], y[frozen & ~above], "o", color=color, ms=3, zorder=z, label=label)
        for i in np.flatnonzero(above):
            ax.plot(n[i], Y_MAX, marker="^", color=color, ms=5, zorder=z, clip_on=False)
        for k in mp.collision_laps(result):  # 1-based lap in which the collision happened
            if k <= len(lt):
                ax.plot(k, y[k - 1], marker="x", color="#d62728", ms=6, mew=1.4, zorder=z + 10)
        return above

    for n_s, color, z in (("2000", mp.MPPI_COLOR, 2),):  # MPPI 50 left out
        r = mp.load(mppi_paths[n_s])
        lt = np.array([l["lap_time"] for l in r["laps"]])
        med = np.median(lt[1:])
        coll = f", {r['n_collisions']} coll." if r["n_collisions"] else ""
        above = draw(lt, r, color, f"MPPI, {n_s} rollouts ({med:.2f} s){coll}", z)
        ax.hlines(med, 2, len(lt), color=color, lw=0.8, ls="--", alpha=0.9, zorder=1)
        if inline:
            note("mppi", f"MPPI,\n{n_s} rollouts\nmedian {med:.2f} s{coll}", color, ha="center", va="bottom")
        if above.any():
            above_txt.append(f"MPPI {n_s}: " + ", ".join(f"{t:.1f}" for t in lt[above]) + " s")

    ats = [(f"{mp.AT}, 50 rollouts", "50", at_path, mp.HW_COLOR, 3)]
    if at1000_path:
        ats.append((f"{mp.AT}, 1000 rollouts", "1000", at1000_path, AT1000_COLOR, 4))
    label_laps, inline_above = [], []
    for label, k, path, c, z in ats:
        hw = mp.load(path)
        lt = np.array([l["lap_time"] for l in hw["laps"]])
        f = hw["freeze_after_lap"]
        med = np.median(hw["after_freeze"]["lap_times"])
        n_af = hw["after_freeze"]["n_collisions"]
        coll = f", {n_af} coll. after freeze" if n_af else ""
        above = draw(lt, hw, c, f"{label} ({med:.2f} s){coll}", z, learn_until=f)
        ax.axvline(f + 0.5, color=c, lw=1.0, ls=":", zorder=1)
        if inline:
            note(k, f"{label}\nmedian after freeze {med:.2f} s{coll}", c, ha="center", va="top")
            right = inline[f"frozen {k}"][0][0] > f
            note(f"frozen {k}", f"memory frozen ({k})" + ("\nfilled: frozen →" if right else "\n← open: learning"),
                 c, ha="left" if right else "right", va="center")
            if above.any():
                inline_above.append((f"{k} rollouts: " + ", ".join(f"{t:.1f}" for t in lt[above]), c))
        else:
            dy = 0.45 * sum(abs(f - g) < 4 for g in label_laps)
            label_laps.append(f)
            ax.text(f + 0.8, Y_MAX - 0.08 - dy, f"frozen\n({k})", color=c, fontsize=6.5, va="top", ha="left")
        ax.hlines(med, f + 0.5, len(lt), color=c, lw=0.8, ls="--", alpha=0.9, zorder=1)
        if above.any():
            above_txt.insert(len(ats) - 1 if k == "1000" else 0,
                             f"{mp.AT} {k}: " + ", ".join(f"{t:.1f}" for t in lt[above]) + " s")
        print(f"noise {noise}, seed {SEEDS[noise]}, AT {k}: freeze after lap {f} ({hw['freeze_t']:.0f} s), first lap "
              f"{lt[0]:.1f} s, median after freeze {med:.2f} s, mean {np.mean(hw['after_freeze']['lap_times']):.2f} "
              f"+- {np.std(hw['after_freeze']['lap_times']):.2f}, coll {hw['n_collisions']} ({n_af} after freeze)")

    if inline and inline_above:  # coloured, one line per curve, in the gap right of the last MPPI spikes
        y = Y_MAX - 0.05
        ax.text(n_max, y, f"laps above {Y_MAX:g} s (▲), in s:", fontsize=7, va="top", ha="right", color="0.3")
        for i, (t, c) in enumerate(inline_above, start=1):
            ax.text(n_max, y - 0.17 * i, t, fontsize=7, va="top", ha="right", color=c)
    elif above_txt:
        ax.text(n_max, Y_MAX - 0.08, f"laps above {Y_MAX:g} s (▲):\n" + "\n".join(above_txt), fontsize=7,
                va="top", ha="right", color="0.3")
    y_min = Y_MIN - (0.5 if inline else 0)
    ax.set_ylim(y_min, Y_MAX)
    if inline:
        ax.set_xlim(0, n_max + 1)
        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap time (s)")
        fig.tight_layout()
        sf.save(fig, f"icra_laptimes_noise{noise}")
        return
    ax.plot([], [], "o", ms=3.5, mfc="white", mec="0.3", mew=0.9, label="learning (before freeze)")
    ax.plot([], [], "--", color="0.3", lw=0.8, label="median after freeze (MPPI: from lap 2)")
    if any(mp.collision_laps(mp.load(p)) for p in (at_path, *([at1000_path] if at1000_path else []), mppi_paths["2000"])):
        ax.plot([], [], "x", color="#d62728", ms=6, mew=1.4, ls="none", label="collision")
    ax.set_xlim(0, n_max + 1)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout()
    sf.save(fig, f"icra_laptimes_noise{noise}")
    for n_s, p in mppi_paths.items():
        r = mp.load(p)
        l = np.array([x["lap_time"] for x in r["laps"]])[1:]
        print(f"   MPPI {n_s}: median {np.median(l):.2f}, mean {l.mean():.2f} +- {l.std():.2f}, "
              f"coll {r['n_collisions']}, laps > median+1 s: {(l > np.median(l) + 1).sum()} of {len(l)}")


if __name__ == "__main__":
    for nz in SEEDS:
        plot(nz)

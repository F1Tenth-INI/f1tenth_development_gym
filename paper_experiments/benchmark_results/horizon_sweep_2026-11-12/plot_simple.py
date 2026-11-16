#!/usr/bin/env python3
"""Simple horizon plot: MPPI 2048 vs ATMPPI 1024, mean over seeds with +-1 std error bars, H = 15..35.

    python plot_simple.py                ->  figures/horizon_simple.{pdf,png}
    python plot_simple.py --noise half   ->  figures/horizon_simple_noisehalf.{pdf,png}: runs_noisehalf/ (observation
        noise +-0.025, 1000 s), with ATMPPI 1024 at steering noise 0.08 (paper) and 0.16

Same runs and metrics as ``horizon_sweep.py`` (icra2022, raceline, mu 0.5, no noise, 800 s, seeds 1..5):
crash rate per seed = collisions / (laps + collisions), ATMPPI after the freeze, MPPI from lap 2. A seed that
completed no lap and did not collide (stuck) counts as 100 % crashes and has no lap time; the number of such
seeds is written next to the point.
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import horizon_sweep as hs

HORIZONS = [15, 20, 25, 30, 35]
STYLE = {"mppi": ("MPPI, 2048 rollouts", "#444444", "s", 0.3),
         "atmppi1024": ("Auto-Tuning MPPI, 1024 rollouts", "#1f5fa8", "o", -0.3)}
STYLE_NOISE = {"mppi": ("MPPI, 2048 rollouts, steering noise 0.2", "#444444", "s", 0.3),
               "atmppi1024": ("Auto-Tuning MPPI, 1024 rollouts, steering noise 0.08", "#1f5fa8", "o", -0.3),
               "atmppi1024s16": ("Auto-Tuning MPPI, 1024 rollouts, steering noise 0.16", "#d95f02", "D", 0.0)}


def main(noise):
    hs.SETTING["noise"] = noise
    style = STYLE_NOISE if noise == "half" else STYLE
    rows = hs.collect()
    n_seeds = len({r["seed"] for r in rows})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.0, 4.6), sharex=True)
    for ctrl, (label, color, marker, dx) in style.items():
        crash_m, crash_s, lap_m, lap_s, stuck = [], [], [], [], []
        for h in HORIZONS:
            sub = [r for r in rows if r["controller"] == ctrl and r["horizon"] == h]
            crash = [100.0 if r["laps"] == 0 and r["collisions"] == 0 else r["crash_pct"] for r in sub]
            laps = [r["mean_lap"] for r in sub if np.isfinite(r["mean_lap"])]
            crash_m.append(np.mean(crash))
            crash_s.append(np.std(crash))
            lap_m.append(np.mean(laps) if laps else np.nan)
            lap_s.append(np.std(laps) if laps else np.nan)
            stuck.append(sum(1 for r in sub if r["laps"] == 0 and r["collisions"] == 0))
            print(f"{ctrl:10s} H={h:2d} n={len(sub)} crash {crash_m[-1]:5.1f} +- {crash_s[-1]:5.1f} %  "
                  f"lap {lap_m[-1]:6.2f} +- {lap_s[-1]:5.2f} s  stuck {stuck[-1]}")
        x = np.array(HORIZONS) + dx
        kw = dict(color=color, marker=marker, ms=5, lw=1.5, capsize=3, elinewidth=1.0, label=label)
        a1.errorbar(x, crash_m, yerr=crash_s, **kw)
        a2.errorbar(x, lap_m, yerr=lap_s, **kw)
        for xi, yi, k in zip(x, crash_m, stuck):
            if k:
                a1.annotate(f"{k}/{n_seeds} stuck" if n_seeds > 1 else "stuck", (xi, yi), xytext=(-7, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=7, color=color)
        for xi, yi, k in zip(x, lap_m, stuck):
            if k and np.isfinite(yi) and n_seeds > 1:
                a2.annotate(f"{n_seeds - k} seed" + ("s" if n_seeds - k > 1 else ""), (xi, yi), xytext=(-7, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=7, color=color)
    a1.set_ylabel("Crashes per lap (%)")
    a1.set_ylim(-5, 115)
    a1.set_xlim(10.5, 36.5)
    a2.set_ylabel("Mean lap time (s)")
    a2.set_xlabel("Horizon H (steps of 0.04 s)")
    a2.set_xticks(HORIZONS)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("icra2022, raceline, μ = 0.5, " + ("observation noise ±0.025" if noise == "half" else "no noise")
                 + (f"; mean ± std over {n_seeds} seeds" if n_seeds > 1 else "; single seed (seed 1)"), fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(hs.HERE / "figures" / f"horizon_simple{hs.tag()}.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", hs.HERE / "figures" / f"horizon_simple{hs.tag()}.{{pdf,png}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="off", choices=["off", "half"])
    main(ap.parse_args().noise)

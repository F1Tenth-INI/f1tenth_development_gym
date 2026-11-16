#!/usr/bin/env python3
"""Tracking horizon: ATMPPI (learning at H = 35, tracking at H after the freeze) vs MPPI at horizon H.

    python tracking_horizon.py            ->  figures/tracking_horizon.{pdf,png}: MPPI vs ATMPPI 1024
    python tracking_horizon.py --at50     ->  figures/tracking_horizon_at50.{pdf,png}: MPPI vs ATMPPI 50

Setting (the paper's noisy Fig. 2): icra2022, raceline, mu 0.5, observation noise +-0.025, 1000 s per run.
ATMPPI uses the Fig. 2 config wm10_tsf103 (as ATMPPI 1000 in Fig. 2) with steering noise 0.08. It learns with H = 35
(driving samples, window updates, stored sequences); after the permanent freeze the driving samples use H
(MPC_DRIVE_HORIZON + MPC_DRIVE_HORIZON_AFTER_FREEZE, hw50_tune_2026-11-10/drive_horizon.patch on the snapshot).

  ATMPPI 1024: runs_noisehalf/atmppi1024af_h<H>/seed1-6 (horizon_sweep.py, controller atmppi1024af)
  ATMPPI 50:   hw50_tune_2026-11-10/runs/hsweep (seeds 1-4) and runs/hsweep_seeds5to14 (seeds 5-14),
               configs h35/af30/af25/af20/af17 (atmppi50_drive_horizon.json)
  MPPI (reused): MPPI 2000 seeds 1-2 (mppi_tune_2026-11-10/runs/hsweep) + MPPI 2048 seed 1 (runs_noisehalf/mppi_h<H>),
               steering noise 0.2; at H = 17 only the two MPPI 2000 runs.

Counting. ATMPPI: from the end of the first completed lap with the frozen memory (the lap in which the memory
switches to its frozen state is not counted, nor a collision in it). MPPI: from lap 2. Crash % per run =
collisions / (laps + collisions); mean lap time of those laps. A run whose memory froze broken (> 30 % slower than
the MPPI 35 median, or collisions on at least half the counted laps, judged on its H = 35 run) is dropped for all
H: the learning phase does not depend on H, so a seed learns the same memory for every H. Plot: mean +- std.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BR = HERE.parent
HORIZONS = [17, 20, 25, 30, 35]
BROKEN_SLOW = 30.0
AT50_DIRS = [BR / "hw50_tune_2026-11-10" / "runs" / "hsweep", BR / "hw50_tune_2026-11-10" / "runs" / "hsweep_seeds5to14"]
AT50_NAME = {35: "h35", 30: "af30", 25: "af25", 20: "af20", 17: "af17"}


def stats(laps, coll):
    n = len(laps) + coll
    return {"laps": len(laps), "coll": coll, "crash": 100.0 * coll / n if n else 100.0,
            "lap": float(np.mean(laps)) if laps else np.nan}


def at_stats(path):
    """From the end of the first completed frozen lap."""
    r = json.load(open(path))
    frozen = [l for l in r["laps"] if l.get("frozen")]
    if len(frozen) < 2:
        return stats([], 0) | {"laps": 0, "crash": np.nan, "lap": np.nan}
    t0 = frozen[0]["t"]
    raw = r["collisions_raw"]  # one entry per contact event; collision_clusters groups them into collisions
    clusters = r.get("collision_clusters") or [[i] for i in range(len(raw))]
    coll = sum(1 for cl in clusters if raw[cl[0]]["t"] > t0)
    return stats([l["lap_time"] for l in frozen[1:]], coll)


def mppi_stats(path):
    r = json.load(open(path))
    return stats([l["lap_time"] for l in r["laps"]][1:], int(r["n_collisions"]))


def load(at50):
    mppi = {h: [] for h in HORIZONS}
    for h in HORIZONS:
        for p in sorted((BR / "mppi_tune_2026-11-10" / "runs" / "hsweep" / f"h{h}").glob("seed*/result.json")):
            mppi[h].append(mppi_stats(p))
        p = HERE / "runs_noisehalf" / f"mppi_h{h}" / "seed1" / "result.json"
        if p.exists():
            mppi[h].append(mppi_stats(p))
    ref = np.median([s["lap"] for s in mppi[35]])
    at = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        if at50:
            paths = [p for d in AT50_DIRS for p in (d / AT50_NAME[h]).glob("seed*/result.json")]
        else:
            paths = list((HERE / "runs_noisehalf" / f"atmppi1024af_h{h}").glob("seed*/result.json"))
        for p in paths:
            at[h][int(p.parent.name[4:])] = at_stats(p)
    broken = sorted(s for s, v in at[35].items()
                    if not v["lap"] <= ref * (1 + BROKEN_SLOW / 100) or 2 * v["coll"] >= max(v["laps"], 1))
    seeds = sorted(set.intersection(*(set(at[h]) for h in HORIZONS)) - set(broken))
    return mppi, at, seeds, broken, ref


def main(at50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mppi, at, seeds, broken, ref = load(at50)
    k = 50 if at50 else 1024
    print(f"ATMPPI {k}; MPPI 35 median lap {ref:.2f} s; seeds used {seeds} ({len(seeds)}), dropped (froze broken) "
          f"{broken}")
    print(f"{'H':>3s} | {'MPPI: runs, crash % (pooled), lap':36s} | ATMPPI {k}: seeds, crash % (pooled), clean, "
          "collisions / counted laps, lap")
    series = {"mppi": ([], [], [], []), "at": ([], [], [], [])}
    pool = lambda runs: 100 * sum(r["coll"] for r in runs) / max(sum(r["laps"] + r["coll"] for r in runs), 1)
    for h in HORIZONS:
        m = mppi[h]
        a = [at[h][s] for s in seeds]
        for key, runs in (("mppi", m), ("at", a)):
            cr = [r["crash"] for r in runs if np.isfinite(r["crash"])]
            lp = [r["lap"] for r in runs if np.isfinite(r["lap"])]
            series[key][0].append(np.mean(cr) if cr else np.nan)
            series[key][1].append(np.std(cr) if cr else np.nan)
            series[key][2].append(np.mean(lp) if lp else np.nan)
            series[key][3].append(np.std(lp) if lp else np.nan)
        print(f"{h:3d} | {len(m)} runs, {series['mppi'][0][-1]:5.1f} % ({pool(m):5.1f}), {series['mppi'][2][-1]:6.2f} s"
              f"{'':6s} | {len(a)} seeds, {series['at'][0][-1]:5.1f} % ({pool(a):5.1f}), "
              f"{sum(r['coll'] == 0 for r in a)}/{len(a)} clean, {sum(r['coll'] for r in a)} / "
              f"{sum(r['laps'] for r in a)}, {series['at'][2][-1]:6.2f} s")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.0, 4.6), sharex=True)
    style = {"mppi": ("MPPI, 2000–2048 rollouts, horizon H", "#444444", "s", 0.3),
             "at": (f"Auto-Tuning MPPI, {k} rollouts, tracking horizon H\n(learning at 35; {len(seeds)} seeds)",
                    "#1f5fa8", "o", -0.3)}
    for key, (label, color, marker, dx) in style.items():
        cm, cs, lm, ls = series[key]
        x = np.array(HORIZONS) + dx
        kw = dict(color=color, marker=marker, ms=5, lw=1.5, capsize=3, elinewidth=1.0, label=label)
        a1.errorbar(x, cm, yerr=cs, **kw)
        a2.errorbar(x, lm, yerr=ls, **kw)
    a1.set_ylabel("Crashes per lap (%)")
    a1.set_ylim(-5, 80)
    a2.set_ylabel("Mean lap time (s)")
    a2.set_xlabel("Horizon H (steps of 0.04 s)")
    a2.set_xticks(HORIZONS)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.legend(frameon=False, fontsize=7.5, loc="upper right")
    a1.set_title("icra2022, raceline, μ = 0.5, observation noise ±0.025; mean ± std over runs", fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    name = "tracking_horizon" + ("_at50" if at50 else "")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "figures" / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", HERE / "figures" / f"{name}.{{pdf,png}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--at50", action="store_true", help="ATMPPI 50 instead of 1024")
    main(ap.parse_args().at50)

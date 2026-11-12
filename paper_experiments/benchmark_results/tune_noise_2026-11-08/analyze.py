#!/usr/bin/env python3
"""Per config: collisions per 100 laps and lap time relative to the grid's MPPI 2000, split by mu.

Lap time per seed: mean of the laps after the permanent freeze (as in the paper figure). The reference is
the median of the five MPPI 2000 seeds of the grid of 2026-11-05 in the same setting.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "grid_2026-11-05", "summary.csv")
LATE = float(os.environ.get("LATE", 0))  # >0: score laps and collisions after this simulated time instead
UNTIL = float(os.environ.get("UNTIL", "inf"))  # with LATE: ... and before this time, so longer runs score like short ones
NOISE = os.environ.get("NOISE")  # only runs with this noise (on, half, off)
ONLY = os.environ.get("ONLY")  # regex on the run name, e.g. "_midline_"
RUN = re.compile(r"(.+)_(midline|raceline)_mu([\d.]+)_noise(on|half|off)_seed(\d+)")


def reference():
    ref = defaultdict(list)
    grid_hw = defaultdict(list)
    with open(GRID) as f:
        for r in csv.DictReader(f):
            key = (r["map"], r["wp"], r["mu"], r["noise"])
            if r["controller"] == "mppi" and r["rollouts"].startswith("2000"):
                ref[key].append(float(r["lap_mean"]))
            if r["controller"] == "hw" and r["rollouts"].startswith("50"):
                grid_hw[key].append(r)
    # Halved noise exists only here: the reference is the median of this folder's MPPI 2000 seeds.
    for path in glob.glob(os.path.join(HERE, "runs", "mppi2000", "*_noisehalf_*", "result.json")):
        m = RUN.match(os.path.basename(os.path.dirname(path)))
        ref[(m[1], m[2], m[3], m[4])].append(float(np.mean([l["lap_time"] for l in json.load(open(path))["laps"]])))
    return {k: float(np.median(v)) for k, v in ref.items()}, grid_hw


def summarize(name, rows, ref):
    """rows: list of (key, seed, lap_after_freeze, laps_after_freeze, collisions_after_freeze, collisions_total)."""
    out = {}
    for mu in ("0.5", "0.9", "all"):
        sel = [r for r in rows if mu == "all" or r[0][2] == mu]
        if not sel:
            continue
        rel = [100 * (r[2] / ref[r[0]] - 1) for r in sel if r[2] == r[2]]
        laps = sum(r[3] for r in sel)
        coll = sum(r[4] for r in sel)
        out[mu] = (len(sel), np.nanmedian(rel) if rel else np.nan, np.mean(np.array(rel) < 0) * 100, 100 * coll / max(laps, 1),
                   sum(1 for v in rel if v > 30))
    line = f"{name:12s}"
    for mu, (n, med, faster, cr, slow) in out.items():
        line += f" | mu {mu:3s}: n={n:2d} rel {med:+6.1f}% faster {faster:3.0f}% coll/100 {cr:5.2f} slow>30% {slow}"
    print(line)


def main():
    ref, grid_hw = reference()
    configs = sorted(c for c in os.listdir(os.path.join(HERE, "runs")) if not c.startswith("mppi"))
    tuned_keys = set()
    for c in configs:
        rows = []
        for path in glob.glob(os.path.join(HERE, "runs", c, "*", "result.json")):
            m = RUN.match(os.path.basename(os.path.dirname(path)))
            key = (m[1], m[2], m[3], m[4])
            if (NOISE and m[4] != NOISE) or key not in ref or (ONLY and not re.search(ONLY, m[0])):
                continue
            r = json.load(open(path))
            if r.get("error"):
                print("ERROR", path, r["error"][:200])
                continue
            af = r.get("after_freeze") or {}
            lt = af.get("lap_times") or [l["lap_time"] for l in r["laps"]]
            if LATE:
                # Same window for every config, frozen or not: laps and collisions after LATE seconds.
                lt = [l["lap_time"] for l in r["laps"] if l["t"] - l["lap_time"] >= LATE and l["t"] <= UNTIL]
                late_coll = sum(1 for cl in r["collision_clusters"]
                                if LATE <= r["collisions_raw"][cl[0]]["t"] <= UNTIL)
                rows.append((key, int(m[5]), float(np.mean(lt)) if lt else float("nan"), len(lt), late_coll,
                             int(r["n_collisions"])))
            else:
                # Laps and collisions with the frozen memory only.
                rows.append((key, int(m[5]), float(np.mean(lt)) if lt else float("nan"), len(af.get("lap_times") or []),
                             int(af.get("n_collisions", 0)), int(r["n_collisions"])))
            tuned_keys.add((key, int(m[5])))
        if rows:
            summarize(c, rows, ref)
    # The grid's HW 50 on the same settings and seeds, for comparison.
    rows = []
    for (key, seed) in tuned_keys:
        for r in grid_hw[key]:
            if int(r["seed"]) == seed:
                rows.append((key, seed, float(r["after_freeze_lap_mean"]), int(r["after_freeze_laps"]),
                             int(float(r["after_freeze_collisions"])), int(r["collisions"])))
    if rows:
        summarize("grid hw50", rows, ref)


if __name__ == "__main__":
    main()

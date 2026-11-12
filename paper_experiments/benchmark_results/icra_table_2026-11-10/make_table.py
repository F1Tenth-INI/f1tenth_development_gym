#!/usr/bin/env python3
"""Comparison table for the paper: icra2022, raceline start, mu 0.9, state noise 0.05, control delay 0.08 s,
1000 s simulated per run, seeds 3-6, working-copy code of 2026-11-10.

    python make_table.py   ->  icra_table.tex, icra_table.csv

Lap times: Auto-Tuning MPPI laps after the permanent freeze (the learned line), the others every lap but the
first (standing start). Collisions count every collision in the run (bench_run clusters, one per lap at most).
"""
import csv
import json
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BR = HERE.parent
ROWS = [  # label, run folder, rollouts shown
    ("Auto-Tuning MPPI", BR / "hw_tune_2026-11-10/runs/table/hw50", "50"),
    ("Auto-Tuning MPPI", BR / "hw_tune_2026-11-10/runs/confirm/base", "1000"),
    ("MPPI", BR / "mppi_tune_2026-11-10/runs/table/mppi2000_paper", "2000"),
    ("MPC (acados)", BR / "hw_tune_2026-11-10/runs/table/acados", "--"),
]


def stats(folder):
    results = [json.load(open(p)) for p in sorted(folder.glob("seed*/result.json"))]
    laps, coll, coll_after, total_laps, freeze = [], 0, 0, 0, []
    for r in results:
        all_laps = [l["lap_time"] for l in r["laps"]]
        if r.get("after_freeze") is not None and r.get("freeze_t") is not None:
            laps += r["after_freeze"]["lap_times"]
            freeze.append(r["freeze_after_lap"])
            coll_after += r["after_freeze"]["n_collisions"]
        else:
            laps += all_laps[1:]
            coll_after += r["n_collisions"]
        coll += r["n_collisions"]
        total_laps += len(all_laps)
    return {"runs": len(results), "laps": len(laps), "mean": statistics.mean(laps), "median": statistics.median(laps),
            "best": min(laps), "std": statistics.stdev(laps), "coll": coll, "coll_after": coll_after, "total_laps": total_laps,
            "coll_per_100": 100.0 * coll / total_laps, "freeze": statistics.mean(freeze) if freeze else None}


def main():
    rows = [(label, n, stats(folder)) for label, folder, n in ROWS]
    with open(HERE / "icra_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "rollouts", "runs", "laps_scored", "mean_s", "median_s", "best_s", "std_s",
                    "collisions", "collisions_scored_laps", "laps_total", "collisions_per_100_laps", "freeze_after_lap"])
        for label, n, s in rows:
            w.writerow([label, n, s["runs"], s["laps"], f"{s['mean']:.3f}", f"{s['median']:.3f}", f"{s['best']:.3f}",
                        f"{s['std']:.3f}", s["coll"], s["coll_after"], s["total_laps"], f"{s['coll_per_100']:.2f}",
                        "" if s["freeze"] is None else f"{s['freeze']:.1f}"])
    best_mean = min(s["mean"] for _, _, s in rows)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{icra2022, $\mu=0.9$, state noise and \SI{80}{ms} control delay: lap times and collisions over"
        r" 4 runs of \SI{1000}{s} each. Auto-Tuning MPPI laps after the permanent freeze, all others every lap"
        r" but the first. Collisions: whole run / laps scored (for Auto-Tuning MPPI, after the freeze).}",
        r"  \label{tab:icra_comparison}",
        r"  \begin{tabular}{lrrrrrrr}",
        r"    \toprule",
        r"    Controller & Rollouts & Mean (s) & Median (s) & Best (s) & Std (s) & \multicolumn{2}{c}{Collisions} \\",
        r"    \cmidrule(lr){7-8}",
        r"    & & & & & & all / laps & scored laps \\",
        r"    \midrule",
    ]
    for label, n, s in rows:
        mean = f"{s['mean']:.2f}"
        if abs(s["mean"] - best_mean) < 1e-9:
            mean = rf"\textbf{{{mean}}}"
        lines.append(f"    {label} & {n} & {mean} & {s['median']:.2f} & {s['best']:.2f} & {s['std']:.2f} & "
                     f"{s['coll']} / {s['total_laps']} & {s['coll_after']} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (HERE / "icra_table.tex").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

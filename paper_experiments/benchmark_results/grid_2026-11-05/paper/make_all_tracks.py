#!/usr/bin/env python3
"""The all-tracks figure (halved noise, Auto-Tuning MPPI 50 = soft25w02), without the rest of make_sim_figures.main
(which would overwrite figures/icra_laptimes.*, now made by make_icra_laptimes_rollouts_v2.py).

    python make_all_tracks.py   ->  figures/all_tracks.{pdf,png}  and  figures/all_tracks_1000.{pdf,png}

``all_tracks_1000`` adds Auto-Tuning MPPI 1000 (config wm10_tsf103 of the lap-time plots,
tune_noise_2026-11-08/runs/wm10_tsf103). Broken freezes are dropped as for 50 rollouts (``drop_broken``).
"""
import os

import make_sim_figures as sf

AT1000_DIR = os.path.join(sf.TUNE_RUNS, "wm10_tsf103")
AT1000_COLOR = "#08306b"  # as in the lap-time plots
CONTROLLERS_1000 = {
    ("hw", "50"): ("Auto-Tuning MPPI, 50 rollouts", sf.mp.HW_COLOR, -0.27),
    ("hw", "1000"): ("Auto-Tuning MPPI, 1000 rollouts", AT1000_COLOR, -0.09),
    ("mppi", "50"): ("MPPI, 50 rollouts", "#b0b0b0", 0.09),
    ("mppi", "2000"): ("MPPI, 2000 rollouts (reference, 0 %)", sf.mp.MPPI_COLOR, 0.27),
}


def main():
    sf.HW_CONFIG = sf.PAPER_CONFIG
    rows = sf.drop_broken(sf.half_noise_rows())
    sf.save(sf.all_tracks_figure(rows), "all_tracks")
    sf.table_numbers(rows)
    if os.path.isdir(AT1000_DIR):
        rows = sf.half_noise_rows()
        sf.replace_runs(rows, ("hw", "1000"), AT1000_DIR, ("half", "off"))
        n = sum(len(v) for k, v in rows.items() if k[:2] == ("hw", "1000"))
        rows = sf.drop_broken(rows)
        kept = sum(len(v) for k, v in rows.items() if k[:2] == ("hw", "1000"))
        print(f"\nAuto-Tuning MPPI 1000: {n} of 160 runs finished, {kept} kept")
        sf.save(sf.all_tracks_figure(rows, CONTROLLERS_1000), "all_tracks_1000")


if __name__ == "__main__":
    main()

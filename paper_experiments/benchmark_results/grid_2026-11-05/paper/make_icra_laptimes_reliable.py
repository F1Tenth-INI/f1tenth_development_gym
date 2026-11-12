#!/usr/bin/env python3
"""``icra_laptimes`` of the paper (compact, soft25w02, seed 1) with the MPPI 2000 run replaced by one
with steering sampling noise 0.2 instead of 0.1 (same cost, same code snapshot otherwise).

    python make_icra_laptimes_reliable.py   ->  figures/icra_laptimes_mppi_reliable.{pdf,png}
"""
import os

import make_sim_figures as sf

mp = sf.mp
SEED = 1
MPPI_RELIABLE = os.path.join(sf.GRID, "..", "mppi_tune_2026-11-10", "runs", "plotrun", "ns_s02", f"seed{SEED}")

mp.configure("off", SEED, 50)
# HW: soft25w02 (paper) or, with HW_TUNED=1, soft25w02 + window margin 0.10/0.08 + target speed factor 1.03
HW_TUNED = os.environ.get("HW_TUNED") == "1"
mp.HW_RUN = (os.path.join(sf.GRID, "..", "hw50_tune_2026-11-10", "runs", "r2", "wm10_tsf103", f"seed{SEED}") if HW_TUNED
             else os.path.join(sf.TUNE_RUNS, "soft25w02", f"icra2022_raceline_mu0.5_noiseoff_seed{SEED}"))
hw, mppi = mp.load(mp.HW_RUN), mp.load(MPPI_RELIABLE)
assert hw["n_collisions"] == 0 and mppi["n_collisions"] == 0, "the paper figure assumes a collision-free run"
sf.save(mp.lap_time_plot(hw, mppi, show_collisions=False, title=False, figsize=(3.5, 2.5), legend=False, compact=True),
        "icra_laptimes_hw_tuned" if HW_TUNED else "icra_laptimes_mppi_reliable")
print(f"HW 50: freeze after lap {hw['freeze_after_lap']} ({hw['freeze_t']:.0f} s), {hw['after_freeze']['lap_stats']['mean']:.2f} "
      f"+- {hw['after_freeze']['lap_stats']['std']:.2f} s after the freeze")
laps = [l["lap_time"] for l in mppi["laps"]]
print(f"MPPI 2000 (steering noise 0.2): {len(laps)} laps, mean {sum(laps) / len(laps):.2f} s, max after lap 1 "
      f"{max(laps[1:]):.2f} s")

#!/usr/bin/env python3
"""Fig. 2 setting (icra2022, raceline, mu 0.5, tuned configs) with observation noise fitted to a physical recording.

    python make_icra_laptimes_recnoise.py [SEED]   ->  figures/icra_laptimes_recnoise_seed<SEED>.{pdf,png}

Noise: AnalyseData/NoiseAnalysis (IPZ20 recording, 2026-01-16), per-state sigma; the simulator's uniform noise
half-width is sqrt(3) sigma, so the standard deviation matches. Runs: hw50_tune runs/recnoise_n{50,1000},
mppi_tune runs/recnoise (MPPI 2000, steering noise 0.2).
"""
import os
import sys

import make_icra_laptimes_rollouts_v2 as v2
from make_icra_laptimes_rollouts import BR

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
hw = os.path.join(BR, "hw50_tune_2026-11-10", "runs")
runs = {k: os.path.join(hw, f"recnoise_n{k}", "rec_wm10_tsf103", f"seed{seed}") for k in (50, 1000)}
mppi = os.path.join(BR, "mppi_tune_2026-11-10", "runs", "recnoise", "rec_ns_s02", f"seed{seed}")
v2.main(runs, mppi, f"icra_laptimes_recnoise_seed{seed}", y_max=float(os.environ.get("Y_MAX", 16.0)))

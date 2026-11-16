# Fig. 3 re-run (2026-11-12/13)

The paper's Fig. 3 (`all_tracks.pdf`): 4 tracks x {centerline, raceline} x mu {0.5, 0.9}, half observation noise,
1000 s per run, seeds 0..4, the same 80 runs for every controller. Here with **1024 / 2048 rollouts** instead of
1000 / 2000. Script, settings and metrics: `fig3.py` (docstring).

```bash
python fig3.py run --workers 30     # 320 runs on snapshot aac2064a, ~5.3 h wall with 30 workers (MPPI on CPU)
python fig3.py plot                 # figures/all_tracks_repro.{pdf,png} + numbers
python fig3.py plot --source paper  # the paper's runs, drawn by the same script: figures/all_tracks_paper.*
```

The plotting script of the paper's figure is not in the repository; `fig3.py plot` redraws its layout.
Check: `--source paper` gives the paper's figure point for point and the paper's numbers (collisions per 100 laps
0.11 / 0.10 / 0.29 for ATMPPI 50 / MPPI 2000 / MPPI 50, MPPI 50 +15.8 %, ATMPPI 1000 3.0 % faster than ATMPPI 50,
faster in 15 of 16 settings, two ATMPPI runs dropped).

Wall time per run (30 in parallel): ATMPPI 50 450 s, ATMPPI 1024 3700 s, MPPI 50 200 s, MPPI 2048 2800 s.
One ATMPPI 1024 run (icra2022, centerline, mu 0.9, seed 3) died with a segfault at t = 967 s (log kept in
`segfault_at1024_icra2022_midline_mu0.9_seed3.log`) and was re-run alone.

## Result (paper vs re-run)

| | paper (1000 / 2000) | re-run (1024 / 2048) |
|---|---|---|
| ATMPPI 50 vs MPPI ref, median | -5.6 % | -5.3 % |
| per track (icra2022, RCA1, IPZ50, hangar3) | -12.8, -10.0, -5.6, -8.8 | -12.2, -10.4, -4.5, -8.4 |
| centerline / raceline start | -22.4 / -1.8 % | -21.1 / -1.8 % |
| ATMPPI 50 faster than ref (seed mean) | 16 of 16 settings | 15 of 16 |
| ATMPPI large vs MPPI ref, median (pooled over settings) | -10.9 % | -16.1 % |
| ATMPPI large vs ATMPPI 50 (per setting) | -3.0 %, faster in 15 of 16 | -2.4 %, faster in 14 of 16 |
| MPPI 50 vs ref | +15.8 % | +15.0 % |
| collisions / 100 laps: AT50, AT large, MPPI 50, MPPI ref | 0.11, 0.07, 0.29, 0.10 | 0.30, 0.12, 0.19, 0.10 |
| ATMPPI seeds dropped (froze broken) | 1 + 1 | 3 (AT50, all IPZ50 centerline) + 1 |
| median freeze time ATMPPI 50 / large | 288 / 213 s | 297 / 220 s |

All 320 runs finished (the segfaulted run re-ran cleanly). The pooled median of ATMPPI large jumps between -13.8 and
-16.1 % with that one run: it falls between the centerline and raceline clusters (see the conventions in
`../../README.md`); compare per setting instead.

The lap-time panel reproduces: same ordering, same clusters, same magnitudes. The differences are in the rare
events. ATMPPI 50 collides more after the freeze on icra2022 (15 collisions in 6 of its 20 runs, against 2 in the
paper), and three ATMPPI 50 seeds on IPZ50 centerline froze a slow memory (51 to 113 % slower; paper: one).
Runs are not bitwise reproducible (other machine, MPPI on CPU instead of GPU), and with ~6000 laps per controller a
few runs decide the collision rate, so the collision bars are noisy at five seeds.

## Check: more steering sampling noise for ATMPPI 50 on icra2022 (`noise_check.py`)

Paper config soft25w02 with `MPC_DRIVE_NOISE[0]` raised from 0.08 to 0.12 and 0.16 (like MPPI's 0.1 -> 0.2 of
Fig. 2), icra2022, 4 settings x seeds 0..4, half observation noise, 1000 s. Lap time vs the MPPI 2048 median.

| steering std | collisions after freeze | per 100 laps | median lap vs MPPI 2048 |
|---|---|---|---|
| 0.08 (paper, Fig. 3 re-run) | 15 in 888 laps (6 of 20 runs) | 1.69 | -12.2 % |
| 0.12 | 6 in 977 laps (5 runs) | 0.61 | -13.0 % |
| 0.16 | 5 in 983 laps (3 runs) | 0.51 | -13.6 % |

(Paper's own icra2022 runs at 0.08: 2 collisions, 0.22 per 100 laps.) More sampling noise cuts the collisions
to about a third and is not slower (slightly faster, and more frozen laps because fewer resets). It does not
reach the paper's 0.22, and with 20 runs each a difference of a few collisions is within seed-to-seed spread.

Fig. 3 with these runs in place of ATMPPI 50 on icra2022 (`fig3.py plot --icra-at50 0.12|0.16`):
`figures/all_tracks_repro_icra_at50_s0.12.*`, `..._s0.16.*`. ATMPPI 50 then has 0.15 / 0.13 collisions per 100
laps over all tracks (0.30 with 0.08; paper 0.11). ATMPPI 1024 on icra2022 (still at 0.08) is then the highest bar.

# Horizon sweep (2026-11-12)

Shortest rollout horizon H at which MPPI (2048 rollouts) and ATMPPI (256 rollouts) still drive reliably.
Setting of Fig. 2: icra2022, raceline start, mu 0.5, no noise. Seeds 1 and 2, 800 s simulated time per run.
Script, configs and metric definitions: `horizon_sweep.py` (docstring). Uses the snapshots of
`../fig2_repro_2026-11-12` (`collect_fig2.py setup`).

```bash
python horizon_sweep.py run      # 2 seeds: 24 runs, ~21 min wall with 24 in parallel (MPPI H=50 is the slowest)
python horizon_sweep.py plot     # figures/horizon_sweep.{pdf,png}, summary.csv, table on stdout
```

ATMPPI: `MPC_TRAJECTORY_LENGTH = MPC_FREE_TRAJECTORY_LENGTH = H` (drive samples, window samples and stored
sequences all H). MPPI: `p:horizon = H`, steering noise 0.2 (`ns_s02`). Crash rate = collisions / (laps +
collisions); ATMPPI after the freeze, MPPI from lap 2.

## Result (5 seeds, with ATMPPI 1024; 2026-11-13)

`python horizon_sweep.py run --seeds 1,2,3,4,5` (ATMPPI 1024 added as `atmppi1024_h<H>`; seeds 3 to 5 added for the
other two; 66 new runs, ~1.5 h wall with 30 in parallel). Same setting and 800 s per run as below.
`figures/horizon_sweep.{pdf,png}`, `summary.csv`. Crash % pooled over the 5 seeds (collisions / (laps + collisions)),
"clean" = seeds with no collision, lap = median of the seed means.

| H | MPPI 2048 | ATMPPI 256 (after freeze) | ATMPPI 1024 (after freeze) |
|---|---|---|---|
| 50 | 2.3 %, clean 0/5, 15.07 s | — | — |
| 35 | **0 %, clean 5/5, 14.52 s** | 7.4 %, clean 4/5, 14.44 s | **0 %, clean 5/5, 14.11 s** |
| 30 | 25 %, clean 0/5, 15.26 s | 12.6 %, clean 3/5, 14.83 s | 6.0 %, clean 2/5, 14.21 s |
| 25 | 53 %, clean 0/5, 16.64 s | 19.5 %, clean 1/5, 14.25 s | 4.1 %, clean 3/5, 14.22 s |
| 20 | 54 %, clean 0/5, 17.71 s | 9.3 %, clean 4/5, but 18 to 35 s laps | 3.4 %, clean 4/5, but 19 to 39 s laps |
| 15 | 59 %, clean 0/5, 18.42 s | 4 of 5 seeds stuck (no lap) | 4 of 5 stuck, 1 at 28 s laps |
| 10 | — | all stuck | all stuck |

- ATMPPI 1024 is the only ATMPPI that is clean at H = 35 in all 5 seeds, and the fastest (14.11 s vs 14.52 s for
  MPPI). With 5 seeds, ATMPPI 256 is not fully reliable even at the paper's H = 35: seed 3 froze a memory that
  crashes in 37.5 % of the laps (the 2-seed sweep did not show this).
- Below 35 every MPPI seed crashes (25 % at H = 30, over 50 % at 25 and below). ATMPPI degrades more gently:
  ATMPPI 1024 crashes 4 to 6 % at H = 25 to 30 with the lap time unchanged (14.2 s), but 2 to 3 of 5 seeds
  have collisions, so it is not reliable there either.
- At H = 20 ATMPPI barely crashes but freezes slow memories (seed means of 18 to 39 s); at 15 and below it gets stuck.
  More rollouts do not move this limit: the horizon, not the sample count, sets it.
- So: shortest reliable horizon 35 for MPPI 2048 and ATMPPI 1024; with a few percent of crashes ATMPPI 1024 still
  drives at full speed down to H = 25, where MPPI already crashes in half the laps.

## Result (2 seeds)

| H | MPPI 2048: crash %, mean lap | ATMPPI 256 (after freeze): crash %, mean lap |
|---|---|---|
| 50 | 1.9 / 3.8, 15.1 s | — |
| 35 | 0 / 0, 14.52 s | 0 / 0, 14.41 s |
| 30 | 22 / 24, 15.2 s | 0 / 0, 14.08 / 14.83 s |
| 25 | 52 / 56, 16.7 s | 2.5 / 41, 14.20 / 14.55 s |
| 20 | 53 / 56, 17.8 s | 0 / 0, but 35 / 21 s laps (slow frozen memory) |
| 15 | 63 / 100, 18.6 s | 0 % at 22 s laps / no lap completed |
| 10 | — | no lap completed (both seeds, no collision: stuck) |

- MPPI 2048 drives reliably only at H = 35. At 30 it already crashes in about a quarter of the laps, and
  H = 50 is slower (15.1 s) with a few crashes.
- ATMPPI 256 is reliable down to H = 30 (both seeds clean, but seed 2 is 0.75 s slower). At H = 25 one seed
  is clean and fast (14.20 s), the other crashes in 41 % of the frozen laps. At 20 and below it does not
  crash, but the memory freezes at 21 to 35 s laps or the car never finishes a lap.
- So the shortest reliable horizon is 35 for MPPI and 30 for ATMPPI, and 25 is borderline for ATMPPI.
  Two seeds are not enough to separate 25 and 30 for ATMPPI.
- With noise (earlier sweep in `hw50_tune_2026-11-10`, `opt25`): ATMPPI with all horizons at 25 crashed a
  lot on icra2022, consistent with 25 being borderline.

## RCA1 (2 seeds, 1000 s; `--map RCA1`, runs_RCA1/, figures/horizon_sweep_RCA1.*, summary_RCA1.csv)

| H | MPPI 2048: crash %, mean lap | ATMPPI 256 (after freeze): crash %, mean lap |
|---|---|---|
| 50 | 0 %, 22.30 s | — |
| 35 | 0 %, 22.01 s | 0 %, 21.04 / 21.48 s |
| 30 | 0 %, 22.66 s | 0 %, 21.33 / 20.99 s |
| 25 | 0 %, 23.92 s | 0 %, 21.58 s / 58 s laps (froze only at 866 s) |
| 20 | 0 %, 25.76 s | 0 %, 30.3 / 26.5 s |
| 15 | 21 / 27 %, 26.1 s | no lap completed |
| 10 | — | no lap completed |

On RCA1 MPPI does not crash down to H = 20, but it slows down steadily
below 35 (+0.65 s at 30, +1.9 s at 25, +3.7 s at 20). ATMPPI is fastest at 30 to 35 and still beats MPPI 35 at
H = 30 (both seeds); at 25 one seed is fine, the other learns too slowly; at 20 and below it fails by
driving slowly or getting stuck, not by crashing.

**Harness fix (2026-11-12):** `mppi_tune.py` now sets `Settings.MAP_NAME` before importing the MPPI planner.
`WaypointUtils.__init__` has the map as default arguments, which Python evaluates at import, so with any map
other than the Settings default (icra2022) the planner followed icra2022's waypoints. All icra2022 runs are
unaffected; the first RCA1 MPPI runs (0 laps each) were deleted and re-run.

`python horizon_sweep.py plot --new`: the same plot with only the runs of 2026-11-13 (ATMPPI 1024 seeds 1-5, MPPI and
ATMPPI 256 seeds 3-5): `figures/horizon_sweep_new.*`, `summary_new.csv`. `plot` without `--new` pools all seeds.
`python horizon_sweep.py plot --controllers mppi,atmppi1024`: only MPPI 2048 vs ATMPPI 1024, 5 seeds each:
`figures/horizon_sweep_mppi_vs_atmppi1024.*`.
`python plot_simple.py`: MPPI 2048 vs ATMPPI 1024, mean +- std over the 5 seeds, H = 15 to 35, no per-seed markers:
`figures/horizon_simple.*` (stuck seeds count as 100 % crashes and are labelled).

## With observation noise, single seed (2026-11-13; proof of concept)

`python horizon_sweep.py run --noise half --seeds 1 --sim-seconds 1000 --only mppi,atmppi1024,atmppi1024s16
--horizons 35,30,25,20,15` (15 runs, ~30 min wall), then `python plot_simple.py --noise half`
(`figures/horizon_simple_noisehalf.*`). Observation noise +-0.025 (the paper's), runs in `runs_noisehalf/`.
MPPI 2048 with steering noise 0.2 (paper, icra2022); ATMPPI 1024 with the paper's 0.08 and with 0.16
(`atmppi1024s16`, `MPC_DRIVE_NOISE = [0.16, 1.5]`). Seed 1 only, crash % / mean lap after the freeze (MPPI from lap 2):

| H | MPPI 2048 (0.2) | ATMPPI 1024 (0.08) | ATMPPI 1024 (0.16) |
|---|---|---|---|
| 35 | 0 %, 14.56 s | 2.8 %, 14.76 s | 0 %, 14.06 s |
| 30 | 26 %, 15.32 s | 14 %, 14.29 s | 0 %, 13.85 s |
| 25 | 51 %, 16.88 s | 0 %, 14.21 s | 18 %, 14.37 s |
| 20 | 52 %, 17.86 s | 0 %, 20.9 s | 0 %, 20.1 s |
| 15 | 57 %, 18.76 s | stuck | 0 %, 21.4 s |

### H = 35 for learning, H = 25 for driving after the freeze (3 seeds, observation noise)

`python horizon_sweep.py run --noise half --seeds 1,2,3 --sim-seconds 1000 --only atmppi1024af --horizons 25`
(needs `hw50_tune_2026-11-10/drive_horizon.patch` on the snapshot; `atmppi1024af`: `MPC_DRIVE_HORIZON = H` with
`MPC_DRIVE_HORIZON_AFTER_FREEZE`, everything else at 35, steering noise 0.08). ATMPPI 1024, icra2022 raceline mu 0.5:

| seed | freeze | laps after freeze | collisions after freeze | mean lap after freeze | last laps before freeze |
|---|---|---|---|---|---|
| 1 | 472 s (lap 27) | 35 | 0 | 14.87 s | 14.8 to 14.9 s |
| 2 | 300 s (lap 20) | 49 | 0 | 14.21 s | 13.9 to 14.3 s |
| 3 | 449 s (lap 18) | 18 | 0 | 30.2 s | 19 to 28 s |

No collision in any of the 3 runs, and the drive horizon 25 keeps the lap time of the frozen memory (seeds 1, 2).
Seed 3 learned a slow memory already before the freeze (with H = 35, i.e. not caused by the short drive horizon);
the paper's rule drops such runs (> 30 % slower than MPPI). For comparison, seed 1 with H = 35 throughout
(`atmppi1024_h35`) has the identical learning phase and 1 collision, 14.76 s after the freeze.

## Tracking horizon: ATMPPI 50 (learning at 35) vs MPPI (2026-11-13) — `tracking_horizon.py`

`python tracking_horizon.py` -> `figures/tracking_horizon.*`. icra2022 raceline mu 0.5, observation noise +-0.025,
1000 s. ATMPPI 50 learns at H = 35 and tracks with H after the freeze (configs `atmppi50_drive_horizon.json`, the
af* configs of hw50_tune_2026-11-10); seeds 1-4 from `hw50_tune_2026-11-10/runs/hsweep`, seeds 5-14 new
(`runs/hsweep_seeds5to14`, 50 runs, ~25 min). MPPI reused: 2000 (2 seeds, mppi_tune hsweep) + 2048 (seed 1).
Seed 4 froze a broken memory (dropped for all H; same learning phase for every H). Crash % = mean over runs (pooled):

| H | MPPI (3 runs) | ATMPPI 50 (13 seeds) | ATMPPI clean seeds |
|---|---|---|---|
| 35 | 0 %, 14.56 s | 2.4 % (2.4), 14.38 s | 7/13 |
| 30 | 28 %, 15.37 s | 3.3 % (3.6), 14.25 s | 10/13 |
| 25 | 52 %, 16.86 s | 3.9 % (3.8), 14.15 s | 8/13 |
| 20 | 53 %, 17.93 s | 10.5 % (8.7), 14.25 s | 7/13 |
| 17 | 54 %, 18.45 s | 15.7 % (11.3), 14.45 s | 5/13 |

Collisions after the freeze per seed: at H = 35 and 25 most seeds have 0 or 1 (35: s2 s3 s5 s6 s9 one each;
25: s3 s7 s9 s13 one each). Most of the rate comes from seed 14, whose memory crashes at every horizon
(10/40 laps at 35, 20/28 at 25). Without seed 14 (and the dropped seed 4): 5 collisions at H = 35, 4 at H = 25, in about 550 frozen laps each.
Shortening the tracking horizon to 25 keeps the crash rate of 35 and makes the laps faster (14.15 vs 14.38 s);
MPPI at 25 crashes in half the laps. Below 25 (20, 17) more seeds fail.

## ATMPPI 1024, learning at 35, tracking at 25: config options (6 seeds, observation noise; 2026-11-13)

`python horizon_sweep.py run --noise half --seeds 1,2,3,4,5,6 --sim-seconds 1000 --only
atmppi1024af,atmppi1024afs16,atmppi1024afsoft --horizons 25` (runs_noisehalf/). Same seeds for all three, so the
comparison is per seed. Collisions / laps after the freeze, mean lap after the freeze:

| seed | Fig. 2 config `wm10_tsf103` (`atmppi1024af`) | + steering noise 0.16 (`atmppi1024afs16`) | paper config `soft25w02` (`atmppi1024afsoft`) |
|---|---|---|---|
| 1 | 0/35, 14.87 s | 0/46, 13.83 s | 0/22, 29.3 s (slow memory) |
| 2 | 0/49, 14.21 s | 0/48, 13.74 s | 0/53, 14.16 s |
| 3 | 0/18, 30.2 s (slow memory) | 0/46, 14.12 s | 4/37, 14.39 s |
| 4 | 0/49, 14.02 s | 12/35, 14.20 s | 0/46, 14.07 s |
| 5 | 0/50, 13.99 s | 1/49, 14.00 s | 0/52, 14.23 s |
| 6 | segfault at t = 697 s (re-run) | 0/48, 13.90 s | 1/51, 14.28 s |
| total | **0 collisions in 201 laps** (no collision at all in the 5 runs, also not while learning) | 13 in 272 | 5 in 261 |

ATMPPI 1024 with the Fig. 2 config tracks crash-free at H = 25 (5 of 5 seeds; one learned a slow memory before
the freeze, at H = 35). Neither option helps: more steering noise gives faster laps but one seed with 12
collisions, and soft25w02 is slower with collisions in 2 seeds. Also ATMPPI 100 (af25, seed 1,
hw50_tune_2026-11-10/runs/hsweep_n100): 0 collisions in the whole run, 50 frozen laps at 14.02 s.

## Tracking horizon with ATMPPI 1024 (Fig. 2 config), icra2022, 6 seeds (2026-11-13)

`python tracking_horizon.py` -> `figures/tracking_horizon.*` (now ATMPPI 1024 vs MPPI; `--at50` for ATMPPI 50).
Counting changed: ATMPPI from the end of the first completed frozen lap (the freeze lap and a crash in it are not
counted), collisions = clustered events (`collision_clusters`). Seed 3 froze a slow memory and is dropped for all H.
Segfaults (exit -11, logs kept): atmppi1024af_h25 seed 6 and atmppi1024af_h35 seed 5, both re-run.

| H | MPPI 2000/2048 (3 runs) | ATMPPI 1024, tracking H (5 seeds) |
|---|---|---|
| 35 | 0 %, 14.56 s | 1 collision / 229 laps, 4/5 clean, 14.26 s |
| 30 | 28 %, 15.37 s | **0 / 228, 5/5 clean**, 14.25 s |
| 25 | 52 %, 16.86 s | 9 / 223 (all seed 6), 4/5 clean, 14.29 s |
| 20 | 53 %, 17.93 s | 42 / 192, 1/5 clean, 14.56 s |
| 17 | 54 %, 18.45 s | 60 / 147, 0/5 clean, 15.04 s |

Seed 6 is clean at 35 and 30 and crashes only at 25, so those crashes come from the short tracking horizon.

## RCA1, Fig. 3 settings, 1 seed (2026-11-13)

`python horizon_sweep.py run --map RCA1 --noise half --seeds 1 --sim-seconds 1000 --only mppi01,atmppi1024af
--horizons 35,30,25,20,17` (runs_RCA1_noisehalf/, ~25 min). RCA1 raceline mu 0.5, observation noise +-0.025.
MPPI 2048 with the Fig. 3 steering noise 0.1 (`mppi01`), horizon H; ATMPPI 1024 (Fig. 3 config of ATMPPI 1000,
wm10_tsf103) learning at 35, tracking at H after the freeze (452 s, the same for every H). Collisions / laps:

| H | MPPI 2048 (from lap 2) | ATMPPI 1024 (after the first frozen lap) |
|---|---|---|
| 35 | 0 / 44, 22.15 s | 0 / 24, 21.26 s |
| 30 | 0 / 42, 22.84 s | 0 / 24, 21.16 s |
| 25 | 0 / 40, 24.17 s | 0 / 24, 21.10 s |
| 20 | 5 / 35, 25.70 s | 0 / 24, 21.05 s |
| 17 | 2 laps, then a collision at 74 s and stuck for the rest of the run | 0 / 24, 21.06 s |

On RCA1 the tracking horizon of ATMPPI can go down to 17 without a crash or a loss of lap time; MPPI loses 2 s per
lap at 25, crashes at 20 and fails at 17.

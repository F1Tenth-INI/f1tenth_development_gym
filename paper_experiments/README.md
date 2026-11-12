# Paper experiments: Auto-Tuning MPPI for Autonomous Racing (L4DC 2026)

Scripts, configs and results behind the simulation figures and numbers of the paper. The data for every
figure is included, so the plots can be regenerated without running any simulation.

In the code, Auto-Tuning MPPI (ATMPPI) is the **HirarchicalPlanner**, called `hw` (planner:
`TrainingLite/hwm/hirarchical_planner.py`, settings: `utilities/Settings.py`, prefixes `HWM_` and `MPC_`).
The MPPI baseline is **mppi-lite-jax** (`Control_Toolkit_ASF/Controllers/MPPILite/`).

## Quick start: regenerate the paper figures

Run everything from the repository root with the repo's Python environment (conda env `f1t`, see the main
README). Figures are written to `paper_experiments/benchmark_results/grid_2026-11-05/paper/figures/`.

```bash
cd paper_experiments/benchmark_results/grid_2026-11-05/paper
python make_icra_laptimes_rollouts_v2.py   # Fig. 2: icra_laptimes_rollouts_v2.pdf
python make_all_tracks.py                  # Fig. 3: all_tracks.pdf, and all_tracks_1000.pdf (with ATMPPI 1000)
python make_icra_laptimes_noise.py         # icra_laptimes_noise{half,on}.pdf: Fig. 2 setting with observation noise
python make_icra_laptimes_recnoise.py 3    # icra_laptimes_recnoise_seed3.pdf: noise fitted to a car recording
cd ../../icra_table_2026-11-10 && python make_table.py   # comparison table (mu 0.9, noise on), printed as LaTeX
```

Check the setup with `make_all_tracks.py`: it must print `Auto-Tuning MPPI 50 faster than MPPI 2000 (seed
mean) in 31 of 32 settings` and per-track gains of 9.1 to 13.1 %.

Figure names in the paper vs. here:

| Paper | File here | Script |
|---|---|---|
| Fig. 2 (`icra_laptimes.pdf` in the paper) | `icra_laptimes_rollouts_v2.pdf` | `make_icra_laptimes_rollouts_v2.py` |
| Fig. 3 (`all_tracks.pdf`) | `all_tracks.pdf` | `make_all_tracks.py` |
| Appendix overlay (`icra_overlay.pdf`) | `icra_overlay.pdf` | `make_sim_figures.py` |
| Method overview (Fig. 1, `method_overview_new.pdf` in the paper) | `method_overview.pdf` (older version) | `method_overview/` (`capture.py`, `make_figure.py`) |

**Do not run `make_sim_figures.py` to update Fig. 2 or Fig. 3.** Its `main()` rewrites `icra_laptimes.pdf`
with an older single-run plot and copies over `all_tracks.pdf`. Use the scripts in the table; they write only
their own files. `make_sim_figures.py` is still the library the other scripts import (loading, filtering,
the all-tracks plot, the track miniatures).

Not included: the physical-car figure (`physical_laptimes.pdf`), the track maps (`track_maps.pdf`) and the
car/track photos. Their data comes from the car recordings, not from these simulations.

## Folder map

The folders keep the names (and dates) of the original result folders, so the relative paths in the scripts
work unchanged. Each run is a folder with `result.json` (laps, collisions, freeze), and for ATMPPI also
`initial_raceline.npz` / `final_raceline.npz` (the memory before and after). Run logs are not included.

| Folder | What | Used for |
|---|---|---|
| `grid_2026-11-05/` | The first full grid: 4 tracks x {midline, raceline} x mu {0.5, 0.9} x noise {on, off} x 5 seeds, for ATMPPI 50/200, MPPI 50 to 2000 and RPGD. `summary.csv` = one row per run. | MPPI 50 / 2000 rows of Fig. 3 (noise off) |
| `grid_2026-11-05/paper/` | All paper plot scripts and figures | |
| `grid_2026-11-05/thesis_icra2022/` | `make_plots.py` (loading, lap plots, overlays; imported by the paper scripts), `record_traj.py` (re-run with the true state logged, for the overlay), per-step trajectories in `runs/` | |
| `tune_noise_2026-11-08/` | ATMPPI configs re-run on the full grid; `soft25w02` = the paper config. `launch.py` runs a config over the grid, `tune_run.py` runs one simulation with Settings overrides. `runs/mppi50`, `runs/mppi2000` = MPPI at half noise. `runs/wm10_tsf103` = ATMPPI 1000 on the grid (`all_tracks_1000`). | Fig. 3 (ATMPPI 50, MPPI at half noise), overlay |
| `hw50_tune_2026-11-10/` | Tuning on icra2022 (raceline, mu 0.5) with `hw50_tune.py`. `wm10_tsf103` = the Fig. 2 config. Also the horizon experiments (below). | Fig. 2 (ATMPPI 50, 1000), noise plots |
| `mppi_tune_2026-11-10/` | MPPI tuning with `mppi_tune.py`. `runs/plotrun/ns_s02` = the Fig. 2 baseline (steering noise 0.2). | Fig. 2 (MPPI 2000) |
| `hw_tune_2026-11-10/`, `icra_table_2026-11-10/` | Runs and script of the comparison table (ATMPPI 50 / 1000, MPPI 2000, acados MPC; mu 0.9, noise on) | Table |
| `atmppi_verify_2026-11-10/` | Checks that the standalone repo (github.com/F1Tenth-INI/L4DC) reproduces the snapshot step by step | |

## Conventions (how the numbers are computed)

- **ATMPPI lap time** = laps after the permanent memory freeze (`result["after_freeze"]`). Collisions while
  learning are not counted for it. **MPPI lap time** = all laps from lap 2 (lap 1 starts from standstill).
- **Fig. 3** shows per seed the mean lap time relative to the median of MPPI 2000 in the same setting.
  ATMPPI seeds that froze broken are dropped (`drop_broken` in `make_sim_figures.py`: more than 30 % slower
  than MPPI 2000, or a collision on at least half the frozen laps). Per-track or per-setting comparisons
  between two ATMPPI variants should be computed **per setting** (median over seeds, then compared), not as
  the median over all runs of a track: midline and raceline starts form two clusters, and a pooled median
  falls between them.
- **Observation noise**: uniform in +-`NOISE_LEVEL_CAR_STATE` on each observed state component, drawn every
  control step. "on" in the grid = 0.05, the paper's noise = **half** = 0.025 (`noisehalf` in run names).
- **Broken first lap** (horizon experiments): a seed whose first lap is far slower than usual (over 30 s on
  icra2022, or 1.3x the track's median first lap) or that crashes in lap 1 is discarded.
- **Known artefact**: a collision resets the car to the start, so laps are counted from there.
  See also the comments in `grid_2026-11-05/thesis_icra2022/make_plots.py`.

## Running new simulations

The paper runs used a frozen copy of the code (a "snapshot"), not the current branch head:

| Runs | Snapshot |
|---|---|
| `grid_2026-11-05` | commit `986f92d7` + `grid_2026-11-05/code/BENCH_SNAPSHOT_DIFF.patch` |
| `tune_noise_2026-11-08`, `hw50_tune_2026-11-10`, `hw_tune_2026-11-10` | commit `aac2064a` + `tune_noise_2026-11-08/code/BENCH_SNAPSHOT_DIFF.patch` |
| `mppi_tune_2026-11-10` | `mppi_tune_2026-11-10/code_snapshot/BENCH_SNAPSHOT_*` (same MPPI files) |

The current code has changed since (e.g. `c57cbc71` swapped the `d_left`/`d_right` column labels of
`icra2022_wp.csv`), so for numbers comparable with the paper, run on the snapshot:

```bash
git worktree add ../snapshot_aac2064a aac2064a
cd ../snapshot_aac2064a && git apply ../f1tenth_development_gym/paper_experiments/benchmark_results/tune_noise_2026-11-08/code/BENCH_SNAPSHOT_DIFF.patch
cp -r ../f1tenth_development_gym/utilities/maps/IPZ50 utilities/maps/   # IPZ50 was untracked at that commit
export SNAP=$PWD
```

Without `HW_GYM` / `MPPI_GYM` / `CODE_DIR`, the harnesses run the repository they are in. All harnesses skip
runs whose `result.json` already exists, and use `PY` (default: the current Python) for the workers.

**ATMPPI on icra2022 (or any map), one config file, several seeds** (`hw50_tune.py`):

```bash
cd paper_experiments/benchmark_results/hw50_tune_2026-11-10
# CONFIGS.json: {"name": {"SETTING": value, ...}}; runs go to runs/<TAG>/<name>/seed<k>/
HW_GYM=$SNAP HW_ROLLOUTS=50 HW_MU=0.5 HW_NOISE=off HW_SEED_OFFSET=1 python hw50_tune.py sweep r2.json TAG 5 1000 5
python hw50_tune.py summary r2.json TAG
```

Env: `HW_MAP` (default icra2022), `HW_MU` (0.5), `HW_NOISE` (off/on; with on, the noise level comes from
`NOISE_LEVEL_CAR_STATE` in the config, e.g. `[0.025]*10` for half noise), `HW_ROLLOUTS`, `HW_SEED_OFFSET`.
Start line is always the raceline. Arguments: `sweep CONFIGS TAG N_SEEDS SIM_SECONDS PARALLEL`.
`HWM_ROLLOUT_THREADS` in the config sets threads per run (1 for 50 rollouts, 2 to 8 for 1000).

**MPPI** (`mppi_tune.py`, GPU by default, `--cpu` otherwise): config keys `w:<cost weight>`, `p:<planner
attribute>` (e.g. `p:horizon`, `p:noise_std`), or a Settings name.

```bash
cd paper_experiments/benchmark_results/mppi_tune_2026-11-10
MPPI_GYM=$SNAP MPPI_ROLLOUTS=2000 MPPI_MU=0.5 MPPI_NOISE=off MPPI_SEED_OFFSET=1 python mppi_tune.py sweep plotrun.json TAG 2 1000 2
```

GPU runs need a JAX with CUDA: set `JAX_CUDA` to a `pip install --target` folder with `jax-cuda12-plugin`,
it is put on `PYTHONPATH` of the workers.

**ATMPPI or MPPI over the whole grid** (`tune_noise_2026-11-08/launch.py`; configs are the `CONFIGS` dict in
`tune_run.py`, runs go to `runs/<config>/<map>_<wp>_mu<mu>_noise<nz>_seed<s>/`):

```bash
cd paper_experiments/benchmark_results/tune_noise_2026-11-08
CODE_DIR=$SNAP python launch.py --configs soft25w02 --noise half,off --seeds 0,1,2,3,4 --workers 26
CODE_DIR=$SNAP python launch.py --configs wm10_tsf103 --rollouts 1000 --noise half,off --seeds 0,1,2,3,4 --workers 26
CODE_DIR=$SNAP python launch.py --configs none --controller mppi --rollouts 2000 --noise half --gpu --workers 8
```

After new grid runs, `make_all_tracks.py` picks them up from these folders.

## Horizon experiments (2026-11-12, not in the paper yet)

Question: how short can the rollout horizon of the driving samples be? Half noise, icra2022 raceline,
mu 0.5, config `wm10_tsf103`. These need `hw50_tune_2026-11-10/drive_horizon.patch` applied to the snapshot
(adds `MPC_DRIVE_HORIZON` and `MPC_DRIVE_HORIZON_AFTER_FREEZE`; both off by default):

```bash
cd $SNAP && git apply <repo>/paper_experiments/benchmark_results/hw50_tune_2026-11-10/drive_horizon.patch
```

Configs in `hw50_tune_2026-11-10/`: `hsweep_n50.json` (horizon 35 before the freeze and for window updates,
drive horizon 30/25/20/17 after the freeze, `af*`; `opt25` = stored sequences and window updates 25 too),
`hsweep_maps.json` (the same on RCA1/IPZ50/hangar3 with `HW_MAP`). MPPI: `mppi_tune_2026-11-10/hsweep.json`
(`p:horizon` 50/35/30/25/20/17). Summaries: `python reports/hsweep_report.py`, `reports/maps_report.py`.

Result so far (3 seeds, ATMPPI 50; 2 seeds, MPPI 2000): ATMPPI keeps or improves lap time and crash rate down
to a drive horizon of 25 after the freeze (14.06 s vs 14.28 s at 35); 20 is unreliable, 17 fails. MPPI 2000
needs 35: 30 and below crash 10 to 22 times per run, 50 is 0.5 s slower. Runs that were still going when
this branch was made are missing from `runs/`.

# Fig. 2 re-run (2026-11-12)

Recreates the data of the paper's Fig. 2 (icra2022, raceline start, mu 0.5, no noise, seed 1, 1000 s) from
scratch and redraws the figure. Everything is in `collect_fig2.py`:

```bash
python collect_fig2.py setup   # code snapshots as git worktrees next to the repo (../snapshot_aac2064a, ../snapshot_986f92d7)
python collect_fig2.py run     # ATMPPI 50, ATMPPI 1000, MPPI 2000 in parallel -> runs/<name>/result.json
python collect_fig2.py plot    # figures/icra_laptimes_repro.{pdf,png}, figures/icra_laptimes_paper.{pdf,png}, table
```

| Run | Harness, config | Code |
|---|---|---|
| `at50` | `hw50_tune.py`, `r2.json:wm10_tsf103`, 50 rollouts, 1 thread | `aac2064a` + tune_noise patch |
| `at1024` | `hw50_tune.py`, `r3_n1000.json:wm10_tsf103`, 1024 rollouts (paper: 1000), 4 threads | `aac2064a` + tune_noise patch |
| `mppi2048` | `mppi_tune.py`, `plotrun.json:ns_s02` (steering noise 0.2), 2048 rollouts (paper: 2000) | `986f92d7` + grid patch |

The figure is drawn with the paper's script (`grid_2026-11-05/paper/make_icra_laptimes_rollouts_v2.py`), only
the run folders differ. `icra_laptimes_paper` is the same script on the paper's runs, for comparison.

## Differences from the paper runs

- **MPPI on the CPU**: the paper's MPPI run used JAX on the GPU (`"gpu": true` in its `result.json`). The f1t
  env here has a CPU-only JAX, so `run` uses the CPU unless `--gpu` is given (with `JAX_CUDA`).
- **Snapshot fixes done by `setup`**:
  - The MPPI patch replaces `utilities/maps/icra2022/centerline.csv`, which is stored with CRLF line endings.
    The patch file lost the CRs, so `git apply` rejects it. `setup` writes the hunk's new content directly; it
    hashes to the blob in the patch header (`fdb375ef`).
  - Commit `986f92d7` has no `scripts/benchmark/` (first committed in `be2256bd`); the paper used an
    uncommitted copy. `setup` copies it from `aac2064a`. `bench_run.py` is identical in `be2256bd` and
    `aac2064a`, and the MPPI code is identical in `986f92d7` and `aac2064a`.
  - Submodules (`SI_Toolkit`, `Control_Toolkit`) are cloned from the main checkout's `.git/modules` at the
    pinned commits (both snapshots pin the same ones as the current branch).
- **Python env**: workers run with `PYTHONNOUSERSITE=1`, because `~/.local` has a numba that does not import
  with the env's llvmlite. `USER_SITE=1` turns this off.
- **Not bitwise reproducible**: different machine and library versions give slightly different floating-point
  results, which grow over a 1000 s run. First laps of a 30 s test: ATMPPI 50 22.68 s (paper 23.00 s),
  ATMPPI 1000 26.36 s (paper 22.12 s), MPPI 2000 16.36 s (paper 16.44 s). Compare the runs statistically,
  not lap by lap.

## Result with 1000 / 2000 rollouts as in the paper (runs/at1000, runs/mppi2000; 2026-11-12, MPPI on CPU)

| Run | Paper: freeze lap, median after freeze | Re-run | Paper, other seeds (median) |
|---|---|---|---|
| ATMPPI 50 | lap 23 (349 s), 14.16 s, 0 coll. | lap 22 (335 s), 14.24 s, 0 coll. | seeds 0-4: 14.04 to 14.32 s |
| ATMPPI 1000 | lap 17 (260 s), 14.28 s, 0 coll. | lap 18 (279 s), 13.96 s, 0 coll. | seeds 0-2: 13.88 to 14.28 s |
| MPPI 2000 | 14.52 s (from lap 2), 0 coll. | 14.52 s, 0 coll. | seeds 0-4: 14.52 s |

MPPI 2000 is reproduced exactly in its median. Both ATMPPI runs are again faster than MPPI 2000 and fall within
the spread of the paper's seeds, but their order flips: the paper's ATMPPI 1000 seed 1 was the slowest of its
three seeds, the re-run is close to the fastest.

## Result with 1024 / 2048 rollouts (current `collect_fig2.py`; runs/at1024, runs/mppi2048)

| Run | Freeze | Median after freeze (MPPI: from lap 2) | Collisions |
|---|---|---|---|
| ATMPPI 50 (same run as above) | lap 22 (335 s) | 14.24 s | 0 |
| ATMPPI 1024 | lap 25 (403 s) | 14.04 s | 0 |
| MPPI 2048 | | 14.52 s | 0 |

`make_icra_laptimes_rollouts_v2.py` got an `mppi_rollouts` argument (default 2000) for the legend label.

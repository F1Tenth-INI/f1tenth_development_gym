# Fig. 1 re-run (2026-11-12)

Recreates the paper's Fig. 1 (`method_overview_new.pdf`, one control step of ATMPPI) from a new simulation,
with the paper's scripts (`grid_2026-11-05/paper/method_overview/capture.py` and `make_figure.py`).

```bash
python recreate_fig1.py capture   # icra2022, raceline, mu 0.5, no noise, seed 1, ATMPPI 50 (soft25w02), ~1 min
python recreate_fig1.py plot      # figures/method_overview.{pdf,png}; figures/paper/ = the paper's step, same script
```

Code: snapshot `aac2064a` + tune_noise patch (`../fig2_repro_2026-11-12/collect_fig2.py setup`).
`capture.py` and `make_figure.py` got env overrides for their folders (`CAPTURE_DIR`, `FIG1_CODE`,
`FIG1_STEPS`, `FIG1_OUT`); their defaults are the original paths. The original `make_figure.py` expects the full
code snapshot in `tune_noise_2026-11-08/code/`, which in this repo only holds the patch, so it needs `FIG1_CODE`.

## Which step

The paper shows step 2604 (lap 6): car at waypoint 60 of 155, window at waypoint 41, targets moved by up to
0.155 m. The re-run is not bitwise identical and its early laps are faster (lap 6 = steps 2295 to 2685), so
the capture starts at step 2200 instead of 2400. `plot` picks the step in lap 6 with the same car and window
waypoints, then the largest target shift: **step 2360** (car 60, window 41, lap 6), targets moved by up to
0.036 m. Same corner, same layout; the update is smaller because this run had already learned more of the
corner by lap 6. Other steps: `python recreate_fig1.py plot --step N` (steps/ holds 2200 to 3300).

Check: drawing the paper's saved step 2604 with the same script gives the paper's PNG pixel for pixel
(`figures/paper/`).

## Compact variant

`make_figure_compact.py` (drawn by `plot` too): `figures/method_overview_compact.{pdf,png}`, and for the paper's
step in `figures/paper/`. Same two map panels, but no waypoint-range panel (c), no panel letters or titles
(the zoom is linked to its rectangle by magnifier lines from the rectangle's right corners to the zoom's left
corners), and "update window" / "rejoin section" written out instead of W_win / W_down.
Labels ("car", the window names, r_w, "cross-track line") and the zoom's scale bar are placed automatically
(class `Labels`): each takes the first of its candidate positions that hides the fewest drawn points and
overlaps no other label, so they stay readable for any step. Frames with larger updates (4 targets moved,
like the paper's): `plot --step 2430` (lap 6, up to 13.8 cm) and `plot --step 3203` (lap 8, up to 14.9 cm);
drafts in `figures/step2430/`, `figures/step3203/`.

**Paper version and v2 (2026-11-13).** The paper's Fig. 1 (`method_overview_compact.pdf` in the paper repo) is
`figures/step2430/method_overview_compact.pdf` (byte-identical). Since then `make_figure_compact.py` no longer draws
the retrieval points (green dots; they still enter the axis limits, so the layout is unchanged) and draws the
updated targets in the zoom in purple (`C_UPD`) instead of the orange of the accepted candidate:
`figures/step2430_v2/` (`FIG1_OUT=figures/step2430_v2 python make_figure_compact.py 2430` with the env of
`recreate_fig1.py`).

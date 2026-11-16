#!/usr/bin/env python3
"""Recreate the paper's Fig. 1 (method overview, ``method_overview_new.pdf``) from a new simulation.

The paper figure is ``grid_2026-11-05/paper/method_overview/make_figure.py`` on step 2604 (lap 6) of the run
saved by ``capture.py``: icra2022, raceline, mu 0.5, no noise, seed 1, ATMPPI 50 with config soft25w02, code
snapshot aac2064a + tune_noise_2026-11-08 patch. This script reruns that capture on the snapshot, picks the
equivalent step and draws the figure with the paper's script.

    python recreate_fig1.py capture     # simulation, saves steps/step_<n>.npz for steps 2200..3300 (~1 min)
    python recreate_fig1.py plot [--step N]   # figures/method_overview.{pdf,png} and the compact variant
                                              # method_overview_compact.{pdf,png} (make_figure_compact.py)

The run is not bitwise identical to the paper's (other machine and library versions), so step 2604 does not
show the same moment. Without --step, the step is the captured one (each has an accepted window update)
in lap 6 (--lap) whose car is at the paper step's waypoint (60 of 155, within --tol waypoints): the same
car and window waypoints as the paper step if possible, then the update that moves the targets most. The capture starts at step 2200 (paper: 2400), because this run reaches lap 6 earlier. Snapshot: ../fig2_repro_2026-11-12/collect_fig2.py setup (SNAP_AT overrides its path).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BR = HERE.parent
REPO = HERE.parents[2]
MO = BR / "grid_2026-11-05" / "paper" / "method_overview"
SNAP_AT = Path(os.environ.get("SNAP_AT", REPO.parent / "snapshot_aac2064a"))
PAPER_STEP = 2604
PAPER_STEPS = MO / "steps"


def env():
    e = dict(os.environ, CODE_DIR=str(SNAP_AT), CAPTURE_DIR=str(HERE), FIG1_CODE=str(SNAP_AT),
             FIG1_STEPS=str(HERE / "steps"), FIG1_OUT=str(HERE / "figures"))
    if os.environ.get("USER_SITE", "0") != "1":
        e["PYTHONNOUSERSITE"] = "1"  # ~/.local has a numba that does not import with the env's llvmlite
    return e


def capture(first, last):
    if not (SNAP_AT / "scripts" / "benchmark" / "bench_run.py").exists():
        raise SystemExit(f"{SNAP_AT} missing: python ../fig2_repro_2026-11-12/collect_fig2.py setup")
    (HERE / "steps").mkdir(exist_ok=True)
    with open(HERE / "capture.log", "w") as log:
        code = subprocess.run([sys.executable, str(MO / "capture.py"), str(first), str(last)], env=env(), cwd=str(HERE),
                              stdout=log, stderr=subprocess.STDOUT).returncode
    n = len(list((HERE / "steps").glob("step_*.npz")))
    print(f"capture exit {code}, {n} steps saved in {HERE / 'steps'} (log: capture.log)")


def info(path):
    """step, car waypoint, window waypoint, number of waypoints, largest target shift of the update [m]."""
    d = np.load(path)
    shift = float(np.linalg.norm(d["stored_after"] - d["stored_before"], axis=1).max())
    return int(d["step"]), int(d["car_waypoint"]), int(d["w"]), len(d["anchor"]), shift


def lap_ends():
    """Simulated times [s] at which laps end, from the capture log."""
    return [float(l.split("t=")[1].split()[0]) for l in open(HERE / "capture.log") if l.startswith("BENCH_LAP")]


def lap_of(step, ends):
    t = step * 0.04
    return 1 + sum(e < t for e in ends)


def pick_step(tol, lap):
    ref_step, ref_wp, ref_w, n, ref_shift = info(PAPER_STEPS / f"step_{PAPER_STEP}.npz")
    rows = [info(p) for p in (HERE / "steps").glob("step_*.npz")]
    if not rows:
        raise SystemExit("no captured steps: run capture first")

    def wp_dist(a, b):
        return min((a - b) % n, (b - a) % n)

    ends = lap_ends()
    near = [r for r in rows if wp_dist(r[1], ref_wp) <= tol and (not lap or lap_of(r[0], ends) == lap)]
    if not near:
        raise SystemExit(f"no captured step in lap {lap} with the car within {tol} waypoints of {ref_wp}; "
                         "try --tol, --lap 0, an earlier --first for capture, or --step")
    # same car and window waypoint as the paper step first, then the update that moves the targets most
    # (the arrows of panel b), then closest in time
    best = min(near, key=lambda r: (wp_dist(r[1], ref_wp) + wp_dist(r[2], ref_w), -r[4], abs(r[0] - ref_step)))
    print(f"paper step {ref_step}: car waypoint {ref_wp}, window waypoint {ref_w}, largest target shift {ref_shift:.3f} m")
    print(f"chosen step {best[0]} (lap {lap_of(best[0], ends)}): car waypoint {best[1]}, window waypoint {best[2]}, "
          f"largest target shift {best[4]:.3f} m "
          f"({len(near)} captured steps within {tol} waypoints)")
    return best[0]


def plot(step, tol, lap):
    step = step if step is not None else pick_step(tol, lap)
    subprocess.run([sys.executable, str(MO / "make_figure.py"), str(step)], env=env(), check=True)
    subprocess.run([sys.executable, str(HERE / "make_figure_compact.py"), str(step)], env=env(), check=True)
    # the paper's own step, drawn by the same script, for side-by-side comparison
    e = env()
    e["FIG1_STEPS"] = str(PAPER_STEPS)
    e["FIG1_OUT"] = str(HERE / "figures" / "paper")
    subprocess.run([sys.executable, str(MO / "make_figure.py"), str(PAPER_STEP)], env=e, check=True)
    subprocess.run([sys.executable, str(HERE / "make_figure_compact.py"), str(PAPER_STEP)], env=e, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["capture", "plot"])
    ap.add_argument("--step", type=int, help="step to draw (default: the equivalent of the paper's step 2604)")
    ap.add_argument("--first", type=int, default=2200, help="capture: first step saved (paper: 2400)")
    ap.add_argument("--last", type=int, default=3300, help="capture: last step saved")
    ap.add_argument("--lap", type=int, default=6, help="plot: lap the step must be in (as the paper's, lap 6); 0: any")
    ap.add_argument("--tol", type=int, default=2, help="waypoints the car may be from the paper step's waypoint")
    a = ap.parse_args()
    capture(a.first, a.last) if a.mode == "capture" else plot(a.step, a.tol, a.lap)

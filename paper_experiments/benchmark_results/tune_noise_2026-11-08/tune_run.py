#!/usr/bin/env python3
"""``bench_run.py`` of the code snapshot with Settings overrides applied first.

    python tune_run.py <config> <bench_run args...>

Env: ``NOISE_SCALE`` scales the observation noise, ``RECORD_TRAJ=1`` runs ``record_traj.py`` instead.

The overrides are set on ``utilities.Settings.Settings`` before ``bench_run`` configures the run,
so everything ``bench_run`` does not set itself comes from the config below.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# Code to run: CODE_DIR (absolute, or relative to this folder); default the repository root.
# The paper runs used the snapshot in code/ (see paper_experiments/README.md).
CODE = os.path.join(HERE, os.environ["CODE_DIR"]) if os.environ.get("CODE_DIR") else REPO
RECORD = os.path.join(HERE, "..", "grid_2026-11-05", "thesis_icra2022")  # record_traj.py: also log the true state

CONFIGS: dict[str, dict] = {
    # The grid of 2026-11-05: collision margin 0.07 m.
    "base07": {"HWM_MIN_BORDER_DISTANCE": 0.07},
    "border12": {"HWM_MIN_BORDER_DISTANCE": 0.12},
    "border20": {"HWM_MIN_BORDER_DISTANCE": 0.20},
    # The learned line keeps more room: window candidates need more wall clearance.
    "win25": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_WAYPOINT_WINDOW_MARGIN": 0.25, "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.20},
    # Soft wall cost on the driving rollouts from 0.25 m down to the margin.
    "soft25": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.5},
    "combo": {"HWM_MIN_BORDER_DISTANCE": 0.12, "MPC_WAYPOINT_WINDOW_MARGIN": 0.25, "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.20,
              "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.5},
    # No permanent freeze: window updates and reverts go on; the best-lap reset stays.
    "nofreeze": {"HWM_MIN_BORDER_DISTANCE": 0.07, "HWM_PERMANENT_FREEZE_LAPS": 0},
    "win25_nofreeze": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_WAYPOINT_WINDOW_MARGIN": 0.25,
                       "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.20, "HWM_PERMANENT_FREEZE_LAPS": 0},
    # Longer patience before the freeze: more laps to improve at high friction.
    "freeze10": {"HWM_MIN_BORDER_DISTANCE": 0.07, "HWM_PERMANENT_FREEZE_LAPS": 10, "HWM_BEST_LAP_RESET_LAPS": 10},
    # Between soft25 and base07.
    "soft18": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.18, "MPC_DRIVE_WALL_WEIGHT": 0.5},
    "soft25w02": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2},
    "soft25_win25": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.5,
                     "MPC_WAYPOINT_WINDOW_MARGIN": 0.25, "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.20},
    "soft25_freeze10": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.5,
                        "HWM_PERMANENT_FREEZE_LAPS": 10, "HWM_BEST_LAP_RESET_LAPS": 10},
    # The wall cost keeps rising inside the margin, down to the real wall.
    "ramp0": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
              "MPC_DRIVE_WALL_RAMP_END": 0.0},
    "ramp0_w03": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.3,
                  "MPC_DRIVE_WALL_RAMP_END": 0.0},
    # A crash costs more the deeper the rollout gets into the margin.
    "depth": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
              "MPC_DRIVE_CRASH_DEPTH_WEIGHT": 1.0},
    # Inflated walls with the learned line kept outside them (window margin > margin + noise).
    "b15": {"HWM_MIN_BORDER_DISTANCE": 0.15, "MPC_WAYPOINT_WINDOW_MARGIN": 0.25, "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.20,
            "MPC_DRIVE_WALL_CLEARANCE": 0.30, "MPC_DRIVE_WALL_WEIGHT": 0.2},
    # The same with the cost graded all the way to the real wall.
    "b15_fix": {"HWM_MIN_BORDER_DISTANCE": 0.15, "MPC_WAYPOINT_WINDOW_MARGIN": 0.25, "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.20,
                "MPC_DRIVE_WALL_CLEARANCE": 0.30, "MPC_DRIVE_WALL_WEIGHT": 0.2, "MPC_DRIVE_WALL_RAMP_END": 0.0,
                "MPC_DRIVE_CRASH_DEPTH_WEIGHT": 1.0},
    # TEST (code_reuse only): window updates from the driving rollouts, strict 0.3 m gating.
    "reuse": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
              "HWM_WINDOW_FROM_DRIVE": True, "MPC_WAYPOINT_UPDATE_DISTANCE": 0.3},
    "reuse_tol3": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
                   "HWM_WINDOW_FROM_DRIVE": True, "MPC_WAYPOINT_UPDATE_DISTANCE": 0.3,
                   "MPC_WAYPOINT_WINDOW_TOLERANCE": 0.3},
    "reuse_tgt105": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
                     "HWM_WINDOW_FROM_DRIVE": True, "MPC_WAYPOINT_UPDATE_DISTANCE": 0.3,
                     "MPC_WAYPOINT_WINDOW_TOLERANCE": 0.3, "MPC_DRIVE_TARGET_SPEED_FACTOR": 1.05},
    "reuse_arr": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
                  "HWM_WINDOW_FROM_DRIVE": True, "MPC_WAYPOINT_UPDATE_DISTANCE": 0.3,
                  "MPC_WAYPOINT_WINDOW_TOLERANCE": 0.3, "MPC_DRIVE_ARRIVAL_WEIGHT": 1.0,
                  "MPC_DRIVE_OVERSPEED_WEIGHT": 0.5},
    # Control (production code): soft25w02 with only the strict 0.3 m gating.
    "gate03": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
               "MPC_WAYPOINT_UPDATE_DISTANCE": 0.3},
    # Auto-Tuning MPPI 1000 of the lap-time plots (hw50_tune_2026-11-10 wm10_tsf103): soft25w02 with tighter window
    # margins and 3 % faster target speed. Run with --rollouts 1000; runs go to runs/wm10_tsf103/.
    "wm10_tsf103": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
                    "MPC_WAYPOINT_WINDOW_MARGIN": 0.1, "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.08,
                    "MPC_DRIVE_TARGET_SPEED_FACTOR": 1.03},
    # Baselines (MPPI) and checks.
    "none": {},
    "depth_ramp0_torch": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
                          "MPC_DRIVE_WALL_RAMP_END": 0.0, "MPC_DRIVE_CRASH_DEPTH_WEIGHT": 1.0, "HWM_JAX_REWARDS": False},
    "depth_ramp0_jax": {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
                        "MPC_DRIVE_WALL_RAMP_END": 0.0, "MPC_DRIVE_CRASH_DEPTH_WEIGHT": 1.0},
}


def main() -> int:
    config = sys.argv[1]
    sys.argv = [os.path.join(CODE, "scripts", "benchmark", "bench_run.py")] + sys.argv[2:]
    os.chdir(CODE)
    sys.path.insert(0, CODE)
    sys.path.insert(0, os.path.join(CODE, "scripts", "benchmark"))
    from utilities.Settings import Settings

    overrides = dict(CONFIGS[config])
    overrides.setdefault("HWM_ROLLOUT_THREADS", 1)  # many runs in parallel
    for key, value in overrides.items():
        if not hasattr(Settings, key):
            raise KeyError(f"Settings has no {key}")
        setattr(Settings, key, value)
    scale = float(os.environ.get("NOISE_SCALE", "1"))
    Settings.NOISE_LEVEL_CAR_STATE = [scale * n for n in Settings.NOISE_LEVEL_CAR_STATE]
    print(f"TUNE_CONFIG {config} {overrides} noise_scale={scale}", flush=True)
    if os.environ.get("RECORD_TRAJ") == "1":
        sys.path.insert(0, RECORD)
        import record_traj  # noqa: F401  (patches bench_run)
    import bench_run

    return bench_run.main()


if __name__ == "__main__":
    sys.exit(main())

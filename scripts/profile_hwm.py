"""Profile the HWM control step headless, with Pacejka and with the LLD as the planner's dynamics model.

    python scripts/profile_hwm.py --model hwm_weighted_SE_test_7 --steps 400 --mode pacejka lld lld_residual

Each mode runs in its own subprocess (clean JIT/compile state) and writes ``<out>_<mode>.txt/.json``.
Run from the ``f1tenth_development_gym`` directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

MODES = {
    "pacejka": {"RUN_MPC_WITH_PACEJKA": True},
    "lld": {"RUN_MPC_WITH_PACEJKA": False, "HWM_LLD_RESIDUAL_ONLY": False},
    "lld_residual": {"RUN_MPC_WITH_PACEJKA": False, "HWM_LLD_RESIDUAL_ONLY": True},
}


def child(mode: str, model: str, steps: int, out: str, skip: int, samples: int | None) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, os.getcwd())
    from utilities.Settings import Settings

    Settings.CONTROLLER = "HirarchicalPlanner"
    Settings.RENDER_MODE = None
    Settings.HWM_INFERENCE_MODEL_NAME = model
    Settings.HWM_SIM_SPEED = None
    Settings.MAX_SIM_FREQUENCY = None
    Settings.HWM_PROFILE = True
    Settings.HWM_PROFILE_FILE = f"{out}_{mode}"
    Settings.HWM_PROFILE_SKIP_STEPS = skip
    Settings.HWM_PP_DRIVES = False
    if samples is not None:
        Settings.MPC_NUM_OF_TRAJECTORIES = int(samples)
    for key, value in MODES[mode].items():
        setattr(Settings, key, value)
    Settings.SIMULATION_LENGTH = steps + skip  # counted in control steps by the simulation loop

    from run.run_simulation import RacingSimulation

    RacingSimulation().run_experiments()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HWM_INFERENCE_MODEL_NAME (folder in TrainingLite/hwm/models)")
    parser.add_argument("--steps", type=int, default=400, help="control steps measured, after the skipped warmup")
    parser.add_argument("--skip", type=int, default=30, help="warmup control steps not measured")
    parser.add_argument("--mode", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--out", default="hwm_profile")
    parser.add_argument("--samples", type=int, default=None, help="MPC_NUM_OF_TRAJECTORIES override")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.model, args.steps, args.out, args.skip, args.samples)
        return
    for mode in args.mode:
        print(f"=== profiling mode: {mode}", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--model", args.model, "--steps", str(args.steps), "--skip", str(args.skip),
             "--out", args.out, "--child", mode] + (["--samples", str(args.samples)] if args.samples else []),
            check=False,
        )


if __name__ == "__main__":
    main()

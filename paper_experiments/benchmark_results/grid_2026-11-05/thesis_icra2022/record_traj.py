#!/usr/bin/env python3
"""bench_run.py with the true car state logged every control step.

Same arguments as bench_run.py. Run from the code snapshot root. Additionally writes
``<out>/trajectory.npz`` with t, x, y, v (noise-free simulator state).
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.getcwd(), "scripts", "benchmark")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, BENCH)

import bench_run  # noqa: E402

_out = sys.argv[sys.argv.index("--out") + 1]
_rows: list[tuple[float, float, float, float]] = []


def _install():
    import numpy as np
    from run.run_simulation import RacingSimulation
    from utilities.Settings import Settings
    from utilities.state_utilities import POSE_X_IDX, POSE_Y_IDX, LINEAR_VEL_X_IDX

    original = RacingSimulation.simulation_step

    def simulation_step(self):
        out = original(self)
        s = self.world_sim.agents[0].state
        t = (int(self.sim_index) + 1) * float(Settings.TIMESTEP_CONTROL)
        _rows.append((t, float(s[POSE_X_IDX]), float(s[POSE_Y_IDX]), float(s[LINEAR_VEL_X_IDX])))
        return out

    RacingSimulation.simulation_step = simulation_step

    real_exit = os._exit

    def _exit(code):
        a = np.asarray(_rows, dtype=np.float64).reshape(-1, 4)
        np.savez_compressed(os.path.join(_out, "trajectory.npz"), t=a[:, 0], x=a[:, 1], y=a[:, 2], v=a[:, 3])
        sys.stdout.flush()
        real_exit(code)

    os._exit = _exit


_orig_configure = bench_run.configure_settings


def configure_settings(args):
    Settings = _orig_configure(args)
    if os.environ.get("HWM_FROZEN_REVERT") == "1":
        # After the permanent freeze, leaving the raceline or crashing still puts the waypoints
        # around the car one lap further back (utilities/Settings.py: HWM_FROZEN_REVERT).
        Settings.HWM_FROZEN_REVERT = True
        print("HWM_FROZEN_REVERT=True", flush=True)
    _install()
    return Settings


bench_run.configure_settings = configure_settings

if __name__ == "__main__":
    sys.exit(bench_run.main())

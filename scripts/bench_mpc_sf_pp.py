#!/usr/bin/env python3
"""Headless PP + MPC safety filter benchmark."""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=4 "
        "inter_op_parallelism_threads=1"
    )

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utilities.Settings import Settings
from utilities import mpc_safety_filter as mpc_sf

Settings.CONTROLLER = "pp"
Settings.MPC_SAFETY_FILTER = True
mpc_sf.BACKEND = "grid"
Settings.RENDER_MODE = None
Settings.RECORDING = False
Settings.NUMBER_OF_EXPERIMENTS = 1
Settings.SIMULATION_LENGTH = 8000
Settings.MAX_SIM_FREQUENCY = None
Settings.RESET_ON_DONE = False
Settings.MAX_EPISODE_LENGTH = 0  # disable RL episode cutoffs for bench
Settings.TRUNCATE_ON_LEAVE_TRACK = False

from run.run_simulation import RacingSimulation


def main():
    sim = RacingSimulation()
    t0 = time.perf_counter()
    certified = intervened = 0
    try:
        sim.prepare_simulation()
        sim.reset()
        for sim.sim_index in range(int(Settings.SIMULATION_LENGTH)):
            sim.simulation_step()
            info = getattr(sim.drivers[0], "mpc_info", {}) or {}
            if info.get("certified"):
                certified += 1
            if info.get("intervene"):
                intervened += 1
            if sim.drivers[0].env_state and sim.world_sim.agents[0].in_collision:
                print(f"CRASH at step {sim.sim_index}", flush=True)
                break
            if getattr(sim.drivers[0], "lap_limit_reached", False):
                break
    except Exception as exc:
        print("run ended:", exc, flush=True)
    elapsed = time.perf_counter() - t0
    driver = sim.drivers[0]
    steps = sim.sim_index + 1
    print(
        f"steps={steps} wall={elapsed:.2f}s hz={steps/max(elapsed,1e-9):.1f} "
        f"certified={certified} intervened={intervened} "
        f"laptimes={getattr(driver, 'laptimes', [])} backend={mpc_sf.BACKEND}",
        flush=True,
    )


if __name__ == "__main__":
    main()

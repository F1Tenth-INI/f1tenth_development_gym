#!/usr/bin/env python3
"""One benchmark run: one controller on one map for a fixed simulated time.

Run from the root of the code it should use (the launcher passes a snapshot of the
repository). Everything not set here comes from ``utilities/Settings.py``.

The episode resets only on a collision, for every controller: leaving the track,
spinning, standing still, MAX_EPISODE_LENGTH and EXPERIMENT_MAX_LENGTH do not end it.

Writes ``<out>/result.json``. For the HirarchicalPlanner also ``initial_raceline.svg``
and ``final_raceline.svg`` (the waypoint table at the first control step and at the end).

    python bench_run.py --out DIR --controller hw --rollouts 50 --map RCA1 --wp raceline \
        --mu 0.5 --noise on --seed 0 --sim-seconds 1000
    python bench_run.py --out DIR --map RCA1 --wp midline --initial-svg-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
    )

CONTROLLERS = {"hw": "HirarchicalPlanner", "mppi": "mppi-lite-jax", "rpgd": "rpgd-lite-jax"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--controller", choices=sorted(CONTROLLERS), default="hw")
    p.add_argument("--rollouts", type=int, default=None, help="hw: MPC_NUM_OF_TRAJECTORIES. mppi: batch size. rpgd: unused")
    p.add_argument("--map", required=True)
    p.add_argument("--wp", choices=["midline", "raceline"], required=True)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--noise", choices=["on", "off"], default="on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-seconds", type=float, default=1000.0)
    p.add_argument("--gpu", action="store_true", help="let JAX use the GPU (needs a CUDA jaxlib)")
    p.add_argument("--initial-svg-only", action="store_true")
    return p.parse_args()


def configure_settings(args):
    from utilities.Settings import Settings

    Settings.MAP_NAME = args.map
    Settings.WAYPOINT_SOURCE = args.wp
    Settings.recalculate_paths()
    Settings.CONTROLLER = CONTROLLERS[args.controller]
    if args.mu is not None:
        Settings.SURFACE_FRICTION = float(args.mu)
    if args.noise == "off":
        Settings.NOISE_LEVEL_CAR_STATE = [0.0] * len(Settings.NOISE_LEVEL_CAR_STATE)
    Settings.RANDOM_SEED = int(args.seed)
    if args.controller == "hw" and args.rollouts is not None:
        Settings.MPC_NUM_OF_TRAJECTORIES = int(args.rollouts)
    # No learner: the planner drives with the Pacejka model. A closed port keeps the TCP
    # client from reaching a learner that someone else runs on the default port.
    Settings.LEARNER_TCP_HOST = "127.0.0.1"
    Settings.LEARNER_TCP_PORT = 1
    # Only collisions reset, the run is a fixed simulated time.
    Settings.SIMULATION_LENGTH = int(round(args.sim_seconds / Settings.TIMESTEP_CONTROL))
    Settings.TRUNCATE_ON_LEAVE_TRACK = False
    Settings.MAX_EPISODE_LENGTH = 0
    Settings.EXPERIMENT_MAX_LENGTH = 10 ** 12
    Settings.RESET_ON_DONE = True
    Settings.RESPAWN_ON_RESET = False
    Settings.STOP_AFTER_N_LAPS = None
    Settings.NUMBER_OF_EXPERIMENTS = 1
    # Harness only: no rendering, no recordings, no real-time pacing.
    Settings.RENDER_MODE = None
    Settings.WEB_RENDER_AUTO_OPEN = False
    Settings.SAVE_RECORDINGS = False
    Settings.SAVE_PLOTS = False
    Settings.SAVE_REWARDS = False
    Settings.SAVE_VIDEOS = False
    Settings.HWM_SIM_SPEED = None
    Settings.MAX_SIM_FREQUENCY = None
    if args.gpu:
        Settings.DISABLE_GPU = False

    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    return Settings


def v_max(Settings) -> float:
    from utilities.car_files.vehicle_parameters import VehicleParameters

    return float(VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).v_max)


def initial_svg(args, Settings) -> None:
    import numpy as np
    from utilities.waypoint_utils import WaypointUtils, WP_X_IDX, WP_Y_IDX, WP_VX_IDX
    from bench_svg import write_waypoint_svg

    wpu = WaypointUtils()
    wp = np.asarray(wpu.waypoints)
    path = os.path.join(args.out, f"initial_{args.map}_{args.wp}.svg")
    write_waypoint_svg(path, Settings.MAP_PATH, args.map, wp[:, [WP_X_IDX, WP_Y_IDX]], wp[:, WP_VX_IDX],
                       v_max=v_max(Settings), title=f"{args.map} {args.wp} (waypoint file speeds)")
    print(f"wrote {path} with {len(wp)} waypoints", flush=True)


def install_crash_only_resets() -> None:
    from utilities.episode_termination import EpisodeTerminator

    original = EpisodeTerminator.evaluate

    def evaluate(self, controller_obs, driver_obs, **_):
        return original(self, controller_obs, driver_obs, ignore_leave_track=True,
                        ignore_episode_length=True, crash_only=True)

    EpisodeTerminator.evaluate = evaluate


def hw_table(planner):
    """Copy of the HW waypoint table: xy, heading, target speed. None before it exists."""
    wps = getattr(planner, "waypoints", None)
    if wps is None:
        return None
    from TrainingLite.hwm.hirarchical_planner import WP_HEADING_IDX, WP_TARGET_SPEED_IDX

    a = wps.detach().cpu().numpy()
    return {"xy": a[:, 1:3].copy(), "heading": a[:, WP_HEADING_IDX].copy(), "speed": a[:, WP_TARGET_SPEED_IDX].copy()}


def merge_collisions(collisions, laps):
    """Collisions with no completed lap between them count as one.

    Returns the clusters, each the list of indices into ``collisions``.
    """
    lap_t = sorted(l["t"] for l in laps)
    clusters = []
    for i, c in enumerate(collisions):
        if clusters:
            prev_t = collisions[clusters[-1][-1]]["t"]
            if not any(prev_t < t <= c["t"] for t in lap_t):
                clusters[-1].append(i)
                continue
        clusters.append([i])
    return clusters


def lap_stats(times):
    import statistics

    if not times:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None, "std": None}
    return {"n": len(times), "min": min(times), "mean": statistics.mean(times),
            "median": statistics.median(times), "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0}


def main() -> int:
    args = parse_args()
    if not args.gpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(args.out, exist_ok=True)
    Settings = configure_settings(args)
    if args.initial_svg_only:
        initial_svg(args, Settings)
        return 0

    install_crash_only_resets()
    import numpy as np
    from utilities.car_system import CarSystem
    from run.run_simulation import RacingSimulation

    dt = float(Settings.TIMESTEP_CONTROL)
    hw = args.controller == "hw"
    st = {"sim": None, "laps": [], "collisions": [], "freeze_t": None, "freeze_lap": None,
          "initial": None, "planner": None}

    def now() -> float:
        sim = st["sim"]
        return 0.0 if sim is None else (int(sim.sim_index) + 1) * dt

    def frozen() -> bool:
        p = st["planner"]
        return bool(p is not None and getattr(p, "_wp_permanently_frozen", False))

    original_init_controller = CarSystem.initialize_controller

    def initialize_controller(self, controller_name):
        original_init_controller(self, controller_name)
        if st["planner"] is not None or self.planner is None:
            return
        st["planner"] = self.planner
        if args.controller in ("mppi", "rpgd"):
            import jax

            self.planner.key = jax.random.PRNGKey(int(args.seed))
            if args.controller == "mppi" and args.rollouts is not None:
                self.planner.batch_size = int(args.rollouts)

    CarSystem.initialize_controller = initialize_controller

    original_lap = CarSystem.lap_complete_cb

    def lap_complete_cb(self, lap_time, *rest):
        before = frozen()
        st["laps"].append({"t": now(), "lap_time": float(lap_time), "frozen": before})
        print(f"BENCH_LAP t={now():.2f} lap_time={float(lap_time):.4f} frozen={before}", flush=True)
        out = original_lap(self, lap_time, *rest)
        if hw and not before and frozen():
            st["freeze_t"], st["freeze_lap"] = now(), len(st["laps"])
            print(f"BENCH_FREEZE t={now():.2f} after_lap={len(st['laps'])}", flush=True)
        return out

    CarSystem.lap_complete_cb = lap_complete_cb

    original_step_end = CarSystem.on_step_end

    def on_step_end(self, observation=None):
        out = original_step_end(self, observation)
        if observation is not None and self.planner is st["planner"] and observation.get("collision"):
            st["raw_collision_steps"] = st.get("raw_collision_steps", 0) + 1
            if not observation.get("done"):
                st["collision_without_reset"] = st.get("collision_without_reset", 0) + 1
                print(f"BENCH_COLLISION_WITHOUT_RESET t={now():.2f}", flush=True)
        return out

    CarSystem.on_step_end = on_step_end

    original_done = RacingSimulation.handle_done

    def handle_done(self):
        obs = self.drivers[0].driver_observation or {}
        term = obs.get("episode_termination") or {}
        reasons = [k for k in ("collision", "virtual_opponent_collision", "interrupted") if term.get(k)]
        if not hw:
            # car_system prints this line for the HirarchicalPlanner only.
            print(f"[{Settings.CONTROLLER}] Episode reset: {', '.join(reasons) or 'done'}", flush=True)
        st["collisions"].append({"t": now(), "frozen": frozen(), "reasons": reasons})
        print(f"BENCH_COLLISION t={now():.2f} frozen={frozen()} reasons={reasons}", flush=True)
        return original_done(self)

    RacingSimulation.handle_done = handle_done

    original_step = RacingSimulation.simulation_step

    def simulation_step(self):
        out = original_step(self)
        if hw and st["initial"] is None and st["planner"] is not None:
            st["initial"] = hw_table(st["planner"])
        if hw and st["freeze_t"] is None and frozen():
            st["freeze_t"], st["freeze_lap"] = now(), len(st["laps"])
            print(f"BENCH_FREEZE t={now():.2f} after_lap={len(st['laps'])}", flush=True)
        return out

    RacingSimulation.simulation_step = simulation_step

    print(f"BENCH_START {json.dumps(vars(args))} steps={Settings.SIMULATION_LENGTH} "
          f"noise={Settings.NOISE_LEVEL_CAR_STATE} mu={Settings.SURFACE_FRICTION} "
          f"controller={Settings.CONTROLLER} trajectories={getattr(Settings, 'MPC_NUM_OF_TRAJECTORIES', None)}",
          flush=True)
    t0 = time.time()
    error = None
    sim = None
    try:
        sim = RacingSimulation()
        st["sim"] = sim
        sim.prepare_simulation()
        if args.controller == "mppi":
            print(f"BENCH_MPPI_BATCH {st['planner'].batch_size}", flush=True)
        sim.run_simulation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    wall = time.time() - t0
    sim_seconds = now()

    laps, cols = st["laps"], st["collisions"]
    clusters = merge_collisions(cols, laps)
    result = {
        "config": vars(args),
        "controller_name": CONTROLLERS[args.controller],
        "sim_seconds": sim_seconds,
        "wall_seconds": wall,
        "error": error,
        "noise_level_car_state": list(Settings.NOISE_LEVEL_CAR_STATE),
        "surface_friction": Settings.SURFACE_FRICTION,
        "laps": laps,
        "collisions_raw": cols,
        "collision_clusters": clusters,
        "n_collisions": len(clusters),
        "n_collisions_raw": len(cols),
        "sim_collision_steps": st.get("raw_collision_steps", 0),
        "sim_collision_steps_without_reset": st.get("collision_without_reset", 0),
        "lap_stats": lap_stats([l["lap_time"] for l in laps]),
    }
    if hw:
        def phase(frozen_flag):
            pl = [l["lap_time"] for l in laps if l["frozen"] == frozen_flag]
            pc = [c for c in clusters if cols[c[0]]["frozen"] == frozen_flag]
            pr = [c for c in cols if c["frozen"] == frozen_flag]
            return {"lap_times": pl, "lap_stats": lap_stats(pl), "n_collisions": len(pc), "n_collisions_raw": len(pr)}

        result["freeze_t"] = st["freeze_t"]
        result["freeze_after_lap"] = st["freeze_lap"]
        result["before_freeze"] = phase(False)
        result["after_freeze"] = phase(True)
        try:
            from bench_svg import write_waypoint_svg

            vm = v_max(Settings)
            final = hw_table(st["planner"]) if st["planner"] is not None else None
            for name, table in (("initial", st["initial"]), ("final", final)):
                if table is not None:
                    write_waypoint_svg(os.path.join(args.out, f"{name}_raceline.svg"), Settings.MAP_PATH, args.map,
                                       table["xy"], table["speed"], table["heading"], v_max=vm,
                                       title=f"{name} raceline")
                    np.savez_compressed(os.path.join(args.out, f"{name}_raceline.npz"), **table)
        except Exception:
            traceback.print_exc()
    # A failed run is not written as result.json, so the launcher retries it on resume.
    with open(os.path.join(args.out, "result.json" if error is None else "failed.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"BENCH_DONE sim={sim_seconds:.1f}s wall={wall:.0f}s laps={len(laps)} collisions={len(clusters)} "
          f"(raw {len(cols)}) freeze={st['freeze_t']} error={error}", flush=True)
    # The HW TCP client thread and JAX can keep the interpreter alive.
    sys.stdout.flush()
    os._exit(0 if error is None else 1)


if __name__ == "__main__":
    sys.exit(main())

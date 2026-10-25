"""Grid search of ``MPC_WALL_DISTANCE_WEIGHT`` with the Pacejka planner.

    python scripts/grid_wall_distance.py --model hwm_weighted_SE_test_7 --laps 30 --seeds 0 1 2

Each (weight, seed) pair is a separate process. The report is written to
``Emilio_Journal/profiling/wall_distance_grid.*``. Run from ``f1tenth_development_gym``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WEIGHTS_DEFAULT = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SEEDS_DEFAULT = (0, 1, 2)


def child(weight: float, seed: int, model: str, laps: int, out: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, os.getcwd())
    from utilities.Settings import Settings

    Settings.CONTROLLER = "HirarchicalPlanner"
    Settings.RENDER_MODE = None
    Settings.HWM_INFERENCE_MODEL_NAME = model
    Settings.HWM_SIM_SPEED = None
    Settings.MAX_SIM_FREQUENCY = None
    Settings.HWM_PP_DRIVES = False
    Settings.RUN_MPC_WITH_PACEJKA = True
    Settings.HWM_LIVE_MODE = True
    Settings.HWM_PROFILE = False
    Settings.MPC_WALL_DISTANCE_WEIGHT = float(weight)
    Settings.RANDOM_SEED = int(seed)
    Settings.STOP_AFTER_N_LAPS = None  # car_system.reset() after a crash wipes laptimes; the planner keeps them
    Settings.RESET_ON_DONE = True
    Settings.SAVE_RECORDINGS = False
    Settings.SAVE_PLOTS = False
    Settings.SAVE_REWARDS = False
    # Control steps. Each lap is ~30 s / 0.04 s; extra budget is for crash resets.
    Settings.SIMULATION_LENGTH = int(laps) * 2500 + 50_000

    from run.run_simulation import RacingSimulation

    t0 = time.time()
    sim = RacingSimulation()
    try:
        sim.prepare_simulation()
        sim.reset()
        planner = getattr(sim.drivers[0], "planner", None)
        for sim.sim_index in range(int(Settings.SIMULATION_LENGTH)):
            sim.simulation_step()
            if planner is not None and int(getattr(planner, "_completed_laps", 0)) >= int(laps):
                break
        sim.on_simulation_end(collision=False)
    except Exception as error:
        print(f"[grid] run failed: {type(error).__name__}: {error}")
    driver = sim.drivers[0]
    planner = getattr(driver, "planner", None)
    laps_done = list(getattr(planner, "_grid_laps", None) or getattr(driver, "laptimes", []))
    payload = {
        "weight": float(weight),
        "seed": int(seed),
        "wall_time_s": time.time() - t0,
        "sim_time_s": float(sim.sim_time),
        "laps": [float(x) for x in laps_done],
        "n_laps": int(len(laps_done)),
        "crashes": int(getattr(planner, "_grid_crashes", 0)) if planner is not None else 0,
        "crash_reasons": dict(getattr(planner, "_grid_crash_reasons", {})) if planner is not None else {},
        "n_episodes": int(getattr(planner, "_episode_id", 0)) + 1 if planner is not None else 0,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f)
    print(
        f"[grid] w={weight} seed={seed}: {payload['n_laps']} laps, {payload['crashes']} crashes, "
        f"{payload['sim_time_s']:.1f}s sim, {payload['wall_time_s']:.0f}s wall"
    )


def summarize(rows: list[dict]) -> str:
    import numpy as np

    by_w: dict[float, list[dict]] = {}
    for r in rows:
        by_w.setdefault(float(r["weight"]), []).append(r)
    lines = [
        "MPC_WALL_DISTANCE_WEIGHT grid (Pacejka, 3 seeds, 30-lap target). "
        "A crash is an HWM episode reset (collision, stuck, spinning).",
        "",
        f"{'weight':>8} {'laps':>6} {'mean lap':>9} {'std lap':>8} {'min':>7} {'max':>7} "
        f"{'crashes':>8} {'crash/lap':>9} {'sim h':>7} {'wall min':>9}",
    ]
    for w in sorted(by_w):
        group = by_w[w]
        laps = [t for r in group for t in r["laps"]]
        crashes = sum(int(r["crashes"]) for r in group)
        n_laps = len(laps)
        a = np.asarray(laps, dtype=np.float64)
        mean = float(a.mean()) if n_laps else float("nan")
        std = float(a.std(ddof=1)) if n_laps > 1 else 0.0
        mn = float(a.min()) if n_laps else float("nan")
        mx = float(a.max()) if n_laps else float("nan")
        sim_h = sum(float(r["sim_time_s"]) for r in group) / 3600.0
        wall_min = sum(float(r["wall_time_s"]) for r in group) / 60.0
        rate = crashes / n_laps if n_laps else float("nan")
        lines.append(
            f"{w:8.2f} {n_laps:6d} {mean:9.3f} {std:8.3f} {mn:7.3f} {mx:7.3f} "
            f"{crashes:8d} {rate:9.3f} {sim_h:7.2f} {wall_min:9.1f}"
        )
        for r in sorted(group, key=lambda x: x["seed"]):
            sl = r["laps"]
            sa = np.asarray(sl, dtype=np.float64)
            sm = float(sa.mean()) if sl else float("nan")
            lines.append(
                f"{'':>8} seed {r['seed']:<3} {len(sl):3d} laps, {sm:7.3f}s mean, "
                f"{r['crashes']:3d} crashes {r['crash_reasons']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--laps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--weights", type=float, nargs="+", default=list(WEIGHTS_DEFAULT))
    parser.add_argument("--out", default="Emilio_Journal/profiling/wall_distance_grid")
    parser.add_argument("--jobs", type=int, default=3, help="parallel (weight, seed) runs")
    parser.add_argument("--child", nargs=2, metavar=("WEIGHT", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        w, seed = args.child
        child(float(w), int(seed), args.model, args.laps, f"{args.out}_{w}_{seed}.json")
        return

    jobs = []
    for w in args.weights:
        for seed in args.seeds:
            jobs.append((w, seed))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} runs, {args.jobs} at a time, weights={args.weights}, seeds={args.seeds}", flush=True)
    running: list[tuple[subprocess.Popen, float, int]] = []
    finished = 0
    for w, seed in jobs:
        while len(running) >= args.jobs:
            still = []
            for proc, ww, ss in running:
                if proc.poll() is None:
                    still.append((proc, ww, ss))
                else:
                    finished += 1
                    print(f"  finished {finished}/{len(jobs)} (w={ww} seed={ss} rc={proc.returncode})", flush=True)
            running = still
            if len(running) >= args.jobs:
                time.sleep(2)
        proc = subprocess.Popen(
            [sys.executable, __file__, "--model", args.model, "--laps", str(args.laps),
             "--out", args.out, "--child", str(w), str(seed)],
        )
        running.append((proc, w, seed))
    for proc, w, seed in running:
        proc.wait()
        finished += 1
        print(f"  finished {finished}/{len(jobs)} (w={w} seed={seed} rc={proc.returncode})", flush=True)

    rows = []
    for w, seed in jobs:
        path = Path(f"{args.out}_{w}_{seed}.json")
        if path.is_file():
            rows.append(json.loads(path.read_text()))
    text = summarize(rows)
    Path(args.out + ".txt").write_text(text + "\n")
    Path(args.out + ".json").write_text(json.dumps(rows, indent=2))
    print(text)


if __name__ == "__main__":
    main()

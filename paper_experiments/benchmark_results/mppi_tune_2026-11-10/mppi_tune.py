"""MPPI (mppi-lite-jax) tuning on icra2022, 2000 rollouts, with scripts/benchmark/bench_run.py of the working copy.

    python mppi_tune.py run CONFIGS.json NAME SEED SIM_SECONDS OUTDIR      (one run)
    python mppi_tune.py sweep CONFIGS.json TAG SEEDS SIM_SECONDS PAR [--cpu]
    python mppi_tune.py summary CONFIGS.json TAG

CONFIGS.json: {"name": {key: value}}. Keys:
  "w:<weight>"  entry of DEFAULT_COST_WEIGHTS (new keys are added, e.g. w:progress_sdot)
  "p:<attr>"    planner attribute set before compilation (horizon, noise_std, intermediate_steps, plan_filter_window)
  "<SETTING>"   Settings attribute
Raceline, mu 0.9, state noise on (Settings default), GPU unless --cpu. Runs go to runs/<TAG>/<name>/seed<k>/.
"""
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]  # repository root (paper_experiments/benchmark_results/<dir>/<script>)
GYM = os.environ.get("MPPI_GYM", str(REPO))
PY = os.environ.get("PY", sys.executable)
JAX_CUDA = os.environ.get("JAX_CUDA", "")
ROLLOUTS = os.environ.get("MPPI_ROLLOUTS", "2000")


def run_one(cfg_file, name, seed, sim_s, out, gpu):
    os.chdir(GYM)
    sys.path.insert(0, GYM)
    sys.path.insert(0, os.path.join(GYM, "scripts", "benchmark"))
    from utilities.Settings import Settings
    overrides = dict(json.load(open(cfg_file))[name])
    weights = {k[2:]: v for k, v in overrides.items() if k.startswith("w:")}
    attrs = {k[2:]: v for k, v in overrides.items() if k.startswith("p:")}
    for k, v in overrides.items():
        if k[:2] in ("w:", "p:"):
            continue
        if not hasattr(Settings, k):
            raise KeyError(k)
        setattr(Settings, k, v)

    import numpy as np
    # Set the map before the planner is imported: WaypointUtils.__init__ takes Settings.MAP_NAME / MAP_PATH as
    # default arguments, which Python evaluates at import time. Otherwise the planner keeps the Settings
    # default map (icra2022) on any other --map, and the car cannot follow the waypoints.
    Settings.MAP_NAME = os.environ.get("MPPI_MAP", "icra2022")
    Settings.recalculate_paths()
    from Control_Toolkit_ASF.Controllers.MPPILite import mppi_lite_jax_planner as mp
    mp.DEFAULT_COST_WEIGHTS.update(weights)
    original_precompile = mp.MPPILitePlanner._precompile_functions

    def precompile(self):
        for k, v in attrs.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, tuple(v) if isinstance(v, list) else v)
        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        return original_precompile(self)

    mp.MPPILitePlanner._precompile_functions = precompile
    print("TUNE", name, overrides, flush=True)
    sys.argv = ["bench_run.py", "--out", out, "--controller", "mppi", "--rollouts", ROLLOUTS, "--map", os.environ.get("MPPI_MAP", "icra2022"),
                "--wp", "raceline", "--mu", os.environ.get("MPPI_MU", "0.9"), "--noise", os.environ.get("MPPI_NOISE", "on"), "--seed", str(seed),
                "--sim-seconds", str(sim_s)] + (["--gpu"] if gpu else [])
    import bench_run
    return bench_run.main()


def score(r):
    laps = [l["lap_time"] for l in r["laps"]][1:]  # first lap starts from standstill
    return {"n": len(laps), "best": min(laps) if laps else None,
            "med": statistics.median(laps) if laps else None,
            "mean": statistics.mean(laps) if laps else None,
            "coll": r["n_collisions"], "wall": r["wall_seconds"],
            "slow": sum(l > min(laps) + 1.0 for l in laps) if laps else 0}


def summary(cfg_file, tag):
    cfgs = json.load(open(cfg_file))
    rows = []
    for name in cfgs:
        rs = sorted((HERE / "runs" / tag / name).glob("seed*/result.json"))
        if not rs:
            continue
        sc = [score(json.load(open(p))) for p in rs]
        nan = [float("nan")]
        rows.append((statistics.mean([x["med"] for x in sc if x["med"]] or nan), name,
                     statistics.mean([x["mean"] for x in sc if x["mean"]] or nan),
                     min([x["best"] for x in sc if x["best"]] or nan),
                     sum(x["coll"] for x in sc), sum(x["slow"] for x in sc), sum(x["n"] for x in sc), len(sc),
                     statistics.mean(x["wall"] for x in sc)))
    rows.sort()
    print(f"{'config':30s} {'median':>7s} {'mean':>7s} {'best':>7s} {'coll':>5s} {'slow':>5s} {'laps':>5s} {'runs':>4s} {'wall':>6s}")
    for med, name, mean, best, c, sl, n, k, w in rows:
        print(f"{name:30s} {med:7.3f} {mean:7.3f} {best:7.3f} {c:5d} {sl:5d} {n:5d} {k:4d} {w:6.0f}")


def sweep(cfg_file, tag, seeds, sim_s, par, gpu):
    cfgs = json.load(open(cfg_file))
    off = int(os.environ.get("MPPI_SEED_OFFSET", "0"))
    jobs = [(n, s) for s in range(off, off + seeds) for n in cfgs]

    def go(job):
        n, s = job
        out = HERE / "runs" / tag / n / f"seed{s}"
        if (out / "result.json").exists():
            return
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TQDM_DISABLE="1")
        if gpu:
            env.update(PYTHONPATH=JAX_CUDA, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        else:
            env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        cmd = [PY, str(Path(__file__).resolve()), "run", str(Path(cfg_file).resolve()), n, str(s), str(sim_s), str(out)]
        if gpu:
            cmd.append("--gpu")
        with open(out / "run.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=GYM)

    with ThreadPoolExecutor(par) as ex:
        list(ex.map(go, jobs))
    summary(cfg_file, tag)


if __name__ == "__main__":
    mode = sys.argv[1]
    gpu = "--cpu" not in sys.argv
    a = [x for x in sys.argv[2:] if x not in ("--cpu", "--gpu")]
    if mode == "run":
        sys.exit(run_one(a[0], a[1], int(a[2]), float(a[3]), a[4], gpu))
    elif mode == "sweep":
        sweep(a[0], a[1], int(a[2]), float(a[3]), int(a[4]), gpu)
    else:
        summary(a[0], a[1])

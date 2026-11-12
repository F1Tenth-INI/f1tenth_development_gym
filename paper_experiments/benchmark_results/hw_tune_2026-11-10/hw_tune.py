"""HW tuning on icra2022 with scripts/benchmark/bench_run.py of the working copy.

    python hw_tune.py run CONFIGS.json NAME SEED SIM_SECONDS OUTDIR [--gpu]   (one run)
    python hw_tune.py sweep CONFIGS.json TAG SEEDS SIM_SECONDS PAR [--gpu]
    python hw_tune.py summary CONFIGS.json TAG

CONFIGS.json: {"name": {"SETTING": value, ...}}, applied to Settings before bench_run.
Keys: "_controller" (hw/acados), "env:NAME" sets an environment variable.
Env: HW_NOISE (on/off), HW_ROLLOUTS (default 1000), HW_SEED_OFFSET (first seed), HW_GYM (repo root).
Raceline, mu 0.9, noise on. Runs go to runs/<TAG>/<name>/seed<k>/; finished runs are skipped.
"""
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GYM = os.environ.get("HW_GYM", "/home/emilio/projects/bachelor/f1tenth_development_gym")
PY = "/home/emilio/miniconda3/envs/f1t/bin/python"
JAX_CUDA = "/home/emilio/projects/bachelor/benchmark_results/jax_cuda"
ROLLOUTS = os.environ.get("HW_ROLLOUTS", "1000")


def run_one(cfg_file, name, seed, sim_s, out, gpu):
    os.chdir(GYM)
    sys.path.insert(0, GYM)
    sys.path.insert(0, os.path.join(GYM, "scripts", "benchmark"))
    from utilities.Settings import Settings
    overrides = dict(json.load(open(cfg_file))[name])
    controller = overrides.pop("_controller", "hw")
    for k in [k for k in overrides if k.startswith("env:")]:
        os.environ[k[4:]] = str(overrides.pop(k))
    for k, v in overrides.items():
        if not hasattr(Settings, k):
            raise KeyError(k)
        setattr(Settings, k, v)
    print("TUNE", name, overrides, flush=True)
    sys.argv = ["bench_run.py", "--out", out, "--controller", controller, "--rollouts", ROLLOUTS, "--map", "icra2022",
                "--wp", "raceline", "--mu", "0.9", "--noise", os.environ.get("HW_NOISE", "on"), "--seed", str(seed),
                "--sim-seconds", str(sim_s)] + (["--gpu"] if gpu else [])
    import bench_run
    real_exit = os._exit

    def exit_with_profile(code):  # bench_run leaves through os._exit, which skips the profiler's atexit hook
        try:
            from TrainingLite.hwm.profiling import PROF
            PROF.report_at_exit()
        except Exception as error:
            print("profile dump failed:", error, flush=True)
        real_exit(code)

    os._exit = exit_with_profile
    return bench_run.main()


def score(r):
    laps = [l["lap_time"] for l in r["laps"]]
    tail = laps[len(laps) // 2:]  # second half of the run: after learning
    return {
        "n": len(laps), "best": min(laps) if laps else None,
        "med_tail": statistics.median(tail) if tail else None,
        "mean_tail": statistics.mean(tail) if tail else None,
        "coll": r["n_collisions"], "wall": r["wall_seconds"],
    }


def summary(cfg_file, tag):
    cfgs = json.load(open(cfg_file))
    rows = []
    for name in cfgs:
        rs = sorted((HERE / "runs" / tag / name).glob("seed*/result.json"))
        if not rs:
            continue
        sc = [score(json.load(open(p))) for p in rs]
        med = statistics.mean([x["med_tail"] for x in sc if x["med_tail"]] or [float("nan")])
        mean = statistics.mean([x["mean_tail"] for x in sc if x["mean_tail"]] or [float("nan")])
        best = min([x["best"] for x in sc if x["best"]] or [float("nan")])
        rows.append((med, name, mean, best, sum(x["coll"] for x in sc), sum(x["n"] for x in sc), len(sc),
                     statistics.mean(x["wall"] for x in sc)))
    rows.sort()
    print(f"{'config':28s} {'medTail':>8s} {'meanTail':>8s} {'best':>7s} {'coll':>5s} {'laps':>5s} {'runs':>4s} {'wall':>6s}")
    for med, name, mean, best, c, n, k, w in rows:
        print(f"{name:28s} {med:8.3f} {mean:8.3f} {best:7.3f} {c:5d} {n:5d} {k:4d} {w:6.0f}")


def sweep(cfg_file, tag, seeds, sim_s, par, gpu):
    cfgs = json.load(open(cfg_file))
    off = int(os.environ.get("HW_SEED_OFFSET", "0"))
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
    gpu = "--gpu" in sys.argv
    a = [x for x in sys.argv[2:] if x != "--gpu"]
    if mode == "run":
        sys.exit(run_one(a[0], a[1], int(a[2]), float(a[3]), a[4], gpu))
    elif mode == "sweep":
        sweep(a[0], a[1], int(a[2]), float(a[3]), int(a[4]), gpu)
    else:
        summary(a[0], a[1])

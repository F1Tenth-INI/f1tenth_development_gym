#!/usr/bin/env python3
"""Benchmark grid: HW, MPPI and RPGD on several maps, both waypoint styles, two frictions,
with and without observation noise, several seeds, a fixed simulated time per run.

The code is copied once into <out>/code (tracked and untracked-but-not-ignored files of the
working tree), so editing the repository while the grid runs does not change it. Runs whose
result.json exists are skipped, so the same command resumes an interrupted grid.

    python scripts/benchmark/benchmark.py --out /path/to/benchmark_out --workers 28
    python scripts/benchmark/benchmark.py --out /tmp/bench_smoke --sim-seconds 30 --seeds 0 \
        --maps RCA1 --controllers hw:50,mppi:50,rpgd       # quick check

Output:
    <out>/runs/<run_id>/result.json, run.log.gz, and for HW initial/final_raceline.svg
    <out>/initial_configs/initial_<map>_<wp>.svg
    <out>/results.jsonl   one line per finished run (the result.json content)
    <out>/summary.csv     one row per run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = "/home/emilio/miniconda3/envs/f1t/bin/python"
HERE = Path(__file__).resolve().parent

DEFAULT_CONTROLLERS = "hw:50,hw:200,mppi:50,mppi:250,mppi:500,mppi:1000,mppi:2000,rpgd"


def snapshot_code(dest: Path) -> None:
    if (dest / "run.py").exists():
        print(f"using existing code snapshot {dest}", flush=True)
        return
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO, check=True, capture_output=True,
    ).stdout.split(b"\0")
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        if not f:
            continue
        rel = f.decode()
        src = REPO / rel
        if not src.is_file():
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        n += 1
    # Maps can be gitignored (IPZ50), copy the whole folder.
    shutil.copytree(REPO / "utilities" / "maps", dest / "utilities" / "maps", dirs_exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout
    (dest / "BENCH_SNAPSHOT_HEAD.txt").write_text(head + "\n")
    (dest / "BENCH_SNAPSHOT_DIFF.patch").write_text(diff)
    print(f"copied {n} files of {REPO} (HEAD {head[:8]} + working tree) to {dest}", flush=True)


def run_id(job: dict) -> str:
    roll = f"{job['rollouts']}" if job["rollouts"] is not None else "def"
    return (f"{job['controller']}{roll}_{job['map']}_{job['wp']}_mu{job['mu']:g}_"
            f"noise{job['noise']}_seed{job['seed']}")


def build_jobs(args) -> list[dict]:
    controllers = []
    for item in args.controllers.split(","):
        name, _, roll = item.partition(":")
        controllers.append((name, int(roll) if roll else None))
    jobs = []
    # Seed outermost, so a partial grid already covers every configuration.
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name, roll in controllers:
            for m in args.maps.split(","):
                for wp in args.wps.split(","):
                    for mu in [float(x) for x in args.mus.split(",")]:
                        for noise in args.noise.split(","):
                            jobs.append({"controller": name, "rollouts": roll, "map": m, "wp": wp,
                                         "mu": mu, "noise": noise, "seed": seed})
    # Slow runs first within a seed, so the pool does not end on a few long ones.
    cost = {"hw": 3, "mppi": 1, "rpgd": 1}
    jobs.sort(key=lambda j: (j["seed"], -cost[j["controller"]] * (j["rollouts"] or 50)))
    return jobs


_lock = threading.Lock()


def run_job(job: dict, code: Path, runs: Path, out: Path, sim_seconds: float, timeout: float,
            gpu_path: str | None = None) -> dict | None:
    rid = run_id(job)
    d = runs / rid
    if (d / "result.json").exists():
        return None
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    cmd = [PY, "-u", str(code / "scripts" / "benchmark" / "bench_run.py"), "--out", str(d),
           "--controller", job["controller"], "--map", job["map"], "--wp", job["wp"],
           "--mu", str(job["mu"]), "--noise", job["noise"], "--seed", str(job["seed"]),
           "--sim-seconds", str(sim_seconds)]
    if job["rollouts"] is not None:
        cmd += ["--rollouts", str(job["rollouts"])]
    env = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
    if gpu_path is not None:
        cmd.append("--gpu")
        env["PYTHONPATH"] = gpu_path
    log_path = d / "run.log"
    t0 = time.time()
    timed_out = False
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=str(code), stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    with log_path.open("rb") as src, gzip.open(str(log_path) + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    log_path.unlink()
    res_path = d / "result.json"
    if res_path.exists():
        result = json.loads(res_path.read_text())
    elif (d / "failed.json").exists():
        result = json.loads((d / "failed.json").read_text())
    else:
        result = {"config": job, "error": f"no result (exit {proc.returncode}, timed_out={timed_out})"}
        # Not written as result.json, so a resume retries it.
        (d / "failed.json").write_text(json.dumps(result, indent=1))
    result["run_id"] = rid
    result["device"] = "gpu" if gpu_path is not None else "cpu"
    if res_path.exists():
        saved = json.loads(res_path.read_text())
        saved["device"] = result["device"]
        res_path.write_text(json.dumps(saved, indent=1))
    result["launcher_wall_seconds"] = time.time() - t0
    with _lock:
        with (out / "results.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        ls = result.get("lap_stats") or {}
        extra = ""
        if job["controller"] == "hw" and "before_freeze" in result:
            extra = (f" | before freeze: {result['before_freeze']['lap_stats']['n']} laps, "
                     f"{result['before_freeze']['n_collisions']} coll; after: "
                     f"{result['after_freeze']['lap_stats']['n']} laps min {result['after_freeze']['lap_stats']['min']}, "
                     f"{result['after_freeze']['n_collisions']} coll")
        print(f"[{time.strftime('%H:%M:%S')}] {rid}: laps {ls.get('n')} min {ls.get('min')} "
              f"coll {result.get('n_collisions')} wall {result['launcher_wall_seconds']:.0f}s "
              f"err {result.get('error')}{extra}", flush=True)
    return result


def write_summary(out: Path, runs: Path) -> None:
    rows = []
    for d in sorted(runs.iterdir()):
        p = d / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        c = r["config"]
        row = {"run_id": d.name, "controller": c["controller"], "rollouts": c.get("rollouts"), "map": c["map"],
               "wp": c["wp"], "mu": c["mu"], "noise": c["noise"], "seed": c["seed"],
               "device": r.get("device"), "sim_seconds": r.get("sim_seconds"), "wall_seconds": round(r.get("wall_seconds") or 0),
               "laps": r["lap_stats"]["n"], "lap_min": r["lap_stats"]["min"], "lap_mean": r["lap_stats"]["mean"],
               "lap_median": r["lap_stats"]["median"], "collisions": r["n_collisions"],
               "collisions_raw": r["n_collisions_raw"], "error": r.get("error")}
        for ph in ("before_freeze", "after_freeze"):
            if ph in r:
                row[f"{ph}_laps"] = r[ph]["lap_stats"]["n"]
                row[f"{ph}_lap_min"] = r[ph]["lap_stats"]["min"]
                row[f"{ph}_lap_mean"] = r[ph]["lap_stats"]["mean"]
                row[f"{ph}_collisions"] = r[ph]["n_collisions"]
            row["freeze_t"] = r.get("freeze_t")
        rows.append(row)
    if not rows:
        return
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--maps", default="RCA1,IPZ50,hangar3,icra2022")
    p.add_argument("--wps", default="midline,raceline")
    p.add_argument("--mus", default="0.5,0.9")
    p.add_argument("--noise", default="on,off")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--controllers", default=DEFAULT_CONTROLLERS, help="name[:rollouts],...  names: hw, mppi, rpgd")
    p.add_argument("--sim-seconds", type=float, default=1000.0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    p.add_argument("--timeout", type=float, default=4 * 3600, help="wall seconds per run")
    p.add_argument("--gpu-controllers", default="mppi:1000,mppi:2000",
                   help="name:rollouts entries whose JAX runs on the GPU. Empty: none")
    p.add_argument("--gpu-workers", type=int, default=8)
    p.add_argument("--jax-cuda-path", default=str(REPO.parent / "benchmark_results" / "jax_cuda"),
                   help="pip --target folder with jax-cuda12-plugin, put on PYTHONPATH for GPU runs")
    args = p.parse_args()

    out = Path(args.out).resolve()
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    code = out / "code"
    snapshot_code(code)
    # The runner and SVG writer always come from this script's folder, so fixes to them apply.
    for f in ("bench_run.py", "bench_svg.py"):
        shutil.copy2(HERE / f, code / "scripts" / "benchmark" / f)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    cfg_dir = out / "initial_configs"
    for m in args.maps.split(","):
        for wp in args.wps.split(","):
            if (cfg_dir / f"initial_{m}_{wp}.svg").exists():
                continue
            subprocess.run([PY, str(code / "scripts" / "benchmark" / "bench_run.py"), "--out", str(cfg_dir),
                            "--map", m, "--wp", wp, "--initial-svg-only"], cwd=str(code), check=False,
                           env=dict(os.environ, TQDM_DISABLE="1"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"initial config svg {m} {wp}: {(cfg_dir / f'initial_{m}_{wp}.svg').exists()}", flush=True)

    jobs = build_jobs(args)
    todo = [j for j in jobs if not (runs / run_id(j) / "result.json").exists()]
    gpu_set = {item for item in args.gpu_controllers.split(",") if item}
    use_gpu = bool(gpu_set) and args.gpu_workers > 0 and Path(args.jax_cuda_path).is_dir()

    def on_gpu(j):
        return use_gpu and f"{j['controller']}:{j['rollouts']}" in gpu_set

    n_gpu = sum(on_gpu(j) for j in todo)
    print(f"{len(jobs)} runs, {len(todo)} to do ({n_gpu} on the GPU), {args.workers} CPU workers, "
          f"{args.gpu_workers if use_gpu else 0} GPU workers, {args.sim_seconds:g} s simulated each", flush=True)
    with ThreadPoolExecutor(args.workers) as cpu_pool, ThreadPoolExecutor(max(1, args.gpu_workers)) as gpu_pool:
        futures = [
            (gpu_pool if on_gpu(j) else cpu_pool).submit(
                run_job, j, code, runs, out, args.sim_seconds, args.timeout,
                args.jax_cuda_path if on_gpu(j) else None)
            for j in todo
        ]
        for i, fut in enumerate(futures):
            fut.result()
            if i % 20 == 0:
                write_summary(out, runs)
    write_summary(out, runs)
    print("all done", flush=True)


if __name__ == "__main__":
    main()

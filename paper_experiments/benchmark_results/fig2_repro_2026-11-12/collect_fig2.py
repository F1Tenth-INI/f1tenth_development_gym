#!/usr/bin/env python3
"""Recreate the data of the paper's Fig. 2 (icra2022, raceline start, mu 0.5, no noise, seed 1) and redraw it.

Three runs of 1000 s simulated time each:

    ATMPPI 50     config wm10_tsf103 of hw50_tune_2026-11-10/r2.json        (paper: hw50_tune_2026-11-10/runs/r2/wm10_tsf103/seed1)
    ATMPPI 1024   config wm10_tsf103 of hw50_tune_2026-11-10/r3_n1000.json  (paper: 1000 rollouts, .../runs/r3_n1000/wm10_tsf103/seed1)
    MPPI 2048     config ns_s02 of mppi_tune_2026-11-10/plotrun.json        (paper: 2000 rollouts, mppi_tune_2026-11-10/runs/plotrun/ns_s02/seed1)
                  (steering noise 0.2)

The runs go through the original harnesses (hw50_tune.py / mppi_tune.py, mode "run") with the original
config files, on the code snapshots the paper runs used:

    ATMPPI  commit aac2064a + tune_noise_2026-11-08/code/BENCH_SNAPSHOT_DIFF.patch
    MPPI    commit 986f92d7 + mppi_tune_2026-11-10/code_snapshot/BENCH_SNAPSHOT_DIFF.patch

Usage (from anywhere, with the f1t conda env):

    python collect_fig2.py setup        # git worktrees ../snapshot_aac2064a and ../snapshot_986f92d7 next to the repo
    python collect_fig2.py run [--sim-seconds 1000] [--only at50,at1024,mppi2048] [--gpu]
    python collect_fig2.py plot         # figures/icra_laptimes_repro.{pdf,png} + comparison with the paper runs

Runs go to runs/<name>/ (result.json, run.log); a run whose result.json exists is skipped.
The paper's MPPI run used the GPU (JAX with CUDA). Without --gpu, MPPI runs on the CPU, so its numbers
can differ slightly from the paper's run (different floating-point reductions).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BR = HERE.parent  # paper_experiments/benchmark_results
REPO = HERE.parents[2]
PY = os.environ.get("PY", sys.executable)

SNAP_AT = Path(os.environ.get("SNAP_AT", REPO.parent / "snapshot_aac2064a"))
SNAP_MPPI = Path(os.environ.get("SNAP_MPPI", REPO.parent / "snapshot_986f92d7"))
SNAPSHOTS = {  # worktree: (commit, patch)
    SNAP_AT: ("aac2064a8bbb455e6f47ef4bd19930fd325f2855", BR / "tune_noise_2026-11-08" / "code" / "BENCH_SNAPSHOT_DIFF.patch"),
    SNAP_MPPI: ("986f92d7be456da955b474a035936dc43536e2e5", BR / "mppi_tune_2026-11-10" / "code_snapshot" / "BENCH_SNAPSHOT_DIFF.patch"),
}
SEED = 1
RUNS = {  # name: (harness, config file, config name, gym, env)
    "at50": (BR / "hw50_tune_2026-11-10" / "hw50_tune.py", BR / "hw50_tune_2026-11-10" / "r2.json", "wm10_tsf103", SNAP_AT,
             {"HW_ROLLOUTS": "50", "HW_MU": "0.5", "HW_NOISE": "off", "HW_MAP": "icra2022"}),
    "at1024": (BR / "hw50_tune_2026-11-10" / "hw50_tune.py", BR / "hw50_tune_2026-11-10" / "r3_n1000.json", "wm10_tsf103", SNAP_AT,
               {"HW_ROLLOUTS": "1024", "HW_MU": "0.5", "HW_NOISE": "off", "HW_MAP": "icra2022"}),
    "mppi2048": (BR / "mppi_tune_2026-11-10" / "mppi_tune.py", BR / "mppi_tune_2026-11-10" / "plotrun.json", "ns_s02", SNAP_MPPI,
                 {"MPPI_ROLLOUTS": "2048", "MPPI_MU": "0.5", "MPPI_NOISE": "off"}),
}
PAPER = {  # the runs behind the paper's figure (1000 and 2000 rollouts)
    "at50": BR / "hw50_tune_2026-11-10" / "runs" / "r2" / "wm10_tsf103" / f"seed{SEED}",
    "at1024": BR / "hw50_tune_2026-11-10" / "runs" / "r3_n1000" / "wm10_tsf103" / f"seed{SEED}",
    "mppi2048": BR / "mppi_tune_2026-11-10" / "runs" / "plotrun" / "ns_s02" / f"seed{SEED}",
}


def sh(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def setup():
    for wt, (commit, patch) in SNAPSHOTS.items():
        if wt.exists():
            print(f"{wt} exists, left as it is")
        else:
            make_snapshot(wt, commit, patch)
        copy_bench_scripts(wt)


def copy_bench_scripts(wt):
    # Commit 986f92d7 predates scripts/benchmark (first committed in be2256bd); the paper's MPPI runs used
    # an uncommitted copy. bench_run.py is the same in be2256bd and aac2064a, and the MPPI code is the
    # same in 986f92d7 and aac2064a, so the benchmark scripts are taken from aac2064a.
    bench = wt / "scripts" / "benchmark"
    if not (bench / "bench_run.py").exists():
        bench.mkdir(parents=True, exist_ok=True)
        names = subprocess.run(["git", "ls-tree", "--name-only", f"{SNAPSHOTS[SNAP_AT][0]}:scripts/benchmark"],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
        for f in names:
            blob = subprocess.run(["git", "show", f"{SNAPSHOTS[SNAP_AT][0]}:scripts/benchmark/{f}"], cwd=REPO,
                                  capture_output=True, check=True).stdout
            (bench / f).write_bytes(blob)
        print(f"copied scripts/benchmark/{{{','.join(names)}}} of aac2064a into {wt}")


def make_snapshot(wt, commit, patch):
    sh("git", "worktree", "add", "--detach", wt, commit, cwd=REPO)
    # The saved patch of the MPPI snapshot also replaces utilities/maps/icra2022/centerline.csv, which is
    # stored with CRLF line endings; the patch file lost the CR, so git apply cannot match it. The hunk
    # replaces the whole file, so its "+" lines are the new file (LF, blob fdb375ef as in the patch header).
    text = patch.read_text()
    csv = "utilities/maps/icra2022/centerline.csv"
    if f"+++ b/{csv}" in text:
        hunk = text.split(f"+++ b/{csv}\n", 1)[1].split("\ndiff --git", 1)[0]
        new = [l[1:] for l in hunk.splitlines() if l.startswith("+")]
        (wt / csv).write_text("\n".join(new) + "\n")
        sh("git", "apply", f"--exclude={csv}", patch, cwd=wt)
    else:
        sh("git", "apply", patch, cwd=wt)
    # submodules at the commits the snapshot pins, cloned from the main checkout's module repos (no network)
    for m in ("SI_Toolkit", "Control_Toolkit"):
        sub = wt / m
        pinned = subprocess.run(["git", "ls-tree", "HEAD", m], cwd=wt, capture_output=True, text=True,
                                check=True).stdout.split()[2]
        if sub.exists() and not any(sub.iterdir()):
            sub.rmdir()
        sh("git", "clone", "-q", "--shared", "--no-checkout", REPO / ".git" / "modules" / m, sub)
        sh("git", "checkout", "-q", "--detach", pinned, cwd=sub)
    sh("git", "status", "--short", cwd=wt)

def run(names, sim_s, gpu):
    def go(name):
        harness, cfg, cfg_name, gym, extra = RUNS[name]
        out = HERE / "runs" / name
        if (out / "result.json").exists():
            print(f"{name}: result.json exists, skipped")
            return
        if not (gym / "scripts" / "benchmark" / "bench_run.py").exists():
            raise SystemExit(f"{gym} missing, run: python {Path(__file__).name} setup")
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TQDM_DISABLE="1",
                   HW_GYM=str(gym), MPPI_GYM=str(gym), **extra)
        # Ignore ~/.local site-packages by default: there a newer numba sits next to an older llvmlite and
        # fails to import. USER_SITE=1 keeps them.
        if os.environ.get("USER_SITE", "0") != "1":
            env["PYTHONNOUSERSITE"] = "1"
        cmd = [PY, str(harness), "run", str(cfg), cfg_name, str(SEED), str(sim_s), str(out)]
        if name.startswith("mppi"):
            if gpu:  # as the paper run; needs JAX with CUDA (JAX_CUDA: pip --target folder with jax-cuda12-plugin)
                env.update(PYTHONPATH=os.environ.get("JAX_CUDA", ""), XLA_PYTHON_CLIENT_PREALLOCATE="false")
            else:
                cmd.append("--cpu")
                env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        else:
            env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        t0 = time.time()
        print(f"{name}: started -> {out}", flush=True)
        with open(out / "run.log", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=gym).returncode
        print(f"{name}: exit {code} after {time.time() - t0:.0f} s", flush=True)

    with ThreadPoolExecutor(len(names)) as ex:
        list(ex.map(go, names))


def stats(path):
    import numpy as np
    r = json.load(open(Path(path) / "result.json"))
    lt = np.array([l["lap_time"] for l in r["laps"]])
    frozen = np.array(r["after_freeze"]["lap_times"]) if "after_freeze" in r else lt[1:]
    return {"laps": len(lt), "first": lt[0], "freeze_lap": r.get("freeze_after_lap"), "freeze_t": r.get("freeze_t"),
            "mean": frozen.mean(), "std": frozen.std(), "median": np.median(frozen), "coll": r["n_collisions"],
            "wall": r["wall_seconds"]}


def plot():
    sys.path.insert(0, str(BR / "grid_2026-11-05" / "paper"))
    import matplotlib
    matplotlib.use("Agg")
    import make_sim_figures as sf
    import make_icra_laptimes_rollouts_v2 as v2
    sf.OUT = str(HERE / "figures")
    have = {n: HERE / "runs" / n for n in RUNS if (HERE / "runs" / n / "result.json").exists()}
    if "mppi2048" not in have:
        raise SystemExit("runs/mppi2048/result.json missing")
    v2.BLUES[1024] = v2.BLUES[1000]  # same colour as the paper's 1000-rollout curve
    at = {k: str(have[n]) for k, n in ((50, "at50"), (1024, "at1024")) if n in have}
    v2.main(runs=at, mppi_run=str(have["mppi2048"]), name="icra_laptimes_repro", mppi_rollouts=2048)
    v2.main(name="icra_laptimes_paper")  # the paper's runs, same script, for side-by-side comparison

    print(f"\n{'run':9s} {'':6s} {'laps':>4s} {'lap 1':>6s} {'frz lap':>7s} {'frz t':>7s} {'mean+-std':>12s} {'median':>7s} {'coll':>4s} {'wall':>6s}")
    for n in RUNS:
        for tag, p in (("paper", PAPER[n]), ("repro", have.get(n))):
            if p is None:
                continue
            s = stats(p)
            fl = "" if s["freeze_lap"] is None else s["freeze_lap"]
            ft = "" if s["freeze_t"] is None else f"{s['freeze_t']:.0f}"
            print(f"{n:9s} {tag:6s} {s['laps']:4d} {s['first']:6.2f} {fl!s:>7s} {ft:>7s} "
                  f"{s['mean']:6.2f}+-{s['std']:.2f} {s['median']:7.2f} {s['coll']:4d} {s['wall']:6.0f}")
    print("(ATMPPI: laps after the freeze; MPPI: laps from lap 2)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["setup", "run", "plot"])
    ap.add_argument("--sim-seconds", type=float, default=1000.0)
    ap.add_argument("--only", default=",".join(RUNS), help="comma-separated subset of " + ",".join(RUNS))
    ap.add_argument("--gpu", action="store_true", help="MPPI on the GPU, as the paper run")
    a = ap.parse_args()
    if a.mode == "setup":
        setup()
    elif a.mode == "run":
        run(a.only.split(","), a.sim_seconds, a.gpu)
    else:
        plot()

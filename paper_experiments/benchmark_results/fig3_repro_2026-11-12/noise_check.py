#!/usr/bin/env python3
"""Check: ATMPPI 50 on icra2022 with more steering sampling noise (like MPPI's 0.1 -> 0.2 for Fig. 2).

    python noise_check.py run  [--workers 30]   # 2 x 20 runs: runs_noise/at50_s<std>/...
    python noise_check.py summary               # vs the Fig. 3 re-run (runs/at50, steering std 0.08)

Paper config soft25w02 with MPC_DRIVE_NOISE = [std, 1.5] (paper: 0.08), half observation noise, 1000 s,
4 settings x seeds 0..4, same launcher and snapshot as fig3.py. Worker mode (internal):
``noise_check.py worker <std> <bench_run args>``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig3  # noqa: E402

STDS = ("0.12", "0.16")
ROOT = fig3.HERE / "runs_noise"


def worker(std, rest):
    sys.path.insert(0, str(fig3.TUNE))
    import tune_run
    name = f"soft25w02_s{std}"
    tune_run.CONFIGS[name] = dict(tune_run.CONFIGS["soft25w02"], MPC_DRIVE_NOISE=[float(std), 1.5])
    sys.argv = ["tune_run.py", name] + rest
    return tune_run.main()


def run(workers):
    jobs = [(std, wp, mu, s) for std in STDS for s in range(5) for wp in fig3.WPS for mu in fig3.MUS]

    def one(job):
        std, wp, mu, s = job
        out = ROOT / f"at50_s{std}" / f"icra2022_{wp}_mu{mu}_noisehalf_seed{s}"
        if (out / "result.json").exists():
            return f"skip {out}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, "worker", std, "--out", str(out), "--controller", "hw", "--rollouts", "50",
               "--map", "icra2022", "--wp", wp, "--mu", mu, "--noise", "on", "--seed", str(s), "--sim-seconds", "1000"]
        env = dict(os.environ, CODE_DIR=str(fig3.SNAP_AT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NOISE_SCALE="0.5", TQDM_DISABLE="1", PYTHONNOUSERSITE="1",
                   JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        with open(out / "run.log", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return f"{out.parent.name} {out.name} exit={code}"

    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(one, j) for j in jobs]):
            print(f.result(), flush=True)
    print("ALL_DONE", flush=True)


def summary():
    ref = {(wp, mu): np.nanmedian([json.load(open(p))["lap_stats"]["mean"] for p in
                                   (fig3.HERE / "runs" / "mppi2048").glob(f"icra2022_{wp}_mu{mu}_*/result.json")])
           for wp in fig3.WPS for mu in fig3.MUS}
    print("ATMPPI 50 on icra2022, after the freeze; lap time vs MPPI 2048 median of the same setting")
    print(f"{'steer std':9s} {'setting':16s} {'collisions/laps':16s} {'per seed: rel lap % (coll)'}")
    for std, folder in (("0.08", fig3.HERE / "runs" / "at50"), *((s, ROOT / f"at50_s{s}") for s in STDS)):
        tot_c = tot_l = 0
        rels = []
        for wp in fig3.WPS:
            for mu in fig3.MUS:
                cells = []
                c_set = l_set = 0
                for p in sorted(folder.glob(f"icra2022_{wp}_mu{mu}_noisehalf_seed*/result.json")):
                    a = json.load(open(p))["after_freeze"]
                    laps, c = a["lap_times"], int(a["n_collisions"])
                    rel = 100 * (np.mean(laps) / ref[(wp, mu)] - 1) if laps else np.nan
                    rels.append(rel)
                    cells.append(f"{rel:+.1f}({c})")
                    c_set += c
                    l_set += len(laps)
                tot_c += c_set
                tot_l += l_set
                print(f"{std:9s} {wp + ' mu' + mu:16s} {f'{c_set}/{l_set}':16s} " + " ".join(cells))
        print(f"{std:9s} {'all':16s} {f'{tot_c}/{tot_l}':16s} = {100 * tot_c / max(tot_l, 1):.2f} per 100 laps, "
              f"median rel lap {np.nanmedian(rels):+.1f} %\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.exit(worker(sys.argv[2], sys.argv[3:]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    run(a.workers) if a.mode == "run" else summary()

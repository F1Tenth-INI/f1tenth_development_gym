#!/usr/bin/env python3
"""Run HW 50 for several Settings configs over a grid subset. Finished runs are skipped.

    python launch.py --configs base07,border12 --noise on --seeds 0,1 --workers 30
    python launch.py --configs none --controller mppi --rollouts 2000 --noise half --gpu --workers 8

Noise ``half`` is the observation noise of ``on`` halved. Runs go to ``runs/<config>/`` for HW and to
``runs/<controller><rollouts>/`` otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("PY", sys.executable)
JAX_CUDA = os.environ.get("JAX_CUDA", os.path.join(HERE, "..", "jax_cuda"))  # pip --target folder with jax-cuda12-plugin
sys.path.insert(0, HERE)
from tune_run import CONFIGS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--controller", default="hw")
    p.add_argument("--rollouts", default="50")
    p.add_argument("--maps", default="icra2022,RCA1,IPZ50,hangar3")
    p.add_argument("--wps", default="midline,raceline")
    p.add_argument("--mus", default="0.5,0.9")
    p.add_argument("--noise", default="on", help="comma list of on, half, off")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--sim-seconds", type=float, default=1000.0)
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--gpu", action="store_true", help="JAX on the GPU (MPPI with many rollouts)")
    args = p.parse_args()
    # Seed-major, so the first seeds of every setting finish first.
    jobs = [(c, m, wp, mu, nz, s)
            for s in args.seeds.split(",") for c in args.configs.split(",") for nz in args.noise.split(",")
            for mu in args.mus.split(",") for m in args.maps.split(",") for wp in args.wps.split(",")]
    folder = (lambda c: c) if args.controller == "hw" else (lambda c: f"{args.controller}{args.rollouts}")

    def run(job):
        c, m, wp, mu, nz, s = job
        out = os.path.join(HERE, "runs", folder(c), f"{m}_{wp}_mu{mu}_noise{nz}_seed{s}")
        if os.path.exists(os.path.join(out, "result.json")):
            return f"skip {c} {os.path.basename(out)}"
        os.makedirs(out, exist_ok=True)
        cmd = [PY, os.path.join(HERE, "tune_run.py"), c, "--out", out, "--controller", args.controller,
               "--rollouts", args.rollouts, "--map", m, "--wp", wp, "--mu", mu,
               "--noise", "off" if nz == "off" else "on", "--seed", s, "--sim-seconds", str(args.sim_seconds)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NOISE_SCALE="0.5" if nz == "half" else "1", TQDM_DISABLE="1")
        if args.gpu:
            cmd.append("--gpu")
            env["PYTHONPATH"] = JAX_CUDA
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        else:
            env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        with open(os.path.join(out, "run.log"), "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return f"{folder(c)} {os.path.basename(out)} exit={code}"

    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            print(f.result(), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

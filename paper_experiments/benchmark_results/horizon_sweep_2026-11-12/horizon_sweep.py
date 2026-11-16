#!/usr/bin/env python3
"""Shortest horizon that still drives reliably, for MPPI and ATMPPI with "enough" rollouts.

Setting of the paper's Fig. 2: icra2022, raceline start, mu 0.5, no observation noise. Fixed rollout counts
(about where more rollouts stop helping): MPPI 2048, ATMPPI 256, and ATMPPI 1024 (added 2026-11-13, runs
atmppi1024_h<H>). Swept: the rollout horizon H (control steps of 0.04 s).

    MPPI    config ns_s02 (steering noise 0.2) + p:horizon = H             snapshot 986f92d7 (as Fig. 2)
    ATMPPI  config wm10_tsf103 + MPC_TRAJECTORY_LENGTH = MPC_FREE_TRAJECTORY_LENGTH = H   snapshot aac2064a
            (both the driving samples and the window-update samples use H; stored sequences are H long)

Usage (snapshots from ../fig2_repro_2026-11-12/collect_fig2.py setup):

    python horizon_sweep.py run [--seeds 1,2] [--sim-seconds 800] [--workers 24] [--only mppi,atmppi,atmppi1024]
    python horizon_sweep.py plot     # figures/horizon_sweep.{pdf,png} and summary.csv
    --map RCA1 (both modes): other map, runs in runs_RCA1/, figures/horizon_sweep_RCA1.*, summary_RCA1.csv

Runs go to runs/<controller>_h<H>/seed<k>/ (result.json, run.log); finished runs are skipped.

Metrics per run (paper conventions): ATMPPI after the permanent memory freeze, MPPI from lap 2.
  crash rate [%] = collisions / (completed laps + collisions) * 100   (a collision ends the lap: reset to start)
  mean lap time  = mean of the completed laps in that phase
An ATMPPI run that never froze is counted over all laps from lap 2 and marked in the plot and table.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BR = HERE.parent
REPO = HERE.parents[2]
PY = os.environ.get("PY", sys.executable)
SNAP_AT = Path(os.environ.get("SNAP_AT", REPO.parent / "snapshot_aac2064a"))
SNAP_MPPI = Path(os.environ.get("SNAP_MPPI", REPO.parent / "snapshot_986f92d7"))

ROLLOUTS = {"mppi": 2048, "mppi01": 2048, "atmppi": 256, "atmppi1024": 1024, "atmppi1024s16": 1024, "atmppi1024af": 1024,
            "atmppi1024afs16": 1024, "atmppi1024afsoft": 1024}
HORIZONS = {"mppi": [50, 35, 30, 25, 20, 15], "mppi01": [35, 30, 25, 20, 17], "atmppi": [35, 30, 25, 20, 15, 10],
            "atmppi1024": [35, 30, 25, 20, 15, 10], "atmppi1024s16": [35, 30, 25, 20, 15, 10],
            "atmppi1024af": [35, 30, 25, 20, 17, 15], "atmppi1024afs16": [35, 30, 25, 20, 15],
            "atmppi1024afsoft": [35, 30, 25, 20, 15]}
# *af*: H = 35 while learning and for window updates / stored sequences; only the drive samples use H, and only
# after the permanent freeze (MPC_DRIVE_HORIZON + MPC_DRIVE_HORIZON_AFTER_FREEZE, hw50_tune_2026-11-10/drive_horizon.patch,
# applied to the snapshot). This is the setting of the earlier horizon-25 experiments (af25 in hsweep_n50.json).
DRIVE_ONLY = ("atmppi1024af", "atmppi1024afs16", "atmppi1024afsoft")
# atmppi1024afsoft: the paper's general config soft25w02 (window margins 0.15 / 0.12 m, no target-speed factor)
# instead of the Fig. 2 config wm10_tsf103
SOFT25W02 = {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2}
HALF_NOISE = [0.025] * 10  # the paper's observation noise ("half"): uniform +-0.025 per state component
DEFAULT_H = 35  # horizon of all paper experiments
CRASH_TOP = 105.0  # [%] top of the crash axis; runs without any lap are drawn here
LAP_TOP = 20.0  # [s] mean lap times above are drawn at the top edge
SETTING = {"map": "icra2022", "mu": "0.5", "noise": "off"}  # map: --map


def runs_dir():
    """runs/ for icra2022 without noise (the first sweeps), runs_<map>/ for other maps, + _noisehalf with noise."""
    name = "runs" if SETTING["map"] == "icra2022" else f"runs_{SETTING['map']}"
    return HERE / (name + ("_noisehalf" if SETTING["noise"] == "half" else ""))


def tag():
    return ("" if SETTING["map"] == "icra2022" else "_" + SETTING["map"]) + \
        ("_noisehalf" if SETTING["noise"] == "half" else "")


def configs():
    """{controller: {run name: settings overrides}} for the harnesses."""
    at_base = json.load(open(BR / "hw50_tune_2026-11-10" / "r2.json"))["wm10_tsf103"]
    mppi_base = json.load(open(BR / "mppi_tune_2026-11-10" / "plotrun.json"))["ns_s02"]
    noise = {"NOISE_LEVEL_CAR_STATE": HALF_NOISE} if SETTING["noise"] == "half" else {}
    steer = {k: {"MPC_DRIVE_NOISE": [0.16, 1.5]} for k in ("atmppi1024s16", "atmppi1024afs16")}  # paper: 0.08

    def horizon(at, h):
        if at in DRIVE_ONLY:
            return {"MPC_DRIVE_HORIZON": h, "MPC_DRIVE_HORIZON_AFTER_FREEZE": True}
        return {"MPC_TRAJECTORY_LENGTH": h, "MPC_FREE_TRAJECTORY_LENGTH": h}

    return {
        **{at: {f"{at}_h{h}": dict(SOFT25W02 if at == "atmppi1024afsoft" else at_base, **horizon(at, h), HWM_ROLLOUT_THREADS=1, **steer.get(at, {}), **noise)
                for h in HORIZONS[at]}
           for at in ("atmppi", "atmppi1024", "atmppi1024s16", *DRIVE_ONLY)},
        "mppi": {f"mppi_h{h}": dict(mppi_base, **{"p:horizon": h}, **noise) for h in HORIZONS["mppi"]},
        # mppi01: the default steering noise 0.1 of Fig. 3 (mppi: 0.2, the icra2022 setting of Fig. 2)
        "mppi01": {f"mppi01_h{h}": dict({"p:horizon": h}, **noise) for h in HORIZONS["mppi01"]},
    }


def run(seeds, sim_s, workers, only, horizons=None):
    cfgs = configs()
    (HERE / "configs").mkdir(exist_ok=True)
    jobs = []
    for ctrl in ("mppi", "mppi01", "atmppi", "atmppi1024", "atmppi1024s16", *DRIVE_ONLY):
        if only and ctrl not in only:
            continue
        cfg_file = HERE / "configs" / f"{ctrl}{tag()}.json"
        json.dump(cfgs[ctrl], open(cfg_file, "w"), indent=1)
        jobs += [(ctrl, cfg_file, name, s) for s in seeds for name in cfgs[ctrl]
                 if not horizons or int(name.split("_h")[1]) in horizons]
    # longest first: ATMPPI 1024, then MPPI, then ATMPPI 256, each from the longest horizon
    jobs.sort(key=lambda j: (("atmppi1024", "atmppi1024s16", *DRIVE_ONLY, "mppi", "mppi01", "atmppi").index(j[0]), -int(j[2].split("_h")[1]), j[3]))

    def go(job):
        ctrl, cfg_file, name, seed = job
        out = runs_dir() / name / f"seed{seed}"
        if (out / "result.json").exists():
            return
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TQDM_DISABLE="1",
                   JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        if os.environ.get("USER_SITE", "0") != "1":
            env["PYTHONNOUSERSITE"] = "1"  # ~/.local has a numba that does not import with the env's llvmlite
        if ctrl.startswith("mppi"):
            gym = SNAP_MPPI
            harness = BR / "mppi_tune_2026-11-10" / "mppi_tune.py"
            env.update(MPPI_GYM=str(gym), MPPI_ROLLOUTS=str(ROLLOUTS[ctrl]), MPPI_MU=SETTING["mu"], MPPI_NOISE=("on" if SETTING["noise"] == "half" else "off"), MPPI_MAP=SETTING["map"])
            extra = ["--cpu"]
        else:
            gym = SNAP_AT
            harness = BR / "hw50_tune_2026-11-10" / "hw50_tune.py"
            env.update(HW_GYM=str(gym), HW_ROLLOUTS=str(ROLLOUTS[ctrl]), HW_MU=SETTING["mu"], HW_NOISE=("on" if SETTING["noise"] == "half" else "off"),
                       HW_MAP=SETTING["map"])
            extra = []
        cmd = [PY, str(harness), "run", str(cfg_file), name, str(seed), str(sim_s), str(out)] + extra
        t0 = time.time()
        with open(out / "run.log", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=gym).returncode
        print(f"{name} seed{seed}: exit {code} after {time.time() - t0:.0f} s", flush=True)

    print(f"{len(jobs)} runs, {workers} in parallel", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, jobs))


def metrics(r):
    laps = [l["lap_time"] for l in r["laps"]]
    froze = r.get("freeze_after_lap") is not None
    if froze:
        lt, coll = r["after_freeze"]["lap_times"], r["after_freeze"]["n_collisions"]
    else:  # MPPI, or ATMPPI that never froze: from lap 2
        lt = laps[1:]
        # collisions during lap 1 are not excluded here (rare; lap 1 starts from standstill)
        coll = r["n_collisions"]
    attempts = len(lt) + coll
    return {"laps": len(lt), "collisions": coll, "crash_pct": 100.0 * coll / attempts if attempts else float("nan"),
            "mean_lap": sum(lt) / len(lt) if lt else float("nan"), "froze": froze,
            "freeze_t": r.get("freeze_t"), "coll_total": r["n_collisions"], "laps_total": len(laps)}


def collect():
    rows = []
    for d in sorted(runs_dir().glob("*_h*/seed*/result.json")):
        ctrl, h = d.parent.parent.name.split("_h")
        r = json.load(open(d))
        if r.get("error"):
            print(f"{d.parent}: error {r['error']}")
        rows.append(dict(controller=ctrl, horizon=int(h), seed=int(d.parent.name[4:]), **metrics(r)))
    return rows


def plot(new_only=False, controllers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = collect()
    if new_only:  # only the runs of 2026-11-13: ATMPPI 1024 (seeds 1-5) and seeds 3-5 of the other two
        rows = [r for r in rows if r["controller"] == "atmppi1024" or r["seed"] >= 3]
    if controllers:
        rows = [r for r in rows if r["controller"] in controllers]
    if not rows:
        raise SystemExit("no finished runs")
    global LAP_TOP  # top of the lap-time axis: 1.4 x the fastest seed mean (icra2022: 20 s)
    LAP_TOP = 20.0 if SETTING["map"] == "icra2022" else round(1.4 * min(r["mean_lap"] for r in rows if r["mean_lap"] == r["mean_lap"]))
    suffix = tag() + ("_new" if new_only else "") + ("_" + "_vs_".join(controllers) if controllers else "")
    with open(HERE / f"summary{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["controller"], -r["horizon"], r["seed"])))

    style = {"mppi": ("#3a3a3a", f"MPPI, {ROLLOUTS['mppi']} rollouts", "s", 0.35),
             "atmppi": ("#6baed6", f"Auto-Tuning MPPI, {ROLLOUTS['atmppi']} rollouts (after freeze)", "o", -0.35),
             "atmppi1024": ("#08306b", f"Auto-Tuning MPPI, {ROLLOUTS['atmppi1024']} rollouts (after freeze)", "D", 0.0),
             "atmppi1024s16": ("#d95f02", "Auto-Tuning MPPI, 1024 rollouts, steering noise 0.16 (after freeze)", "v", 0.15)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    print(f"\n{'controller':10s} {'H':>3s} {'seed':>4s} {'laps':>5s} {'coll':>5s} {'crash %':>8s} {'mean lap':>9s} {'froze':>6s} {'freeze t':>8s}")
    for ctrl, (c, label, m, dx) in style.items():
        sub = [r for r in rows if r["controller"] == ctrl]
        dx = dx if not controllers or len(controllers) > 2 else (0.25 if ctrl == "mppi" else -0.25)
        if not sub:
            continue
        hs = sorted({r["horizon"] for r in sub})
        for r in sorted(sub, key=lambda r: (-r["horizon"], r["seed"])):
            ft = "" if r["freeze_t"] is None else f"{r['freeze_t']:.0f}"
            print(f"{ctrl:10s} {r['horizon']:3d} {r['seed']:4d} {r['laps']:5d} {r['collisions']:5d} {r['crash_pct']:8.1f} "
                  f"{r['mean_lap']:9.2f} {str(r['froze']) if ctrl != 'mppi' else '':>6s} {ft:>8s}")
            mfc = c if (r["froze"] or ctrl == "mppi") else "white"  # open: ATMPPI never froze
            if r["laps"] == 0 and r["collisions"] == 0:  # stuck or too slow: no lap counted, no collision
                a1.plot(r["horizon"] + dx, CRASH_TOP, "x", color=c, ms=6, mew=1.5, zorder=3, clip_on=False)
            else:
                a1.plot(r["horizon"] + dx, r["crash_pct"], m, color=c, mfc=mfc, ms=4, alpha=0.6, zorder=3)
            if np.isfinite(r["mean_lap"]):
                y = min(r["mean_lap"], LAP_TOP)
                mk = "^" if r["mean_lap"] > LAP_TOP else m
                a2.plot(r["horizon"] + dx, y, mk, color=c, mfc=mfc, ms=4 if mk == m else 6, alpha=0.6, zorder=3, clip_on=False)
        # runs above LAP_TOP: one label per horizon with their range, stacked by controller
        for h in hs:
            over = [r["mean_lap"] for r in sub if r["horizon"] == h and r["mean_lap"] > LAP_TOP]
            if over:
                text = f"{min(over):.0f}" if len(over) == 1 or round(min(over)) == round(max(over)) else f"{min(over):.0f}–{max(over):.0f}"
                k = list(style).index(ctrl)
                a2.annotate(text + " s", (h + dx, LAP_TOP), xytext=(9, 1 - 8 * k), textcoords="offset points",
                            fontsize=6.5, color=c, ha="left", va="center",
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.9), zorder=5)
        # a run without any counted lap or collision failed completely: 100 % for the seed mean
        crash = [np.mean([100.0 if not np.isfinite(r["crash_pct"]) else r["crash_pct"] for r in sub if r["horizon"] == h]) for h in hs]
        lap = [min(np.nanmean([r["mean_lap"] for r in sub if r["horizon"] == h]), LAP_TOP)
               if any(np.isfinite(r["mean_lap"]) for r in sub if r["horizon"] == h) else np.nan for h in hs]
        a1.plot(hs, crash, "-", color=c, lw=1.2, label=label + ", seed mean")
        a2.plot(hs, lap, "-", color=c, lw=1.2)
    a1.set_ylabel("Crashes per lap (%)")
    a1.set_ylim(-2, CRASH_TOP)
    a2.set_ylim(top=LAP_TOP)
    a2.set_ylabel("Mean lap time (s)")
    a2.set_xlabel("Horizon H (control steps of 0.04 s)")
    for a in (a1, a2):
        a.axvline(DEFAULT_H, color="0.7", lw=0.8, ls=":", zorder=0)
        a.grid(alpha=0.25)
    a2.text(DEFAULT_H + 0.4, a2.get_ylim()[0] + 0.1, "paper H", fontsize=7, color="0.5", va="bottom")
    a1.plot([], [], "o", mfc="white", mec="#1f77b4", ms=4, label="ATMPPI run that never froze (all laps from lap 2)")
    a1.plot([], [], "x", color="0.4", ms=6, mew=1.5, label="no lap completed, no collision (stuck); counted as 100 %")
    a2.plot([], [], "^", color="0.4", ms=6, label=f"above {LAP_TOP:.0f} s (range next to it)")
    a2.legend(fontsize=7, frameon=False, loc="upper right")
    a1.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.36, 1.0))
    a1.set_title(f"{SETTING['map']}, raceline, $\\mu$ = 0.5, "
                 + ("observation noise +-0.025" if SETTING["noise"] == "half" else "no noise") + "; markers: seeds"
                 + ("\nnew runs only: ATMPPI 1024 seeds 1-5, MPPI and ATMPPI 256 seeds 3-5" if new_only else ""), fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "figures" / f"horizon_sweep{suffix}.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", HERE / "figures" / f"horizon_sweep{suffix}.{{pdf,png}}", f"and summary{suffix}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "plot"])
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--sim-seconds", type=float, default=800.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--horizons", help="run: comma list of H to run (default: all of HORIZONS)")
    ap.add_argument("--only", help="comma list of mppi, atmppi (256), atmppi1024")
    ap.add_argument("--controllers", help="plot: comma list to show, e.g. mppi,atmppi1024")
    ap.add_argument("--new", action="store_true", help="plot: only the runs of 2026-11-13 (horizon_sweep_new.*)")
    ap.add_argument("--map", default="icra2022", help="runs go to runs/ for icra2022, runs_<map>/ otherwise")
    ap.add_argument("--noise", default="off", choices=["off", "half"],
                    help="half: the paper's observation noise (+-0.025), runs in runs*_noisehalf/")
    a = ap.parse_args()
    SETTING["map"] = a.map
    SETTING["noise"] = a.noise
    if a.mode == "run":
        run([int(s) for s in a.seeds.split(",")], a.sim_seconds, a.workers, a.only.split(",") if a.only else None,
            [int(h) for h in a.horizons.split(",")] if a.horizons else None)
    else:
        plot(a.new, a.controllers.split(",") if a.controllers else None)

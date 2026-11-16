#!/usr/bin/env python3
"""Recreate the paper's Fig. 3 (``all_tracks.pdf``): all tracks, initial references and friction coefficients.

    python fig3.py run  [--workers 30] [--seeds 0,1,2,3,4] [--only at50,mppi2048]   # 320 simulations
    python fig3.py plot [--source repro|paper]   # figures/all_tracks_<source>.{pdf,png} + numbers on stdout
    python fig3.py plot --icra-at50 0.16         # ATMPPI 50 on icra2022 from noise_check.py (steering std 0.16)

Setting (paper, Sec. 5): icra2022, RCA1, IPZ50, hangar3 x {centerline ("midline"), min-curvature raceline} x
mu {0.5, 0.9}, observation noise on (the "half" noise: uniform +-0.025 per state component), 1000 s per run,
seeds 0..4 -> 16 settings x 5 seeds = 80 runs per controller, the same for all four controllers:

    at50      ATMPPI   50 rollouts, config soft25w02   (tune_noise_2026-11-08/tune_run.py CONFIGS)
    at1024    ATMPPI 1024 rollouts, config wm10_tsf103 (paper: 1000)
    mppi50    MPPI     50 rollouts, config none (sigma_delta 0.1)
    mppi2048  MPPI   2048 rollouts, config none        (paper: 2000; the reference of the lap-time panel)

Runs use the paper's launcher (``tune_run.py``) on the code snapshot aac2064a + tune_noise patch (SNAP_AT, made by
``../fig2_repro_2026-11-12/collect_fig2.py setup``; IPZ50 is copied in from the repository, as in the README).
MPPI runs on the CPU (the paper's ran on the GPU). Runs go to runs/<controller>/<map>_<wp>_mu<mu>_noisehalf_seed<s>/
and finished runs are skipped.

The paper's plotting script for this figure is not in the repository (the one in
``grid_2026-11-05/paper/make_all_tracks.py`` draws the older noise on/off version). ``plot`` redraws the paper's
layout. Its metrics are those of ``make_sim_figures.py``: ATMPPI = laps and collisions after the memory freeze,
MPPI = all laps; per seed the mean lap time relative to the median of MPPI 2048 (paper: 2000) in the same setting; ATMPPI seeds
that froze broken (> 30 % slower than the reference, or collisions on at least half the frozen laps) are dropped.
``--source paper`` draws the paper's own runs (tune_noise_2026-11-08/runs) to check the script against the PDF.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BR = HERE.parent
REPO = HERE.parents[2]
TUNE = BR / "tune_noise_2026-11-08"
PAPER_DIR = BR / "grid_2026-11-05" / "paper"
SNAP_AT = Path(os.environ.get("SNAP_AT", REPO.parent / "snapshot_aac2064a"))

MAPS = ("icra2022", "RCA1", "IPZ50", "hangar3")
WPS = ("midline", "raceline")
MUS = ("0.5", "0.9")
SETTINGS = [(wp, mu) for wp in WPS for mu in MUS]
# name: (controller, rollouts, config, folder of the paper's runs, rollouts of the paper's runs, color, x offset)
# The repro uses 1024 / 2048 rollouts where the paper used 1000 / 2000 (keys keep the names of the large variants).
CTRL = {
    "at50": ("hw", "50", "soft25w02", "soft25w02", "50", "#3a86c8", -0.27),
    "at1024": ("hw", "1024", "wm10_tsf103", "wm10_tsf103", "1000", "#08306b", -0.09),
    "mppi50": ("mppi", "50", "none", "mppi50", "50", "#b0b0b0", 0.09),
    "mppi2048": ("mppi", "2048", "none", "mppi2000", "2000", "#444444", 0.27),
}
REF = "mppi2048"
ORDER = ("at1024", "mppi2048", "at50", "mppi50")
SOURCE = "repro"  # set by plot(): which rollout numbers the labels show


def rollouts(n):
    return CTRL[n][1] if SOURCE == "repro" else CTRL[n][4]


def label(n):
    name = "Auto-Tuning MPPI" if CTRL[n][0] == "hw" else "MPPI"
    return f"{name}, {rollouts(n)} rollouts" + (" (reference, 0 %)" if n == REF else "")  # longest runs first, so the pool drains evenly
CLIP, BROKEN_SLOW, LINTHRESH = 40.0, 30.0, 3.0
Y_TICKS = (-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40)


# ---------------------------------------------------------------- run
def run(workers, seeds, only, maps=MAPS, runs_dir="runs"):
    if not (SNAP_AT / "scripts" / "benchmark" / "bench_run.py").exists():
        raise SystemExit(f"{SNAP_AT} missing: python ../fig2_repro_2026-11-12/collect_fig2.py setup")
    if not (SNAP_AT / "utilities" / "maps" / "IPZ50").is_dir():  # untracked at aac2064a
        subprocess.run(["cp", "-r", str(REPO / "utilities" / "maps" / "IPZ50"), str(SNAP_AT / "utilities" / "maps")],
                       check=True)
    names = [n for n in ORDER if not only or n in only]
    jobs = [(n, m, wp, mu, s) for n in names for s in seeds for m in maps for wp in WPS for mu in MUS]

    def one(job):
        n, m, wp, mu, s = job
        ctrl, rollouts, config = CTRL[n][:3]
        out = HERE / runs_dir / n / f"{m}_{wp}_mu{mu}_noisehalf_seed{s}"
        if (out / "result.json").exists():
            return f"skip {n} {out.name}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(TUNE / "tune_run.py"), config, "--out", str(out), "--controller", ctrl,
               "--rollouts", rollouts, "--map", m, "--wp", wp, "--mu", mu, "--noise", "on", "--seed", str(s),
               "--sim-seconds", "1000"]
        env = dict(os.environ, CODE_DIR=str(SNAP_AT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NOISE_SCALE="0.5", TQDM_DISABLE="1", PYTHONNOUSERSITE="1",
                   JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        with open(out / "run.log", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return f"{n} {out.name} exit={code}"

    print(f"{len(jobs)} jobs, {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(one, j) for j in jobs]):
            print(f.result(), flush=True)
    print("ALL_DONE", flush=True)


# ---------------------------------------------------------------- data
def load(source, icra_at50=None):
    """{(name, map, wp, mu): [per seed {lap, laps, collisions, freeze_t, seed}]} for the half-noise runs."""
    rows = {}
    for n, c in CTRL.items():
        root = HERE / "runs" / n if source == "repro" else TUNE / "runs" / c[3]
        for m in MAPS:
            for wp, mu in SETTINGS:
                seeds = []
                # --icra-at50 <std>: ATMPPI 50 on icra2022 from the steering-noise check (noise_check.py)
                if icra_at50 and n == "at50" and m == "icra2022":
                    root_m = HERE / "runs_noise" / f"at50_s{icra_at50}"
                else:
                    root_m = root
                for path in sorted(root_m.glob(f"{m}_{wp}_mu{mu}_noisehalf_seed*/result.json")):
                    r = json.loads(path.read_text())
                    if c[0] == "hw":  # collisions while learning are not counted
                        laps, coll = r["after_freeze"]["lap_times"], int(r["after_freeze"]["n_collisions"])
                    else:
                        laps, coll = [l["lap_time"] for l in r["laps"]], int(r["n_collisions"])
                    seeds.append({"lap": float(np.mean(laps)) if laps else np.nan, "laps": len(laps),
                                  "collisions": coll, "freeze_t": r.get("freeze_t"),
                                  "laps_all": len(r["laps"]), "seed": path.parent.name.split("seed")[-1]})
                rows[(n, m, wp, mu)] = seeds
    return rows


def relative(rows, n, key):
    ref = np.nanmedian([s["lap"] for s in rows[(REF, *key)]])
    return 100.0 * (np.array([s["lap"] for s in rows[(n, *key)]], float) / ref - 1.0)


def drop_broken(rows):
    dropped = []
    for key in [k for k in rows if CTRL[k[0]][0] == "hw"]:
        keep = []
        for s, r in zip(rows[key], relative(rows, key[0], key[1:])):
            if not r <= BROKEN_SLOW or 2 * s["collisions"] >= max(s["laps"], 1):
                dropped.append((key, s["seed"], r, s["collisions"], s["laps"]))
            else:
                keep.append(s)
        rows[key] = keep
    return rows, dropped


def collision_rate(rows, n, m):
    seeds = [s for wp, mu in SETTINGS for s in rows[(n, m, wp, mu)]]
    return 100.0 * sum(s["collisions"] for s in seeds) / max(sum(s["laps"] for s in seeds), 1)


# ---------------------------------------------------------------- figure
def figure(rows, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    sys.path.insert(0, str(PAPER_DIR))
    import make_sim_figures as sf  # track_miniature

    fig, axes = plt.subplots(3, len(MAPS), figsize=(7.2, 5.0),
                             gridspec_kw={"height_ratios": (0.95, 1.0, 2.3), "hspace": 0.32, "wspace": 0.42})
    max_rate = max(collision_rate(rows, n, m) for n in CTRL for m in MAPS)
    for col, m in enumerate(MAPS):
        sf.track_miniature(axes[0, col], m)
        axes[0, col].set_title(m, fontsize=9, pad=4)

        ax = axes[1, col]  # collisions per 100 laps
        width = 0.8 / len(CTRL)
        for i, (n, c) in enumerate(CTRL.items()):
            ax.bar((i - (len(CTRL) - 1) / 2) * width, collision_rate(rows, n, m), width * 0.9, color=c[5], zorder=3)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylim(0, max(max_rate, 0.1) * 1.12)
        ax.grid(axis="y", color="#e4e4e4", lw=0.6, zorder=0)
        ax.tick_params(labelsize=7.5)
        if col:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Collisions\nper 100 laps", fontsize=8.5)

        ax = axes[2, col]  # lap time relative to the reference
        ax.axhline(0.0, color="#444444", lw=0.9, zorder=1)
        values = []
        for x, (wp, mu) in enumerate(SETTINGS):
            for n, c in CTRL.items():
                rel = relative(rows, n, (m, wp, mu))
                rel = rel[np.isfinite(rel)]
                values.append(rel)
                shown = np.minimum(rel, CLIP)
                xs = np.full(len(shown), x + c[6])
                ax.scatter(xs[rel <= CLIP], shown[rel <= CLIP], s=10, color=c[5], zorder=3, linewidths=0)
                if (rel > CLIP).any():
                    ax.scatter(xs[rel > CLIP], np.full((rel > CLIP).sum(), CLIP), s=16, marker="^", color=c[5],
                               zorder=4, clip_on=False)
        values = np.minimum(np.concatenate(values), CLIP)
        ax.set_xlim(-0.5, len(SETTINGS) - 0.5)
        ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.6)
        ax.set_ylim(min(values.min(), -LINTHRESH) * 1.15, max(values.max(), LINTHRESH) * 1.15)
        ticks = [t for t in Y_TICKS if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:+d}" if t else "0" for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="y", color="#e4e4e4", lw=0.6, zorder=0)
        ax.axvline(0.5, color="#eeeeee", lw=0.6, zorder=0)
        ax.axvline(2.5, color="#eeeeee", lw=0.6, zorder=0)
        ax.axvline(1.5, color="black", lw=0.8, zorder=2)
        ax.set_xticks(range(len(SETTINGS)))
        ax.set_xticklabels([f"μ={mu}" for _wp, mu in SETTINGS], fontsize=5.5)
        ax.tick_params(axis="x", length=2, pad=2)
        ax.tick_params(axis="y", labelsize=7.5)
        for x, text in ((0.5, "centerline"), (2.5, "raceline")):
            ax.text(x, -0.17, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
        if col == 0:
            ax.set_ylabel(f"Lap time vs.\nMPPI {rollouts(REF)} (%)", fontsize=8.5)

    # one rounded box per track around its three panels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    all_bbs = [[inv.transform_bbox(a.get_tightbbox(renderer)) for a in axes[:, col]] for col in range(len(MAPS))]
    y0 = min(b.y0 for bbs in all_bbs for b in bbs) - 0.008  # same height for all boxes
    y1 = max(b.y1 for bbs in all_bbs for b in bbs) + 0.008
    for col in range(len(MAPS)):
        bbs = all_bbs[col]
        x0 = min(b.x0 for b in bbs) - 0.008
        x1 = max(b.x1 for b in bbs) + 0.008
        fig.add_artist(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fill=False, edgecolor="#aaaaaa", lw=0.8))
    handles = [Line2D([], [], ls="", marker="o", ms=4.5, color=c[5], label=label(n)) for n, c in CTRL.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.06), handletextpad=0.2, columnspacing=2.0)
    if title:
        fig.suptitle(title, fontsize=8, y=0.995)
    return fig


# ---------------------------------------------------------------- numbers
def numbers(rows, dropped, raw):
    print("runs per controller (finished / kept):")
    for n in CTRL:
        total = sum(len(raw[(n, m, wp, mu)]) for m in MAPS for wp, mu in SETTINGS)
        kept = sum(len(rows[(n, m, wp, mu)]) for m in MAPS for wp, mu in SETTINGS)
        print(f"  {n:9s} {total:3d} / {kept:3d}")
    for key, seed, r, c, laps in dropped:
        print(f"  dropped {key} seed {seed}: {r:+.1f} %, {c} collisions in {laps} frozen laps")

    def rel(n, maps=MAPS, wps=WPS, mus=MUS):
        v = np.concatenate([relative(rows, n, (m, wp, mu)) for m in maps for wp in wps for mu in mus])
        return v[np.isfinite(v)]

    print(f"\nmedian lap time vs MPPI {rollouts(REF)} (all seeds, all settings):")
    for n in ("at50", "at1024", "mppi50"):
        print(f"  {n:7s} {np.median(rel(n)):+6.1f} %   per track: "
              + ", ".join(f"{m} {np.median(rel(n, maps=(m,))):+.1f}" for m in MAPS))
    at = "at50"
    print(f"  at50 centerline {np.median(rel(at, wps=('midline',))):+.1f} %, raceline "
          f"{np.median(rel(at, wps=('raceline',))):+.1f} %; mu 0.5 {np.median(rel(at, mus=('0.5',))):+.1f} %, "
          f"mu 0.9 {np.median(rel(at, mus=('0.9',))):+.1f} %")
    faster = [np.nanmean(relative(rows, "at50", (m, wp, mu))) < 0 for m in MAPS for wp, mu in SETTINGS]
    print(f"  ATMPPI 50 faster than MPPI {rollouts(REF)} (seed mean) in {sum(faster)} of {len(faster)} settings")
    a1000 = [np.nanmedian(relative(rows, "at1024", k)) < np.nanmedian(relative(rows, "at50", k))
             for k in [(m, wp, mu) for m in MAPS for wp, mu in SETTINGS]]
    d = [np.nanmedian([s["lap"] for s in rows[("at1024", m, wp, mu)]]) /
         np.nanmedian([s["lap"] for s in rows[("at50", m, wp, mu)]]) - 1 for m in MAPS for wp, mu in SETTINGS]
    print(f"  ATMPPI {rollouts('at1024')} vs 50 (per-setting medians): median {100 * np.nanmedian(d):+.1f} %, faster in "
          f"{sum(a1000)} of {len(a1000)}")
    print("\ncollisions per 100 laps (all tracks; ATMPPI after the freeze):")
    for n in CTRL:
        seeds = [s for m in MAPS for wp, mu in SETTINGS for s in rows[(n, m, wp, mu)]]
        c, l = sum(s["collisions"] for s in seeds), sum(s["laps"] for s in seeds)
        print(f"  {n:9s} {c:3d} collisions in {l:5d} laps = {100 * c / max(l, 1):.2f}")
    for n in ("at50", "at1024"):
        f = [s["freeze_t"] for m in MAPS for wp, mu in SETTINGS for s in raw[(n, m, wp, mu)]]
        froze = [x for x in f if x]
        print(f"  {n} freeze: median {np.median(froze):.0f} s, never froze {len(f) - len(froze)} of {len(f)}"
              if froze else f"  {n}: no freeze")


def plot(source, icra_at50=None):
    global SOURCE
    SOURCE = source
    raw = load(source, icra_at50)
    rows, dropped = drop_broken({k: list(v) for k, v in raw.items()})
    numbers(rows, dropped, raw)
    fig = figure(rows)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    name = f"all_tracks_{source}" + (f"_icra_at50_s{icra_at50}" if icra_at50 else "")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", out / f"{name}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "plot"])
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--only", default="", help="comma list of " + ",".join(CTRL))
    ap.add_argument("--source", default="repro", choices=["repro", "paper"])
    ap.add_argument("--icra-at50", help="steering std of noise_check.py (0.12, 0.16) for ATMPPI 50 on icra2022")
    ap.add_argument("--maps", default=",".join(MAPS), help="run: maps")
    ap.add_argument("--runs-dir", default="runs", help="run: output folder (extra seeds: runs_extra)")
    a = ap.parse_args()
    if a.mode == "run":
        run(a.workers, a.seeds.split(","), [x for x in a.only.split(",") if x], a.maps.split(","), a.runs_dir)
    else:
        plot(a.source, a.icra_at50)

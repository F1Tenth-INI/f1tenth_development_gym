#!/usr/bin/env python3
"""Figures and table numbers of the paper's simulation section.

    python make_sim_figures.py

``all_tracks`` uses the Auto-Tuning MPPI 50 runs in ``HW_RUNS``, ``all_tracks_grid`` those of the grid.

1. ``icra_laptimes``: icra2022, raceline start, mu 0.5, noise off, seed 1. Auto-Tuning MPPI with
   50 rollouts vs MPPI with 2000, lap time per lap. That run has no collisions, so none are drawn.
2. ``icra_overlay``: one lap of each after training, on one track.
3. ``all_tracks``: every track, start line, friction and noise setting of the grid. Per seed, lap time
   relative to the median of MPPI 2000 in the same setting, for Auto-Tuning MPPI 50 and MPPI 50
   (same sample budget). Below, collisions per 100 laps.

Seed values are those of ``plot_laptime_boxes.py``: Auto-Tuning MPPI is the mean of its laps after the
permanent freeze, MPPI the mean of all its laps. Auto-Tuning MPPI seeds that froze broken are left out
(``drop_broken``).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GRID, "thesis_icra2022"))
import make_plots as mp  # noqa: E402

OUT = os.path.join(HERE, "figures")
TUNE_RUNS = os.path.join(GRID, "..", "tune_noise_2026-11-08", "runs")
# Auto-Tuning MPPI 50 of the paper: the grid re-run with the stronger soft wall penalty
HW_RUNS = os.path.join(TUNE_RUNS, "soft25w02")
HW_CONFIG = "soft25w02"  # config of the halved-noise figure, a folder in TUNE_RUNS
PAPER_CONFIG = "soft25w02"  # copied to all_tracks, icra_laptimes, icra_overlay
OTHER_HALF_CONFIGS = ("base07",)
# Seed of the icra2022 example (raceline, mu 0.5, no noise): base07 shows its median seed, which is collision-free.
ICRA_SEED = {"base07": 4}
NOISE_SHOWN = {"half": "on"}
MAPS = (("icra2022", "icra2022"), ("RCA1", "RCA1"), ("IPZ50", "IPZ50"), ("hangar3", "hangar3"))
SETTINGS = [(wp, mu) for wp in ("midline", "raceline") for mu in ("0.5", "0.9")]
CONTROLLERS = {  # key: (label, color, x offset)
    ("hw", "50"): ("Auto-Tuning MPPI, 50 rollouts", mp.HW_COLOR, -0.22),
    ("mppi", "50"): ("MPPI, 50 rollouts", "#b0b0b0", 0.0),
    ("mppi", "2000"): ("MPPI, 2000 rollouts (reference, 0 %)", mp.MPPI_COLOR, 0.22),
}
REFERENCE = ("mppi", "2000")
CLIP = 40.0  # [%] relative lap times above this are drawn at the edge
BROKEN_SLOW = 30.0  # [%]
LINTHRESH = 3.0  # [%] the lap-time axis is linear within +-LINTHRESH and logarithmic beyond
Y_TICKS = (-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40)
MAP_DIR = os.path.join(GRID, "..", "tune_noise_2026-11-08", "code", "utilities", "maps")
if not os.path.isdir(MAP_DIR):  # no code snapshot next to the results: the repository's maps
    MAP_DIR = os.path.join(GRID, "..", "..", "..", "utilities", "maps")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def icra_figures(suffix="", compact=False):
    seed = ICRA_SEED.get(HW_CONFIG, 1) if suffix or compact else 1
    mp.configure("off", seed, 50)
    if suffix or compact:
        name = f"icra2022_raceline_mu0.5_noiseoff_seed{seed}"
        mp.HW_RUN = os.path.join(TUNE_RUNS, HW_CONFIG, name)
        mp.HW_TRAJ = os.path.join(TUNE_RUNS, "..", "runs_traj", f"{HW_CONFIG}_{name}")
    hw, mppi = mp.load(mp.HW_RUN), mp.load(mp.MPPI_RUN)
    assert hw["n_collisions"] == 0 and mppi["n_collisions"] == 0, "the paper figure assumes a collision-free run"
    save(mp.lap_time_plot(hw, mppi, show_collisions=False, title=False,
                          figsize=(3.5, 2.5) if compact else (7.2, 3.2), legend=not compact, compact=compact),
         "icra_laptimes" + suffix)
    hw_re, mppi_re = mp.load(mp.HW_TRAJ), mp.load(mp.MPPI_TRAJ)
    mp.check_rerun(hw, hw_re, "HW")
    mp.check_rerun(mppi, mppi_re, "MPPI")
    first = hw_re["freeze_after_lap"] + 1
    laps = [(f"{mp.AT}, 50 rollouts", mp.lap_segment(mp.HW_TRAJ, hw_re, first)),
            ("MPPI, 2000 rollouts", mp.lap_segment(mp.MPPI_TRAJ, mppi_re, first))]
    save(mp.overlay_plot(laps, compact=compact), "icra_overlay" + suffix)
    print(f"icra example{suffix}: Auto-Tuning MPPI {hw['after_freeze']['lap_stats']['mean']:.2f} s after the freeze "
          f"(lap {hw['freeze_after_lap']}, {hw['freeze_t']:.0f} s), MPPI 2000 {mppi['lap_stats']['mean']:.2f} s")


def load_summary():
    rows = defaultdict(list)
    with open(os.path.join(GRID, "summary.csv")) as f:
        for r in csv.DictReader(f):
            rollouts = r["rollouts"].split(".")[0]
            value = r["after_freeze_lap_mean"] if r["controller"] == "hw" else r["lap_mean"]
            hw = r["controller"] == "hw"  # collisions while learning are not counted, only those with the frozen memory
            rows[(r["controller"], rollouts, r["map"], r["wp"], r["mu"], r["noise"])].append({
                "lap": float(value) if value else np.nan,
                "laps": int(r["after_freeze_laps"] if hw else r["laps"]),
                "collisions": int(float(r["after_freeze_collisions"])) if hw else int(r["collisions"]),
            })
    return rows


def replace_runs(rows, ctrl, runs_dir, noises):
    """Swap the grid's seeds of ``ctrl`` with noise in ``noises`` for the runs in ``runs_dir``
    (``<map>_<wp>_mu<mu>_noise<nz>_seed<s>``). Halved noise takes the place of ``on``."""
    shown = {NOISE_SHOWN.get(nz, nz) for nz in noises}
    for key in [k for k in rows if k[:2] == ctrl and k[5] in shown]:
        del rows[key]
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name, "result.json")
        map_name, wp, mu, noise, _seed = name.split("_")
        if noise[5:] not in noises or not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        if ctrl[0] == "hw":  # collisions while learning are not counted, only those with the frozen memory
            laps, collisions = r["after_freeze"]["lap_times"], int(r["after_freeze"]["n_collisions"])
        else:
            laps, collisions = [l["lap_time"] for l in r["laps"]], int(r["n_collisions"])
        rows[(*ctrl, map_name, wp, mu[2:], NOISE_SHOWN.get(noise[5:], noise[5:]))].append({
            "lap": float(np.mean(laps)) if laps else np.nan,
            "laps": len(laps),
            "collisions": collisions,
        })
    return rows


def replace_hw50(rows, runs_dir):
    return replace_runs(rows, ("hw", "50"), runs_dir, ("on", "off"))


def half_noise_rows():
    """The paper's data: noise on is the halved noise, Auto-Tuning MPPI 50 is ``HW_CONFIG``."""
    rows = load_summary()
    replace_runs(rows, ("hw", "50"), os.path.join(TUNE_RUNS, HW_CONFIG), ("half", "off"))
    for n in ("50", "2000"):
        replace_runs(rows, ("mppi", n), os.path.join(TUNE_RUNS, f"mppi{n}"), ("half",))
    return rows


def drop_broken(rows):
    """Remove Auto-Tuning MPPI seeds whose memory froze broken: post-freeze laps more than ``BROKEN_SLOW`` %
    slower than MPPI 2000, or a collision on at least half of them. Such a run would be restarted."""
    for key in [k for k in rows if k[0] == "hw"]:
        rel = relative(rows, key[:2], key[2:])
        keep = []
        for s, r in zip(rows[key], rel):
            if r > BROKEN_SLOW or 2 * s["collisions"] >= max(s["laps"], 1):
                print(f"dropped {key[2:]}: {r:+.1f} %, {s['collisions']} collisions in {s['laps']} laps")
            else:
                keep.append(s)
        rows[key] = keep
    return rows


def relative(rows, ctrl, key):
    ref = np.nanmedian([s["lap"] for s in rows[(*REFERENCE, *key)]])
    return 100.0 * (np.array([s["lap"] for s in rows[(*ctrl, *key)]]) / ref - 1.0)


def collision_rate(rows, ctrl, map_name, noise):
    seeds = [s for (c, n, m, _wp, _mu, nz), v in rows.items() if (c, n) == ctrl and m == map_name and nz == noise for s in v]
    return 100.0 * sum(s["collisions"] for s in seeds) / max(sum(s["laps"] for s in seeds), 1)


def track_miniature(ax, map_name):
    """The map's free area connected to the centerline (light grey, walls outlined) and the raceline."""
    import yaml
    from PIL import Image
    from scipy import ndimage

    folder = os.path.join(MAP_DIR, map_name)
    with open(os.path.join(folder, f"{map_name}.yaml")) as f:
        meta = yaml.safe_load(f)
    image = np.array(Image.open(os.path.join(folder, meta["image"])).convert("L"))[::-1]  # row 0 = lowest y
    labels, _ = ndimage.label(image > 250)
    res, (ox, oy) = meta["resolution"], meta["origin"][:2]

    def pixels(xy):
        return ((xy[:, 0] - ox) / res).astype(int), ((xy[:, 1] - oy) / res).astype(int)

    centre = np.loadtxt(os.path.join(folder, "centerline.csv"), delimiter=",", comments="#")[:, :2]
    cols, rows_ = pixels(centre)
    ids = labels[rows_, cols]
    track = labels == np.bincount(ids[ids > 0]).argmax()
    # Same look on every map: no specks or small holes (< 0.3 m^2), edges smoothed over ~5 cm, drawn as vectors
    small = 0.3 / res ** 2
    holes, n = ndimage.label(~track)
    sizes = ndimage.sum(np.ones_like(holes), holes, range(1, n + 1))
    track |= np.isin(holes, 1 + np.flatnonzero(sizes < small))
    pad = 4
    ys, xs = np.nonzero(track)
    y0, x0 = ys.min() - pad, xs.min() - pad
    sub = np.pad(track, pad)[y0 + pad:ys.max() + 2 * pad + 1, x0 + pad:xs.max() + 2 * pad + 1]
    smooth = ndimage.gaussian_filter(sub.astype(float), 0.05 / res)
    race = np.loadtxt(os.path.join(folder, f"{map_name}_wp.csv"), delimiter=",", skiprows=1)[:, 1:3]
    rc, rr = pixels(race)
    rx, ry = np.append(rc, rc[0]) - x0, np.append(rr, rr[0]) - y0
    if smooth.shape[0] > smooth.shape[1]:  # tall map: turned 90 deg to fit the wide slot
        rx, ry = smooth.shape[0] - 1 - ry, rx
        smooth = np.rot90(smooth, -1)
    ax.contourf(smooth, levels=[0.5, 2.0], colors=["#d9d9d9"])
    ax.contour(smooth, levels=[0.5], colors="0.3", linewidths=0.5)
    ax.plot(rx, ry, color="black", lw=0.45)
    # 5 m scale bar right of the map, at its bottom (each miniature has its own zoom)
    h, w = smooth.shape
    bar = 5.0 / res
    x = w + 0.04 * max(w, h)
    ax.plot([x, x], [0, bar], color="black", lw=1.0, solid_capstyle="butt", clip_on=False)
    ax.text(x + 0.03 * max(w, h), bar / 2, "5 m", fontsize=6, va="center", ha="left", rotation=90, clip_on=False)
    ax.set_xlim(0, x + 0.12 * max(w, h))
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")


def all_tracks_figure(rows, controllers=None):
    """``controllers``: {key: (label, color, x offset)}, default ``CONTROLLERS``."""
    controllers = controllers or CONTROLLERS
    fig, axes = plt.subplots(3, len(MAPS), figsize=(7.2, 4.4),
                             gridspec_kw={"height_ratios": (0.9, 2.4, 1.0), "hspace": 0.38, "wspace": 0.3})
    for col, (map_name, title) in enumerate(MAPS):
        track_miniature(axes[0, col], map_name)
        axes[0, col].set_title(title, fontsize=9, pad=6)
        p = axes[0, col].get_position()  # closer to its plot, with room to the neighbours
        axes[0, col].set_position([p.x0 + 0.01, p.y0 - 0.035, p.width - 0.02, p.height - 0.02])
    axes = axes[1:]
    max_rate = max(collision_rate(rows, c, m, nz) for c in controllers for m, _ in MAPS for nz in ("off", "on"))
    for col, (map_name, title) in enumerate(MAPS):
        ax = axes[0, col]
        ax.axhline(0.0, color="#444444", lw=0.9, zorder=1)
        values = np.concatenate([relative(rows, c, (map_name, wp, mu, nz))
                                 for c in controllers for wp, mu in SETTINGS for nz in ("off", "on")])
        values = np.minimum(values[np.isfinite(values)], CLIP)
        for x, (wp, mu) in enumerate(SETTINGS):
            for ctrl, (_label, color, dx) in controllers.items():
                dn0 = 0.05 if len(controllers) <= 3 else 0.035
                for noise, dn, filled in (("off", -dn0, True), ("on", dn0, False)):
                    rel = relative(rows, ctrl, (map_name, wp, mu, noise))
                    shown = np.minimum(rel, CLIP)
                    xs = np.full(len(shown), x + dx + dn)
                    ax.scatter(xs, shown, s=11, marker="o", zorder=3, linewidths=0.8,
                               facecolors=color if filled else "white", edgecolors=color)
                    over = rel > CLIP
                    if over.any():
                        ax.scatter(xs[over], np.full(over.sum(), CLIP), s=18, marker="^", color=color, zorder=4,
                                   clip_on=False)
        ax.set_xlim(-0.5, len(SETTINGS) - 0.5)
        ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.6)
        ax.set_ylim(min(values.min(), -LINTHRESH) * 1.15, max(values.max(), LINTHRESH) * 1.15)
        ticks = [t for t in Y_TICKS if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:+d}" if t else "0" for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(range(len(SETTINGS)))
        ax.grid(axis="y", color="#e4e4e4", lw=0.6, zorder=0)
        for x in (0.5, 1.5, 2.5):
            ax.axvline(x, color="#eeeeee", lw=0.6, zorder=0)
        if col == 0:
            ax.set_ylabel("Lap time vs.\nMPPI 2000 (%)", fontsize=8.5)

        ax = axes[1, col]
        width = 0.78 / len(controllers)
        for i, ctrl in enumerate(controllers):
            color = controllers[ctrl][1]
            for j, noise in enumerate(("off", "on")):
                rate = collision_rate(rows, ctrl, map_name, noise)
                ax.bar(j + (i - (len(controllers) - 1) / 2) * width, rate, width * 0.9, color=color if noise == "off" else "white",
                       edgecolor=color, linewidth=0.9, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["noise off", "noise on"], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, max_rate * 1.1)
        if col:
            ax.set_yticklabels([])
        ax.grid(axis="y", color="#e4e4e4", lw=0.6, zorder=0)
        if col == 0:
            ax.set_ylabel("Collisions\nper 100 laps", fontsize=8.5)

    for ax in axes[0]:
        ax.set_xticklabels([f"{wp[:4]}\nμ={mu}" for wp, mu in SETTINGS], fontsize=6.5)
    for ax in axes[1]:
        ax.tick_params(labelsize=7.5)
    for ax in axes[0]:
        ax.tick_params(axis="y", labelsize=7.5)
    handles = [Line2D([], [], ls="", marker="o", ms=4.5, color=c, label=l) for (l, c, _dx) in controllers.values()]
    handles += [Line2D([], [], ls="", marker="o", ms=4.5, mfc="#777777", mec="#777777", label="noise off"),
                Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec="#777777", label="noise on")]
    many = len(controllers) > 3
    fig.legend(handles=handles, loc="lower center", ncol=4 if many else 5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.09 if many else -0.05), handletextpad=0.2, columnspacing=1.0)
    return fig


def table_numbers(rows):
    """Median relative lap time over all settings and seeds, and collisions per 100 laps, per track."""
    print("\ntrack     | AT50 vs MPPI2000 | MPPI50 vs MPPI2000 | coll/100 laps off (AT50, MPPI50, MPPI2000) | on")
    for map_name, _ in MAPS:
        out = []
        for ctrl in CONTROLLERS:
            rel = np.concatenate([relative(rows, ctrl, (map_name, wp, mu, nz))
                                  for wp, mu in SETTINGS for nz in ("off", "on")])
            out.append(f"{np.nanmedian(rel):+6.1f} %")
        rates = [[collision_rate(rows, c, map_name, nz) for c in CONTROLLERS] for nz in ("off", "on")]
        print(f"{map_name:9s} | {out[0]:16s} | {out[1]:18s} | "
              + ", ".join(f"{r:.2f}" for r in rates[0]) + " | " + ", ".join(f"{r:.2f}" for r in rates[1]))
    better = [np.mean(relative(rows, ("hw", "50"), (m, wp, mu, nz))) < 0
              for m, _ in MAPS for wp, mu in SETTINGS for nz in ("off", "on")]
    print(f"Auto-Tuning MPPI 50 faster than MPPI 2000 (seed mean) in {sum(better)} of {len(better)} settings")


def main():
    icra_figures()
    print("\n== grid runs (all_tracks_grid)")
    rows = drop_broken(load_summary())
    save(all_tracks_figure(rows), "all_tracks_grid")
    table_numbers(rows)
    print(f"\n== {HW_RUNS}, full noise (all_tracks_fullnoise)")
    rows = drop_broken(replace_hw50(load_summary(), HW_RUNS))
    save(all_tracks_figure(rows), "all_tracks_fullnoise")
    table_numbers(rows)
    global HW_CONFIG
    for config in (*OTHER_HALF_CONFIGS, PAPER_CONFIG):
        HW_CONFIG = config
        print(f"\n== {HW_CONFIG}, halved noise (all_tracks_half_{HW_CONFIG}, icra_*_{HW_CONFIG})")
        icra_figures("_" + HW_CONFIG)
        rows = drop_broken(half_noise_rows())
        save(all_tracks_figure(rows), f"all_tracks_half_{HW_CONFIG}")
        table_numbers(rows)
        paper_numbers(rows)
    print(f"\n== paper figures (all_tracks, icra_laptimes, icra_overlay) = {PAPER_CONFIG}")
    for name in ("icra_laptimes", "icra_overlay"):
        for ext in ("pdf", "png"):
            shutil.copyfile(os.path.join(OUT, f"{name}_{PAPER_CONFIG}.{ext}"), os.path.join(OUT, f"{name}.{ext}"))
    for ext in ("pdf", "png"):
        shutil.copyfile(os.path.join(OUT, f"all_tracks_half_{PAPER_CONFIG}.{ext}"), os.path.join(OUT, f"all_tracks.{ext}"))


def paper_numbers(rows):
    """The numbers quoted in the paper's text."""
    hw = ("hw", "50")

    def rel(ctrl, maps=None, wps=None, mus=None, noises=("off", "on")):
        return np.concatenate([relative(rows, ctrl, (m, wp, mu, nz)) for m, _ in MAPS for wp, mu in SETTINGS
                               for nz in noises if (maps is None or m in maps) and (wps is None or wp in wps)
                               and (mus is None or mu in mus)])

    def coll(ctrl, nz):
        seeds = [s for k, v in rows.items() if k[:2] == ctrl and k[5] == nz for s in v]
        return sum(s["collisions"] for s in seeds), 100 * sum(s["collisions"] for s in seeds) / sum(s["laps"] for s in seeds)

    n_hw = sum(len(v) for k, v in rows.items() if k[:2] == hw)
    print(f"runs kept: {n_hw} of 160")
    print(f"median vs MPPI 2000: AT50 {np.nanmedian(rel(hw)):+.1f} %, MPPI50 {np.nanmedian(rel(('mppi', '50'))):+.1f} %")
    print("per track: " + ", ".join(f"{m} {np.nanmedian(rel(hw, maps=(m,))):+.1f} %" for m, _ in MAPS))
    print(f"midline {np.nanmedian(rel(hw, wps=('midline',))):+.1f} %, raceline {np.nanmedian(rel(hw, wps=('raceline',))):+.1f} %")
    print(f"mu 0.5 {np.nanmedian(rel(hw, mus=('0.5',))):+.1f} %, mu 0.9 {np.nanmedian(rel(hw, mus=('0.9',))):+.1f} %")
    print(f"noise off {np.nanmedian(rel(hw, noises=('off',))):+.1f} %, on {np.nanmedian(rel(hw, noises=('on',))):+.1f} %")
    for nz in ("off", "on"):
        print(f"collisions noise {nz}: " + ", ".join(f"{c[0]}{c[1]} {coll(c, nz)[0]} ({coll(c, nz)[1]:.2f}/100 laps)"
                                                    for c in CONTROLLERS))
    freeze = [json.load(open(p))["freeze_t"] for p in glob.glob(os.path.join(TUNE_RUNS, HW_CONFIG, "*", "result.json"))]
    print(f"freeze: median {np.median([f for f in freeze if f]):.0f} s, never froze {sum(1 for f in freeze if not f)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One PDF per map: mean lap time of each experiment, five seeds per box.

The box spans the minimum to the maximum of the seeds that remain. A line marks
the median and a dot the mean. For the HirarchicalPlanner the seed value is the
mean of the laps after the permanent freeze. For MPPI and RPGD it is the mean
of every lap. Only MPPI with 2000 rollouts is drawn. On each map, seed means
above the 90th percentile of that controller are left out.

    python scripts/benchmark/plot_laptime_boxes.py \
        --summary /home/emilio/projects/bachelor/benchmark_results/grid_2026-11-05/summary.csv \
        --out /home/emilio/projects/bachelor/benchmark_results/grid_2026-11-05/plots
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

CONTROLLERS = [
    ("hw", "50", "HW 50"),
    ("hw", "200", "HW 200"),
    ("mppi", "2000", "MPPI 2000"),
    ("rpgd", "", "RPGD"),
]
WAYPOINTS = ("midline", "raceline")
CONDITIONS = (("0.5", "off"), ("0.5", "on"), ("0.9", "off"), ("0.9", "on"))
COLORS = {"hw": "#3d6f9c", "mppi": "#6b6b6b", "rpgd": "#c47b2b"}


def seed_value(row: dict) -> float | None:
    raw = row["after_freeze_lap_mean"] if row["controller"] == "hw" else row["lap_mean"]
    if raw == "":
        return None
    return float(raw)


def load(summary: Path) -> dict:
    groups: dict[tuple, list[float]] = defaultdict(list)
    with summary.open() as f:
        for row in csv.DictReader(f):
            value = seed_value(row)
            if value is None:
                continue
            rollouts = row["rollouts"] if row["controller"] != "rpgd" else ""
            groups[(row["map"], row["wp"], row["mu"], row["noise"], row["controller"], rollouts)].append(value)
    return groups


def percentile_90(values: list[float]) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = 0.9 * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def without_top_decile(map_name: str, groups: dict) -> dict:
    """Drop seed means above each controller's 90th percentile on this map."""
    kept_controllers = {(controller, rollouts) for controller, rollouts, _label in CONTROLLERS}
    pooled: dict[tuple, list[float]] = defaultdict(list)
    for (m, _wp, _mu, _noise, controller, rollouts), values in groups.items():
        if m == map_name and (controller, rollouts) in kept_controllers:
            pooled[(controller, rollouts)].extend(values)
    caps = {car: percentile_90(values) for car, values in pooled.items()}
    trimmed = {}
    for (m, wp, mu, noise, controller, rollouts), values in groups.items():
        if m != map_name or (controller, rollouts) not in caps:
            continue
        remaining = [value for value in values if value <= caps[(controller, rollouts)]]
        if remaining:
            trimmed[(m, wp, mu, noise, controller, rollouts)] = remaining
    return trimmed


def draw_box(ax, x: float, values: list[float], color: str) -> None:
    ordered = sorted(values)
    n = len(ordered)
    low, high = ordered[0], ordered[-1]
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    mean = sum(ordered) / n
    height = max(high - low, 1e-3)
    width = 0.46
    ax.add_patch(Rectangle((x - width / 2, low), width, height, facecolor="none", edgecolor=color,
                           linewidth=1.3, zorder=2))
    ax.plot([x - width / 2, x + width / 2], [median, median], color="black", linewidth=1.4,
            solid_capstyle="butt", zorder=3)
    ax.plot(x, mean, marker="o", color="black", markersize=3.4, zorder=4)
    if n < 5:
        ax.text(x, high, f"n={n}", ha="center", va="bottom", fontsize=6, color="#444444")


def plot_map(map_name: str, groups: dict, out: Path) -> None:
    groups = without_top_decile(map_name, groups)
    fig, axes = plt.subplots(len(CONDITIONS), len(WAYPOINTS), figsize=(7.2, 11.2), sharex=True)
    for row, (mu, noise) in enumerate(CONDITIONS):
        for col, wp in enumerate(WAYPOINTS):
            ax = axes[row, col]
            for i, (controller, rollouts, _label) in enumerate(CONTROLLERS):
                values = groups.get((map_name, wp, mu, noise, controller, rollouts), [])
                if values:
                    draw_box(ax, i, values, COLORS[controller])
            ax.set_xlim(-0.6, len(CONTROLLERS) - 0.4)
            ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 0:
                ax.set_ylabel("Mean lap time (s)")
            if row == 0:
                ax.set_title(wp)
            if col == len(WAYPOINTS) - 1:
                ax.text(1.02, 0.5, f"μ {mu}\nnoise {noise}", transform=ax.transAxes,
                        va="center", ha="left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xticks(range(len(CONTROLLERS)))
        ax.set_xticklabels([label for *_rest, label in CONTROLLERS], rotation=35, ha="right")
    legend = [
        Patch(facecolor="none", edgecolor=COLORS["hw"], linewidth=1.3, label="HW"),
        Patch(facecolor="none", edgecolor=COLORS["mppi"], linewidth=1.3, label="MPPI 2000"),
        Patch(facecolor="none", edgecolor=COLORS["rpgd"], linewidth=1.3, label="RPGD"),
        Line2D([0], [0], color="black", linewidth=1.4, label="median"),
        Line2D([0], [0], color="black", marker="o", linestyle="", markersize=4, label="mean"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.48, 0.02))
    fig.suptitle(f"{map_name}: mean lap time per seed", fontsize=13)
    fig.text(0.5, 0.008,
             "Min to max of the remaining seeds. Each controller's slowest 10% on this map is omitted.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 0.96, 0.97))
    fig.savefig(out / f"{map_name}.pdf")
    plt.close(fig)
    print(out / f"{map_name}.pdf", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    groups = load(args.summary)
    for map_name in ("RCA1", "IPZ50", "hangar3", "icra2022"):
        plot_map(map_name, groups, args.out)


if __name__ == "__main__":
    main()

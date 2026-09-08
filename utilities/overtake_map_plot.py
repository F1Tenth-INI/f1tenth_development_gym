"""Save overtake and crash events to CSV and overlay them on the occupancy map."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from utilities.Settings import Settings
from utilities.RaceJudge import RaceJudge


INCIDENT_CSV_FIELDS = [
    "event",
    "kind",
    "fault",
    "fault_reason",
    "episode",
    "time",
    "step",
    "slot",
    "ego_x",
    "ego_y",
    "ego_theta",
    "opponent_x",
    "opponent_y",
    "opponent_theta",
    "gap_m",
    "range_m",
    "left_m",
    "ego_wp",
    "opponent_wp",
]

STATS_CSV_FIELDS = [
    "overtakes",
    "crashes",
    "clean_passes",
    "opponent_crashes",
    "ego_fault_opponent_crashes",
    "opponent_fault_opponent_crashes",
    "wall_crashes",
    "leave_track_crashes",
    "spin_crashes",
]


def experiment_data_dir(
    csv_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Return the experiment ``data/`` folder (not ExperimentRecordings root).

    Preference order:
    1. Explicit ``output_dir``
    2. ``<experiment_dir>/data`` next to the recording CSV
    3. SAC model directory when ``SAC_INFERENCE_MODEL_NAME`` / ``DATASET_NAME`` is set
    """
    if output_dir:
        folder = os.path.abspath(os.path.expanduser(str(output_dir)))
        os.makedirs(folder, exist_ok=True)
        return folder

    if csv_filepath:
        from utilities.saving_helpers import experiment_data_path

        return experiment_data_path(csv_filepath)

    model_name = (
        getattr(Settings, "SAC_INFERENCE_MODEL_NAME", None)
        or getattr(Settings, "DATASET_NAME", None)
    )
    if model_name:
        try:
            from TrainingLite.rl_racing.sac_utilities import SacUtilities

            _, model_dir = SacUtilities.resolve_model_paths(str(model_name))
            os.makedirs(model_dir, exist_ok=True)
            return model_dir
        except Exception:
            pass

    raise ValueError(
        "No experiment data folder: pass csv_filepath/output_dir, or set "
        "SAC_INFERENCE_MODEL_NAME / DATASET_NAME."
    )


def default_incident_paths(
    csv_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict[str, str]:
    """CSV/PNG paths inside the experiment data folder."""
    folder = experiment_data_dir(csv_filepath=csv_filepath, output_dir=output_dir)
    return {
        "overtakes_csv": os.path.join(folder, "overtakes.csv"),
        "crashes_csv": os.path.join(folder, "crashes.csv"),
        "stats_csv": os.path.join(folder, "incident_stats.csv"),
        "png": os.path.join(folder, "incidents.png"),
    }


def write_incidents_csv(events: Sequence[Mapping], csv_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INCIDENT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow({field: event.get(field, "") for field in INCIDENT_CSV_FIELDS})
    return csv_path


def write_stats_csv(stats: Mapping, csv_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATS_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: stats.get(field, 0) for field in STATS_CSV_FIELDS})
    return csv_path


def _load_scaled_map_background() -> tuple[np.ndarray, list[float]] | None:
    from utilities.ExperimentAnalyzer import _load_map_background_extent

    map_name = str(getattr(Settings, "MAP_NAME", "") or "")
    map_path = getattr(Settings, "MAP_PATH", None)
    if not map_name:
        return None
    map_dir = Path(map_path) if map_path else Path("utilities") / "maps" / map_name
    if not map_dir.is_absolute():
        map_dir = (Path.cwd() / map_dir).resolve()
    background = _load_map_background_extent(map_dir, map_name)
    if background is None:
        return None
    image, extent = background
    scale = float(getattr(Settings, "MAP_SCALE", 1.0) or 1.0)
    if scale != 1.0:
        extent = [float(value) * scale for value in extent]
    return image, extent


def _xy(events: Sequence[Mapping], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([float(event.get(x_key, np.nan)) for event in events], dtype=np.float64)
    ys = np.array([float(event.get(y_key, np.nan)) for event in events], dtype=np.float64)
    return xs, ys


def plot_incidents_on_map(
    overtakes: Sequence[Mapping],
    crashes: Sequence[Mapping],
    png_path: str,
    *,
    show: bool = False,
) -> Optional[str]:
    """Scatter ego overtakes and crashes on the map image."""
    fig, ax = plt.subplots(figsize=(9, 8))
    background = _load_scaled_map_background()
    if background is not None:
        map_image, map_extent = background
        ax.imshow(map_image, cmap="gray", origin="lower", extent=map_extent, alpha=0.65)

    if overtakes:
        ego_x, ego_y = _xy(overtakes, "ego_x", "ego_y")
        opp_x, opp_y = _xy(overtakes, "opponent_x", "opponent_y")
        opp_ok = np.isfinite(opp_x) & np.isfinite(opp_y)
        if np.any(opp_ok):
            ax.scatter(
                opp_x[opp_ok],
                opp_y[opp_ok],
                s=36,
                c="tab:orange",
                marker="s",
                zorder=3,
                label="Opponent (being passed)",
            )
            ax.quiver(
                opp_x[opp_ok],
                opp_y[opp_ok],
                ego_x[opp_ok] - opp_x[opp_ok],
                ego_y[opp_ok] - opp_y[opp_ok],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
                color="0.35",
                zorder=3,
            )
        ax.scatter(
            ego_x,
            ego_y,
            s=55,
            c="tab:red",
            marker="o",
            zorder=4,
            label="Ego overtake",
        )

    crash_styles = (
        ("opponent", "ego", "tab:purple", "P", "Opponent crash (ego at fault)"),
        ("opponent", "opponent", "tab:green", "P", "Opponent crash (opponent at fault)"),
        ("wall", None, "black", "x", "Wall crash"),
        ("leave_track", None, "tab:blue", "x", "Left track"),
        ("spin", None, "0.45", "+", "Spin"),
    )
    plotted_kinds = set()
    for crash in crashes:
        kinds = str(crash.get("kind", "") or "").split("+")
        fault = str(crash.get("fault", "") or "ego")
        x = float(crash.get("ego_x", np.nan))
        y = float(crash.get("ego_y", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        style = None
        for kind_key, fault_key, color, marker, label in crash_styles:
            if kind_key not in kinds:
                continue
            if fault_key is not None and fault != fault_key:
                continue
            style = (color, marker, label)
            break
        if style is None:
            style = ("tab:brown", "x", "Crash")
        color, marker, label = style
        legend_label = label if label not in plotted_kinds else None
        plotted_kinds.add(label)
        ax.scatter(
            [x],
            [y],
            s=70,
            c=color,
            marker=marker,
            linewidths=1.6,
            zorder=5,
            label=legend_label,
        )
        if "opponent" in kinds:
            ox = float(crash.get("opponent_x", np.nan))
            oy = float(crash.get("opponent_y", np.nan))
            if np.isfinite(ox) and np.isfinite(oy):
                ax.scatter(
                    [ox],
                    [oy],
                    s=28,
                    c=color,
                    marker="s",
                    zorder=4,
                    label="Opponent at crash" if "Opponent at crash" not in plotted_kinds else None,
                )
                plotted_kinds.add("Opponent at crash")
                at_fault_ego = fault != "opponent"
                if at_fault_ego:
                    ax.quiver(
                        x,
                        y,
                        ox - x,
                        oy - y,
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        width=0.004,
                        color=color,
                        zorder=4,
                    )
                else:
                    ax.quiver(
                        ox,
                        oy,
                        x - ox,
                        y - oy,
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        width=0.004,
                        color=color,
                        zorder=4,
                    )

    stats = RaceJudge.summarize(overtakes, crashes)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Incidents on {Settings.MAP_NAME}\n{RaceJudge.format_summary(stats)}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    if overtakes or crashes:
        ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=150)

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return png_path


def save_and_plot_overtakes(
    events: Iterable[Mapping],
    *,
    crashes: Optional[Iterable[Mapping]] = None,
    csv_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
    show: bool = True,
) -> tuple[str, Optional[str]]:
    """Backward-compatible wrapper: overtakes required, crashes optional."""
    return save_and_plot_incidents(
        events,
        crashes or [],
        csv_filepath=csv_filepath,
        output_dir=output_dir,
        show=show,
    )


def save_and_plot_incidents(
    overtakes: Iterable[Mapping],
    crashes: Iterable[Mapping],
    *,
    csv_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
    show: bool = True,
) -> tuple[str, Optional[str]]:
    overtakes = list(overtakes)
    crashes = list(crashes)
    stats = RaceJudge.summarize(overtakes, crashes)
    paths = default_incident_paths(csv_filepath=csv_filepath, output_dir=output_dir)
    write_incidents_csv(overtakes, paths["overtakes_csv"])
    write_incidents_csv(crashes, paths["crashes_csv"])
    write_stats_csv(stats, paths["stats_csv"])
    print(f"[incidents] saved {len(overtakes)} overtake(s) to {paths['overtakes_csv']}")
    print(f"[incidents] saved {len(crashes)} crash(es) to {paths['crashes_csv']}")
    print(f"[incidents] {RaceJudge.format_summary(stats)}")
    print(f"[incidents] stats saved to {paths['stats_csv']}")
    plot_path = plot_incidents_on_map(overtakes, crashes, paths["png"], show=show)
    if plot_path:
        print(f"[incidents] map plot saved to {plot_path}")
    return paths["overtakes_csv"], plot_path

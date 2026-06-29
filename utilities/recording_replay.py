"""Helpers for replaying and recording simulation CSVs (virtual opponents, laptimes)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from utilities.car_system import CarSystem

_VOPP_POSE_SUFFIXES = ("POSE_X", "POSE_Y", "POSE_THETA")


def virtual_opponent_recording_column_name(slot: int, component: int) -> str:
    return f"VOPP_{slot:02d}_{_VOPP_POSE_SUFFIXES[component]}"


def get_virtual_opponent_recording_dict(driver: "CarSystem", slot_count: int) -> dict:
    """CSV columns for virtual opponent poses [x, y, theta] per slot."""
    recording_dict = {}

    def _pose_value(opponent_idx: int, component_idx: int):
        def getter():
            if driver.virtual_opponents is None:
                return float("nan")
            poses = driver.virtual_opponents.get_poses()
            if opponent_idx >= len(poses):
                return float("nan")
            return float(poses[opponent_idx, component_idx])

        return getter

    for slot in range(slot_count):
        for component_idx in range(3):
            recording_dict[virtual_opponent_recording_column_name(slot, component_idx)] = _pose_value(
                slot, component_idx
            )
    return recording_dict


def load_virtual_opponent_replay_poses(csv_path: str) -> Optional[np.ndarray]:
    """Load virtual opponent poses from a recording CSV, or None if absent."""
    import pandas as pd

    header_df = pd.read_csv(csv_path, comment="#", nrows=0)
    slot_indices: set[int] = set()
    for column in header_df.columns:
        if column.startswith("VOPP_") and column.endswith("_POSE_X"):
            try:
                slot_indices.add(int(column.split("_")[1]))
            except (IndexError, ValueError):
                continue
    if not slot_indices:
        return None

    df = pd.read_csv(csv_path, comment="#")
    num_slots = max(slot_indices) + 1
    poses = np.full((len(df), num_slots, 3), np.nan, dtype=np.float64)
    for slot in slot_indices:
        for component_idx, suffix in enumerate(_VOPP_POSE_SUFFIXES):
            column = virtual_opponent_recording_column_name(slot, component_idx)
            if column in df.columns:
                poses[:, slot, component_idx] = df[column].to_numpy(dtype=np.float64)
    return poses


def load_recording_laptimes(csv_path: str, recording_df=None) -> list[float]:
    """Read lap times from CSV header, or infer them from nearest_wpt_idx crossings."""
    from pathlib import Path

    from utilities.ExperimentAnalyzer import (
        _extract_lap_times_from_csv_header,
        _infer_lap_times_from_recording,
    )

    lap_times = _extract_lap_times_from_csv_header(Path(csv_path))
    if lap_times:
        return lap_times
    if recording_df is None:
        import pandas as pd

        recording_df = pd.read_csv(csv_path, comment="#")
    return _infer_lap_times_from_recording(recording_df)


def resolve_recording_csv_path(csv_arg: str | None = None) -> str:
    """Resolve a recording CSV from a full path, filename, or Settings default."""
    from utilities.Settings import Settings

    if not csv_arg:
        return os.path.abspath(Settings.RECORDING_PATH)

    if os.path.isfile(csv_arg):
        return os.path.abspath(csv_arg)

    candidates: list[str] = []
    if not os.path.dirname(csv_arg):
        candidates.append(os.path.join(Settings.RECORDING_FOLDER, csv_arg))
    candidates.append(csv_arg)

    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if os.path.isfile(resolved):
            return resolved

    return os.path.abspath(csv_arg)


def resolve_map_for_recording(csv_path: str, map_override: str | None = None) -> tuple[str, str]:
    """Resolve map render path and name for replay (CSV header, then Settings)."""
    from pathlib import Path

    from utilities.ExperimentAnalyzer import (
        _extract_metadata_from_csv_header,
        _resolve_map_dir_from_metadata,
    )
    from utilities.Settings import Settings

    if map_override:
        map_name = map_override
        map_render_path = os.path.join(Settings.MAP_PATH, map_name)
        return map_render_path, map_name

    metadata = _extract_metadata_from_csv_header(Path(csv_path))
    map_dir, map_name = _resolve_map_dir_from_metadata(metadata)
    if map_dir is not None and map_name:
        return os.path.join(str(map_dir), map_name), map_name

    map_name = metadata.get("map name") or Settings.MAP_NAME
    map_path_meta = metadata.get("map path")
    if map_path_meta:
        map_render_path = os.path.join(map_path_meta, map_name)
    else:
        map_render_path = os.path.join(Settings.MAP_PATH, map_name)
    return map_render_path, map_name


def next_waypoints_from_recording_row(row) -> Optional[np.ndarray]:
    """Rebuild look-ahead waypoint polyline from WYPT_X/Y columns in one CSV row."""
    x_cols = sorted(
        [col for col in row.index if str(col).startswith("WYPT_X_")],
        key=lambda name: int(str(name).split("_")[-1]),
    )
    y_cols = sorted(
        [col for col in row.index if str(col).startswith("WYPT_Y_")],
        key=lambda name: int(str(name).split("_")[-1]),
    )
    if not x_cols or not y_cols or len(x_cols) != len(y_cols):
        return None
    return np.column_stack(
        [
            np.asarray([float(row[col]) for col in x_cols], dtype=np.float32),
            np.asarray([float(row[col]) for col in y_cols], dtype=np.float32),
        ]
    )


def apply_replay_recording_context(
    driver: "CarSystem",
    row_idx: int,
    recording_df,
    replay_laptimes: list[float],
) -> None:
    """Restore per-row replay metadata used for rendering and lap-time display."""
    if recording_df is None or row_idx < 0 or row_idx >= len(recording_df):
        return

    row = recording_df.iloc[row_idx]
    if "time" in recording_df.columns:
        driver.time = float(row["time"])
    if "nearest_wpt_idx" in recording_df.columns:
        driver.waypoint_utils.nearest_waypoint_index = int(row["nearest_wpt_idx"])
    driver.laptimes = list(replay_laptimes)

    next_waypoints = next_waypoints_from_recording_row(row)
    if next_waypoints is not None and driver.render_utils is not None:
        driver.render_utils.next_waypoints = next_waypoints


def get_virtual_opponent_poses_for_render(driver: "CarSystem") -> Optional[np.ndarray]:
    """Virtual opponent poses for the renderer during live simulation."""
    if driver.virtual_opponents is not None:
        poses = driver.virtual_opponents.get_poses()
        return poses if len(poses) > 0 else None
    return None

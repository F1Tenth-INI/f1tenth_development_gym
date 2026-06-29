"""Helpers for replaying and recording simulation CSVs (virtual opponents, laptimes)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from utilities.Settings import Settings

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
    """Virtual opponent poses for the renderer (live sim or CSV replay)."""
    if Settings.REPLAY_RECORDING:
        poses = getattr(driver, "_virtual_opponent_replay_poses", None)
        if poses is None:
            poses = load_virtual_opponent_replay_poses(Settings.RECORDING_PATH)
            driver._virtual_opponent_replay_poses = poses
        if poses is not None:
            row = max(0, driver.control_index - 1)
            if row >= len(poses):
                row = len(poses) - 1
            slot_poses = np.asarray(poses[row], dtype=np.float32)
            if slot_poses.size == 0:
                return None
            valid_mask = ~np.isnan(slot_poses[:, 0])
            if np.any(valid_mask):
                return slot_poses[valid_mask]

    if driver.virtual_opponents is not None:
        poses = driver.virtual_opponents.get_poses()
        return poses if len(poses) > 0 else None
    return None

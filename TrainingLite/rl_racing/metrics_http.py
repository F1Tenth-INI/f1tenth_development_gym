#!/usr/bin/env python3
"""Lightweight asyncio HTTP server for live training metrics (stdlib only)."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

_SKIP_PLOT_COLS = frozenset(
    {
        "timestamp",
        "time",
        "training_duration",
        "post_process_duration",
        "batch_size",
        "gradient_steps",
        "learning_rate",
    }
)

_X_AXIS_ALIASES = {
    "wallclock": "wallclock",
    "wall-clock": "wallclock",
    "wall_clock": "wallclock",
    "time": "wallclock",
    "simulation": "simulation",
    "sim": "simulation",
    "sim_time": "simulation",
    "timesteps": "timesteps",
    "total_timesteps": "timesteps",
    "steps": "timesteps",
}
_X_AXIS_COLUMNS = {
    "wallclock": ("time",),
    "simulation": ("total_timesteps", "transitions_total"),
    "timesteps": ("total_timesteps", "transitions_total"),
}
_X_AXIS_LABELS = {
    "wallclock": "wall-clock time (s)",
    "simulation": "simulation time (s)",
    "timesteps": "timesteps",
}
_STEP_COLUMNS = ("total_timesteps", "transitions_total")
_DEFAULT_TIMESTEP_CONTROL_S = 0.04
_TIMESTEP_CONTROL_RE = re.compile(r"TIMESTEP_CONTROL\s*=\s*([0-9]*\.?[0-9]+)")


def _normalize_x_axis(x_axis: Optional[str]) -> str:
    key = str(x_axis or "wallclock").strip().lower().replace(" ", "_")
    return _X_AXIS_ALIASES.get(key, "wallclock")


def _parse_timestep_control(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _TIMESTEP_CONTROL_RE.search(text)
    if match is None:
        return None
    dt = float(match.group(1))
    return dt if dt > 0.0 else None


def _timestep_dt_s(csv_path: str) -> float:
    model_dir = Path(csv_path).resolve().parent
    for candidate in (
        model_dir / "training_files" / "Settings.py",
        Path(__file__).resolve().parents[2] / "utilities" / "Settings.py",
    ):
        dt = _parse_timestep_control(candidate)
        if dt is not None:
            return dt
    try:
        from utilities.Settings import Settings

        dt = float(getattr(Settings, "TIMESTEP_CONTROL", 0.0) or 0.0)
        if dt > 0.0:
            return dt
    except Exception:
        pass
    return _DEFAULT_TIMESTEP_CONTROL_S


def _select_x_values(
    df: pd.DataFrame,
    x_axis: str,
    dt_s: float,
) -> Tuple[List[float], str, str]:
    requested = _normalize_x_axis(x_axis)

    if requested in ("simulation", "timesteps"):
        for col in _STEP_COLUMNS:
            if col in df.columns:
                steps = df[col].astype(float)
                if requested == "simulation":
                    return (steps * float(dt_s)).tolist(), col, _X_AXIS_LABELS["simulation"]
                return steps.tolist(), col, _X_AXIS_LABELS["timesteps"]

    if "time" in df.columns:
        return df["time"].astype(float).tolist(), "time", _X_AXIS_LABELS["wallclock"]

    for col in _STEP_COLUMNS:
        if col in df.columns:
            return df[col].astype(float).tolist(), col, _X_AXIS_LABELS["timesteps"]

    return list(range(len(df))), "log_index", "log_index"


def _empty_metrics_payload(
    model_name: str,
    csv_path: str,
    x_axis: str,
    *,
    csv_mtime: Optional[float] = None,
    error: Optional[str] = None,
    row_count: int = 0,
) -> Dict[str, Any]:
    requested = _normalize_x_axis(x_axis)
    payload: Dict[str, Any] = {
        "model_name": model_name,
        "csv_path": csv_path,
        "csv_mtime": csv_mtime,
        "row_count": row_count,
        "x_axis": requested,
        "x_key": _X_AXIS_COLUMNS[requested][0],
        "x_label": _X_AXIS_LABELS[requested],
        "sim_dt_s": _timestep_dt_s(csv_path),
        "series": [],
    }
    if error:
        payload["error"] = error
    return payload


_ARRAY_LIKE_COLS = frozenset(
    {
        "episode_lengths",
        "episode_rewards",
        "episode_mean_step_rewards",
        "stream_batch_sizes",
        "lap_times",
    }
)

_STATIC_DIR = Path(__file__).resolve().parent / "TrainingPlot"
_DASHBOARD_PATH = _STATIC_DIR / "index.html"


def _parse_array_like(value: Any) -> List[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        arr = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            arr = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    else:
        return []
    if not isinstance(arr, (list, tuple)):
        return []
    out: List[float] = []
    for item in arr:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _is_array_like_column(series: pd.Series) -> bool:
    if series.empty:
        return False
    first = series.iloc[0]
    if isinstance(first, (list, tuple)):
        return True
    if isinstance(first, str):
        try:
            parsed = ast.literal_eval(first.strip())
            return isinstance(parsed, (list, tuple))
        except (SyntaxError, ValueError):
            return False
    return False


def _reward_component_keys_from_stats_df(df: pd.DataFrame) -> List[str]:
    if df.empty or "component" not in df.columns:
        return []
    keys = []
    for component in df["component"].tolist():
        name = str(component).strip()
        if name:
            keys.append(name)
    return sorted(set(keys))


def _load_reward_components_live_from_stats(model_dir: str) -> Optional[Dict[str, Any]]:
    stats_path = Path(model_dir) / "obs_tracking" / "reward_components_stats.csv"
    summary_path = Path(model_dir) / "obs_tracking" / "tracker_summary.json"
    if not stats_path.is_file():
        return None
    try:
        df = pd.read_csv(stats_path)
    except Exception:
        return None
    if df.empty or "component" not in df.columns or "accumulated" not in df.columns:
        return None

    keys = _reward_component_keys_from_stats_df(df)
    if not keys:
        return None
    total_accumulated = {key: 0.0 for key in keys}
    for _, row in df.iterrows():
        component = str(row.get("component", "")).strip()
        if component in total_accumulated:
            try:
                total_accumulated[component] = float(row.get("accumulated", 0.0))
            except (TypeError, ValueError):
                total_accumulated[component] = 0.0

    total_steps = 0
    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            total_steps = int(summary.get("reward_components_seen") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            total_steps = 0
    if total_steps <= 0 and "count" in df.columns:
        try:
            total_steps = int(df["count"].max())
        except (TypeError, ValueError):
            total_steps = 0

    return {
        "timesteps": None,
        "total_steps": total_steps,
        "last_episode_steps": 0,
        "components": keys,
        "total_accumulated": total_accumulated,
        "last_episode_accumulated": {key: 0.0 for key in keys},
        "is_live": True,
        "source": "reward_components_stats.csv",
    }


def load_reward_components_payload(model_dir: str, model_name: str) -> Dict[str, Any]:
    """Load checkpoint + live reward-component snapshots for the metrics dashboard."""
    history_dir = Path(model_dir) / "obs_tracking" / "reward_components"
    checkpoints: List[Dict[str, Any]] = []
    if history_dir.is_dir():
        for json_path in sorted(history_dir.glob("checkpoint_*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            timesteps = payload.get("timesteps")
            try:
                timesteps_int = int(timesteps)
            except (TypeError, ValueError):
                continue
            payload["timesteps"] = timesteps_int
            checkpoints.append(payload)

        checkpoints.sort(key=lambda item: int(item.get("timesteps", 0)))

        live_path = history_dir / "live.json"
        live_payload: Optional[Dict[str, Any]] = None
        if live_path.is_file():
            try:
                with open(live_path, "r", encoding="utf-8") as f:
                    live_payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                live_payload = None

        if not isinstance(live_payload, dict):
            live_payload = _load_reward_components_live_from_stats(model_dir)

        if isinstance(live_payload, dict):
            live_steps = int(live_payload.get("total_steps") or 0)
            latest_checkpoint_steps = (
                int(checkpoints[-1].get("total_steps") or 0) if checkpoints else 0
            )
            if live_steps >= latest_checkpoint_steps:
                live_payload = dict(live_payload)
                live_payload["timesteps"] = None
                live_payload["is_live"] = True
                if checkpoints and live_steps == latest_checkpoint_steps:
                    checkpoints[-1] = live_payload
                else:
                    checkpoints.append(live_payload)

    return {
        "model_name": model_name,
        "model_dir": model_dir,
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
    }


def _load_obs_tracker_summary(model_dir: str) -> Dict[str, Any]:
    summary_path = Path(model_dir) / "obs_tracking" / "tracker_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _obs_snapshot_from_live_files(model_dir: str) -> Optional[Dict[str, Any]]:
    """Build a snapshot payload from the latest obs_stats.csv + obs_histograms.npz."""
    tracking_dir = Path(model_dir) / "obs_tracking"
    stats_path = tracking_dir / "obs_stats.csv"
    npz_path = tracking_dir / "obs_histograms.npz"
    if not stats_path.is_file() or not npz_path.is_file():
        return None
    try:
        stats_df = pd.read_csv(stats_path)
    except Exception:
        return None
    if stats_df.empty or "obs_idx" not in stats_df.columns:
        return None

    summary = _load_obs_tracker_summary(model_dir)
    obs_seen = int(summary.get("obs_seen") or stats_df["count"].iloc[0] or 0)
    obs_dim = int(summary.get("obs_dim") or len(stats_df))
    hist_bins = int(summary.get("obs_hist_bins") or 40)
    hist_sample_count = int(summary.get("obs_hist_sample_count") or 0)

    dims: List[Dict[str, Any]] = []
    try:
        npz = np.load(npz_path)
    except Exception:
        npz = None

    for _, row in stats_df.iterrows():
        idx = int(row["obs_idx"])
        dim_entry: Dict[str, Any] = {
            "idx": idx,
            "mean": float(row.get("mean", 0.0)),
            "std": float(row.get("std", 0.0)),
            "min": float(row.get("min", 0.0)),
            "max": float(row.get("max", 0.0)),
        }
        if npz is not None:
            counts_key = f"obs_{idx}_counts"
            edges_key = f"obs_{idx}_edges"
            if counts_key in npz.files and edges_key in npz.files:
                dim_entry["counts"] = np.asarray(npz[counts_key], dtype=int).tolist()
                dim_entry["edges"] = np.asarray(npz[edges_key], dtype=float).tolist()
        dims.append(dim_entry)

    return {
        "obs_seen": obs_seen,
        "obs_dim": obs_dim,
        "hist_bins": hist_bins,
        "hist_sample_count": hist_sample_count,
        "dims": dims,
        "is_live": True,
        "source": "obs_histograms.npz",
    }


def _load_obs_history_manifest(model_dir: str) -> List[Dict[str, Any]]:
    manifest_path = Path(model_dir) / "obs_tracking" / "history" / "manifest.json"
    entries: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                entries = [entry for entry in loaded if isinstance(entry, dict)]
        except (OSError, json.JSONDecodeError):
            entries = []

    if entries:
        return sorted(entries, key=lambda item: int(item.get("obs_seen", 0)))

    history_dir = Path(model_dir) / "obs_tracking" / "history"
    if not history_dir.is_dir():
        return []
    for json_path in sorted(history_dir.glob("snapshot_*.json")):
        try:
            obs_seen = int(json_path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        entries.append({"obs_seen": obs_seen, "path": json_path.name})
    return sorted(entries, key=lambda item: int(item.get("obs_seen", 0)))


def _load_obs_snapshot_by_seen(model_dir: str, obs_seen: int) -> Optional[Dict[str, Any]]:
    history_dir = Path(model_dir) / "obs_tracking" / "history"
    snapshot_path = history_dir / f"snapshot_{obs_seen}.json"
    if snapshot_path.is_file():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload["is_live"] = False
                payload["source"] = snapshot_path.name
                return payload
        except (OSError, json.JSONDecodeError):
            pass

    live = _obs_snapshot_from_live_files(model_dir)
    if live is not None and int(live.get("obs_seen", -1)) == int(obs_seen):
        return live
    return None


def _load_obs_stats_history(model_dir: str) -> List[Dict[str, Any]]:
    history_path = Path(model_dir) / "obs_tracking" / "obs_stats_history.csv"
    rows: List[Dict[str, Any]] = []
    if history_path.is_file():
        try:
            df = pd.read_csv(history_path)
        except Exception:
            df = pd.DataFrame()
        if not df.empty and "obs_seen" in df.columns and "obs_idx" in df.columns:
            for _, row in df.iterrows():
                try:
                    rows.append(
                        {
                            "obs_seen": int(row["obs_seen"]),
                            "obs_idx": int(row["obs_idx"]),
                            "mean": float(row.get("mean", 0.0)),
                            "std": float(row.get("std", 0.0)),
                            "min": float(row.get("min", 0.0)),
                            "max": float(row.get("max", 0.0)),
                        }
                    )
                except (TypeError, ValueError):
                    continue
    if rows:
        return rows

    stats_path = Path(model_dir) / "obs_tracking" / "obs_stats.csv"
    if not stats_path.is_file():
        return []
    summary = _load_obs_tracker_summary(model_dir)
    try:
        obs_seen = int(summary.get("obs_seen") or 0)
        stats_df = pd.read_csv(stats_path)
    except Exception:
        return []
    if stats_df.empty or obs_seen <= 0:
        return []
    fallback_rows: List[Dict[str, Any]] = []
    for _, row in stats_df.iterrows():
        try:
            fallback_rows.append(
                {
                    "obs_seen": obs_seen,
                    "obs_idx": int(row["obs_idx"]),
                    "mean": float(row.get("mean", 0.0)),
                    "std": float(row.get("std", 0.0)),
                    "min": float(row.get("min", 0.0)),
                    "max": float(row.get("max", 0.0)),
                }
            )
        except (TypeError, ValueError):
            continue
    return fallback_rows


def load_obs_tracking_payload(
    model_dir: str,
    model_name: str,
    *,
    obs_seen: Optional[int] = None,
    include_stats: bool = False,
) -> Dict[str, Any]:
    """Load observation tracking manifest and optionally one snapshot's histograms."""
    summary = _load_obs_tracker_summary(model_dir)
    manifest = _load_obs_history_manifest(model_dir)
    live = _obs_snapshot_from_live_files(model_dir)

    snapshots_meta: List[Dict[str, Any]] = []
    seen_values = set()
    for entry in manifest:
        try:
            steps = int(entry.get("obs_seen", 0))
        except (TypeError, ValueError):
            continue
        seen_values.add(steps)
        snapshots_meta.append(
            {
                "obs_seen": steps,
                "obs_dim": int(entry.get("obs_dim") or summary.get("obs_dim") or 0),
                "hist_sample_count": int(entry.get("hist_sample_count") or 0),
                "is_live": False,
            }
        )

    if isinstance(live, dict):
        live_steps = int(live.get("obs_seen") or 0)
        if live_steps > 0:
            live_meta = {
                "obs_seen": live_steps,
                "obs_dim": int(live.get("obs_dim") or 0),
                "hist_sample_count": int(live.get("hist_sample_count") or 0),
                "is_live": True,
            }
            if live_steps in seen_values:
                snapshots_meta = [
                    live_meta if item["obs_seen"] == live_steps else item for item in snapshots_meta
                ]
            else:
                snapshots_meta.append(live_meta)
            snapshots_meta.sort(key=lambda item: int(item.get("obs_seen", 0)))

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "model_dir": model_dir,
        "summary": summary,
        "snapshots": snapshots_meta,
        "snapshot_count": len(snapshots_meta),
    }
    if include_stats:
        payload["stats_history"] = _load_obs_stats_history(model_dir)

    if obs_seen is not None:
        snapshot = _load_obs_snapshot_by_seen(model_dir, int(obs_seen))
        if snapshot is None and isinstance(live, dict):
            if int(live.get("obs_seen", -1)) == int(obs_seen):
                snapshot = live
        payload["snapshot"] = snapshot
        if snapshot is None:
            payload["error"] = f"snapshot not found for obs_seen={obs_seen}"

    return payload


def load_metrics_payload(
    csv_path: str,
    model_name: str,
    ingest_csv_path: Optional[str] = None,
    x_axis: str = "wallclock",
) -> Dict[str, Any]:
    """Parse learning_metrics.csv (and optional ingest_metrics.csv) into chart payload."""
    payload = _load_single_metrics_csv(
        csv_path, model_name, source_label="training", x_axis=x_axis
    )
    if not ingest_csv_path:
        return payload

    ingest_payload = _load_single_metrics_csv(
        ingest_csv_path, model_name, source_label="ingest", x_axis=x_axis
    )
    if not ingest_payload.get("series"):
        return payload
    if not payload.get("series"):
        ingest_payload["poll_interval_s"] = payload.get("poll_interval_s")
        return ingest_payload

    training_mtime = payload.get("csv_mtime") or 0.0
    ingest_mtime = ingest_payload.get("csv_mtime") or 0.0
    payload["series"] = ingest_payload["series"] + payload["series"]
    payload["row_count"] = int(payload.get("row_count", 0)) + int(
        ingest_payload.get("row_count", 0)
    )
    payload["csv_mtime"] = max(training_mtime, ingest_mtime)
    payload["ingest_row_count"] = int(ingest_payload.get("row_count", 0))
    return payload


def _load_single_metrics_csv(
    csv_path: str,
    model_name: str,
    *,
    source_label: str = "training",
    x_axis: str = "wallclock",
) -> Dict[str, Any]:
    """Parse one metrics CSV into a JSON-serializable chart payload."""
    if not os.path.isfile(csv_path):
        return _empty_metrics_payload(model_name, csv_path, x_axis)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return _empty_metrics_payload(
            model_name,
            csv_path,
            x_axis,
            csv_mtime=os.path.getmtime(csv_path),
            error=str(exc),
        )

    if df.empty:
        return _empty_metrics_payload(
            model_name,
            csv_path,
            x_axis,
            csv_mtime=os.path.getmtime(csv_path),
        )

    dt_s = _timestep_dt_s(csv_path)
    x_vals, x_key, x_label = _select_x_values(df, x_axis, dt_s)
    skip_cols = set(_SKIP_PLOT_COLS)
    skip_cols.add(x_key)
    if _normalize_x_axis(x_axis) in ("simulation", "timesteps"):
        skip_cols.update(_STEP_COLUMNS)
    columns_to_plot = [col for col in df.columns if col not in skip_cols]
    series: List[Dict[str, Any]] = []

    for col in columns_to_plot:
        col_series = df[col]
        is_array_like = col in _ARRAY_LIKE_COLS or _is_array_like_column(col_series)
        if is_array_like:
            xs: List[float] = []
            ys: List[float] = []
            for idx, raw in enumerate(col_series.values):
                arr = _parse_array_like(raw)
                if not arr:
                    continue
                x_base = float(x_vals[idx])
                xs.extend([x_base] * len(arr))
                ys.extend(arr)
            series.append({
                "name": f"{source_label}:{col}" if source_label != "training" else col,
                "type": "scatter",
                "x": xs,
                "y": ys,
            })
        else:
            ys_numeric: List[Optional[float]] = []
            for raw in col_series.values:
                try:
                    val = float(raw)
                    if np.isnan(val):
                        ys_numeric.append(None)
                    else:
                        ys_numeric.append(val)
                except (TypeError, ValueError):
                    ys_numeric.append(None)
            plot_type = "scatter" if col in {"min_laptime", "avg_laptime"} else "line"
            series.append(
                {
                    "name": f"{source_label}:{col}" if source_label != "training" else col,
                    "type": plot_type,
                    "x": [float(x) for x in x_vals],
                    "y": ys_numeric,
                }
            )

    return {
        "model_name": model_name,
        "csv_path": csv_path,
        "csv_mtime": os.path.getmtime(csv_path),
        "row_count": int(len(df)),
        "x_axis": _normalize_x_axis(x_axis),
        "x_key": x_key,
        "x_label": x_label,
        "sim_dt_s": dt_s,
        "series": series,
        "source": source_label,
    }


def _json_response(payload: Dict[str, Any], status: str = "200 OK") -> bytes:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    header = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n"
    )
    return header.encode("utf-8") + body


def _bytes_response(body: bytes, content_type: str, status: str = "200 OK") -> bytes:
    header = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    )
    return header.encode("utf-8") + body


def _text_response(text: str, status: str = "200 OK", content_type: str = "text/plain; charset=utf-8") -> bytes:
    return _bytes_response(text.encode("utf-8"), content_type, status)


class MetricsHttpServer:
    """Serves GET / (dashboard), GET /api/metrics, GET /api/health."""

    def __init__(
        self,
        host: str,
        port: int,
        csv_path: str,
        model_name: str,
        poll_hint_s: float = 2.0,
        ingest_csv_path: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = int(port)
        self.csv_path = csv_path
        self.ingest_csv_path = ingest_csv_path
        self.model_name = model_name
        self.model_dir = model_dir or str(Path(csv_path).resolve().parent)
        self.poll_hint_s = float(poll_hint_s)
        self._server: Optional[asyncio.AbstractServer] = None

    @staticmethod
    def _parse_request_path(request_header: str) -> Tuple[str, str]:
        first_line = request_header.split("\r\n", 1)[0]
        parts = first_line.split()
        if len(parts) < 2:
            return "GET", "/"
        method = parts[0].upper()
        path = parts[1].split("?", 1)[0]
        return method, path

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header_lines: List[bytes] = []
            while True:
                line = await reader.readline()
                if not line:
                    break
                header_lines.append(line)
                if line in (b"\r\n", b"\n"):
                    break

            request_header = b"".join(header_lines).decode("utf-8", errors="replace")
            method, path = self._parse_request_path(request_header)

            if method != "GET":
                writer.write(_text_response("Method not allowed", status="405 Method Not Allowed"))
                await writer.drain()
                return

            if path in ("/api/health", "/health"):
                body = {"ok": True, "model_name": self.model_name}
                writer.write(_json_response(body))
            elif path in ("/api/metrics", "/metrics"):
                first_line = request_header.split("\r\n", 1)[0]
                raw_path = first_line.split()[1] if len(first_line.split()) >= 2 else "/"
                query = parse_qs(urlparse(raw_path).query)
                x_axis = query.get("x", ["wallclock"])[0]
                payload = await asyncio.to_thread(
                    load_metrics_payload,
                    self.csv_path,
                    self.model_name,
                    self.ingest_csv_path,
                    x_axis,
                )
                payload["poll_interval_s"] = self.poll_hint_s
                writer.write(_json_response(payload))
            elif path in ("/api/reward-components", "/reward-components"):
                payload = await asyncio.to_thread(
                    load_reward_components_payload,
                    self.model_dir,
                    self.model_name,
                )
                payload["poll_interval_s"] = self.poll_hint_s
                writer.write(_json_response(payload))
            elif path.startswith("/api/obs-tracking"):
                first_line = request_header.split("\r\n", 1)[0]
                raw_path = first_line.split()[1] if len(first_line.split()) >= 2 else "/"
                query = parse_qs(urlparse(raw_path).query)
                obs_seen_raw = query.get("obs_seen", [None])[0]
                obs_seen = int(obs_seen_raw) if obs_seen_raw not in (None, "") else None
                stats_raw = query.get("stats", ["0"])[0]
                include_stats = str(stats_raw).lower() in ("1", "true", "yes")
                payload = await asyncio.to_thread(
                    load_obs_tracking_payload,
                    self.model_dir,
                    self.model_name,
                    obs_seen=obs_seen,
                    include_stats=include_stats,
                )
                payload["poll_interval_s"] = self.poll_hint_s
                writer.write(_json_response(payload))
            elif path in ("/", "/dashboard", "/index.html"):
                if _DASHBOARD_PATH.is_file():
                    html = _DASHBOARD_PATH.read_text(encoding="utf-8")
                    writer.write(_bytes_response(html.encode("utf-8"), "text/html; charset=utf-8"))
                else:
                    writer.write(_text_response("Dashboard not found", status="404 Not Found"))
            else:
                writer.write(_text_response("Not found", status="404 Not Found"))
            await writer.drain()
        except Exception as exc:
            try:
                writer.write(_json_response({"error": str(exc)}, status="500 Internal Server Error"))
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        addrs = ", ".join(str(sock.getsockname()) for sock in self._server.sockets or [])
        display_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        print(
            f"[server] Metrics dashboard http://{display_host}:{self.port}/ "
            f"(listening on {addrs})"
        )

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


def dashboard_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{display_host}:{port}/"

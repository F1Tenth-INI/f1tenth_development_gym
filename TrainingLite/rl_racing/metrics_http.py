#!/usr/bin/env python3
"""Lightweight asyncio HTTP server for live training metrics (stdlib only)."""

from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

_ARRAY_LIKE_COLS = frozenset(
    {
        "episode_lengths",
        "episode_rewards",
        "episode_mean_step_rewards",
        "stream_batch_sizes",
        "lap_times",
    }
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DASHBOARD_PATH = _STATIC_DIR / "metrics_dashboard.html"


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


def _reward_component_keys() -> List[str]:
    return [
        "progress",
        "crash_reward",
        "wp_distance_penalty",
        "d_action_penality",
        "speed_cap_penalty",
        "proximity_penalty",
        "stuck_reward",
        "spin_reward",
    ]


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

    keys = _reward_component_keys()
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


def load_metrics_payload(
    csv_path: str,
    model_name: str,
    ingest_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse learning_metrics.csv (and optional ingest_metrics.csv) into chart payload."""
    payload = _load_single_metrics_csv(csv_path, model_name, source_label="training")
    if not ingest_csv_path:
        return payload

    ingest_payload = _load_single_metrics_csv(
        ingest_csv_path, model_name, source_label="ingest"
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
) -> Dict[str, Any]:
    """Parse one metrics CSV into a JSON-serializable chart payload."""
    if not os.path.isfile(csv_path):
        return {
            "model_name": model_name,
            "csv_path": csv_path,
            "csv_mtime": None,
            "row_count": 0,
            "x_key": "time",
            "x_label": "time (s)",
            "series": [],
        }

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return {
            "model_name": model_name,
            "csv_path": csv_path,
            "csv_mtime": os.path.getmtime(csv_path),
            "row_count": 0,
            "x_key": "time",
            "x_label": "time (s)",
            "series": [],
            "error": str(exc),
        }

    if df.empty:
        return {
            "model_name": model_name,
            "csv_path": csv_path,
            "csv_mtime": os.path.getmtime(csv_path),
            "row_count": 0,
            "x_key": "time",
            "x_label": "time (s)",
            "series": [],
        }

    if "time" in df.columns:
        x_vals = df["time"].astype(float).tolist()
        x_key = "time"
        x_label = "time (s)"
    elif "total_timesteps" in df.columns:
        x_vals = df["total_timesteps"].astype(float).tolist()
        x_key = "total_timesteps"
        x_label = "total_timesteps"
    else:
        x_vals = list(range(len(df)))
        x_key = "log_index"
        x_label = "log_index"

    columns_to_plot = [col for col in df.columns if col not in _SKIP_PLOT_COLS]
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
        "x_key": x_key,
        "x_label": x_label,
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
                payload = await asyncio.to_thread(
                    load_metrics_payload,
                    self.csv_path,
                    self.model_name,
                    self.ingest_csv_path,
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

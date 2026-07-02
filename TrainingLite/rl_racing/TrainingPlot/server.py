"""HTTP server for the unified TrainingPlot dashboard."""

from __future__ import annotations

import http.server
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from metrics_http import load_metrics_payload, load_reward_components_payload

from TrainingPlot.plotter_utils import (
    EPISODES_CSV,
    LEARNING_METRICS_CSV,
    MODELS_ROOT,
    TRAINING_PLOT_DIR,
    STATIC_FILES,
    get_latest_model_name,
    list_models,
    resolve_model_dir,
)

DEFAULT_POLL_HINT_S = 2.0


class TrainingPlotHandler(http.server.BaseHTTPRequestHandler):
    static_root = TRAINING_PLOT_DIR
    default_model: str | None = None
    custom_model_dir: Path | None = None
    poll_hint_s: float = DEFAULT_POLL_HINT_S

    def log_message(self, format: str, *args) -> None:
        if self.path.startswith("/api/"):
            return
        super().log_message(format, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/models":
            self._handle_models_api()
            return
        if parsed.path == "/api/episodes.csv":
            self._handle_episodes_api(query)
            return
        if parsed.path in ("/api/metrics", "/metrics"):
            self._handle_metrics_api(query)
            return
        if parsed.path in ("/api/reward-components", "/reward-components"):
            self._handle_reward_components_api(query)
            return
        if parsed.path in ("/api/health", "/health"):
            self._handle_health_api(query)
            return
        self._handle_static(parsed.path)

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _resolve_model_name(self, query: dict[str, list[str]]) -> str | None:
        model_name = query.get("model", [None])[0]
        if model_name:
            return model_name
        return self.default_model or get_latest_model_name()

    def _resolve_model_dir(self, model_name: str | None) -> Path | None:
        if self.custom_model_dir is not None and (
            model_name is None or model_name == self.custom_model_dir.name
        ):
            return self.custom_model_dir.resolve()

        if not model_name:
            model_name = self.default_model or get_latest_model_name()
        if not model_name:
            return None

        try:
            return resolve_model_dir(model_name)
        except FileNotFoundError:
            return None

    def _handle_models_api(self) -> None:
        models = list_models()
        if self.custom_model_dir is not None:
            custom_name = self.custom_model_dir.name
            if not any(model["name"] == custom_name for model in models):
                episodes_path = self.custom_model_dir / EPISODES_CSV
                metrics_path = self.custom_model_dir / LEARNING_METRICS_CSV
                models.insert(
                    0,
                    {
                        "name": custom_name,
                        "mtime": max(
                            episodes_path.stat().st_mtime if episodes_path.is_file() else 0.0,
                            metrics_path.stat().st_mtime if metrics_path.is_file() else 0.0,
                        ),
                        "has_episodes": episodes_path.is_file(),
                        "has_metrics": metrics_path.is_file(),
                    },
                )
        default_model = self.default_model or get_latest_model_name()
        self._send_json({"models": models, "default": default_model})

    def _handle_episodes_api(self, query: dict[str, list[str]]) -> None:
        model_name = self._resolve_model_name(query)
        model_dir = self._resolve_model_dir(model_name)
        if model_dir is None:
            self.send_error(404, "model not found")
            return

        episodes_path = model_dir / EPISODES_CSV
        if not episodes_path.is_file():
            self.send_error(404, "episodes.csv not found")
            return

        body = episodes_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Last-Modified",
            http.server.email.utils.formatdate(episodes_path.stat().st_mtime, usegmt=True),
        )
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_metrics_api(self, query: dict[str, list[str]]) -> None:
        model_name = self._resolve_model_name(query)
        model_dir = self._resolve_model_dir(model_name)
        if model_dir is None or not model_name:
            self._send_json(
                {
                    "model_name": model_name,
                    "row_count": 0,
                    "x_key": "time",
                    "x_label": "time (s)",
                    "series": [],
                    "poll_interval_s": self.poll_hint_s,
                    "error": "model not found",
                },
                status=404,
            )
            return

        csv_path = str(model_dir / LEARNING_METRICS_CSV)
        ingest_csv_path = str(model_dir / "ingest_metrics.csv")
        payload = load_metrics_payload(csv_path, model_name, ingest_csv_path)
        payload["poll_interval_s"] = self.poll_hint_s
        self._send_json(payload)

    def _handle_reward_components_api(self, query: dict[str, list[str]]) -> None:
        model_name = self._resolve_model_name(query)
        model_dir = self._resolve_model_dir(model_name)
        if model_dir is None or not model_name:
            self._send_json(
                {
                    "model_name": model_name,
                    "checkpoints": [],
                    "checkpoint_count": 0,
                    "poll_interval_s": self.poll_hint_s,
                    "error": "model not found",
                },
                status=404,
            )
            return

        payload = load_reward_components_payload(str(model_dir), model_name)
        payload["poll_interval_s"] = self.poll_hint_s
        self._send_json(payload)

    def _handle_health_api(self, query: dict[str, list[str]]) -> None:
        model_name = self._resolve_model_name(query)
        self._send_json({"ok": True, "model_name": model_name})

    def _handle_static(self, url_path: str) -> None:
        relative = url_path.lstrip("/")
        if relative in ("", "index.html", "dashboard"):
            relative = "index.html"
        elif relative not in STATIC_FILES:
            self.send_error(404, "File not found")
            return

        file_path = (self.static_root / relative).resolve()
        if not str(file_path).startswith(str(self.static_root.resolve())):
            self.send_error(403, "Forbidden")
            return
        if not file_path.is_file():
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_handler_class(
    *,
    default_model: str | None,
    custom_model_dir: Path | None,
    poll_hint_s: float,
) -> type[TrainingPlotHandler]:
    return type(
        "ConfiguredTrainingPlotHandler",
        (TrainingPlotHandler,),
        {
            "default_model": default_model,
            "custom_model_dir": custom_model_dir,
            "poll_hint_s": poll_hint_s,
        },
    )

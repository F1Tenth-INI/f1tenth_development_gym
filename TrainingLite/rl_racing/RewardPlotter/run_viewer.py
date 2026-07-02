#!/usr/bin/env python3
"""Serve the reward-components viewer for SAC training model directories."""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import socket
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from RewardPlotter.plotter_utils import (  # noqa: E402
    EPISODES_CSV,
    MODELS_ROOT,
    REWARD_PLOTTER_DIR,
    STATIC_FILES,
    get_latest_model_name,
    install_reward_plotter,
    list_models,
    resolve_model_dir,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8070


def _pick_free_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found near {preferred}")


def _resolve_initial_model(
    model_name: str | None,
    model_dir: Path | None,
) -> tuple[str | None, Path | None]:
    if model_dir is not None:
        resolved = model_dir.resolve()
        if not resolved.is_dir():
            raise SystemExit(f"Model directory not found: {resolved}")
        return resolved.name, resolved

    if model_name:
        resolved = resolve_model_dir(model_name)
        return model_name, resolved

    latest = get_latest_model_name()
    if latest is None:
        raise SystemExit(f"No model directories found under {MODELS_ROOT}")
    return latest, resolve_model_dir(latest)


class RewardPlotterHandler(http.server.BaseHTTPRequestHandler):
    static_root = REWARD_PLOTTER_DIR
    default_model: str | None = None
    custom_model_dir: Path | None = None

    def log_message(self, format: str, *args) -> None:
        if self.path.startswith("/api/"):
            return
        super().log_message(format, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            self._handle_models_api()
            return
        if parsed.path == "/api/episodes.csv":
            self._handle_episodes_api(parse_qs(parsed.query))
            return
        self._handle_static(parsed.path)

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_models_api(self) -> None:
        models = list_models()
        if self.custom_model_dir is not None:
            custom_name = self.custom_model_dir.name
            if not any(model["name"] == custom_name for model in models):
                episodes_path = self.custom_model_dir / EPISODES_CSV
                models.insert(
                    0,
                    {
                        "name": custom_name,
                        "mtime": episodes_path.stat().st_mtime if episodes_path.is_file() else 0.0,
                        "has_episodes": episodes_path.is_file(),
                    },
                )
        default_model = self.default_model or get_latest_model_name()
        self._send_json({"models": models, "default": default_model})

    def _resolve_episodes_path(self, model_name: str | None) -> Path | None:
        if self.custom_model_dir is not None and (
            model_name is None or model_name == self.custom_model_dir.name
        ):
            return self.custom_model_dir / EPISODES_CSV

        if not model_name:
            model_name = self.default_model or get_latest_model_name()
        if not model_name:
            return None

        try:
            return resolve_model_dir(model_name) / EPISODES_CSV
        except FileNotFoundError:
            return None

    def _handle_episodes_api(self, query: dict[str, list[str]]) -> None:
        model_name = query.get("model", [None])[0]
        episodes_path = self._resolve_episodes_path(model_name)
        if episodes_path is None or not episodes_path.is_file():
            self.send_error(404, "episodes.csv not found")
            return

        body = episodes_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Last-Modified", http.server.email.utils.formatdate(episodes_path.stat().st_mtime, usegmt=True))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_static(self, url_path: str) -> None:
        relative = url_path.lstrip("/")
        if relative in ("", "index.html"):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the reward-components viewer for training models.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help=f"Optional initial model name under {MODELS_ROOT} (default: latest)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Direct path to a model directory (must contain or receive episodes.csv)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Copy viewer files into the model directory and exit",
    )
    args = parser.parse_args()

    default_model, model_dir = _resolve_initial_model(args.model, args.model_dir)

    if args.install_only:
        if model_dir is None:
            raise SystemExit("No model directory available for --install-only")
        install_reward_plotter(model_dir)
        print(f"Installed reward plotter into {model_dir}")
        return

    port = _pick_free_port(args.host, args.port)
    url = f"http://{args.host}:{port}/"

    handler_cls = type(
        "ConfiguredRewardPlotterHandler",
        (RewardPlotterHandler,),
        {
            "default_model": default_model,
            "custom_model_dir": model_dir if args.model_dir is not None else None,
        },
    )
    server = http.server.ThreadingHTTPServer((args.host, port), handler_cls)
    server.daemon_threads = True

    try:
        if not args.no_browser:
            webbrowser.open(url, new=0)
        print(f"Reward plotter serving models from {MODELS_ROOT} at {url}")
        if default_model:
            print(f"Initial model: {default_model}")
        print("Press Ctrl+C to stop.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

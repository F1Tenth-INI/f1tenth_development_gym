#!/usr/bin/env python3
"""Serve the unified TrainingPlot dashboard for SAC training models."""

from __future__ import annotations

import argparse
import http.server
import socket
import sys
import webbrowser
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from TrainingPlot.plotter_utils import (  # noqa: E402
    MODELS_ROOT,
    get_latest_model_name,
    install_training_plot,
    resolve_model_dir,
)
from TrainingPlot.server import make_handler_class  # noqa: E402

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve TrainingPlot (training metrics + reward plotter).",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help=f"Optional initial model name under {MODELS_ROOT} (default: latest)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Direct path to a model directory",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Copy static files into the model directory and exit",
    )
    args = parser.parse_args()

    default_model, model_dir = _resolve_initial_model(args.model, args.model_dir)

    if args.install_only:
        if model_dir is None:
            raise SystemExit("No model directory available for --install-only")
        install_training_plot(model_dir)
        print(f"Installed TrainingPlot static files into {model_dir}")
        return

    port = _pick_free_port(args.host, args.port)
    url = f"http://{args.host}:{port}/"

    handler_cls = make_handler_class(
        default_model=default_model,
        custom_model_dir=model_dir if args.model_dir is not None else None,
        poll_hint_s=float(args.poll_interval_s),
    )
    server = http.server.ThreadingHTTPServer((args.host, port), handler_cls)
    server.daemon_threads = True

    try:
        if not args.no_browser:
            webbrowser.open(url, new=0)
        print(f"TrainingPlot serving models from {MODELS_ROOT} at {url}")
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

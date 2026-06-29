#!/usr/bin/env python3
"""View a recorded experiment CSV in the web renderer (scrub/play on the track map)."""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    from utilities.Settings import Settings

    parser = argparse.ArgumentParser(
        description="Replay an experiment recording CSV in the browser web renderer.",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Recording CSV path or filename (default: Settings.RECORDING_PATH; "
        "bare filenames are looked up in Settings.RECORDING_FOLDER)",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Override map name (default: read from CSV header, then Settings.MAP_NAME)",
    )
    parser.add_argument(
        "--host",
        default=str(getattr(Settings, "WEB_RENDER_HOST", "127.0.0.1")),
        help="Web renderer host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(getattr(Settings, "WEB_RENDER_PORT", 8765)),
        help="Web renderer port",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Default browser replay speed multiplier",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from utilities.Settings import Settings
    from utilities.recording_replay import (
        load_recording_laptimes,
        resolve_map_for_recording,
        resolve_recording_csv_path,
    )
    from sim.f110_sim.envs.rendering.WebRenderer.web_renderer import WebEnvRenderer

    csv_path = resolve_recording_csv_path(args.csv)
    if not os.path.isfile(csv_path):
        print(f"Recording not found: {csv_path}", file=sys.stderr)
        return 1

    map_render_path, map_name = resolve_map_for_recording(csv_path, map_override=args.map)

    lap_times = load_recording_laptimes(csv_path)
    if lap_times:
        print(f"Lap times: {lap_times}")

    renderer = WebEnvRenderer(
        host=args.host,
        port=args.port,
        auto_open_browser=not args.no_browser,
        recording_csv_path=csv_path,
        recording_playback_speed=args.speed,
    )
    if not Settings.BLANK_MAP:
        renderer.update_map(map_render_path, ".png")

    url = f"http://{args.host}:{args.port}"
    if renderer._replay_archive_preloaded:
        print(
            f"Replay viewer at {url} — map={map_name}, "
            f"{len(renderer._state_history)} frames, "
            f"{renderer._replay_total_time:.2f}s. "
            "Use the browser slider to scrub/play. Press Ctrl+C to exit."
        )
    else:
        print(
            f"Replay viewer at {url} — archive preload failed or CSV missing pose columns. "
            "Press Ctrl+C to exit.",
            file=sys.stderr,
        )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Closing replay viewer.")
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

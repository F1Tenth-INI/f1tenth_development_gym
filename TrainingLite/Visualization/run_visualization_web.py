#!/usr/bin/env python3
"""Launch the state comparison visualization webapp."""

import os
import sys

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(VIS_DIR, "web")
sys.path.insert(0, VIS_DIR)
sys.path.insert(0, WEB_DIR)

from browser_session import is_browser_session_active

HOST = "127.0.0.1"
PORT = 8050
APP_URL = f"http://{HOST}:{PORT}"


def _should_open_browser(argv: list) -> bool:
    if "--no-browser" in argv:
        return False
    if "--open-browser" in argv:
        return True
    return not is_browser_session_active()


if __name__ == "__main__":
    import uvicorn

    if _should_open_browser(sys.argv):
        os.environ["VIZ_OPEN_BROWSER"] = APP_URL
    print(f"State Comparison Visualizer running at {APP_URL}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)

#!/usr/bin/env python3
"""Track whether a visualization browser tab is already active."""

import os
import time

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
HEARTBEAT_FILE = os.path.join(VIS_DIR, ".viz_browser_heartbeat")
HEARTBEAT_TTL_SEC = 45


def touch_browser_heartbeat() -> None:
    with open(HEARTBEAT_FILE, "w") as f:
        f.write(f"{time.time():.3f}")


def is_browser_session_active() -> bool:
    if not os.path.exists(HEARTBEAT_FILE):
        return False
    try:
        with open(HEARTBEAT_FILE) as f:
            last_seen = float(f.read().strip())
        return (time.time() - last_seen) < HEARTBEAT_TTL_SEC
    except (OSError, ValueError):
        return False

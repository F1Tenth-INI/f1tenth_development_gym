#!/usr/bin/env python3
"""Launch the HWM learner and the simulation client on the track midline.

Same server, client and Settings as ``run_training.py``. The waypoint table is
loaded from ``centerline.csv`` in the map folder instead of the minimum-curvature
raceline. Initial speeds follow ``HWM_INITIAL_SPEED_GRIP_SHARE`` or
``HWM_INITIAL_SPEED_LIMIT`` when those are set, and otherwise
``MIDLINE_DEFAULT_SPEED``.

The client is started automatically. Pass ``--no-auto-start-client`` to run the
server only.

Examples:
    python TrainingLite/hwm/learner/run_midline.py --model-name hwm_midline
    python TrainingLite/hwm/learner/run_midline.py --model-name hwm_midline --MAP_NAME RCA2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_argv(argv: Optional[list[str]] = None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    start_client = "--no-auto-start-client" not in raw
    cleaned: list[str] = []
    skip_next = False
    for arg in raw:
        if skip_next:
            skip_next = False
            continue
        if arg == "--no-auto-start-client":
            continue
        if arg.split("=", 1)[0] == "--WAYPOINT_SOURCE":
            if "=" not in arg:
                skip_next = True
            continue
        cleaned.append(arg)
    if start_client and "--auto-start-client" not in cleaned:
        cleaned.insert(0, "--auto-start-client")
    cleaned.extend(["--WAYPOINT_SOURCE", "midline"])
    return cleaned


def main(argv: Optional[list[str]] = None) -> None:
    from TrainingLite.hwm.learner.run_training import main as run_training_main

    run_training_main(build_argv(argv))


if __name__ == "__main__":
    main()

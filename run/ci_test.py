"""CI smoke test runner using explicit `run.py --SETTING value` commands."""

import re
import subprocess
import sys
from typing import List


def format_command(args: List[str]) -> str:
    """Return a shell-like command string for readable logs."""
    return " ".join(["python", "run.py", *args])


def run_case(name: str, args: List[str]) -> None:
    cmd = [sys.executable, "run.py", *args]
    print(f"\n=== Running case: {name} ===")
    print(format_command(args))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines: List[str] = []
    assert process.stdout is not None, f"Case '{name}' has no stdout stream."
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    completed = process.wait()
    output = "".join(output_lines)

    # Basic process-level failure check.
    assert completed == 0, (
        f"Case '{name}' failed with return code {completed}."
    )

    # Crash signals that can appear in stdout even when run returns 0.
    lowered = output.lower()
    assert not (
        ("controller " in lowered and "crashed the car." in lowered)
        or "collision detected" in lowered
        or "carcrashexception" in lowered
        or "traceback" in lowered
    ), f"Case '{name}' shows crash/error markers in output."

    # Ensure at least one lap was completed.
    lap_hits = re.findall(r"Lap time:\s*([0-9]+(?:\.[0-9]+)?)", output)
    assert len(lap_hits) > 0, f"Case '{name}' completed with no lap times recorded."


if __name__ == "__main__":
    base = [
        "--MAP_NAME", "RCA2",
        "--RENDER_MODE", "None",
        "--START_FROM_RANDOM_POSITION", "False",
        "--SIMULATION_LENGTH", "2000",
        "--SURFACE_FRICTION", "0.75",
        "--NOISE_LEVEL_CONTROL", "[0.0, 0.0]",
    ]

    cases = [
        # (
        #     "pp",
        #     [*base, "--CONTROLLER", "pp", "--GLOBAL_WAYPOINT_VEL_FACTOR", "0.5", "--CONTROL_DELAY", "0.0"],
        # ),
        # (
        #     "mpc",
        #     [*base, "--CONTROLLER", "mpc", "--GLOBAL_WAYPOINT_VEL_FACTOR", "1.0", "--CONTROL_DELAY", "0.08"],
        # ),
        # (
        #     "mppi-lite-jax",
        #     [*base, "--CONTROLLER", "mppi-lite-jax", "--GLOBAL_WAYPOINT_VEL_FACTOR", "1.0", "--CONTROL_DELAY", "0.08"],
        # ),
        # (
        #     "rpgd-lite-jax",
        #     [*base, "--CONTROLLER", "rpgd-lite-jax", "--GLOBAL_WAYPOINT_VEL_FACTOR", "1.0", "--CONTROL_DELAY", "0.08"],
        # ),
        (
            "SAC agent",
            [*base, "--MAP_NAME", "RCA1",  "--CONTROLLER", "sac_agent", "--GLOBAL_WAYPOINT_VEL_FACTOR", "1.0", "--CONTROL_DELAY", "0.08", "--SAC_INFERENCE_MODEL_NAME", "Example-1b", "--SURFACE_FRICTION", "0.9"],
        ),
        
    ]

    for name, args in cases:
        run_case(name, args)

    print("\nAll CI test cases passed: no crashes/errors detected and lap(s) completed.")


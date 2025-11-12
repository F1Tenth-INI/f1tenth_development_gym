#!/usr/bin/env python3
"""
Launcher that starts both the learner server (via run_training.py) and run.py.

Server-related arguments are parsed locally and forwarded to run_training.py,
while any remaining arguments are passed through to run.py to override Settings.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from TrainingLite.rl_racing.sac_utilities import SacUtilities  # type: ignore
from utilities.parser_utilities import parse_settings_args, save_settings_snapshot


def parse_arguments() -> tuple[argparse.Namespace, List[str]]:
    """Parse server arguments; return remaining args for run.py."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-name", default="SAC_RCA1_0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--train-every-seconds", type=float, default=0.2)
    parser.add_argument("--gradient-steps", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--control_penalty_factor", type=float, default=0.1)
    parser.add_argument("--d_control_penalty_factor", type=float, default=0.1)

    return parser.parse_known_args()


def save_settings_to_model_folder(model_name: str):
    """Save Settings snapshot in the model directory."""
    _, model_dir = SacUtilities.resolve_model_paths(model_name)
    os.makedirs(model_dir, exist_ok=True)
    snapshot_path = os.path.join(model_dir, "settings_snapshot.yml")
    save_settings_snapshot(output_path=snapshot_path, format="yaml")
    print(f"[start_training] Settings snapshot saved to: {snapshot_path}")


def build_server_command(server_args: argparse.Namespace) -> List[str]:
    script = PROJECT_ROOT / "TrainingLite" / "rl_racing" / "run_training.py"
    cmd = [sys.executable, str(script)]
    for key, value in vars(server_args).items():
        if value is not None:
            cmd.append(f"--{key}")
            if not isinstance(value, bool):
                cmd.append(str(value))
    return cmd


def build_run_command(run_args: List[str]) -> List[str]:
    script = PROJECT_ROOT / "run.py"
    cmd = [sys.executable, str(script)]
    cmd.extend(run_args)
    return cmd


def main():
    server_args, run_args = parse_arguments()

    print(f"[start_training] Starting training with model: {server_args.model_name}")
    print(f"[start_training] Server arguments: {len(vars(server_args))} parameters")
    if run_args:
        print(f"[start_training] Forwarding {len(run_args)} argument(s) to run.py: {' '.join(run_args)}")

    if run_args:
        original_argv = sys.argv
        try:
            sys.argv = ["run.py"] + run_args
            parse_settings_args(description="Applying run.py overrides", save_snapshot=False, verbose=False)
        finally:
            sys.argv = original_argv

    save_settings_to_model_folder(server_args.model_name)

    server_cmd = build_server_command(server_args)
    run_cmd = build_run_command(run_args)

    print("[start_training] Starting learner server...")
    print(f"[start_training] Command: {' '.join(server_cmd)}")
    print("[start_training] Starting run.py...")
    print(f"[start_training] Command: {' '.join(run_cmd)}")

    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        server_proc = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT))
        processes.append(("learner_server", server_proc))
        print(f"[start_training] Learner server started (PID: {server_proc.pid})")

        time.sleep(2.0)

        run_proc = subprocess.Popen(run_cmd, cwd=str(PROJECT_ROOT))
        processes.append(("run.py", run_proc))
        print(f"[start_training] run.py started (PID: {run_proc.pid})")

        print("[start_training] Both processes started. Press Ctrl+C to stop.")

        while processes:
            for name, proc in list(processes):
                if proc.poll() is not None:
                    print(f"[start_training] {name} exited with code {proc.returncode}")
                    processes.remove((name, proc))
            if not processes:
                break
            time.sleep(0.5)

        print("[start_training] All processes finished.")

    except KeyboardInterrupt:
        print("\n[start_training] KeyboardInterrupt received, terminating processes...")
        for name, proc in processes:
            try:
                print(f"[start_training] Terminating {name} (PID: {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"[start_training] {name} did not terminate, killing...")
                    proc.kill()
                    proc.wait()
                print(f"[start_training] {name} terminated")
            except Exception as exc:
                print(f"[start_training] Error terminating {name}: {exc}")
    except Exception as exc:
        print(f"[start_training] Error: {exc}")
        for _name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        raise


if __name__ == "__main__":
    main()

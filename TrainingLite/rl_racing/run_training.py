#!/usr/bin/env python3
"""
Command-line entry point for launching the learner server training loop.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if PROJECT_ROOT.exists():
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from learner_server import LearnerServer  # noqa: E402
from utilities.parser_utilities import parse_settings_args  # noqa: E402


def parse_args(argv: list[str] | None = None) -> Tuple[argparse.Namespace, list[str]]:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Learner server: collect episodes, train SAC, broadcast weights.",
    )

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-name", default="SAC_RCA1_0")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--train-every-seconds", type=float, default=0.2)
    parser.add_argument("--gradient-steps", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument(
        "--auto-start-client",
        default=True,
        action="store_true",
        help="Automatically start the client/simulation process",
    )

    parser.add_argument(
        "--forward-client-output",
        action="store_true",
        default=True,
        help=(
            "Forward client output to terminal "
            "(output forwarding is enabled by default)"
        ),
    )
   
    known_args, remaining = parser.parse_known_args(argv)
    return known_args, remaining


def parse_settings_overrides(settings_args: list[str]) -> argparse.Namespace:
    original_argv = sys.argv.copy()
    try:
        sys.argv = [original_argv[0], *settings_args]
        return parse_settings_args(
            description="Learner server: override simulation Settings for training run.",
        )
    finally:
        sys.argv = original_argv


def resolve_run_script_path(run_script_path: str) -> Optional[Path]:
    candidate = PROJECT_ROOT / run_script_path
    if candidate.exists():
        return candidate
    print(f"[run_training] Warning: Could not find client script at {candidate}")
    return None


def start_client_process(
    run_script_path: str,
    forward_output: bool,
    extra_args: Optional[list[str]] = None,
) -> Optional[subprocess.Popen]:
    script_path = resolve_run_script_path(run_script_path)
    if script_path is None:
        return None

    if extra_args is None:
        extra_args = []

    cmd = [sys.executable, str(script_path), *extra_args]
    cwd = str(script_path.parent)

    try:
        if forward_output:
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                cwd=cwd,
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                bufsize=1,
            )
        print(f"[run_training] Started client process (PID: {process.pid})")
        return process
    except Exception as exc:
        print(f"[run_training] Failed to start client process: {exc}")
        return None


async def forward_client_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return

    try:
        while process.poll() is None:
            line = await asyncio.to_thread(process.stdout.readline)
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                print(f"[client] {decoded}")
    except Exception as exc:
        if process.poll() is None:
            print(f"[run_training] Error reading client output: {exc}")


async def _run_with_optional_client(server: LearnerServer, args: argparse.Namespace) -> None:
    forward_task: asyncio.Task | None = None
    client_process: Optional[subprocess.Popen] = None

    try:
        if args.auto_start_client:
            await asyncio.sleep(1.0)
            client_process = start_client_process(
                "run.py",
                forward_output=args.forward_client_output,
                extra_args=getattr(args, "forwarded_settings_args", []),
            )
            if (
                not args.forward_client_output
                and client_process is not None
                and client_process.stdout is not None
            ):
                forward_task = asyncio.create_task(
                    forward_client_output(client_process)
                )

        await server.run()
    finally:
        if forward_task is not None and not forward_task.done():
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass
        if client_process is not None and client_process.poll() is None:
            try:
                client_process.terminate()
                try:
                    client_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[run_training] Client process did not terminate, killing...")
                    client_process.kill()
                    client_process.wait()
            except Exception as exc:
                print(f"[run_training] Error terminating client process: {exc}")


def main() -> None:
    run_args, settings_args = parse_args()
    settings_namespace = parse_settings_overrides(settings_args)

    setattr(run_args, "forwarded_settings_args", settings_args)
    setattr(run_args, "settings_namespace", settings_namespace)

    server = LearnerServer(
        host=run_args.host,
        port=run_args.port,
        model_name=run_args.model_name,
        device=run_args.device,
        train_every_seconds=run_args.train_every_seconds,
        grad_steps=run_args.gradient_steps,
        replay_capacity=run_args.replay_capacity,
        learning_starts=run_args.learning_starts,
        batch_size=run_args.batch_size,
        learning_rate=run_args.learning_rate,
        discount_factor=run_args.discount_factor,
        train_frequency=run_args.train_frequency,
    )

    try:
        asyncio.run(_run_with_optional_client(server, run_args))
    except KeyboardInterrupt:
        print("\n[server] KeyboardInterrupt received in run_training, exiting...")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Launch the HWM learner server, optionally with the simulation client.

Examples:
    python TrainingLite/hwm/learner/run_training.py --model-name hwm_v0 --auto-start-client --MAP_NAME RCA1
    python run.py --CONTROLLER HirarchicalPlanner                          # client only (server already running)
    python run.py --CONTROLLER HirarchicalPlanner --HWM_INFERENCE_MODEL_NAME hwm_v0   # inference, no server

``--model-name X``: if models/X exists it is loaded and saved back (finetune),
otherwise training starts from scratch. Unknown arguments are forwarded as
Settings overrides to both the server and the auto-started client.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from TrainingLite.hwm.paths import model_exists  # noqa: E402
from TrainingLite.hwm.learner.server import HWMLearnerServer  # noqa: E402
from utilities.parser_utilities import parse_settings_args  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="HWM learner server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-name", default=None, help="Save name; also load name if it already exists.")
    parser.add_argument("--load-model-name", default=None)
    parser.add_argument("--save-model-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--train-every-seconds", type=float, default=0.0)
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--auto-start-client", action="store_true", default=False)
    return parser.parse_known_args(argv if argv is not None else sys.argv[1:])


def apply_settings_overrides(settings_args: list[str]) -> None:
    original = sys.argv.copy()
    try:
        sys.argv = [original[0], *settings_args]
        parse_settings_args(description="HWM learner: Settings overrides", save_snapshot=False)
    finally:
        sys.argv = original


def _child_death_signal() -> None:
    if os.name != "posix":
        return
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def start_client(settings_args: list[str], port: int) -> Optional[subprocess.Popen]:
    script = PROJECT_ROOT / "run.py"
    if not script.exists():
        print(f"[run_training] run.py not found at {script}")
        return None
    args = list(settings_args)
    if not any(a.split("=", 1)[0] == "--LEARNER_TCP_PORT" for a in args):
        args += ["--LEARNER_TCP_PORT", str(port)]
    if not any(a.split("=", 1)[0] == "--CONTROLLER" for a in args):
        args += ["--CONTROLLER", "HirarchicalPlanner"]
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), preexec_fn=_child_death_signal if os.name == "posix" else None)
    print(f"[run_training] Started client (PID {proc.pid}): {' '.join(cmd)}")
    return proc


async def _run(server: HWMLearnerServer, args: argparse.Namespace, settings_args: list[str]) -> None:
    client: Optional[subprocess.Popen] = None
    try:
        if args.auto_start_client:
            await asyncio.sleep(1.0)
            client = start_client(settings_args, args.port)
        await server.run()
    finally:
        if client is not None and client.poll() is None:
            client.terminate()
            try:
                client.wait(timeout=5)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait()


def main() -> None:
    args, settings_args = parse_args()
    apply_settings_overrides(settings_args)

    save_name = args.save_model_name or args.model_name
    load_name = args.load_model_name
    if load_name is None and args.model_name is not None and model_exists(args.model_name):
        load_name = args.model_name
        print(f"[run_training] '{args.model_name}' exists -> loading and saving back to it")
    if save_name is None:
        print("[run_training] Error: pass --model-name NAME or --save-model-name NAME", file=sys.stderr)
        sys.exit(2)

    server = HWMLearnerServer(
        host=args.host,
        port=args.port,
        save_model_name=save_name,
        load_model_name=load_name,
        device=args.device,
        train_every_seconds=args.train_every_seconds,
        grad_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
    )
    try:
        asyncio.run(_run(server, args, settings_args))
    except KeyboardInterrupt:
        print("\n[run_training] Interrupted")


if __name__ == "__main__":
    main()

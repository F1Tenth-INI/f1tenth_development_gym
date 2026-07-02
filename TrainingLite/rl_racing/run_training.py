#!/usr/bin/env python3
"""
Command-line entry point for launching the learner server training loop.
example usage:

python TrainingLite/rl_racing/run_training.py --model-name ServerClientTest1 --RENDER_MODE human_fast --MAP_NAME RCA2

"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch

RUN_EVALUATION = False

def _child_death_signal() -> None:
    """On Linux, make the child receive SIGKILL when the parent dies.
    Prevents accumulation of orphaned run.py processes when run_training
    is killed (Ctrl+C, terminal closed, etc.). Uses prctl via ctypes."""
    if os.name != "posix":
        return
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            pass  # Ignore failure (e.g. in containers)
    except (OSError, AttributeError, TypeError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if PROJECT_ROOT.exists():
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from learner_server import LearnerServer  # noqa: E402
from utilities.Settings import Settings  # noqa: E402
from utilities.parser_utilities import parse_settings_args  # noqa: E402

_TQDM_PROGRESS_RE = re.compile(
    r"(?P<pct>\d+%)?\s*\|[^\n|]*\|\s*(?P<cur>\d+)\s*/\s*(?P<tot>\d+)\s*\[(?P<timing>[^\]]+)\]"
)

MODELS_ROOT = PROJECT_ROOT / "TrainingLite" / "rl_racing" / "models"


def _model_dir_for_name(model_name: str) -> Path:
    return MODELS_ROOT / model_name


def model_exists(model_name: str) -> bool:
    """
    Return True if a trained model with this name is already present on disk.
    Checks both the local layout (models/{name}/{name}.zip) and the server
    layout (models/{name}/server/{name}.zip).
    """
    if not model_name:
        return False
    model_dir = _model_dir_for_name(model_name)
    if not model_dir.is_dir():
        return False
    candidates = (
        model_dir / f"{model_name}.zip",
        model_dir / "server" / f"{model_name}.zip",
    )
    return any(candidate.exists() for candidate in candidates)


def backup_existing_model_if_present(save_model_name: str) -> Optional[Path]:
    """
    If models/{save_model_name} already exists with content, copy it to
    models/{save_model_name}_backup_{timestamp}, then remove the original
    directory so training starts from scratch (no weights/metrics/replay resume).
    """
    model_dir = _model_dir_for_name(save_model_name)
    if not model_dir.is_dir():
        return None
    try:
        has_content = any(model_dir.iterdir())
    except OSError as exc:
        print(f"[run_training] Warning: could not inspect model dir {model_dir}: {exc}")
        return None
    if not has_content:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = MODELS_ROOT / f"{save_model_name}_backup_{timestamp}"
    suffix = 1
    while backup_dir.exists():
        backup_dir = MODELS_ROOT / f"{save_model_name}_backup_{timestamp}_{suffix}"
        suffix += 1

    try:
        shutil.copytree(model_dir, backup_dir)
    except Exception as exc:
        print(f"[run_training] Failed to back up {model_dir} -> {backup_dir}: {exc}")
        raise

    try:
        shutil.rmtree(model_dir)
    except Exception as exc:
        print(f"[run_training] Failed to remove model dir after backup {model_dir}: {exc}")
        raise

    print(
        f"[run_training] Backed up existing model: {model_dir} -> {backup_dir}; "
        f"removed {model_dir} for a fresh training run"
    )
    return backup_dir


class CombinedTrainingStatus:
    """One terminal line: tqdm sim progress + learner train stats."""

    def __init__(self) -> None:
        self._sim = ""
        self._train = ""
        self._last_len = 0

    def set_train(self, msg: str) -> None:
        self._train = str(msg).strip()
        self._redraw()

    def handle_client_text(self, text: str) -> None:
        """Handle one stdout segment (tqdm uses \\r without newlines when piped)."""
        stripped = text.strip()
        if not stripped or stripped.startswith("[server]"):
            return
        parsed = _parse_tqdm_progress(stripped)
        if parsed is not None:
            self._sim = parsed
            self._redraw()
            return
        self._println(stripped)

    def _redraw(self) -> None:
        parts = [p for p in (self._sim, self._train) if p]
        if not parts:
            return
        line = " | ".join(parts)
        pad = max(0, self._last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def _println(self, msg: str) -> None:
        if self._last_len > 0:
            sys.stdout.write("\r" + " " * self._last_len + "\r")
            sys.stdout.flush()
            self._last_len = 0
        print(msg, flush=True)

    def finish(self) -> None:
        if self._last_len > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_len = 0


def _parse_tqdm_progress(line: str) -> Optional[str]:
    """Extract compact progress from a tqdm trange line."""
    m = _TQDM_PROGRESS_RE.search(line)
    if m is None:
        return None
    pct = m.group("pct") or ""
    cur, tot, timing = m.group("cur"), m.group("tot"), m.group("timing")
    parts = [p.strip() for p in timing.split(",") if p.strip()]
    timing_short = ", ".join(parts[:2]) if parts else timing.strip()
    prefix = f"{pct} " if pct else ""
    return f"{prefix}{cur}/{tot} [{timing_short}]"


def _parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true/false."
    )


def parse_args(argv: list[str] | None = None) -> Tuple[argparse.Namespace, list[str]]:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Learner server: collect episodes, train SAC, broadcast weights.",
    )

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    # Convenience model name: sets BOTH load (if it already exists) and save
    # names, so a single --model-name can train, retrain and finetune.
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Convenience model name. Used as the save name when --save-model-name "
            "is not given, and also loaded as the training base when a model with "
            "this name already exists and --load-model-name is not given. Lets you "
            "train, retrain and finetune with a single argument. No default: pass "
            "this or --save-model-name explicitly."
        ),
    )
    parser.add_argument(
        "--load-model-name",
        default=None,
        help=("Model name to load as a base for training. If omitted (None), falls back to --model-name when that model already exists; otherwise training starts from scratch."),
    )
    parser.add_argument(
        "--save-model-name",
        default=None,
        help=("Model name to save training progress to. If omitted, falls back to --model-name."),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--train-every-seconds", type=float, default=0.0)
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Legacy alias for SAC training minibatch size. Prefer --SAC_BATCH_SIZE "
            "(Settings); when set, overrides --SAC_BATCH_SIZE for this run."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument(
        "--save_replay_buffer",
        nargs="?",
        const=True,
        type=_parse_bool_arg,
        default=False,
        help=(
            "Persist replay buffer transitions to replay_buffer.csv on saves/checkpoints. "
            "Supports '--save_replay_buffer' or '--save_replay_buffer true/false'."
        ),
    )
    parser.add_argument(
        "--load-replay-buffer",
        "--load_replay_buffer",
        dest="load_replay_buffer",
        nargs="?",
        const=True,
        type=_parse_bool_arg,
        default=False,
        help=(
            "Load replay buffer transitions from replay_buffer.csv if present. "
            "Supports '--load-replay-buffer' or '--load-replay-buffer true/false'."
        ),
    )
    parser.add_argument(
        "--auto-start-client",
        default=False,
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
    *,
    attach_tty: bool,
    extra_args: Optional[list[str]] = None,
) -> Optional[subprocess.Popen]:
    script_path = resolve_run_script_path(run_script_path)
    if script_path is None:
        return None

    if extra_args is None:
        extra_args = []

    cmd = [sys.executable, str(script_path), *extra_args]
    cwd = str(script_path.parent)

    preexec = _child_death_signal if os.name == "posix" else None
    try:
        if attach_tty:
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                cwd=cwd,
                preexec_fn=preexec,
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                bufsize=1,
                preexec_fn=preexec,
            )
        print(f"[run_training] Started client process (PID: {process.pid})")
        return process
    except Exception as exc:
        print(f"[run_training] Failed to start client process: {exc}")
        return None


def _read_stdout_chunk(stdout, timeout_s: float = 0.05) -> bytes:
    """Non-blocking read so tqdm \\r updates are visible before a final newline."""
    if stdout is None:
        return b""
    fd = stdout.fileno()
    try:
        import select

        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            return b""
        return os.read(fd, 4096)
    except (ValueError, OSError):
        return b""


async def forward_client_output(
    process: subprocess.Popen, status: Optional[CombinedTrainingStatus] = None
) -> None:
    if process.stdout is None:
        return

    pending = ""
    try:
        while True:
            chunk_b = await asyncio.to_thread(_read_stdout_chunk, process.stdout)
            if chunk_b:
                pending += chunk_b.decode("utf-8", errors="replace")
            while True:
                split_at = None
                for i, ch in enumerate(pending):
                    if ch in "\r\n":
                        split_at = i
                        break
                if split_at is None:
                    break
                segment = pending[:split_at]
                pending = pending[split_at + 1 :]
                if not segment.strip():
                    continue
                if status is not None:
                    status.handle_client_text(segment)
                else:
                    print(f"[client] {segment.strip()}")
            if not chunk_b and process.poll() is not None:
                break
        if pending.strip():
            if status is not None:
                status.handle_client_text(pending)
            else:
                print(f"[client] {pending.strip()}")
    except Exception as exc:
        if process.poll() is None:
            print(f"[run_training] Error reading client output: {exc}")
    finally:
        if status is not None:
            status.finish()


async def _run_with_optional_client(
    server: LearnerServer, args: argparse.Namespace, status: Optional[CombinedTrainingStatus]
) -> None:
    forward_task: asyncio.Task | None = None
    client_process: Optional[subprocess.Popen] = None

    try:
        if args.auto_start_client:
            await asyncio.sleep(1.0)
            use_combined = status is not None
            client_process = start_client_process(
                "run.py",
                attach_tty=not use_combined and args.forward_client_output,
                extra_args=getattr(args, "forwarded_settings_args", []),
            )
            if client_process is not None and client_process.stdout is not None:
                forward_task = asyncio.create_task(
                    forward_client_output(client_process, status)
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

    # Resolve model naming from --model-name convenience.
    # --model-name sets BOTH the save name (when --save-model-name is omitted)
    # and the load name (when --load-model-name is omitted *and* a model with
    # that name already exists on disk). Explicit --load/--save names always
    # take precedence. This lets a single --model-name train, retrain and
    # finetune the same model.
    model_name = getattr(run_args, "model_name", None)

    if getattr(run_args, "save_model_name", None) is None:
        run_args.save_model_name = model_name

    if run_args.load_model_name is None and model_name is not None:
        if model_exists(model_name):
            run_args.load_model_name = model_name
            print(
                f"[run_training] --model-name '{model_name}' already exists -> "
                f"loading it as the training base and saving back to the same name"
            )
        else:
            print(
                f"[run_training] --model-name '{model_name}' not found -> "
                f"training from scratch and saving to '{run_args.save_model_name}'"
            )

    if run_args.save_model_name is None:
        print(
            "[run_training] Error: no model save name given. "
            "Pass --save-model-name NAME or --model-name NAME.",
            file=sys.stderr,
        )
        sys.exit(2)

    # SAC training minibatch (replay sample size per grad step), not SAC_STREAM_BATCH_SIZE.
    if run_args.batch_size is not None:
        train_batch_size = int(run_args.batch_size)
    else:
        train_batch_size = int(Settings.SAC_BATCH_SIZE)

    # Only reset the save-model directory when not loading a checkpoint.
    if run_args.load_model_name is None:
        backup_existing_model_if_present(str(run_args.save_model_name))

    combined_status = CombinedTrainingStatus() if run_args.auto_start_client else None
    server = LearnerServer(
        host=run_args.host,
        port=run_args.port,
        # explicit load/save names (load may be None -> scratch)
        load_model_name=run_args.load_model_name,
        save_model_name=run_args.save_model_name,
        device=run_args.device,
        train_every_seconds=run_args.train_every_seconds,
        grad_steps=run_args.gradient_steps,
        replay_capacity=run_args.replay_capacity,
        learning_starts=run_args.learning_starts,
        batch_size=train_batch_size,
        learning_rate=run_args.learning_rate,
        discount_factor=run_args.discount_factor,
        train_frequency=run_args.train_frequency,
        save_replay_buffer=run_args.save_replay_buffer,
        load_replay_buffer=run_args.load_replay_buffer,
        status_line_callback=(
            combined_status.set_train if combined_status is not None else None
        ),
    )

    try:
        asyncio.run(_run_with_optional_client(server, run_args, combined_status))
    except KeyboardInterrupt:
        print("\n[server] KeyboardInterrupt received in run_training, exiting...")
        run_completed = False
    else:
        # Normal completion of asyncio.run -> training/server terminated
        run_completed = True
    
    # After the server has terminated normally, optionally run a single
    # evaluation client using the trained model for inference. We only run
    # this when training completed (not when interrupted by Ctrl-C).
    if run_completed and RUN_EVALUATION:
        model_name = getattr(run_args, "save_model_name", None)
        if model_name is not None:
            # In some workflows (e.g. short smoke tests or interrupted runs), the
            # expected `model_name.zip` may not exist yet. Avoid failing hard by
            # skipping evaluation when no SAC zip is present.
            try:
                from TrainingLite.rl_racing.sac_utilities import SacUtilities

                model_path, model_dir = SacUtilities.resolve_model_paths(str(model_name))
                server_model_path = os.path.join(model_dir, "server", str(model_name))
                server_zip_path = server_model_path + ".zip"
                local_zip_path = model_path + ".zip"

                if not os.path.exists(local_zip_path) and not os.path.exists(server_zip_path):
                    print(
                        f"[run_training] Skipping evaluation: model zip not found for '{model_name}'. "
                        f"Tried: {local_zip_path} or {server_zip_path}"
                    )
                    return
            except Exception as e:
                # Evaluation is non-critical for training; just warn and continue.
                print(f"[run_training] Warning: could not verify evaluation model existence: {e}")

            eval_args = [
                "--CONTROLLER",
                "sac_agent",
                "--SIMULATION_LENGTH",
                "2000",
                "--SAVE_RECORDINGS",
                "True",
                "--SAC_INFERENCE_MODEL_NAME",
                str(model_name),
                "--DATASET_NAME",                
                str(model_name),
            ]
            print(f"[run_training] Launching evaluation client with model '{model_name}'")
            try:
                eval_proc = start_client_process(
                    "run.py",
                    attach_tty=run_args.forward_client_output,
                    extra_args=eval_args,
                )
                if eval_proc is not None:
                    # Wait for evaluation client to finish and report exit code
                    ret = eval_proc.wait()
                    print(f"[run_training] Evaluation client exited with code {ret}")
                else:
                    print("[run_training] Failed to start evaluation client")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[run_training] Error running evaluation client: {exc}")


if __name__ == "__main__":
    main()


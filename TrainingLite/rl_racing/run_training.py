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
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# ### 1. IMPORT SETTINGS (Required to apply overrides)
# Depending on your folder structure, ensure this import points to your Settings file
try:
    from utilities.Settings import Settings
except ImportError:
    # Fallback if running from root without module install
    sys.path.append(os.getcwd())
    from utilities.Settings import Settings

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

# Import LearnerServer will be done dynamically in main() based on --learner-impl flag
from utilities.parser_utilities import parse_settings_args  # noqa: E402
from utilities.command_logger import save_run_metadata, print_run_summary  # noqa: E402


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
    # Legacy model name (kept for backward compatibility).
    parser.add_argument("--model-name", default="SAC_RCA1_0", help="(legacy) model name used as default save name if --save-model-name not provided")
    parser.add_argument(
        "--load-model-name",
        default=None,
        help=("Model name to load as a base for training. If omitted (None), the server will start training from scratch."),
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
    parser.add_argument("--train-every-seconds", type=float, default=0.1)
    parser.add_argument("--gradient-steps", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
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

    parser.add_argument(
        "--learner-impl",
        choices=["threaded", "original"],
        default="original",
        help=(
            "Choose learner server implementation: "
            "'original' (default, standard blocking implementation) or "
            "'threaded' (episodes flow during training)"
        ),
    )

    #Sweep bash script args
    parser.add_argument("--USE_CUSTOM_SAC_SAMPLING", type=str, default="True", help="Enable custom sampling")
    parser.add_argument("--alpha", type=float, default=None, help="Override SAC_PRIORITY_FACTOR")
    parser.add_argument("--beta_start", type=float, default=None, help="Override SAC_IMPORANCE_SAMPLING_CORRECTOR")
    parser.add_argument("--td_ratio", type=float, default=None, help="Override SAC_STATE_TO_TD_RATIO")
    parser.add_argument("--SIMULATION_LENGTH", type=int, default=400000, help="Total training steps")
    
    known_args, remaining = parser.parse_known_args(argv)
    return known_args, remaining


def load_learner_server_class(impl_choice: str):
    """Dynamically import the chosen LearnerServer implementation."""
    if impl_choice == "threaded":
        from learner_server_threaded import LearnerServer
        print("[run_training] Using threaded LearnerServer implementation")
    else:  # original
        from learner_server import LearnerServer
        print("[run_training] Using original LearnerServer implementation")
    return LearnerServer


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

    preexec = _child_death_signal if os.name == "posix" else None
    try:
        if forward_output:
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

            client_settings_args = list(getattr(args, "forwarded_settings_args", []))
            if getattr(args, "client_simulation_length", None) is not None:
                client_settings_args.extend(["--SIMULATION_LENGTH", str(args.client_simulation_length)])

            client_process = start_client_process(
                "run.py",
                forward_output=args.forward_client_output,
                extra_args=client_settings_args,
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

    # Load the appropriate LearnerServer implementation
    LearnerServer = load_learner_server_class(run_args.learner_impl)

    #TODO: check if needed
    settings_args.extend(["--SIMULATION_LENGTH", str(run_args.SIMULATION_LENGTH)])
    settings_args.extend(["--USE_CUSTOM_SAC_SAMPLING", str(run_args.USE_CUSTOM_SAC_SAMPLING)])

    # PER Parameter Overrides
    if run_args.alpha is not None:
        Settings.SAC_PRIORITY_FACTOR = run_args.alpha
        print(f"[run_training] Override: Alpha set to {run_args.alpha}")
        
    if run_args.beta_start is not None:
        Settings.SAC_IMPORANCE_SAMPLING_CORRECTOR = run_args.beta_start
        print(f"[run_training] Override: Beta Start set to {run_args.beta_start}")
        
    if run_args.td_ratio is not None:
        Settings.SAC_STATE_TO_TD_RATIO = run_args.td_ratio
        print(f"[run_training] Override: TD Ratio set to {run_args.td_ratio}")

    
    settings_namespace = parse_settings_overrides(settings_args)

    setattr(run_args, "forwarded_settings_args", settings_args)
    setattr(run_args, "settings_namespace", settings_namespace)

    prefill_enabled = bool(getattr(settings_namespace, "SAC_PREFILL_BUFFER_WITH_PP", False))
    prefill_amount_raw = getattr(settings_namespace, "SAC_PREFILL_BUFFER_WITH_PP_AMOUNT", 0)
    prefill_amount = int(prefill_amount_raw) if prefill_amount_raw is not None else 0
    client_simulation_length = int(run_args.SIMULATION_LENGTH)
    if prefill_enabled and prefill_amount > 0:
        client_simulation_length += prefill_amount
        print(
            f"[run_training] Prefill enabled: client SIMULATION_LENGTH set to {client_simulation_length} "
            f"(server target remains {run_args.SIMULATION_LENGTH}, prefill={prefill_amount})."
        )
    setattr(run_args, "client_simulation_length", client_simulation_length)

    # Backwards-compatible handling: if save_model_name not provided, fall back to legacy model_name
    if getattr(run_args, "save_model_name", None) is None:
        run_args.save_model_name = run_args.model_name

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
        batch_size=run_args.batch_size,
        learning_rate=run_args.learning_rate,
        discount_factor=run_args.discount_factor,
        train_frequency=run_args.train_frequency,
        save_replay_buffer=run_args.save_replay_buffer,
    )

    #TODO: check if needed
    if server.replay_buffer is not None:
        if run_args.alpha is not None:
            server.replay_buffer.alpha = run_args.alpha
        if run_args.beta_start is not None:
            server.replay_buffer.initial_beta = run_args.beta_start
            server.replay_buffer.beta = run_args.beta_start
        if run_args.td_ratio is not None:
            server.replay_buffer.state_to_TD_ratio = run_args.td_ratio
            
        # Ensure annealing horizon uses the new simulation length
        ratio = getattr(Settings, 'SAC_BETA_ANNEALING_RATIO', 0.75)
        server.replay_buffer.beta_annealing_horizon = ratio * run_args.SIMULATION_LENGTH

    # Save run metadata (command line args + Settings) for experiment reproducibility
    try:
        csv_path = os.path.join(server.model_dir, "learning_metrics.csv")
        cli_args_dict = vars(run_args)
        save_run_metadata(csv_path, cli_args_dict, Settings)
        print_run_summary(cli_args_dict, Settings)
    except Exception as e:
        print(f"[run_training] Warning: Failed to save run metadata: {e}")

    try:
        asyncio.run(_run_with_optional_client(server, run_args))
    except KeyboardInterrupt:
        print("\n[server] KeyboardInterrupt received in run_training, exiting...")
        run_completed = False
    else:
        # Normal completion of asyncio.run -> training/server terminated
        run_completed = True
    
    # After the server has terminated normally, optionally run a single
    # evaluation client using the trained model for inference. We only run
    # this when training completed (not when interrupted by Ctrl-C).
    if run_completed:
        model_name = getattr(run_args, "save_model_name", None)
        if model_name is not None:
            # In some workflows (e.g. short smoke tests or interrupted runs), the
            # expected `model_name.zip` may not exist yet. Avoid failing hard by
            # skipping evaluation when no SAC zip is present.
            try:
                from TrainingLite.rl_racing.sac_utilities import SacUtilities

                model_path, model_dir = SacUtilities.resolve_model_paths(str(model_name))
                server_model_path = os.path.join(model_dir, "server", str(model_name))
                local_stem_path = model_path[:-4] if str(model_path).endswith(".zip") else model_path
                local_zip_path = local_stem_path + ".zip"
                server_stem_path = server_model_path
                server_zip_path = server_model_path + ".zip"

                if not any(
                    os.path.exists(candidate)
                    for candidate in (local_stem_path, local_zip_path, server_stem_path, server_zip_path)
                ):
                    print(
                        f"[run_training] Skipping evaluation: model zip not found for '{model_name}'. "
                        f"Tried: {local_stem_path}, {local_zip_path}, {server_stem_path}, {server_zip_path}"
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

            # ensure test lap uses same map as the training
            map_name = getattr(run_args.settings_namespace, 'MAP_NAME', None)

            if map_name is not None:
                eval_args.extend(["--MAP_NAME", str(map_name)])

            print(f"[run_training] Launching evaluation client with model '{model_name}'")
            try:
                eval_proc = start_client_process(
                    "run.py", forward_output=run_args.forward_client_output, extra_args=eval_args
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
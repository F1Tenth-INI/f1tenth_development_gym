"""HWM learner server.

Skeleton adapted from ``TrainingLite/rl_racing/learner_server.py`` with the SAC
specifics removed. It accepts planner connections, ingests ``[state, action]``
rows into a ``MemoryManager`` (the same class the planner uses), runs training
rounds in a worker thread, saves checkpoints and broadcasts weights.

Insertion points (the only methods meant to be edited):

    build_models()                  -> ModelBundle      (shared with the planner)
    client_modules()                -> list[str]        which modules to broadcast
    on_ingest(states, actions, episode_id, episode_end)
    train_round(n_steps)            -> dict             metrics; called in a worker thread
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from TrainingLite.hwm.comm.protocol import (
    MSG_ACK,
    MSG_BATCH_ACK,
    MSG_CLEAR_BUFFER,
    MSG_CLEAR_BUFFER_ACK,
    MSG_RAW_BATCH,
    MSG_TERMINATE,
    MSG_TERMINATE_ACK,
    MSG_TRAINING_INFO,
    pack_handshake,
    pack_simple,
    pack_weights,
    unpack_raw_batch,
)
from TrainingLite.hwm.models.bundle import ModelBundle
from TrainingLite.hwm.models.networks import build_models as build_model_bundle
from TrainingLite.hwm.paths import resolve_model_dir
from utilities.Settings import Settings
from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from TrainingLite.rl_racing.tcp_utilities import pack_frame, read_frame


class HWMLearnerServer:
    def __init__(
        self,
        host: str,
        port: int,
        save_model_name: str,
        load_model_name: Optional[str] = None,
        device: str = "cpu",
        train_every_seconds: float = 0.0,
        grad_steps: int = 32,
        learning_starts: int = 500,
        autosave_interval_s: float = 60.0,
        status_line_callback: Optional[Callable[[str], None]] = None,
    ):
        self.host = host
        self.port = port
        self.save_model_name = str(save_model_name)
        self.load_model_name = load_model_name
        self.device = device
        self.train_every_seconds = float(train_every_seconds)
        self.grad_steps = int(grad_steps)
        self.learning_starts = int(learning_starts)
        self.autosave_interval_s = float(autosave_interval_s)
        self._status_cb = status_line_callback

        self.model_dir: Path = resolve_model_dir(self.save_model_name, create=True)

        # Shared components (identical construction on the planner, both from Settings).
        self.wall_geometry = WallGeometry(device=self.device)
        self.memory = MemoryManager()
        self.models: ModelBundle = self.build_models().to(self.device)
        if self.load_model_name is not None:
            loaded = self.models.load(resolve_model_dir(self.load_model_name), device=self.device)
            print(f"[server] Loaded modules {loaded} from '{self.load_model_name}'")
            self._weights_blob: Optional[bytes] = pack_frame(pack_weights(self.models.state_dicts(self.client_modules())))
        else:
            # Random init: first broadcast happens in _train_loop once learning_starts is reached.
            self._weights_blob = None

        # Networking / lifecycle.
        self._clients: set[asyncio.StreamWriter] = set()
        self._client_lock = asyncio.Lock()
        self._should_terminate = False
        self._train_event = asyncio.Event()
        self._pending_rows = 0
        self._train_lock = asyncio.Lock()
        self.total_rows = 0
        self.total_updates = 0
        self._last_autosave = time.time()
        self._latest_training_info: Optional[dict] = None

    # ------------------------------------------------------------------
    # INSERTION POINTS
    # ------------------------------------------------------------------
    def build_models(self) -> ModelBundle:
        return build_model_bundle()

    def client_modules(self) -> list[str]:
        modules = Settings.HWM_CLIENT_MODULES
        if isinstance(modules, str):
            return [modules]
        return list(modules)

    def on_ingest(self, states, actions, episode_id: int, episode_end: bool) -> int:
        """Called for every received raw batch. Default: append to memory."""
        return self.memory.add_batch(states, actions, episode_id, episode_end=episode_end)

    def train_round(self, n_steps: int) -> dict[str, Any]:
        """Run ``n_steps`` gradient steps on ``self.models`` using ``self.memory``.

        Runs in a worker thread; do not touch asyncio state here. Return metrics
        (JSON-serialisable) that are logged and sent to the planner as training_info.
        """
        # Placeholder: no learning yet.
        return {"steps": 0}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    async def _train_loop(self) -> None:
        while not self._should_terminate:
            if self.train_every_seconds > 0:
                await asyncio.sleep(self.train_every_seconds)
            else:
                await self._train_event.wait()
                self._train_event.clear()
            if self._should_terminate:
                break
            if self.memory.total_steps < self.learning_starts:
                continue
            n_new = self._pending_rows
            self._pending_rows = 0
            if n_new <= 0 and self.train_every_seconds <= 0:
                continue

            t0 = time.time()
            async with self._train_lock:
                metrics = await asyncio.to_thread(self._train_round_safe, self.grad_steps)
            dt = time.time() - t0
            self.total_updates += int(metrics.get("steps", 0))

            payload = {
                "total_rows": int(self.total_rows),
                "total_updates": int(self.total_updates),
                "episodes": int(self.memory.num_episodes),
                "train_time_s": float(dt),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
            }
            self._latest_training_info = payload
            self._status(
                f"rows={payload['total_rows']} eps={payload['episodes']} "
                f"updates={payload['total_updates']} round={dt:.2f}s"
            )

            self._weights_blob = pack_frame(pack_weights(self.models.state_dicts(self.client_modules())))
            await self._broadcast_raw(self._weights_blob)
            await self._broadcast_raw(pack_frame(pack_simple(MSG_TRAINING_INFO, **payload)))

            if time.time() - self._last_autosave >= self.autosave_interval_s:
                self._save_models()

    def _train_round_safe(self, n_steps: int) -> dict[str, Any]:
        try:
            self.models.train()
            out = self.train_round(n_steps)
            return out if isinstance(out, dict) else {}
        except Exception as e:
            print(f"[server] train_round failed: {e}")
            return {"steps": 0, "error": str(e)}
        finally:
            self.models.eval()

    def _save_models(self) -> None:
        try:
            self.models.save(self.model_dir)
            self._last_autosave = time.time()
            print(f"[server] Saved models to {self.model_dir}")
        except Exception as e:
            print(f"[server] Failed to save models: {e}")

    def _status(self, msg: str) -> None:
        if self._status_cb is not None:
            self._status_cb(msg)
        else:
            print(f"[server] {msg}")

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------
    async def _broadcast_raw(self, frame_bytes: bytes, exclude: Optional[asyncio.StreamWriter] = None) -> None:
        async with self._client_lock:
            dead = []
            for w in self._clients:
                if w is exclude:
                    continue
                try:
                    w.write(frame_bytes)
                    await w.drain()
                except Exception:
                    dead.append(w)
            for w in dead:
                self._clients.discard(w)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        print(f"[server] Client connected: {addr}")
        async with self._client_lock:
            self._clients.add(writer)

        try:
            writer.write(pack_frame(pack_simple(MSG_ACK, msg="connected")))
            writer.write(pack_frame(pack_handshake(self.save_model_name, str(Settings.MAP_NAME))))
            if self._weights_blob is not None:
                writer.write(self._weights_blob)
            if self._latest_training_info is not None:
                writer.write(pack_frame(pack_simple(MSG_TRAINING_INFO, **self._latest_training_info)))
            await writer.drain()
        except Exception as e:
            print(f"[server] Failed initial send to {addr}: {e}")

        try:
            while not self._should_terminate:
                msg = await read_frame(reader)
                typ = msg.get("type")
                data = msg.get("data", {}) or {}

                if typ == MSG_RAW_BATCH:
                    states, actions, episode_id, episode_end = unpack_raw_batch(data)
                    n = self.on_ingest(states, actions, episode_id, episode_end)
                    self.total_rows += n
                    self._pending_rows += n
                    self._train_event.set()
                    try:
                        writer.write(pack_frame(pack_simple(MSG_BATCH_ACK, n=n, episode_id=episode_id)))
                        await writer.drain()
                    except Exception:
                        pass

                elif typ == MSG_CLEAR_BUFFER:
                    self.memory.clear()
                    self._pending_rows = 0
                    print("[server] Cleared memory (requested by planner)")
                    try:
                        writer.write(pack_frame(pack_simple(MSG_CLEAR_BUFFER_ACK)))
                        await writer.drain()
                    except Exception:
                        pass

                elif typ == MSG_TERMINATE:
                    print("[server] Received terminate from planner")
                    async with self._train_lock:
                        self._save_models()
                    # Ack the requester first (it stops reading once it sees a terminate),
                    # then tell any other connected planners to stop.
                    try:
                        writer.write(pack_frame(pack_simple(MSG_TERMINATE_ACK, msg="terminating")))
                        await writer.drain()
                    except Exception:
                        pass
                    await self._broadcast_raw(
                        pack_frame(pack_simple(MSG_TERMINATE, reason="terminate_requested_by_planner")),
                        exclude=writer,
                    )
                    self._should_terminate = True
                    self._train_event.set()
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[server] Client loop error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            async with self._client_lock:
                self._clients.discard(writer)
            print(f"[server] Client disconnected: {addr}")

    async def run(self) -> None:
        train_task = asyncio.create_task(self._train_loop())
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(s.getsockname()) for s in server.sockets or [])
        print(
            f"[server] Listening on {addrs} | save_model='{self.save_model_name}' "
            f"| load_model='{self.load_model_name}' | device='{self.device}' | map='{Settings.MAP_NAME}'"
        )

        async def monitor() -> None:
            while not self._should_terminate:
                await asyncio.sleep(0.5)

        try:
            async with server:
                await monitor()
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n[server] Interrupted, shutting down...")
        finally:
            self._should_terminate = True
            self._train_event.set()
            if not train_task.done():
                train_task.cancel()
                try:
                    await train_task
                except asyncio.CancelledError:
                    pass
            async with self._client_lock:
                for w in list(self._clients):
                    try:
                        w.close()
                        await w.wait_closed()
                    except Exception:
                        pass
                self._clients.clear()
            server.close()
            await server.wait_closed()
            self._save_models()
            print("[server] Shutdown complete")

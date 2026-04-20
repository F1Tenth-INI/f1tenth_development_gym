import threading
import asyncio
import queue
import os
import shutil

from typing import Optional

try:
    from TrainingLite.rl_racing.tcp_utilities import pack_frame, read_frame, np_to_blob, bytes_to_state_dict
except ModuleNotFoundError:
    from f1tenth_development_gym.TrainingLite.rl_racing.tcp_utilities import (
        pack_frame,
        read_frame,
        np_to_blob,
        bytes_to_state_dict,
    )
import numpy as np


class _TCPActorClient:
    def __init__(self, host: str, port: int, actor_id: int = 0):
        self.host = host
        self.port = port
        self.actor_id = actor_id

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._send_q: "queue.Queue[dict]" = queue.Queue(maxsize=10000)
        # Terminate (and similar control) must not sit behind large transition backlogs.
        self._urgent_q: "queue.Queue[dict]" = queue.Queue(maxsize=32)
        self._terminate_ack_evt = threading.Event()
        self._terminate_sync_lock = threading.Lock()
        self._waiting_for_terminate_ack = False
        self._latest_sd_lock = threading.Lock()
        self._latest_state_dict: Optional[dict] = None
        self._latest_model_sync_lock = threading.Lock()
        self._latest_model_sync: Optional[dict] = None
        self._latest_training_info_lock = threading.Lock()
        self._latest_training_info: Optional[dict] = None
        self._server_terminate_lock = threading.Lock()
        self._server_terminate_payload: Optional[dict] = None
        self._stop_evt = threading.Event()
        self._connect_failures = 0
        self._episode_ack_count = 0

        #NIKITA: pp prefill artifacts
        self._latest_weights_fingerprint: Optional[str] = None
        self._training_status_lock = threading.Lock()
        self._training_ready: Optional[bool] = None
        self._bc_in_progress: bool = False

    # --- public API ---
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, name="TCPActorClient", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_evt.set()
        if self._loop and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def send_transition_batch(self, batch: list) -> None:
        # batch: list of dicts with keys obs, action, next_obs, reward, done, info
        data = {
            "type": "transition_batch",
            "data": {
                "actor_id": int(self.actor_id),
                "obs": [np_to_blob(t["obs"].astype(np.float32)) for t in batch],
                "action": [np_to_blob(t["action"].astype(np.float32)) for t in batch],
                "next_obs": [np_to_blob(t["next_obs"].astype(np.float32)) for t in batch],
                "reward": [float(t["reward"]) for t in batch],
                "done": [bool(t["done"]) for t in batch],
                "info": [t.get("info", {}) for t in batch],
            },
        }
        try:
            self._send_q.put_nowait(data)
        except queue.Full:
            pass

    def send_clear_buffer(self) -> None:
        """Send a message to clear the server's replay buffer."""
        data = {
            "type": "clear_buffer",
            "data": {
                "actor_id": int(self.actor_id),
            },
        }
        try:
            self._send_q.put_nowait(data)
        except queue.Full:
            pass

    def send_terminate(
        self,
        wait_for_ack: bool = True,
        ack_timeout: float = 30.0,
        urgent: bool = True,
    ) -> bool:
        """Send terminate so the server saves and shuts down.

        Uses an urgent queue so the message is not stuck behind transition batches.
        If ``wait_for_ack`` is True, blocks until the server sends ``terminate_ack``
        or ``ack_timeout`` elapses (TCP may still have delivered the frame).
        """
        data = {
            "type": "terminate",
            "data": {
                "actor_id": int(self.actor_id),
            },
        }
        if wait_for_ack:
            with self._terminate_sync_lock:
                self._waiting_for_terminate_ack = True
                self._terminate_ack_evt.clear()
        try:
            target_queue = self._urgent_q if urgent else self._send_q
            target_queue.put(data, timeout=max(5.0, float(ack_timeout)))
        except queue.Full:
            if wait_for_ack:
                with self._terminate_sync_lock:
                    self._waiting_for_terminate_ack = False
            queue_name = "urgent" if urgent else "normal"
            print(f"[TCPActorClient] {queue_name.capitalize()} queue full; could not send terminate")
            return False
        except Exception as exc:
            if wait_for_ack:
                with self._terminate_sync_lock:
                    self._waiting_for_terminate_ack = False
            print(f"[TCPActorClient] Failed to enqueue terminate: {exc}")
            return False
        if not wait_for_ack:
            return True
        ok = self._terminate_ack_evt.wait(timeout=float(ack_timeout))
        with self._terminate_sync_lock:
            self._waiting_for_terminate_ack = False
        if not ok:
            print(
                f"[TCPActorClient] Timeout ({ack_timeout}s) waiting for terminate_ack "
                "(server may still have received terminate)"
            )
        return ok

    def pop_latest_state_dict(self) -> Optional[dict]:
        with self._latest_sd_lock:
            sd = self._latest_state_dict
            self._latest_state_dict = None
            return sd

    def get_latest_weights_fingerprint(self) -> Optional[str]:
        with self._latest_sd_lock:
            return self._latest_weights_fingerprint

    def get_training_ready(self) -> Optional[bool]:
        with self._training_status_lock:
            return self._training_ready

    def get_bc_in_progress(self) -> bool:
        with self._training_status_lock:
            return self._bc_in_progress
    def pop_latest_model_sync(self) -> Optional[dict]:
        with self._latest_model_sync_lock:
            payload = self._latest_model_sync
            self._latest_model_sync = None
            return payload

    def pop_latest_training_info(self) -> Optional[dict]:
        with self._latest_training_info_lock:
            payload = self._latest_training_info
            self._latest_training_info = None
            return payload

    def pop_server_terminate(self) -> Optional[dict]:
        with self._server_terminate_lock:
            payload = self._server_terminate_payload
            self._server_terminate_payload = None
            return payload

    # --- internals ---
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()

    async def _shutdown_async(self):
        self._stop_evt.set()
        await asyncio.sleep(0)

    async def _main(self):
        while not self._stop_evt.is_set():
            try:
                print(f"[TCPActorClient] Connecting to {self.host}:{self.port} (actor_id={self.actor_id})")
                reader, writer = await asyncio.open_connection(self.host, self.port)
                self._connect_failures = 0
                print(f"[TCPActorClient] Connected to {self.host}:{self.port}")
                # Expect ack and maybe initial weights
                try:
                    msg = await asyncio.wait_for(read_frame(reader), timeout=5.0)
                    await self._handle_msg(msg)
                except asyncio.TimeoutError:
                    pass

                # Try to read an immediate weights frame if available
                reader_task = asyncio.create_task(self._reader_loop(reader))
                writer_task = asyncio.create_task(self._writer_loop(writer))
                done, pending = await asyncio.wait(
                    {reader_task, writer_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            except Exception:
                self._connect_failures += 1
                if self._connect_failures == 1 or self._connect_failures % 10 == 0:
                    print(f"[TCPActorClient] Connection failed x{self._connect_failures}; retrying in 1s ({self.host}:{self.port})")
                # reconnect after short delay
                await asyncio.sleep(1.0)

    async def _reader_loop(self, reader: asyncio.StreamReader):
        while not self._stop_evt.is_set():
            try:
                msg = await read_frame(reader)
                await self._handle_msg(msg)
            except (ConnectionResetError, asyncio.IncompleteReadError):
                # Connection lost, exit loop gracefully
                break
            except Exception as e:
                print(f"[TCPActorClient] Reader loop error: {e}")
                break

    async def _handle_msg(self, msg: dict):
        typ = msg.get("type")
        if typ == "weights":
            data = msg.get("data", {})
            blob = data["blob"]
            sd = bytes_to_state_dict(blob)
            with self._latest_sd_lock:
                self._latest_state_dict = sd
                self._latest_weights_fingerprint = data.get("fingerprint")
        elif typ == "training_status":
            data = msg.get("data", {})
            ready = bool(data.get("training_ready", False))
            bc_in_progress = bool(data.get("bc_in_progress", False))
            with self._training_status_lock:
                self._training_ready = ready
                self._bc_in_progress = bc_in_progress
        elif typ == "episode_ack":
            self._episode_ack_count += 1
            if self._episode_ack_count % 5 == 0:
                print(f"[TCPActorClient] Received episode_ack count={self._episode_ack_count}")
        elif typ in ("ack", "model_sync"):
            payload = msg.get("data", {})
            if isinstance(payload, dict):
                self._maybe_mirror_model_folder(payload)
        elif typ == "training_info":
            payload = msg.get("data", {})
            if isinstance(payload, dict):
                with self._latest_training_info_lock:
                    self._latest_training_info = payload
        elif typ == "terminate_ack":
            with self._terminate_sync_lock:
                if self._waiting_for_terminate_ack:
                    self._terminate_ack_evt.set()
        elif typ == "terminate":
            payload = msg.get("data", {})
            if not isinstance(payload, dict):
                payload = {}
            with self._server_terminate_lock:
                self._server_terminate_payload = payload
            # Stop transport loop as soon as server asked to terminate.
            self._stop_evt.set()
        # ignore others

    def _maybe_mirror_model_folder(self, payload: dict) -> None:
        model_name = payload.get("model_name")
        source_model_dir = payload.get("source_model_dir") or payload.get("model_dir")
        if not isinstance(model_name, str) or not model_name:
            return
        if not isinstance(source_model_dir, str) or not source_model_dir:
            return
        if not os.path.isdir(source_model_dir):
            # Usually remote server path not visible locally.
            return

        source_client_dir = os.path.join(source_model_dir, "client")
        source_dir = source_client_dir if os.path.isdir(source_client_dir) else source_model_dir
        target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", model_name, "client"))
        try:
            if os.path.abspath(source_dir) == os.path.abspath(target_dir):
                with self._latest_model_sync_lock:
                    self._latest_model_sync = {"model_name": model_name, "local_client_dir": target_dir, "mirrored": False}
                return

            os.makedirs(target_dir, exist_ok=True)
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            with self._latest_model_sync_lock:
                self._latest_model_sync = {"model_name": model_name, "local_client_dir": target_dir, "mirrored": True}
            print(f"[TCPActorClient] Mirrored client folder: {source_dir} -> {target_dir}")
        except Exception as e:
            # Copy can fail if source/target effectively point to identical files.
            # In that case, still publish model_sync so the planner can load the builder.
            builder_in_target = os.path.join(target_dir, "observation_builder.py")
            fallback_dir = target_dir if os.path.isfile(builder_in_target) else source_dir
            with self._latest_model_sync_lock:
                self._latest_model_sync = {
                    "model_name": model_name,
                    "local_client_dir": fallback_dir,
                    "mirrored": False,
                }
            print(
                f"[TCPActorClient] Client folder mirror skipped ({source_dir} -> {target_dir}): {e}. "
                f"Using local_client_dir={fallback_dir}"
            )

    async def _writer_loop(self, writer: asyncio.StreamWriter):
        while not self._stop_evt.is_set():
            frame = None
            try:
                frame = self._urgent_q.get_nowait()
            except queue.Empty:
                try:
                    frame = self._send_q.get_nowait()
                except queue.Empty:
                    # Avoid busy-spin (get_nowait + sleep(0) pegs a core and slows the sim via GIL).
                    await asyncio.sleep(0.02)
                    continue
            try:
                writer.write(pack_frame(frame))
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                # Connection lost, exit loop gracefully
                break
            except Exception as e:
                print(f"[TCPActorClient] Writer loop error: {e}")
                break





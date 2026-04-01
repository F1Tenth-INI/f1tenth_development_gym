import threading
import asyncio
import queue

from typing import Optional

from TrainingLite.rl_racing.tcp_utilities import pack_frame, read_frame, np_to_blob, bytes_to_state_dict
import numpy as np


class _TCPActorClient:
    def __init__(self, host: str, port: int, actor_id: int = 0):
        self.host = host
        self.port = port
        self.actor_id = actor_id

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._send_q: "queue.Queue[dict]" = queue.Queue(maxsize=10000)
        self._latest_sd_lock = threading.Lock()
        self._latest_state_dict: Optional[dict] = None
        self._latest_weights_fingerprint: Optional[str] = None
        self._training_status_lock = threading.Lock()
        self._training_ready: Optional[bool] = None
        self._bc_in_progress: bool = False
        self._stop_evt = threading.Event()
        self._connect_failures = 0
        self._episode_ack_count = 0

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

    def send_terminate(self) -> None:
        """Send a terminate message to the server, causing it to save the model and shutdown."""
        data = {
            "type": "terminate",
            "data": {
                "actor_id": int(self.actor_id),
            },
        }
        try:
            self._send_q.put_nowait(data)
        except queue.Full:
            pass

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
                    # ignore ack; next may be weights
                    if msg.get("type") != "ack":
                        # some servers might send weights first
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
        # ignore others

    async def _writer_loop(self, writer: asyncio.StreamWriter):
        while not self._stop_evt.is_set():
            try:
                frame = self._send_q.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.0)
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





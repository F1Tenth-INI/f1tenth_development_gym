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
        self._stop_evt = threading.Event()

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

    def pop_latest_state_dict(self) -> Optional[dict]:
        with self._latest_sd_lock:
            sd = self._latest_state_dict
            self._latest_state_dict = None
            return sd

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
                reader, writer = await asyncio.open_connection(self.host, self.port)
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
            blob = msg["data"]["blob"]
            sd = bytes_to_state_dict(blob)
            with self._latest_sd_lock:
                self._latest_state_dict = sd
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





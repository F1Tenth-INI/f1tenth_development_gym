"""Planner-side TCP client for the HWM learner.

Adapted from ``TrainingLite/rl_racing/tcp_client.py``: same daemon thread with
its own asyncio loop, same queues and reconnect/terminate handling. Differences:
payloads are blocks of ``[state, action]`` rows, weights arrive as a dict of
module state dicts, and there is no model-folder mirroring.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Optional

import numpy as np

from TrainingLite.hwm.comm.protocol import (
    MSG_ACK,
    MSG_CLEAR_BUFFER,
    MSG_HANDSHAKE,
    MSG_TERMINATE,
    MSG_TERMINATE_ACK,
    MSG_TRAINING_INFO,
    MSG_WEIGHTS,
    pack_raw_batch,
    pack_simple,
    unpack_weights,
)
from TrainingLite.rl_racing.tcp_utilities import pack_frame, read_frame


class HWMTCPClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._send_q: "queue.Queue[dict]" = queue.Queue(maxsize=10000)
        self._urgent_q: "queue.Queue[dict]" = queue.Queue(maxsize=32)
        self._stop_evt = threading.Event()
        self._send_drops = 0

        self._terminate_ack_evt = threading.Event()
        self._terminate_sync_lock = threading.Lock()
        self._waiting_for_terminate_ack = False

        self._lock = threading.Lock()
        self._latest_weights: Optional[dict[str, dict[str, Any]]] = None
        self._latest_handshake: Optional[dict] = None
        self._latest_training_info: Optional[dict] = None
        self._server_terminate_payload: Optional[dict] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, name="HWMTCPClient", daemon=True)
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

    @property
    def connected(self) -> bool:
        return self._connected

    def send_raw_batch(
        self, states: np.ndarray, actions: np.ndarray, episode_id: int, episode_end: bool = False
    ) -> bool:
        frame = pack_raw_batch(states, actions, episode_id, episode_end)
        try:
            self._send_q.put_nowait(frame)
            return True
        except queue.Full:
            self._send_drops += 1
            if self._send_drops == 1 or self._send_drops % 50 == 0:
                print(
                    f"[HWMTCPClient] Send queue full: dropped batch of {len(states)} row(s) "
                    f"(total_drops={self._send_drops})."
                )
            return False

    def send_clear_buffer(self) -> None:
        try:
            self._send_q.put_nowait(pack_simple(MSG_CLEAR_BUFFER))
        except queue.Full:
            pass

    def send_terminate(self, wait_for_ack: bool = True, ack_timeout: float = 30.0, urgent: bool = True) -> bool:
        frame = pack_simple(MSG_TERMINATE)
        if wait_for_ack:
            with self._terminate_sync_lock:
                self._waiting_for_terminate_ack = True
                self._terminate_ack_evt.clear()
        try:
            target = self._urgent_q if urgent else self._send_q
            target.put(frame, timeout=max(5.0, float(ack_timeout)))
        except queue.Full:
            if wait_for_ack:
                with self._terminate_sync_lock:
                    self._waiting_for_terminate_ack = False
            print("[HWMTCPClient] Queue full; could not send terminate")
            return False
        if not wait_for_ack:
            return True
        ok = self._terminate_ack_evt.wait(timeout=float(ack_timeout))
        with self._terminate_sync_lock:
            self._waiting_for_terminate_ack = False
        if not ok:
            print(f"[HWMTCPClient] Timeout ({ack_timeout}s) waiting for terminate_ack")
        return ok

    def pop_latest_weights(self) -> Optional[dict[str, dict[str, Any]]]:
        with self._lock:
            sds = self._latest_weights
            self._latest_weights = None
            return sds

    def pop_handshake(self) -> Optional[dict]:
        with self._lock:
            payload = self._latest_handshake
            self._latest_handshake = None
            return payload

    def pop_latest_training_info(self) -> Optional[dict]:
        with self._lock:
            payload = self._latest_training_info
            self._latest_training_info = None
            return payload

    def pop_server_terminate(self) -> Optional[dict]:
        with self._lock:
            payload = self._server_terminate_payload
            self._server_terminate_payload = None
            return payload

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
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

    async def _shutdown_async(self) -> None:
        self._stop_evt.set()
        await asyncio.sleep(0)

    async def _main(self) -> None:
        while not self._stop_evt.is_set():
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
                self._connected = True
                reader_task = asyncio.create_task(self._reader_loop(reader))
                writer_task = asyncio.create_task(self._writer_loop(writer))
                _done, pending = await asyncio.wait({reader_task, writer_task}, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                self._connected = False
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            except Exception:
                self._connected = False
                await asyncio.sleep(1.0)

    async def _reader_loop(self, reader: asyncio.StreamReader) -> None:
        while not self._stop_evt.is_set():
            try:
                msg = await read_frame(reader)
                self._handle_msg(msg)
            except (ConnectionResetError, asyncio.IncompleteReadError):
                break
            except Exception as e:
                print(f"[HWMTCPClient] Reader loop error: {e}")
                break

    def _handle_msg(self, msg: dict) -> None:
        typ = msg.get("type")
        data = msg.get("data", {})
        if not isinstance(data, dict):
            data = {}
        if typ == MSG_WEIGHTS:
            sds = unpack_weights(data)
            with self._lock:
                self._latest_weights = sds
        elif typ == MSG_HANDSHAKE:
            with self._lock:
                self._latest_handshake = data
        elif typ == MSG_TRAINING_INFO:
            with self._lock:
                self._latest_training_info = data
        elif typ == MSG_TERMINATE_ACK:
            with self._terminate_sync_lock:
                if self._waiting_for_terminate_ack:
                    self._terminate_ack_evt.set()
        elif typ == MSG_TERMINATE:
            with self._lock:
                self._server_terminate_payload = data
            self._stop_evt.set()
        elif typ == MSG_ACK:
            pass
        # other message types are ignored

    async def _writer_loop(self, writer: asyncio.StreamWriter) -> None:
        while not self._stop_evt.is_set():
            try:
                frame = self._urgent_q.get_nowait()
            except queue.Empty:
                try:
                    frame = self._send_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
            try:
                writer.write(pack_frame(frame))
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                break
            except Exception as e:
                print(f"[HWMTCPClient] Writer loop error: {e}")
                break

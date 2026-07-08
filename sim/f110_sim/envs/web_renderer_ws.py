"""WebSocket push channel for the F1TENTH web renderer."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable, Dict, Optional, Set

try:
    import websockets
    from websockets.server import WebSocketServerProtocol, serve
except ImportError:  # pragma: no cover - optional at import time
    websockets = None
    WebSocketServerProtocol = Any  # type: ignore[misc, assignment]
    serve = None


class WebRendererSocketHub:
    """Thread-safe WebSocket broadcaster running asyncio in a daemon thread."""

    def __init__(
        self,
        host: str,
        port: int,
        hello_builder: Callable[[], Dict[str, Any]],
        max_port_tries: int = 25,
    ):
        self._host = str(host)
        self._port = int(port)
        self._requested_port = int(port)
        self._max_port_tries = int(max_port_tries)
        self._hello_builder = hello_builder
        self._clients: Set[WebSocketServerProtocol] = set()
        self._clients_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._closed = False
        self._server = None
        self._shutdown_future: Optional[asyncio.Future] = None
        self._pending_message: Optional[str] = None
        self._flush_scheduled = False
        self._flush_lock = threading.Lock()

    @property
    def port(self) -> int:
        return int(self._port)

    def start(self) -> bool:
        if serve is None:
            print("Web renderer: websockets package not installed; WS push disabled.")
            return False
        if self._thread is not None:
            return True
        self._thread = threading.Thread(target=self._run_loop, name="web-renderer-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            print("Web renderer: WebSocket server failed to start within 5s.")
            return False
        return True

    def close(self) -> None:
        self._closed = True
        loop = self._loop
        if loop is None:
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            return

        async def _shutdown() -> None:
            with self._clients_lock:
                clients = list(self._clients)
            for websocket in clients:
                try:
                    await websocket.close()
                except Exception:
                    pass
            shutdown_future = self._shutdown_future
            if shutdown_future is not None and not shutdown_future.done():
                shutdown_future.set_result(None)

        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            future.result(timeout=2.0)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def has_clients(self) -> bool:
        with self._clients_lock:
            return len(self._clients) > 0

    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def broadcast_json(self, payload: Dict[str, Any]) -> None:
        """Coalesce bursts: only the latest frame is sent if the client is behind."""
        if self._loop is None or not self.has_clients():
            return
        message = json.dumps(payload, separators=(",", ":"))
        with self._flush_lock:
            self._pending_message = message
            if self._flush_scheduled:
                return
            self._flush_scheduled = True
        asyncio.run_coroutine_threadsafe(self._flush_pending(), self._loop)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve_forever())
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()
            self._loop = None

    async def _serve_forever(self) -> None:
        host = self._host
        if host in ("", "0.0.0.0", "::"):
            bind_host = "0.0.0.0"
        else:
            bind_host = host

        last_error: Optional[Exception] = None
        for attempt in range(self._max_port_tries):
            candidate_port = self._requested_port + attempt
            try:
                async with serve(
                    self._connection_handler,
                    bind_host,
                    candidate_port,
                    ping_interval=30,
                    ping_timeout=30,
                    max_size=4 * 1024 * 1024,
                    max_queue=4,
                ) as server:
                    self._port = int(candidate_port)
                    self._server = server
                    if attempt > 0:
                        print(
                            f"Web renderer: WebSocket using fallback port {candidate_port} "
                            f"(requested {self._requested_port} in use)."
                        )
                    self._ready.set()
                    self._shutdown_future = asyncio.get_running_loop().create_future()
                    try:
                        await self._shutdown_future
                    finally:
                        self._shutdown_future = None
                    return
            except OSError as exc:
                last_error = exc
                if getattr(exc, "errno", None) != 98:
                    raise
        if last_error is not None:
            raise OSError(
                f"Could not bind web renderer WebSocket on {bind_host}:{self._requested_port} "
                f"or the next {self._max_port_tries - 1} ports."
            ) from last_error

    async def _connection_handler(self, websocket: WebSocketServerProtocol) -> None:
        self._register(websocket)
        try:
            hello = self._hello_builder()
            await websocket.send(json.dumps(hello, separators=(",", ":")))
            async for _raw in websocket:
                pass
        finally:
            self._unregister(websocket)

    def _register(self, websocket: WebSocketServerProtocol) -> None:
        with self._clients_lock:
            self._clients.add(websocket)

    def _unregister(self, websocket: WebSocketServerProtocol) -> None:
        with self._clients_lock:
            self._clients.discard(websocket)

    async def _flush_pending(self) -> None:
        try:
            while True:
                with self._flush_lock:
                    message = self._pending_message
                    self._pending_message = None
                if message is None:
                    break
                await self._broadcast(message)
        finally:
            with self._flush_lock:
                more = self._pending_message is not None
                if more:
                    asyncio.create_task(self._flush_pending())
                else:
                    self._flush_scheduled = False

    async def _broadcast(self, message: str) -> None:
        with self._clients_lock:
            clients = list(self._clients)
        if not clients:
            return
        dead = []
        for websocket in clients:
            try:
                await websocket.send(message)
            except Exception:
                dead.append(websocket)
        if dead:
            with self._clients_lock:
                for websocket in dead:
                    self._clients.discard(websocket)

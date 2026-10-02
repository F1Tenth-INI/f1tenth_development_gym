#!/usr/bin/env python3
"""Launch the state comparison visualization webapp."""

import os
import socket
import sys

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(VIS_DIR, "web")
sys.path.insert(0, VIS_DIR)
sys.path.insert(0, WEB_DIR)

from browser_session import is_browser_session_active

HOST = "127.0.0.1"
PORT = 8050


def _bind_localhost(preferred: int, attempts: int = 20) -> tuple[socket.socket, int]:
    """Bind and keep the socket before announcing the URL.

    This container uses the host network. The editor binds 127.0.0.1 as soon
    as it sees a localhost URL, which takes the port out from under uvicorn
    while the app is still importing.
    """
    last_error: OSError | None = None
    for port in range(preferred, preferred + attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((HOST, port))
            sock.set_inheritable(True)
            return sock, port
        except OSError as exc:
            last_error = exc
            sock.close()
    span = f"{preferred}-{preferred + attempts - 1}"
    raise SystemExit(f"No free TCP port for the visualizer in {span}: {last_error}")


def _should_open_browser(argv: list) -> bool:
    if "--no-browser" in argv:
        return False
    if "--open-browser" in argv:
        return True
    return not is_browser_session_active()


if __name__ == "__main__":
    import uvicorn

    sock, port = _bind_localhost(PORT)
    app_url = f"http://{HOST}:{port}"
    if port != PORT:
        print(f"Port {PORT} is already in use; using {port} instead", file=sys.stderr)
    if _should_open_browser(sys.argv):
        os.environ["VIZ_OPEN_BROWSER"] = app_url
    print(f"State Comparison Visualizer running at {app_url}", flush=True)

    config = uvicorn.Config("app:app", host=HOST, port=port, reload=False)
    server = uvicorn.Server(config)
    server.run(sockets=[sock])

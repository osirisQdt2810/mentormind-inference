"""``python -m vlm_server container``: the Docker entrypoint.

``VLM_SERVER_AUTOSTART=true`` execs the server at once; otherwise the container idles and the server
is started on demand with ``docker compose exec <service> python -m vlm_server serve``.
"""

from __future__ import annotations

import signal
from collections.abc import Callable

from vlm_server.config import ServerConfig
from vlm_server.serve.launch import main as serve


def main(
    args: list[str] | None = None,
    *,
    start: Callable[[list[str]], None] = serve,
    idle: Callable[[], None] = signal.pause,
) -> None:
    config = ServerConfig()
    if config.autostart:
        print("[vlm_server] VLM_SERVER_AUTOSTART=true: starting the server", flush=True)
        start(list(args or []))
        return
    print(
        "[vlm_server] idle (VLM_SERVER_AUTOSTART=false). Start the server with:\n"
        "    docker compose exec <service> python -m vlm_server serve",
        flush=True,
    )
    # As PID 1 in a container, Python ignores SIGTERM unless it installs a handler.
    signal.signal(signal.SIGTERM, _stop)
    idle()


def _stop(signum: int, frame: object) -> None:
    # A second signal while shutting down (e.g. forwarded by ``uv run``) just ends the process.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    raise SystemExit(0)

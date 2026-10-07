"""Reverse proxy to the VLM upstream.

Method, path, query, headers (minus hop-by-hop ones and ``Host``) and body go upstream; status,
headers and body come back. The upstream body is streamed through as it arrives, so
``"stream": true`` (SSE) reaches the client chunk by chunk. Bytes are passed raw: a compressed
upstream answer keeps its ``Content-Encoding`` and ``Content-Length``.

Heartbeat: a non-streaming POST to ``/v1/chat/completions`` or ``/v1/completions`` whose answer has
not come after ``heartbeat_s`` gets ``200`` + ``application/json`` at once and then one space every
``heartbeat_s`` until the upstream answer, which follows as is. A tunnel in front (ngrok free) cuts a
response that sends nothing for ~5 minutes; a Thinking model can decode longer. JSON parsers skip
the leading spaces. The early ``200`` cannot change any more: an upstream error that comes after it
keeps its JSON body but arrives with status 200 (``X-Gateway-Heartbeat`` says the answer took this
path). Such requests ask the upstream for an uncompressed body: the early headers carry no
``Content-Encoding``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from vlm_server.gateway.errors import UpstreamUnavailable

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
#: httpx recomputes Host and Content-Length for the upstream request.
_REQUEST_DROP = HOP_BY_HOP | {"host", "content-length"}
#: The gateway's own server writes Date and Server.
_RESPONSE_DROP = HOP_BY_HOP | {"date", "server"}

RawHeaders = list[tuple[bytes, bytes]]
#: Paths whose non-streaming answers get the heartbeat.
HEARTBEAT_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})
HEARTBEAT_HEADER = "X-Gateway-Heartbeat"

log = logging.getLogger(__name__)


def filter_headers(raw: Iterable[tuple[bytes, bytes]], drop: frozenset[str]) -> RawHeaders:
    """``raw`` without ``drop`` and without the headers its ``Connection`` header names."""
    headers = list(raw)
    named = {
        token.strip().lower()
        for name, value in headers
        if name.lower() == b"connection"
        for token in value.decode("latin-1").split(",")
    }
    skip = drop | named
    return [(name, value) for name, value in headers if name.decode("latin-1").lower() not in skip]


def wants_heartbeat(method: str, path: str, body: bytes, heartbeat_s: float) -> bool:
    """A non-streaming JSON POST to a generation path, with the heartbeat on."""
    if heartbeat_s <= 0 or method != "POST" or path not in HEARTBEAT_PATHS:
        return False
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    return isinstance(payload, dict) and not payload.get("stream")


async def forward(
    request: Request, client: httpx.AsyncClient, upstream: str, *, heartbeat_s: float = 0.0
) -> Response:
    """Send ``request`` to ``upstream`` (scheme://host:port) and stream the answer back."""
    url = upstream + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    headers = filter_headers(request.headers.raw, _REQUEST_DROP)
    body = await request.body()
    heartbeat = wants_heartbeat(request.method, request.url.path, body, heartbeat_s)
    if heartbeat:  # the early 200 carries no Content-Encoding: the body must come plain
        headers = [(name, value) for name, value in headers if name.lower() != b"accept-encoding"]
    if not any(name.lower() == b"accept-encoding" for name, _ in headers):
        headers.append((b"accept-encoding", b"identity"))  # not httpx's default gzip
    outgoing = client.build_request(request.method, url, headers=headers, content=body or None)
    sending = asyncio.ensure_future(client.send(outgoing, stream=True))
    if heartbeat:
        done, _ = await asyncio.wait({sending}, timeout=heartbeat_s)
        if not done:
            return StreamingResponse(
                _with_heartbeat(sending, heartbeat_s, upstream),
                status_code=200,
                media_type="application/json",
                headers={HEARTBEAT_HEADER: f"{heartbeat_s:g}"},
            )
    try:
        incoming = await sending
    except httpx.TransportError as exc:
        raise UpstreamUnavailable(_unreachable(upstream, exc)) from exc
    response = StreamingResponse(
        incoming.aiter_raw(),
        status_code=incoming.status_code,
        background=BackgroundTask(incoming.aclose),
    )
    response.raw_headers = filter_headers(incoming.headers.raw, _RESPONSE_DROP)
    return response


async def _with_heartbeat(
    sending: asyncio.Future[httpx.Response], every: float, upstream: str
) -> AsyncIterator[bytes]:
    """Spaces while the upstream works, then its body. A client that leaves cancels the upstream."""
    try:
        while not sending.done():
            done, _ = await asyncio.wait({sending}, timeout=every)
            if not done:
                yield b" "
        try:
            incoming = sending.result()
        except httpx.TransportError as exc:
            yield json.dumps({"error": _unreachable(upstream, exc)}).encode()
            return
        if incoming.status_code >= 400:
            log.warning("VLM upstream answered %s after the heartbeat's 200", incoming.status_code)
        try:
            async for chunk in incoming.aiter_raw():
                yield chunk
        finally:
            await incoming.aclose()
    finally:
        if not sending.done():
            sending.cancel()


def _unreachable(upstream: str, exc: Exception) -> str:
    return f"VLM upstream {upstream} unreachable: {type(exc).__name__}: {exc}"

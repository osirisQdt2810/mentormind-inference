"""Reverse proxy to the VLM upstream.

Method, path, query, headers (minus hop-by-hop ones and ``Host``) and body go upstream; status,
headers and body come back. The upstream body is streamed through as it arrives, so
``"stream": true`` (SSE) reaches the client chunk by chunk. Bytes are passed raw: a compressed
upstream answer keeps its ``Content-Encoding`` and ``Content-Length``.
"""

from __future__ import annotations

from collections.abc import Iterable

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


async def forward(request: Request, client: httpx.AsyncClient, upstream: str) -> Response:
    """Send ``request`` to ``upstream`` (scheme://host:port) and stream the answer back."""
    url = upstream + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    headers = filter_headers(request.headers.raw, _REQUEST_DROP)
    if not any(name.lower() == b"accept-encoding" for name, _ in headers):
        headers.append((b"accept-encoding", b"identity"))  # not httpx's default gzip
    body = await request.body()
    outgoing = client.build_request(request.method, url, headers=headers, content=body or None)
    try:
        incoming = await client.send(outgoing, stream=True)
    except httpx.TransportError as exc:
        raise UpstreamUnavailable(
            f"VLM upstream {upstream} unreachable: {type(exc).__name__}: {exc}"
        ) from exc
    response = StreamingResponse(
        incoming.aiter_raw(),
        status_code=incoming.status_code,
        background=BackgroundTask(incoming.aclose),
    )
    response.raw_headers = filter_headers(incoming.headers.raw, _RESPONSE_DROP)
    return response

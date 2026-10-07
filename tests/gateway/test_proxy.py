"""Every other /v1/* path goes to the VLM: request and answer passed through, SSE streamed."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi.testclient import TestClient

from tests.gateway.fakes import FakeEmbedder, make_config
from vlm_server.gateway.app import create_app

UPSTREAM = "http://vlm.test:18000"
Handler = Callable[[httpx.Request], httpx.Response]


class _Body(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


def upstream(status: int, *chunks: bytes, **headers: str) -> httpx.Response:
    """An answer whose body is still unread, as from the network (``content=`` is pre-read).

    Header names come as keywords: ``content_type=`` is Content-Type.
    """
    named = {name.replace("_", "-"): value for name, value in headers.items()}
    return httpx.Response(status, headers=named, stream=_Body(*chunks))


def upstream_json(status: int, body: Any) -> httpx.Response:
    return upstream(status, json.dumps(body).encode(), content_type="application/json")


def proxy_client(handler: Handler, **config: Any) -> TestClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(make_config(vlm_upstream=UPSTREAM, **config), http=http)
    return TestClient(app)


def test_health_reports_the_upstream_and_each_service() -> None:
    client = proxy_client(lambda request: httpx.Response(500))
    assert client.get("/health").json() == {
        "status": "ok",
        "vlm_upstream": UPSTREAM,
        "services": {"asr": "small", "embeddings": "BAAI/bge-m3", "documents": "docling"},
    }


def test_request_is_forwarded_with_method_query_body_and_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return upstream_json(200, {"id": "chatcmpl-1"})

    payload = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = proxy_client(handler).post(
        "/v1/chat/completions?trace=1",
        content=json.dumps(payload),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer token",
            "X-Request-Id": "abc",
            "Keep-Alive": "timeout=5",
            "TE": "trailers",
            "Proxy-Authorization": "Basic x",
            "Connection": "x-hop",
            "X-Hop": "1",
        },
    )
    assert response.status_code == 200 and response.json() == {"id": "chatcmpl-1"}
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == f"{UPSTREAM}/v1/chat/completions?trace=1"
    assert json.loads(request.content) == payload
    assert request.headers["host"] == "vlm.test:18000"  # the client's Host is not forwarded
    assert request.headers["authorization"] == "Bearer token"
    assert request.headers["x-request-id"] == "abc"
    assert request.headers["content-length"] == str(len(request.content))
    for hop in ("keep-alive", "te", "proxy-authorization", "x-hop"):
        assert hop not in request.headers


def test_upstream_status_headers_and_body_come_back() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return upstream(
            404,
            b'{"error":{"message":"The model `x` does not exist."}}',
            content_type="application/json",
            x_upstream="vllm",
        )

    response = proxy_client(handler).get("/v1/models/x")
    assert response.status_code == 404
    assert response.headers["x-upstream"] == "vllm"
    assert response.json() == {"error": {"message": "The model `x` does not exist."}}


def test_models_and_other_methods_on_known_paths_are_proxied() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return upstream_json(200, {"object": "list", "data": []})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(make_config(vlm_upstream=UPSTREAM), embedder=FakeEmbedder({}), http=http)
    client = TestClient(app)
    assert client.get("/v1/models").json() == {"object": "list", "data": []}
    assert client.get("/v1/embeddings").status_code == 200  # only POST is served locally
    assert seen == ["GET /v1/models", "GET /v1/embeddings"]


def test_streaming_answer_is_passed_through() -> None:
    events = [
        b'data: {"choices":[{"delta":{"content":"Bu"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"oc 1"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return upstream(200, *events, content_type="text/event-stream")

    client = proxy_client(handler)
    with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        assert b"".join(response.iter_bytes()) == b"".join(events)


def test_upstream_down_is_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    response = proxy_client(handler).get("/v1/models")
    assert response.status_code == 502
    assert "unreachable" in response.json()["error"] and UPSTREAM in response.json()["error"]


def test_upstream_down_is_502_with_the_real_client() -> None:
    with socket.socket() as probe:  # a port nobody listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    upstream = f"http://127.0.0.1:{port}"
    with TestClient(create_app(make_config(vlm_upstream=upstream))) as client:
        response = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert response.status_code == 502 and upstream in response.json()["error"]


def test_unknown_non_v1_path_is_json_404() -> None:
    response = proxy_client(lambda request: httpx.Response(200)).get("/nope")
    assert response.status_code == 404 and response.json() == {"error": "Not Found"}


# --- heartbeat: slow non-streaming answers keep a tunnel open -----------------------------------

ANSWER = {"id": "chatcmpl-1", "choices": [{"message": {"content": "{}"}}]}


def slow(seconds: float, status: int = 200, body: Any = ANSWER) -> Callable[[httpx.Request], Any]:
    """An upstream that answers after ``seconds`` (a fresh body for every request)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(seconds)
        return upstream_json(status, body)

    return handler


def chat(client: TestClient, **body: Any) -> httpx.Response:
    return client.post("/v1/chat/completions", json={"model": "m", "messages": [], **body})


def test_slow_answer_gets_200_then_spaces_then_the_upstream_json() -> None:
    client = proxy_client(slow(0.35), heartbeat_s=0.1)
    res = chat(client)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/json")
    assert res.headers["x-gateway-heartbeat"] == "0.1"
    assert res.content.startswith(b" ") and res.content.lstrip(b" ") == json.dumps(ANSWER).encode()
    assert res.json() == ANSWER  # leading spaces are valid JSON whitespace


def test_fast_answer_is_passed_through_untouched() -> None:
    client = proxy_client(lambda request: upstream_json(400, {"error": "bad"}), heartbeat_s=5)
    res = chat(client)
    assert res.status_code == 400 and res.json() == {"error": "bad"}
    assert "x-gateway-heartbeat" not in res.headers


def test_streaming_requests_and_other_paths_get_no_heartbeat() -> None:
    client = proxy_client(slow(0.3), heartbeat_s=0.05)
    res = chat(client, stream=True)
    assert "x-gateway-heartbeat" not in res.headers and res.content == json.dumps(ANSWER).encode()
    res = client.get("/v1/models")
    assert "x-gateway-heartbeat" not in res.headers


def test_heartbeat_zero_turns_it_off() -> None:
    client = proxy_client(slow(0.2), heartbeat_s=0)
    res = chat(client)
    assert "x-gateway-heartbeat" not in res.headers and res.content == json.dumps(ANSWER).encode()


def test_heartbeat_requests_an_uncompressed_body() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        await asyncio.sleep(0.25)
        return upstream_json(200, ANSWER)

    client = proxy_client(handler, heartbeat_s=0.1)
    res = client.post(
        "/v1/chat/completions", json={"messages": []}, headers={"Accept-Encoding": "gzip"}
    )
    assert res.json() == ANSWER
    assert seen[0].headers["accept-encoding"] == "identity"


def test_upstream_lost_during_the_heartbeat_ends_with_an_error_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.25)
        raise httpx.ConnectError("refused", request=request)

    client = proxy_client(handler, heartbeat_s=0.1)
    res = chat(client)
    assert res.status_code == 200
    assert "unreachable" in res.json()["error"]

"""POST /v1/embeddings: bge-m3 [CLS] vectors, L2-normalised, in input order (torch faked)."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.gateway.fakes import FakeEmbedder, make_config
from vlm_server.gateway.app import create_app
from vlm_server.gateway.embeddings import BgeM3Embedder, l2_normalise, load_forward

URL = "/v1/embeddings"
VECTORS = {"a": [3.0, 4.0], "b": [0.0, 0.0], "c": [1.0, 2.0, 2.0], "xin chào": [2.0, 0.0]}


def client_with(embedder: Any) -> TestClient:
    return TestClient(create_app(make_config(), embedder=embedder))


def test_string_input_gives_one_normalised_vector() -> None:
    response = client_with(FakeEmbedder(VECTORS)).post(URL, json={"input": "xin chào"})
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "model": "BAAI/bge-m3",
        "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }


def test_list_input_keeps_order_and_normalises_each_vector() -> None:
    embedder = FakeEmbedder(VECTORS)
    body = {"model": "BAAI/bge-m3", "input": ["c", "a", "b"]}
    data = client_with(embedder).post(URL, json=body).json()["data"]
    assert embedder.calls == [["c", "a", "b"]]
    assert [item["index"] for item in data] == [0, 1, 2]
    assert data[0]["embedding"] == [1 / 3, 2 / 3, 2 / 3]
    assert data[1]["embedding"] == [0.6, 0.8]
    assert data[2]["embedding"] == [0.0, 0.0]  # a zero vector stays zero


def test_normalisation_is_mentorminds_arithmetic() -> None:
    raw = [0.123456789, -1.5, 2.25, 1e-3]
    norm = math.sqrt(sum(v * v for v in raw))
    assert l2_normalise(raw) == [v / norm for v in raw]
    assert math.isclose(sum(v * v for v in l2_normalise(raw)), 1.0)


def test_usage_counts_tokens_and_model_comes_from_config() -> None:
    app = create_app(make_config(embed_model="BAAI/bge-m3-local"), embedder=FakeEmbedder(VECTORS))
    body = TestClient(app).post(URL, json={"input": ["a", "b"]}).json()
    assert body["model"] == "BAAI/bge-m3-local"
    assert body["usage"] == {"prompt_tokens": 4, "total_tokens": 4}


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"input": []}, 400),
        ({}, 422),
        ({"input": [1, 2]}, 422),
        ({"input": "a", "encoding_format": "base64"}, 422),
    ],
)
def test_bad_input_is_rejected(body: dict[str, Any], status: int) -> None:
    response = client_with(FakeEmbedder(VECTORS)).post(URL, json=body)
    assert response.status_code == status and response.json()["error"]


def test_engine_batches_in_order_and_loads_once() -> None:
    batches: list[list[str]] = []
    loads: list[str] = []

    def loader(name: str) -> Any:
        loads.append(name)

        def forward(texts: list[str]) -> tuple[list[list[float]], int]:
            batches.append(texts)
            return [[float(t)] for t in texts], len(texts)

        return forward

    embedder = BgeM3Embedder("BAAI/bge-m3", batch_size=16, loader=loader)
    texts = [str(i) for i in range(35)]
    vectors, tokens = embedder.encode(texts)
    embedder.encode(["1"])
    assert [len(b) for b in batches[:3]] == [16, 16, 3]
    assert vectors == [[float(i)] for i in range(35)] and tokens == 35
    assert loads == ["BAAI/bge-m3"]


class _Tensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __getitem__(self, key: Any) -> _Tensor:
        assert key == (slice(None), 0)  # every row, token 0 = [CLS]
        return _Tensor([row[0] for row in self.data])

    def tolist(self) -> Any:
        return self.data

    def sum(self) -> int:
        return sum(sum(row) for row in self.data)


def test_forward_is_the_cls_row_with_mentorminds_tokenizer_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def tokenizer(texts: list[str], **options: Any) -> dict[str, Any]:
        seen["options"] = options
        return {"input_ids": "ids", "attention_mask": _Tensor([[1, 1, 1], [1, 1, 0]])}

    class Model:
        def eval(self) -> Model:
            seen["eval"] = True
            return self

        def __call__(self, **batch: Any) -> Any:
            seen["no_grad"] = torch.grad_disabled
            hidden = [[[0.5, 1.0], [9.0, 9.0]], [[-2.0, 0.25], [9.0, 9.0]]]
            return SimpleNamespace(last_hidden_state=_Tensor(hidden))

    class NoGrad:
        def __enter__(self) -> None:
            torch.grad_disabled = True

        def __exit__(self, *exc: object) -> None:
            torch.grad_disabled = False

    torch: Any = ModuleType("torch")
    torch.no_grad = NoGrad
    torch.grad_disabled = False
    transformers: Any = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: tokenizer)
    transformers.AutoModel = SimpleNamespace(from_pretrained=lambda name: Model())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    vectors, tokens = load_forward("BAAI/bge-m3")(["first", "second"])
    assert vectors == [[0.5, 1.0], [-2.0, 0.25]] and tokens == 5
    assert seen["options"] == {
        "padding": True,
        "truncation": True,
        "max_length": 1024,
        "return_tensors": "pt",
    }
    assert seen["eval"] is True and seen["no_grad"] is True


def test_missing_torch_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    response = TestClient(create_app(make_config())).post(URL, json={"input": "a"})
    assert response.status_code == 503 and "torch" in response.json()["error"]

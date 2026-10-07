"""Dense embeddings: bge-m3 [CLS] vectors, computed exactly like MentorMind's local embedder.

``AutoTokenizer`` + ``AutoModel(...).eval()``, ``max_length=1024`` truncation, the [CLS] row of
``last_hidden_state`` under ``torch.no_grad()``, then ``l2_normalise`` in Python: the same steps
and arithmetic as ``mentormind.knowledge_base.store.embedder``, so a remote vector equals the local
one. ``torch``/``transformers`` are imported when the model is first loaded.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel

from vlm_server.gateway.deps import install_hint
from vlm_server.gateway.errors import EngineUnavailable

DEFAULT_MODEL = "BAAI/bge-m3"
MAX_LENGTH = 1024


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    #: Vectors are always plain JSON floats; base64 (the OpenAI SDK's implicit default) is refused
    #: rather than answered in a shape the client would misread.
    encoding_format: Literal["float"] | None = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingItem]
    usage: Usage


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Raw (unnormalised) vectors in input order, and the number of tokens read."""
        ...


#: One batch of texts -> (raw [CLS] vectors, tokens read). The only part that touches torch.
Forward = Callable[[list[str]], tuple[list[list[float]], int]]


def l2_normalise(vector: list[float]) -> list[float]:
    """Unit length; a zero vector stays as it is (MentorMind's ``_normalise``)."""
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def load_forward(model_name: str) -> Forward:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise EngineUnavailable(f"torch/transformers are not installed: {install_hint()}") from exc
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModel.from_pretrained(model_name).eval()
    except (OSError, ValueError) as exc:
        raise EngineUnavailable(f"could not load embedding model {model_name!r}: {exc}") from exc

    def forward(texts: list[str]) -> tuple[list[list[float]], int]:
        batch = tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        vectors = [[float(x) for x in row] for row in hidden[:, 0].tolist()]  # dense = [CLS]
        return vectors, int(batch["attention_mask"].sum())

    return forward


class BgeM3Embedder:
    """The model loads on the first call; ``batch_size`` texts per forward pass, serialised."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 16,
        loader: Callable[[str], Forward] = load_forward,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_name = model_name
        self.batch_size = batch_size
        self._loader = loader
        self._forward: Forward | None = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]:
        if not texts:
            return [], 0
        vectors: list[list[float]] = []
        tokens = 0
        with self._lock:
            if self._forward is None:
                self._forward = self._loader(self.model_name)
            for start in range(0, len(texts), self.batch_size):
                batch_vectors, batch_tokens = self._forward(texts[start : start + self.batch_size])
                vectors.extend(batch_vectors)
                tokens += batch_tokens
        return vectors, tokens

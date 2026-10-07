"""Fakes for the gateway tests: no faster-whisper, torch, transformers or docling needed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vlm_server.gateway.config import GatewayConfig
from vlm_server.gateway.errors import ConversionFailed


def make_config(**values: Any) -> GatewayConfig:
    return GatewayConfig(_env_file=None, **values)  # type: ignore[call-arg]


def word(text: str, start: float, end: float, probability: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end, probability=probability)


def segment(seg_id: int, text: str, start: float, end: float, words: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(id=seg_id, text=text, start=start, end=end, words=words)


#: What faster-whisper yields for a short Thai clip: raw word text, Thai words joined without a
#: space, one start slightly below 0 (VAD padding).
THAI_SEGMENTS = [
    segment(
        1,
        " สวัสดีครับ",
        -0.02,
        1.4,
        [word(" สวัสดี", -0.02, 0.8, 0.91), word("ครับ", 0.8, 1.4, 0.88)],
    ),
    segment(
        2,
        " ขันน็อต",
        2.0,
        3.1,
        [word(" ขัน", 2.0, 2.5, 0.75), word("น็อต", 2.5, 3.1, 0.66)],
    ),
]
THAI_INFO = SimpleNamespace(language="th", duration=3.5)


class FakeWhisperModel:
    """Records ``transcribe`` calls; returns a lazy generator like faster-whisper does."""

    def __init__(self, segments: list[Any] = THAI_SEGMENTS, info: Any = THAI_INFO) -> None:
        self.segments = segments
        self.info = info
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: str, **options: Any) -> tuple[Any, Any]:
        self.calls.append({"audio": Path(audio).name, "bytes": Path(audio).read_bytes(), **options})
        return (piece for piece in self.segments), self.info


class FakeEmbedder:
    """Raw (unnormalised) vectors by text; 2 tokens per text."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]:
        self.calls.append(list(texts))
        return [list(self.vectors[text]) for text in texts], 2 * len(texts)


class FakeConverter:
    def __init__(self, result: tuple[int, dict[str, Any]] | None = None, error: str = "") -> None:
        self.result = result or (2, {"schema_name": "DoclingDocument", "texts": []})
        self.error = error
        self.calls: list[tuple[str, bytes]] = []

    def convert(self, path: Path) -> tuple[int, dict[str, Any]]:
        self.calls.append((path.name, path.read_bytes()))
        if self.error:
            raise ConversionFailed(self.error)
        return self.result

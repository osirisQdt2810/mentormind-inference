"""Speech to text: faster-whisper with word timestamps, answered in OpenAI's ``verbose_json`` shape.

Decodes like MentorMind's local ``faster_whisper`` provider (``word_timestamps=True``, beam size and
VAD from the config), clamps negative times to 0 and keeps each word's text RAW: Whisper's leading
space is how words are joined, and Thai words join without one. ``faster_whisper`` is imported when
the model is first loaded.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from vlm_server.gateway.config import GatewayConfig
from vlm_server.gateway.deps import install_hint
from vlm_server.gateway.errors import BadRequest, EngineUnavailable


class Word(BaseModel):
    #: Raw token text exactly as faster-whisper returns it (leading space kept).
    word: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word]


class Transcription(BaseModel):
    text: str
    #: ISO 639-1 code as faster-whisper reports it (e.g. "th").
    language: str
    duration: float
    segments: list[Segment]
    #: Every segment's words, flattened in order.
    words: list[Word]


class AsrEngine(Protocol):
    def transcribe(self, audio: Path, *, language: str | None) -> Transcription:
        """Transcribe ``audio``; ``language`` None = auto-detect."""
        ...


#: ``(model, device, compute_type) -> faster_whisper.WhisperModel``.
WhisperLoader = Callable[[str, str, str], Any]


def load_whisper(model: str, device: str, compute_type: str) -> Any:
    """A faster-whisper ``WhisperModel`` (the first load downloads it into the HF cache)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise EngineUnavailable(f"faster-whisper is not installed: {install_hint()}") from exc
    try:
        return WhisperModel(model, device=device, compute_type=compute_type)
    except (OSError, RuntimeError, ValueError) as exc:
        raise EngineUnavailable(
            f"could not load Whisper model {model!r} ({device}, {compute_type}): {exc}"
        ) from exc


def build_transcription(pieces: Iterable[Any], info: Any) -> Transcription:
    """faster-whisper ``(segments, info)`` -> the response. ``pieces`` may be its lazy generator."""
    segments: list[Segment] = []
    raw_text: list[str] = []
    for piece in pieces:
        raw_text.append(str(piece.text))
        segments.append(
            Segment(
                id=int(piece.id),
                start=_time(piece.start),
                end=_time(piece.end),
                text=str(piece.text).strip(),
                words=[_word(w) for w in piece.words or ()],
            )
        )
    return Transcription(
        text="".join(raw_text).strip(),
        language=str(info.language),
        duration=float(info.duration),
        segments=segments,
        words=[word for segment in segments for word in segment.words],
    )


def _word(word: Any) -> Word:
    return Word(
        word=str(word.word),
        start=_time(word.start),
        end=_time(word.end),
        probability=float(word.probability),
    )


def _time(value: Any) -> float:
    return max(0.0, float(value))


class FasterWhisperAsr:
    """One ``WhisperModel``, loaded on the first call and kept; calls are serialised."""

    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
        loader: WhisperLoader = load_whisper,
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._loader = loader
        self._engine: Any | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls, config: GatewayConfig, *, loader: WhisperLoader = load_whisper
    ) -> FasterWhisperAsr:
        return cls(
            config.asr_model,
            device=config.asr_device,
            compute_type=config.asr_compute_type,
            beam_size=config.asr_beam_size,
            vad_filter=config.asr_vad,
            loader=loader,
        )

    def transcribe(self, audio: Path, *, language: str | None) -> Transcription:
        with self._lock:
            if self._engine is None:
                self._engine = self._loader(self.model, self.device, self.compute_type)
            try:
                pieces, info = self._engine.transcribe(
                    str(audio),
                    language=language,
                    beam_size=self.beam_size,
                    word_timestamps=True,
                    vad_filter=self.vad_filter,
                )
                # A generator: decoding happens while iterating, so it stays under the lock.
                return build_transcription(pieces, info)
            except (OSError, ValueError) as exc:  # undecodable audio, unknown language code
                raise BadRequest(f"could not transcribe {audio.name}: {exc}") from exc

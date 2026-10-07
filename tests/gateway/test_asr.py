"""POST /v1/audio/transcriptions: verbose_json with raw word text, faster-whisper faked."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.gateway.fakes import THAI_INFO, FakeWhisperModel, make_config, segment, word
from vlm_server.gateway.app import create_app
from vlm_server.gateway.asr import FasterWhisperAsr, build_transcription

URL = "/v1/audio/transcriptions"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake audio bytes"


def client_with(model: FakeWhisperModel, **config: Any) -> tuple[TestClient, list[tuple]]:
    loads: list[tuple] = []

    def loader(name: str, device: str, compute_type: str) -> FakeWhisperModel:
        loads.append((name, device, compute_type))
        return model

    cfg = make_config(**config)
    asr = FasterWhisperAsr.from_config(cfg, loader=loader)
    return TestClient(create_app(cfg, asr=asr)), loads


def post(client: TestClient, data: dict[str, Any] | None = None, content: bytes = WAV) -> Any:
    return client.post(URL, files={"file": ("clip.wav", content, "audio/wav")}, data=data or {})


def test_response_has_the_verbose_json_shape_with_words() -> None:
    model = FakeWhisperModel()
    client, _ = client_with(model)
    response = post(client, {"language": "th", "response_format": "verbose_json"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"text", "language", "duration", "segments", "words"}
    assert body["language"] == "th" and body["duration"] == 3.5
    assert body["text"] == "สวัสดีครับ ขันน็อต"
    first = body["segments"][0]
    assert set(first) == {"id", "start", "end", "text", "words"}
    assert first["id"] == 1 and first["text"] == "สวัสดีครับ"  # segment text is stripped
    assert first["words"][0] == {"word": " สวัสดี", "start": 0.0, "end": 0.8, "probability": 0.91}
    assert [w["word"] for w in body["words"]] == [" สวัสดี", "ครับ", " ขัน", "น็อต"]
    assert body["words"] == [w for s in body["segments"] for w in s["words"]]


def test_word_text_is_raw_so_thai_joins_without_spaces() -> None:
    client, _ = client_with(FakeWhisperModel())
    words = post(client).json()["words"]
    assert words[0]["word"].startswith(" ")  # leading space kept, not stripped
    assert "".join(w["word"] for w in words[:2]).strip() == "สวัสดีครับ"


def test_negative_times_are_clamped_to_zero() -> None:
    result = build_transcription(
        [segment(1, " a", -0.5, -0.1, [word(" a", -0.5, -0.1)])],
        THAI_INFO,
    )
    assert (result.segments[0].start, result.segments[0].end) == (0.0, 0.0)
    assert (result.words[0].start, result.words[0].end) == (0.0, 0.0)


def test_decoding_uses_word_timestamps_beam_and_vad_from_config() -> None:
    model = FakeWhisperModel()
    client, loads = client_with(model, asr_model="medium", asr_beam_size=3, asr_vad=False)
    post(client, {"language": "th"})
    post(client, {"language": "th"})
    assert loads == [("medium", "cpu", "int8")]  # loaded once, then kept
    call = model.calls[0]
    assert call["word_timestamps"] is True and call["beam_size"] == 3
    assert call["vad_filter"] is False and call["language"] == "th"
    assert call["audio"] == "clip.wav" and call["bytes"] == WAV


@pytest.mark.parametrize("data", [{}, {"language": ""}, {"language": "auto"}, {"language": " "}])
def test_missing_empty_or_auto_language_means_detect(data: dict[str, str]) -> None:
    model = FakeWhisperModel()
    client, _ = client_with(model)
    assert post(client, data).status_code == 200
    assert model.calls[0]["language"] is None


def test_model_and_timestamp_granularities_are_accepted() -> None:
    client, _ = client_with(FakeWhisperModel())
    data = {"model": "whisper-1", "timestamp_granularities[]": ["word", "segment"]}
    assert post(client, data).status_code == 200


def test_only_verbose_json_is_supported() -> None:
    client, _ = client_with(FakeWhisperModel())
    response = post(client, {"response_format": "srt"})
    assert response.status_code == 400 and "verbose_json" in response.json()["error"]


def test_missing_file_is_rejected() -> None:
    client, _ = client_with(FakeWhisperModel())
    response = client.post(URL, data={"language": "th"})
    assert response.status_code == 422 and "file" in response.json()["error"]


def test_empty_file_is_rejected() -> None:
    client, _ = client_with(FakeWhisperModel())
    response = post(client, content=b"")
    assert response.status_code == 400 and "empty" in response.json()["error"]


def test_undecodable_audio_is_bad_input() -> None:
    class Broken(FakeWhisperModel):
        def transcribe(self, audio: str, **options: Any) -> tuple[Any, Any]:
            raise ValueError("Invalid data found when processing input")

    client, _ = client_with(Broken())
    response = post(client)
    assert response.status_code == 400 and "Invalid data" in response.json()["error"]


def test_missing_faster_whisper_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    client = TestClient(create_app(make_config()))
    response = post(client)
    assert response.status_code == 503 and "faster-whisper" in response.json()["error"]

"""Gateway settings from ``INFERENCE_*`` variables (process environment, then ``<repo>/.env``)."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vlm_server.config import SERVER_DIR


class GatewayConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=SERVER_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Loopback by default: the public side is Caddy (token check) + ngrok in front of this port.
    host: str = "127.0.0.1"
    port: int = 18080
    #: The VLM every other ``/v1/*`` path is forwarded to (vLLM via ``python -m vlm_server serve``).
    vlm_upstream: str = "http://127.0.0.1:18000"
    #: One VLM request may decode for minutes; the proxy waits this long.
    proxy_timeout_s: float = Field(default=1800.0, gt=0)
    #: A non-streaming ``/v1/chat/completions`` answer slower than this many seconds gets its headers
    #: (200, JSON) at once and a space every this many seconds until the JSON comes: a tunnel cuts a
    #: response that sends nothing for minutes (ngrok free: 503 after ~5 min; a Thinking model or a
    #: long merge can decode longer). JSON allows the leading spaces. 0 = off.
    heartbeat_s: float = Field(default=15.0, ge=0)

    #: faster-whisper model size or Hugging Face id ("small" = Systran/faster-whisper-small).
    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_beam_size: int = Field(default=5, ge=1)
    asr_vad: bool = True

    #: Must stay the model of MentorMind's local embedder: vectors are compared with stored ones.
    embed_model: str = "BAAI/bge-m3"
    embed_batch: int = Field(default=16, ge=1)

    @field_validator("vlm_upstream")
    @classmethod
    def _no_trailing_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

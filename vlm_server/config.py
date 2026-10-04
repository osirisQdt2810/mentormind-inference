"""Server configuration from ``VLM_SERVER_*`` variables.

Sources, later wins: ``<vlm-engine>/.env``, the host repo's ``.env`` (``VLM_SERVER_SECRETS_FILE``
overrides that path; the default assumes vlm-engine sits in its ``3rdparty/vlm_server``), then the
process environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVER_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.environ.get("VLM_SERVER_SECRETS_FILE", SERVER_DIR.parent.parent / ".env"))

#: 24 GB cards (A5000/3090/4090/L4) fit 16k tokens of KV cache next to the bf16 weights; bigger
#: cards (an MI250 GCD has 64 GB, A100/H100 40-80 GB) get 32k.
SMALL_GPU_GIB = 32.0
SMALL_GPU_MAX_MODEL_LEN = 16384
LARGE_GPU_MAX_MODEL_LEN = 32768


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLM_SERVER_",
        env_file=(SERVER_DIR / ".env", ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Container mode (``python -m vlm_server container``): start the server right away, or keep
    #: the container idle so an operator starts it on demand with ``python -m vlm_server serve``.
    autostart: bool = False
    #: Hugging Face model id (or local path) to serve.
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    #: Names clients may use in ``"model"``; the first is the canonical one.
    served_model_names: list[str] = Field(
        default_factory=lambda: ["Qwen/Qwen3-VL-8B-Instruct", "qwen3-vl-8b"]
    )
    #: GPU vendor stack: NVIDIA (CUDA) or AMD (ROCm, e.g. MI250). "auto" = whichever tool exists.
    platform: Literal["auto", "cuda", "rocm"] = "auto"
    #: Loopback by default: reach it from another machine through an SSH tunnel.
    host: str = "127.0.0.1"
    #: 8000 is the FastAPI backend (spec 08); the VLM server lives next to it.
    port: int = 8100
    #: ONE GPU index (an MI250 GCD counts as one GPU), or "auto" = the one with most free memory.
    gpu: str = "auto"
    #: "auto" refuses GPUs with less free memory than this.
    min_free_gib: float = 21.0
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    #: Empty = by GPU size. Measured on a 24 GB A5000 at 0.90: bf16 weights take 16.6 GiB,
    #: leaving 2.3 GiB of KV cache = ~16.5k tokens (32k does not start); 48 frames at 448x252
    #: cost ~5.5k tokens, so 16k fits one worst-case request with room for two more.
    max_model_len: int | None = None
    max_num_seqs: int = 4
    #: Per-request multimodal caps; frames are sent as images (spec 04: KNOWHOW_VLM_MAX_FRAMES=48).
    limit_mm_per_prompt: dict[str, int] = Field(default_factory=lambda: {"image": 64, "video": 0})
    #: Upper bound of one image's pixels (<= ~400 tokens each): a client sending 4K frames cannot
    #: blow the context. Clients send 448 px frames (~200k px), so nothing is lost.
    max_pixels: int = 640 * 640
    #: Optional bearer token clients must send. Handed to vLLM as ``VLLM_API_KEY`` (never on the
    #: command line, which every user of a shared box can read with ``ps``).
    api_key: SecretStr | None = None
    #: Anything else to pass to ``vllm serve`` verbatim.
    extra_args: list[str] = Field(default_factory=list)
    #: Extra environment for vLLM (e.g. {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"} on ROCm).
    extra_env: dict[str, str] = Field(default_factory=dict)

    @field_validator("gpu")
    @classmethod
    def _one_gpu_only(cls, value: str) -> str:
        value = value.strip()
        if value == "auto" or value.isdigit():
            return value
        raise ValueError(
            f"VLM_SERVER_GPU={value!r}: exactly one GPU index (e.g. 3) or 'auto' — "
            "this server never spans several GPUs."
        )

    def mm_processor_kwargs(self) -> dict[str, Any]:
        return {"max_pixels": self.max_pixels}

    def max_model_len_for(self, total_gib: float) -> int:
        """The configured context length, or the default for a GPU of ``total_gib``."""
        if self.max_model_len is not None:
            return self.max_model_len
        return SMALL_GPU_MAX_MODEL_LEN if total_gib <= SMALL_GPU_GIB else LARGE_GPU_MAX_MODEL_LEN

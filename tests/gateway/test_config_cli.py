"""INFERENCE_* settings and ``python -m vlm_server gateway``."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest
from fastapi import FastAPI

from tests.gateway.fakes import make_config
from vlm_server.__main__ import main


def test_defaults_match_the_contract() -> None:
    cfg = make_config()
    assert (cfg.host, cfg.port) == ("127.0.0.1", 18080)
    assert cfg.vlm_upstream == "http://127.0.0.1:18000" and cfg.proxy_timeout_s == 1800
    assert (cfg.asr_model, cfg.asr_device, cfg.asr_compute_type) == ("small", "cpu", "int8")
    assert cfg.asr_beam_size == 5 and cfg.asr_vad is True
    assert cfg.embed_model == "BAAI/bge-m3" and cfg.embed_batch == 16


def test_inference_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_PORT", "18181")
    monkeypatch.setenv("INFERENCE_VLM_UPSTREAM", "http://10.0.0.2:8000/")
    monkeypatch.setenv("INFERENCE_ASR_VAD", "false")
    monkeypatch.setenv("INFERENCE_EMBED_BATCH", "4")
    cfg = make_config()
    assert cfg.port == 18181 and cfg.vlm_upstream == "http://10.0.0.2:8000"
    assert cfg.asr_vad is False and cfg.embed_batch == 4


def test_gateway_command_runs_uvicorn_on_the_configured_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    uvicorn: Any = ModuleType("uvicorn")
    uvicorn.run = lambda app, **options: calls.update(app=app, **options)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setenv("INFERENCE_HOST", "0.0.0.0")
    monkeypatch.setenv("INFERENCE_PORT", "18181")
    main(["gateway"])
    assert calls["host"] == "0.0.0.0" and calls["port"] == 18181
    assert isinstance(calls["app"], FastAPI)

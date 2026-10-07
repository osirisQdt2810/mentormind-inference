"""Gateway tests run on fakes only; the developer's INFERENCE_* variables must not leak in."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _no_inference_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("INFERENCE_"):
            monkeypatch.delenv(name)

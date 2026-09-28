"""Docker entrypoint: VLM_SERVER_AUTOSTART decides between serving now and idling."""

from __future__ import annotations

import pytest

from vlm_server.serve import container


def run(monkeypatch: pytest.MonkeyPatch, autostart: str) -> list[str]:
    monkeypatch.setenv("VLM_SERVER_AUTOSTART", autostart)
    calls: list[str] = []
    container.main([], start=lambda args: calls.append("serve"), idle=lambda: calls.append("idle"))
    return calls


@pytest.mark.parametrize("value", ["true", "1", "yes"])
def test_autostart_serves(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert run(monkeypatch, value) == ["serve"]


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_default_is_idle(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert run(monkeypatch, value) == ["idle"]

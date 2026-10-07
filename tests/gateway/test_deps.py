"""The "not installed" errors name this platform's requirements file and its install command."""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from tests.deploy.shell import REPO
from vlm_server.__main__ import main
from vlm_server.gateway import asr, documents, embeddings
from vlm_server.gateway.deps import GATEWAY_REQUIREMENTS, install_hint
from vlm_server.gateway.errors import EngineUnavailable


def test_linux_gets_the_cpu_file_and_macos_its_own() -> None:
    assert install_hint("Linux") == (
        "uv pip install -r requirements/gateway-linux-cpu.txt --torch-backend=cpu"
    )
    assert install_hint("Darwin") == "uv pip install -r requirements/gateway-macos.txt"
    assert "requirements/gateway-*.txt" in install_hint("Windows")


def test_every_named_file_exists() -> None:
    for path, _ in GATEWAY_REQUIREMENTS.values():
        assert (REPO / path).is_file(), path


@pytest.mark.parametrize(
    ("module", "load"),
    [
        ("faster_whisper", lambda: asr.load_whisper("small", "cpu", "int8")),
        ("docling.document_converter", documents.load_converter),
        ("transformers", lambda: embeddings.load_forward("BAAI/bge-m3")),
    ],
)
def test_engine_errors_carry_the_install_hint(
    monkeypatch: pytest.MonkeyPatch, module: str, load: Callable[[], object]
) -> None:
    monkeypatch.setitem(sys.modules, module, None)  # import -> ImportError
    with pytest.raises(EngineUnavailable) as caught:
        load()
    assert caught.value.status_code == 503
    assert install_hint() in caught.value.message and "requirements/gateway-" in install_hint()


def test_the_gateway_command_without_uvicorn_names_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(SystemExit) as caught:
        main(["gateway"])
    assert install_hint() in str(caught.value)

"""The Ollama backend: VLM_SERVER_* settings become scripts/serve-ollama.sh flags."""

from __future__ import annotations

from vlm_server.config import ServerConfig
from vlm_server.serve.ollama import SCRIPT, build_ollama_command


def config(**values: object) -> ServerConfig:
    return ServerConfig(_env_file=None, **values)  # type: ignore[arg-type]


def flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_ollama_is_the_default_backend_on_its_usual_port() -> None:
    cfg = config()
    assert cfg.backend == "ollama" and cfg.serve_port == 11434
    assert config(backend="vllm").serve_port == 8100


def test_the_settings_reach_the_script() -> None:
    argv = build_ollama_command(
        config(image_min_tokens=512, context_length=65536, gpu="2", port=11440)
    )
    assert argv[:2] == ["bash", str(SCRIPT)] and SCRIPT.is_file()
    assert flag(argv, "--image-min-tokens") == "512" and flag(argv, "--context") == "65536"
    assert flag(argv, "--gpu") == "2" and flag(argv, "--port") == "11440"
    assert flag(argv, "--model") == "qwen3-vl:8b" and flag(argv, "--version") == "0.35.1"


def test_auto_gpu_leaves_the_choice_to_ollama() -> None:
    assert "--gpu" not in build_ollama_command(config())

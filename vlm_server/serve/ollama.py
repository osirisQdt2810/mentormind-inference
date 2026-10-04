"""The Ollama backend: ``scripts/serve-ollama.sh`` with the settings of ``ServerConfig``.

The script does the work on macOS and Linux alike (pinned Ollama release, image-min-tokens wrapper,
model pull, a token check, optional ngrok); this module only turns ``VLM_SERVER_*`` into its flags.
"""

from __future__ import annotations

from vlm_server.config import SERVER_DIR, ServerConfig

SCRIPT = SERVER_DIR / "scripts" / "serve-ollama.sh"


def build_ollama_command(config: ServerConfig) -> list[str]:
    """``bash scripts/serve-ollama.sh ...`` for ``config``."""
    argv = [
        "bash",
        str(SCRIPT),
        "--model",
        config.ollama_model,
        "--version",
        config.ollama_version,
        "--context",
        str(config.context_length),
        "--image-min-tokens",
        str(config.image_min_tokens),
        "--host",
        config.host,
        "--port",
        str(config.serve_port),
    ]
    if config.gpu != "auto":
        argv += ["--gpu", config.gpu]
    return argv

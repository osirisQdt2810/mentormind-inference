#!/usr/bin/env bash
# Installed as `llama-server` next to Ollama's binary; the real one is `llama-server.real`.
#
# Ollama 0.35.1 starts llama-server with a hardcoded `--image-min-tokens 1024` for Qwen-VL models
# (llm/llama_server.go, qwenVLServerArgs): every image costs >= 1024 tokens, so a 448x252 frame
# (112 tokens at its own size) is upscaled ~9x. OLLAMA_IMAGE_MIN_TOKENS, when set, replaces that
# value; unset keeps Ollama's own behaviour. Works with macOS's bash 3.2.
set -euo pipefail
real="$(cd "$(dirname "$0")" && pwd)/llama-server.real"
args=("$@")
if [[ -n "${OLLAMA_IMAGE_MIN_TOKENS:-}" ]]; then
  for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--image-min-tokens" ]]; then
      args[i + 1]="${OLLAMA_IMAGE_MIN_TOKENS}"
    fi
  done
fi
exec "$real" ${args[@]+"${args[@]}"}

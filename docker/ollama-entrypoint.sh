#!/usr/bin/env bash
# Container entrypoint: `ollama serve` with the VLM_SERVER_* settings, then pull the model once
# (models persist in the /root/.ollama volume). Same defaults as scripts/serve-ollama.sh.
set -euo pipefail
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_CONTEXT_LENGTH="${VLM_SERVER_CONTEXT_LENGTH:-${OLLAMA_CONTEXT_LENGTH:-98304}}"
export OLLAMA_IMAGE_MIN_TOKENS="${VLM_SERVER_IMAGE_MIN_TOKENS:-${OLLAMA_IMAGE_MIN_TOKENS:-512}}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
model="${VLM_SERVER_OLLAMA_MODEL:-qwen3-vl:8b}"
echo "[vlm-ollama] $OLLAMA_HOST | model $model | context $OLLAMA_CONTEXT_LENGTH | image-min-tokens $OLLAMA_IMAGE_MIN_TOKENS"
ollama serve &
server=$!
for _ in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done
ollama pull "$model" || echo "[vlm-ollama] could not pull $model (offline?): serving what is cached"
wait "$server"

#!/usr/bin/env bash
# Serve Qwen3-VL-8B (Q4_K_M) with Ollama on this machine — macOS or Linux — in one command:
#
#   bash scripts/serve-ollama.sh                       # serve on 127.0.0.1:11434
#   NGROK_BASIC_AUTH=user:pass bash scripts/serve-ollama.sh --ngrok   # + a public ngrok URL
#   bash scripts/serve-ollama.sh --stop                # stop what this script started
#
# What it does: downloads the pinned official Ollama release into its own folder (an installed
# Ollama.app is not touched; models are shared through ~/.ollama/models), puts the
# image-min-tokens wrapper in front of llama-server, starts `ollama serve` with the context length
# and image token floor below, pulls the model, checks one image costs what it should, and prints
# the KNOWHOW_* lines for the mentormind .env.
#
# Options (each also as an environment variable):
#   --image-min-tokens N  VLM_SERVER_IMAGE_MIN_TOKENS  token floor per image (Ollama alone: 1024)
#   --context N           VLM_SERVER_CONTEXT_LENGTH    prompt + answer per request (macOS 49152, Linux 98304)
#   --model NAME          VLM_SERVER_OLLAMA_MODEL
#   --host H / --port N   VLM_SERVER_HOST / VLM_SERVER_PORT
#   --gpu N               VLM_SERVER_GPU (or VLM_GPU)  Linux: the ONE GPU index (ROCm or CUDA)
#   --version V           VLM_SERVER_OLLAMA_VERSION    Ollama release (the wrapper was written for 0.35.1)
#   --ngrok               also expose the port with ngrok; needs NGROK_BASIC_AUTH=user:pass
#   --foreground          run `ollama serve` in this terminal instead of the background
#   --stop                stop the server (and ngrok) this script started
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE_MIN_TOKENS="${VLM_SERVER_IMAGE_MIN_TOKENS:-512}"
# A Mac shares its RAM with the GPU (a 16 GB M4 spilled 29% of a 32k context to the CPU), so it
# gets less: at 512 tokens per frame a 10 s segment (<= 40 frames, ~22k) + the answer fits 48k.
if [[ "$(uname -s)" == Darwin ]]; then DEFAULT_CONTEXT=49152; else DEFAULT_CONTEXT=98304; fi
CONTEXT="${VLM_SERVER_CONTEXT_LENGTH:-$DEFAULT_CONTEXT}"
MODEL="${VLM_SERVER_OLLAMA_MODEL:-qwen3-vl:8b}"
HOST="${VLM_SERVER_HOST:-127.0.0.1}"
PORT="${VLM_SERVER_PORT:-11434}"
GPU="${VLM_SERVER_GPU:-${VLM_GPU:-}}"
[[ "$GPU" == "auto" ]] && GPU=""
VERSION="${VLM_SERVER_OLLAMA_VERSION:-0.35.1}"
STATE_ROOT="${VLM_SERVER_OLLAMA_HOME:-$HOME/.cache/vlm-engine/ollama}"
NGROK=0 FOREGROUND=0 STOP=0

usage() { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while (($#)); do
  case "$1" in
    --image-min-tokens) IMAGE_MIN_TOKENS="$2"; shift 2 ;;
    --context) CONTEXT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --ngrok) NGROK=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --stop) STOP=1; shift ;;
    -h | --help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

log() { printf '[serve-ollama] %s\n' "$*"; }
die() { printf '[serve-ollama] ERROR: %s\n' "$*" >&2; exit 1; }

STATE="$STATE_ROOT/run-$PORT"
mkdir -p "$STATE"

stop_pid() {
  local file="$STATE/$1.pid"
  if [[ -f "$file" ]] && kill -0 "$(cat "$file")" 2>/dev/null; then
    kill "$(cat "$file")" && log "stopped $1 (pid $(cat "$file"))"
  fi
  rm -f "$file"
}
if ((STOP)); then
  stop_pid ngrok
  stop_pid ollama
  exit 0
fi

# --- 1. the pinned Ollama release, in its own folder -------------------------------------------
os="$(uname -s)" arch="$(uname -m)"
case "$os/$arch" in
  Darwin/*) assets=("ollama-darwin.tgz") ;;
  Linux/x86_64) assets=("ollama-linux-amd64.tar.zst")
    # AMD GPUs need the ROCm libraries on top of the generic build.
    [[ -e /dev/kfd ]] && assets+=("ollama-linux-amd64-rocm.tar.zst") ;;
  Linux/aarch64 | Linux/arm64) assets=("ollama-linux-arm64.tar.zst") ;;
  *) die "unsupported platform $os/$arch" ;;
esac
INSTALL="$STATE_ROOT/$VERSION-$(printf '%s' "$os-$arch" | tr '[:upper:]' '[:lower:]')"
if [[ ! -f "$INSTALL/.complete" ]]; then
  mkdir -p "$INSTALL"
  base="https://github.com/ollama/ollama/releases/download/v$VERSION"
  for asset in "${assets[@]}"; do
    log "downloading Ollama $VERSION: $asset"
    curl -fL --retry 3 --progress-bar -o "$INSTALL/$asset" "$base/$asset"
    expected="$(curl -fsL "$base/sha256sum.txt" | awk -v f="$asset" '$2 == f || $2 == "./"f {print $1}')"
    actual="$( (shasum -a 256 "$INSTALL/$asset" 2>/dev/null || sha256sum "$INSTALL/$asset") | awk '{print $1}')"
    [[ -z "$expected" || "$expected" == "$actual" ]] || die "checksum mismatch for $asset"
    case "$asset" in
      *.tgz) tar -xzf "$INSTALL/$asset" -C "$INSTALL" ;;
      *.tar.zst) command -v zstd >/dev/null || die "install zstd to unpack $asset"
        tar --use-compress-program=unzstd -xf "$INSTALL/$asset" -C "$INSTALL" ;;
    esac
    rm -f "$INSTALL/$asset"
  done
  touch "$INSTALL/.complete"
fi
OLLAMA="$(find "$INSTALL" -name ollama -type f -perm -u+x | head -1)"
SERVER="$(find "$INSTALL" -name llama-server -type f | head -1)"
[[ -n "$OLLAMA" && -n "$SERVER" ]] || die "ollama or llama-server not found under $INSTALL"

# --- 2. the image-min-tokens wrapper in front of llama-server ----------------------------------
if [[ ! -f "$SERVER.real" ]]; then
  mv "$SERVER" "$SERVER.real"
fi
cp "$HERE/ollama/llama-server-wrapper.sh" "$SERVER"
chmod 0755 "$SERVER"

# --- 3. ollama serve ---------------------------------------------------------------------------
URL="http://$HOST:$PORT"
if curl -fs -m 2 "$URL/api/version" >/dev/null 2>&1; then
  die "something already serves $URL (an Ollama app?). Quit it, or run with --port 11435 (or --stop)."
fi
export OLLAMA_HOST="$HOST:$PORT" OLLAMA_CONTEXT_LENGTH="$CONTEXT" OLLAMA_IMAGE_MIN_TOKENS="$IMAGE_MIN_TOKENS"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
if [[ -n "$GPU" ]]; then
  if [[ -e /dev/kfd ]]; then export ROCR_VISIBLE_DEVICES="$GPU"; else export CUDA_VISIBLE_DEVICES="$GPU"; fi
fi
log "Ollama $VERSION | $URL | model $MODEL | context $CONTEXT | image-min-tokens $IMAGE_MIN_TOKENS${GPU:+ | GPU $GPU}"
if ((FOREGROUND)); then
  log "foreground mode: pull the model from another terminal with  OLLAMA_HOST=$HOST:$PORT $OLLAMA pull $MODEL"
  exec "$OLLAMA" serve
fi
"$OLLAMA" serve >"$STATE/ollama.log" 2>&1 &
echo $! >"$STATE/ollama.pid"
for _ in $(seq 1 60); do curl -fs -m 2 "$URL/api/version" >/dev/null 2>&1 && break; sleep 1; done
curl -fs -m 2 "$URL/api/version" >/dev/null || die "ollama did not start; see $STATE/ollama.log"
log "server up (pid $(cat "$STATE/ollama.pid"), log $STATE/ollama.log)"

# --- 4. the model, and a check that the token floor took effect ---------------------------------
"$OLLAMA" pull "$MODEL"
image="$(base64 <"$HERE/ollama/probe-448x252.jpg" | tr -d '\n')"
probe() {
  curl -fs -m 600 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"x\"}$1]}]}" |
    sed -n 's/.*"prompt_tokens":\([0-9]*\).*/\1/p'
}
text_only="$(probe "")"
with_image="$(probe ",{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$image\"}}")"
if [[ -n "$text_only" && -n "$with_image" ]]; then
  log "one 448x252 frame = $((with_image - text_only)) tokens (floor $IMAGE_MIN_TOKENS; 112 at its own size)"
else
  log "warning: could not measure the image cost (the model may still be loading)"
fi

# --- 5. optional ngrok, and the lines for mentormind's .env -------------------------------------
public="$URL"
if ((NGROK)); then
  command -v ngrok >/dev/null || die "ngrok is not installed (https://ngrok.com/download)"
  [[ "${NGROK_BASIC_AUTH:-}" == *:* ]] || die "set NGROK_BASIC_AUTH=user:password (8+ char password)"
  ngrok http "$PORT" --host-header="localhost:$PORT" --basic-auth="$NGROK_BASIC_AUTH" --log=stdout >"$STATE/ngrok.log" 2>&1 &
  echo $! >"$STATE/ngrok.pid"
  for _ in $(seq 1 30); do
    public="$(curl -fs http://127.0.0.1:4040/api/tunnels 2>/dev/null | sed -n 's/.*"public_url":"\(https:[^"]*\)".*/\1/p' | head -1)"
    [[ -n "$public" ]] && break
    sleep 1
  done
  [[ -n "$public" ]] || die "ngrok did not report a URL; see $STATE/ngrok.log"
  public="${public/https:\/\//https://$NGROK_BASIC_AUTH@}"
  log "ngrok up (pid $(cat "$STATE/ngrok.pid"))"
fi
cat <<ENV

# ---- mentormind .env (send to whoever runs the app) ----
KNOWHOW_VLM_PROVIDER=local_openai
KNOWHOW_VLM_BASE_URL=$public/v1
KNOWHOW_VLM_MODEL=$MODEL
KNOWHOW_LLM_PROVIDER=
ENV
log "stop with: bash $0 --stop${PORT:+ --port $PORT}"

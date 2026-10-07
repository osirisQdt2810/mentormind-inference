#!/bin/bash
# MentorMind inference server on a Vast.ai PyTorch instance (run as root, after every Start):
#
#   vllm      (GPU) Qwen3-VL-8B-Instruct BF16 on 127.0.0.1:18000       supervisor service "vllm"
#   gateway   (CPU) ASR + embeddings + documents, every other /v1/*    supervisor service "gateway"
#                   proxied to vLLM, on 127.0.0.1:18080
#   Caddy     token edge (Authorization: Bearer $OPEN_BUTTON_TOKEN), external 10100 -> gateway
#   ngrok     static HTTPS domain -> Caddy (a Cloudflare quick tunnel when ngrok has no authtoken)
#
# One public URL serves everything. Usage: bash run.sh (this file alone is enough: it clones
# mentormind-inference itself). Safe to run again: it installs/creates only what is missing and
# restarts a service only when its generated script changed or it is not RUNNING. Prints the
# .env lines for the team at the end.
set -euo pipefail

main() { # parsed whole before it runs: updating the clone cannot change the script mid-run

MODEL="Qwen/Qwen3-VL-8B-Instruct"
VENV=/opt/vllm                     # vLLM (GPU)
CPU_VENV=/opt/inference-cpu        # gateway (CPU): requirements-gateway.txt, torch CPU build
ENGINE=/opt/mentormind-inference   # this repo: vLLM launcher + gateway
ENGINE_REPO=https://github.com/osirisQdt2810/mentormind-inference.git
ENGINE_REF=${ENGINE_REF:-main}     # branch, tag or commit; pin a commit for identical instances
VLLM_PORT=18000                    # vLLM, localhost only
GATEWAY_PORT=18080                 # gateway, localhost only
EXTERNAL_PORT=10100                # Caddy edge with token auth (an open port of this instance)
PORTAL_NAME="vLLM"                 # Caddy portal entry; both services' exit_portal guard needs it
ASR_MODEL=${ASR_MODEL:-small}            # faster-whisper "small" = Systran/faster-whisper-small
EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}  # must stay MentorMind's local embedder model
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}   # context; must fit the KV cache (A5000: ~33k tokens in BF16, ~66k in FP8)
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}   # fp8 halves the KV cache per token: 64K context on 24 GB
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}     # weights moved to RAM to free VRAM for KV; slow over PCIe
GPU_UTIL=${GPU_UTIL:-0.94}             # share of the 24 GB vLLM may use (64K FP8 needs 0.94)
MIN_PIXELS=524288            # exactly 512 tokens per image (min = max = 512 * 32 * 32); no ceiling costs ~1.5 GiB VRAM → no 64K
NGROK_DOMAIN=${NGROK_DOMAIN:-tiptop-ritzy-finisher.ngrok-free.dev}   # ngrok free static domain (authtoken: ngrok config add-authtoken …)
TARGET_ENC="http%3A%2F%2Flocalhost%3A${EXTERNAL_PORT}"
SCRIPTS=/opt/supervisor-scripts
ENV_OUT=/root/mentormind-inference.env

set -a; . /etc/environment; set +a
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
DOCLING_MODELS="$HF_HOME/docling-models"   # docling-tools models download -> DOCLING_ARTIFACTS_PATH
log() { printf '\033[1;36m[run.sh]\033[0m %s\n' "$*"; }

# Write file $1 from stdin; CHANGED=1 when its content differs from what was there.
write_if_changed() {
  cat > "$1.new"
  if cmp -s "$1.new" "$1"; then
    rm "$1.new"; CHANGED=0
  else
    mv "$1.new" "$1"; CHANGED=1
  fi
}

# (Re)start supervisor program $1 when $2 = 1 (its script changed) or when it is not RUNNING.
ensure_running() {
  if [ "$2" = 1 ] || ! supervisorctl status "$1" | grep -q RUNNING; then
    log "Khởi động $1…"
    supervisorctl restart "$1" >/dev/null
  fi
}

# 1. vLLM (once per instance; kept across Stop/Start)
if [ ! -x "$VENV/bin/vllm" ]; then
  log "Cài vLLM vào $VENV (~5 phút)…"
  uv venv "$VENV" --python 3.12 -q
  uv pip install --python "$VENV/bin/python" vllm --torch-backend=auto
fi

# 2. VLM weights (17 GB, once)
if ! ls "$HF_HOME"/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/config.json >/dev/null 2>&1; then
  log "Tải $MODEL (~17 GB)…"
  /venv/main/bin/hf download "$MODEL"
fi

# 3. mentormind-inference at ENGINE_REF (vLLM launcher + gateway)
if [ ! -d "$ENGINE/.git" ]; then
  log "Tải mentormind-inference ($ENGINE_REPO @ $ENGINE_REF)…"
  rm -rf "$ENGINE"
  git clone -q "$ENGINE_REPO" "$ENGINE"
fi
git -C "$ENGINE" remote set-url origin "$ENGINE_REPO"
git -C "$ENGINE" fetch -q --tags origin 2>/dev/null || log "Không fetch được $ENGINE_REPO: dùng bản đã có."
if git -C "$ENGINE" rev-parse -q --verify "origin/$ENGINE_REF^{commit}" >/dev/null; then
  git -C "$ENGINE" checkout -q --detach "origin/$ENGINE_REF"   # a branch: its latest commit
else
  git -C "$ENGINE" checkout -q --detach "$ENGINE_REF"          # a tag or a commit
fi
ENGINE_COMMIT=$(git -C "$ENGINE" rev-parse --short HEAD)
log "mentormind-inference @ $ENGINE_COMMIT ($ENGINE_REF)"
"$VENV/bin/python" -c "import pydantic_settings, httpx, PIL" 2>/dev/null \
  || uv pip install --python "$VENV/bin/python" -q "pydantic-settings>=2.3" "httpx>=0.27" "pillow>=10"

# 4. CPU venv of the gateway: (re)installed when missing or when requirements-gateway.txt changed
REQS="$ENGINE/requirements-gateway.txt"
REQS_HASH=$(sha256sum "$REQS" | cut -d' ' -f1)
if [ ! -x "$CPU_VENV/bin/python" ] || [ "$(cat "$CPU_VENV/.requirements.sha256" 2>/dev/null)" != "$REQS_HASH" ]; then
  log "Cài môi trường CPU cho gateway vào $CPU_VENV (faster-whisper, transformers, docling; ~5 phút)…"
  [ -x "$CPU_VENV/bin/python" ] || uv venv "$CPU_VENV" --python 3.12 -q
  uv pip install --python "$CPU_VENV/bin/python" -q -r "$REQS" --torch-backend=cpu
  echo "$REQS_HASH" > "$CPU_VENV/.requirements.sha256"
fi

# 5. CPU models, once per model set and requirements: Whisper, bge-m3, Docling (layout, tables, OCR)
MODELS_MARK="$HF_HOME/.mentormind-inference-models"
MODELS_WANT="asr=$ASR_MODEL embed=$EMBED_MODEL docling=$DOCLING_MODELS reqs=$REQS_HASH"
if [ "$(cat "$MODELS_MARK" 2>/dev/null)" != "$MODELS_WANT" ]; then
  log "Tải model CPU: faster-whisper $ASR_MODEL, $EMBED_MODEL (~2.3 GB), Docling…"
  "$CPU_VENV/bin/python" -c 'import sys; from faster_whisper import download_model; download_model(sys.argv[1])' "$ASR_MODEL"
  "$CPU_VENV/bin/python" -c 'import sys; from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1]); AutoModel.from_pretrained(sys.argv[1])' "$EMBED_MODEL"
  "$CPU_VENV/bin/docling-tools" models download -o "$DOCLING_MODELS"
  echo "$MODELS_WANT" > "$MODELS_MARK"
fi

# 6. Supervisor services + Caddy entry
EXTRA_ARGS='["--kv-cache-dtype","'$KV_CACHE_DTYPE'","--cpu-offload-gb","'$CPU_OFFLOAD_GB'","--mm-processor-kwargs","{\"min_pixels\":'$MIN_PIXELS',\"max_pixels\":'$MIN_PIXELS'}"]'
write_if_changed "$SCRIPTS/vllm.sh" <<EOF
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
. "\${utils}/exit_portal.sh" "$PORTAL_NAME"
export HF_HOME="\${HF_HOME:-/workspace/.hf_home}"
export VLM_SERVER_BACKEND=vllm VLM_SERVER_PLATFORM=cuda VLM_SERVER_GPU=0
export VLM_SERVER_HOST=127.0.0.1 VLM_SERVER_PORT=$VLLM_PORT
export VLM_SERVER_MODEL=$MODEL VLM_SERVER_SERVED_MODEL_NAMES='["$MODEL"]'
export VLM_SERVER_MAX_MODEL_LEN=$MAX_MODEL_LEN VLM_SERVER_GPU_MEMORY_UTILIZATION=$GPU_UTIL
export VLM_SERVER_MAX_PIXELS=$MIN_PIXELS
export VLM_SERVER_LIMIT_MM_PER_PROMPT='{"image":999,"video":0}'   # launcher always passes one: 999 = no cap
export VLM_SERVER_EXTRA_ARGS='$EXTRA_ARGS'
cd $ENGINE
pty $VENV/bin/python -m vlm_server serve 2>&1
EOF
chmod +x "$SCRIPTS/vllm.sh"
VLLM_CHANGED=$CHANGED
# The commit and the requirements hash are part of the gateway script: new code or new libraries
# restart the gateway (seconds), never vLLM (minutes).
write_if_changed "$SCRIPTS/gateway.sh" <<EOF
#!/bin/bash
# mentormind-inference @ $ENGINE_COMMIT, requirements-gateway.txt $REQS_HASH
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
. "\${utils}/exit_portal.sh" "$PORTAL_NAME"
export HF_HOME="\${HF_HOME:-/workspace/.hf_home}"
export DOCLING_ARTIFACTS_PATH=$DOCLING_MODELS PYTHONUNBUFFERED=1
export INFERENCE_HOST=127.0.0.1 INFERENCE_PORT=$GATEWAY_PORT
export INFERENCE_VLM_UPSTREAM=http://127.0.0.1:$VLLM_PORT
export INFERENCE_ASR_MODEL=$ASR_MODEL INFERENCE_EMBED_MODEL=$EMBED_MODEL
cd $ENGINE
pty $CPU_VENV/bin/python -m vlm_server gateway 2>&1
EOF
chmod +x "$SCRIPTS/gateway.sh"
GATEWAY_CHANGED=$CHANGED
for program in vllm gateway; do
  cat > "/etc/supervisor/conf.d/$program.conf" <<EOF
[program:$program]
environment=PROC_NAME="%(program_name)s"
command=$SCRIPTS/$program.sh
autostart=true
autorestart=unexpected
exitcodes=0
startsecs=10
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF
done
# The public port goes to the gateway, which proxies the VLM (older instances pointed it at vLLM).
PORTAL_CHANGED=$(/venv/main/bin/python - <<EOF
import yaml
path = "/etc/portal.yaml"
with open(path) as f:
    d = yaml.safe_load(f) or {}
apps = d.get("applications") or {}
want = {"hostname": "localhost", "external_port": $EXTERNAL_PORT, "internal_port": $GATEWAY_PORT,
        "open_path": "/v1/models", "name": "$PORTAL_NAME"}
if apps.get("$PORTAL_NAME") == want:
    print(0)
else:
    apps["$PORTAL_NAME"] = want
    d["applications"] = apps
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print(1)
EOF
)
if [ "$PORTAL_CHANGED" = 1 ]; then
  log "Caddy: cổng $EXTERNAL_PORT → gateway $GATEWAY_PORT."
  supervisorctl restart caddy >/dev/null
fi
supervisorctl reread >/dev/null && supervisorctl update >/dev/null
ensure_running vllm "$VLLM_CHANGED"
ensure_running gateway "$GATEWAY_CHANGED"

# 7. Wait for both services (vLLM: 2-3 min after a Start)
log "Chờ vLLM nạp model…"
for _ in $(seq 1 90); do
  curl -sf -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null && break
  if ! supervisorctl status vllm | grep -q RUNNING; then
    log "vLLM không chạy. Log: tail -50 /var/log/portal/vllm.log"; exit 1
  fi
  sleep 5
done
curl -sf -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null || { log "Hết giờ chờ vLLM."; exit 1; }
log "vLLM sẵn sàng."
for _ in $(seq 1 30); do
  curl -sf -m 3 "http://127.0.0.1:$GATEWAY_PORT/health" >/dev/null && break
  sleep 2
done
curl -sf -m 10 "http://127.0.0.1:$GATEWAY_PORT/v1/models" >/dev/null \
  || { log "Gateway không chạy hoặc không tới được vLLM. Log: tail -50 /var/log/portal/gateway.log"; exit 1; }
log "Gateway sẵn sàng (ASR, embeddings, documents; phần /v1 còn lại → vLLM)."

# 8. Public HTTPS URL to the Caddy port (token auth stays on).
#    ngrok static domain when this box has an ngrok authtoken (fixed URL across Stop/Start);
#    otherwise a Cloudflare quick tunnel (new URL after every Start).
if ngrok config check >/dev/null 2>&1 && grep -q "authtoken:" /root/.config/ngrok/ngrok.yml 2>/dev/null; then
  command -v ngrok >/dev/null || curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar -xz -C /usr/local/bin
  write_if_changed "$SCRIPTS/ngrok.sh" <<EOF
#!/bin/bash
export HOME=/root
exec ngrok http $EXTERNAL_PORT --url=https://$NGROK_DOMAIN --log stdout --log-format logfmt
EOF
  chmod +x "$SCRIPTS/ngrok.sh"
  NGROK_CHANGED=$CHANGED
  cat > /etc/supervisor/conf.d/ngrok.conf <<'EOF'
[program:ngrok]
command=/opt/supervisor-scripts/ngrok.sh
autostart=true
autorestart=true
startsecs=5
stopasgroup=true
killasgroup=true
stdout_logfile=/var/log/portal/ngrok.log
redirect_stderr=true
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=1
EOF
  supervisorctl reread >/dev/null && supervisorctl update >/dev/null
  ensure_running ngrok "$NGROK_CHANGED"
  curl -s -X POST "http://localhost:11111/stop-quick-tunnel/$TARGET_ENC" >/dev/null 2>&1 || true
  URL="https://$NGROK_DOMAIN"
else
  URL=$(curl -s "http://localhost:11111/get-existing-quick-tunnel/$TARGET_ENC" | tr -d '"')
  case "$URL" in https://*) ;; *)
    URL=$(curl -s -X POST "http://localhost:11111/start-quick-tunnel/$TARGET_ENC" \
          | /venv/main/bin/python -c "import json,sys; print(json.load(sys.stdin)['tunnel_url'])") ;;
  esac
fi

# 9. Self-test through the public URL: the VLM (proxied) and the embeddings (CPU)
AUTH="Authorization: Bearer $OPEN_BUTTON_TOKEN"
log "Chờ URL public $URL…"
code=000
for _ in $(seq 1 30); do
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' -H "$AUTH" "$URL/v1/models" || true)
  [ "$code" = 200 ] && break
  sleep 5
done
[ "$code" = 200 ] || { log "URL public chưa thông (HTTP $code). Log: tail -20 /var/log/portal/ngrok.log; chạy lại: bash run.sh"; exit 1; }
log "GET $URL/v1/models: HTTP 200."
log "Thử embeddings qua URL public (lần đầu gateway nạp bge-m3, khoảng 30 giây)…"
DIM=$(curl -s -m 600 -H "$AUTH" -H "Content-Type: application/json" -d '{"input": "xin chào"}' "$URL/v1/embeddings" \
      | "$CPU_VENV/bin/python" -c 'import json, sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))' 2>/dev/null || true)
[ "$DIM" = 1024 ] || { log "Embeddings lỗi (số chiều: '$DIM'). Log: tail -50 /var/log/portal/gateway.log"; exit 1; }
log "POST $URL/v1/embeddings: vector $DIM chiều."

# 10. What the team needs
cat > "$ENV_OUT" <<EOF
KNOWHOW_VLM_BASE_URL=$URL/v1
KNOWHOW_VLM_MODEL=$MODEL
KNOWHOW_VLM_API_KEY=$OPEN_BUTTON_TOKEN
KNOWHOW_VLM_MAX_FRAMES=40
KNOWHOW_VLM_MAX_TOKENS=8192
KNOWHOW_INFERENCE_URL=$URL/v1
KNOWHOW_INFERENCE_API_KEY=$OPEN_BUTTON_TOKEN
KNOWHOW_ASR_PROVIDER=remote
KNOWHOW_EMBEDDER=remote
KNOWHOW_DOC_EXTRACTOR=remote
EOF
log "XONG. Gửi các dòng sau cho cả nhóm (dán vào .env của repo mentormind, rồi khởi động lại API):"
echo "------------------------------------------------------------------"
cat "$ENV_OUT"
echo "------------------------------------------------------------------"
case "$URL" in
  *ngrok*) log "Đã lưu ở $ENV_OUT. URL ngrok cố định: nhóm không cần đổi .env sau mỗi lần Start." ;;
  *) log "Đã lưu ở $ENV_OUT. URL Cloudflare đổi sau mỗi lần Stop/Start: chạy lại run.sh và gửi URL mới." ;;
esac
}

main "$@"

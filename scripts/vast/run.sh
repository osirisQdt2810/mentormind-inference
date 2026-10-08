#!/bin/bash
# MentorMind inference server on a Vast.ai PyTorch instance (run as root, after every Start):
#
#   vllm      (GPU) Qwen3-VL-8B-Instruct BF16 on 127.0.0.1:18000       supervisor service "vllm"
#                   (VLM_VARIANT / VLM_MODEL pick another model; SPEC_CONFIG = speculative decoding)
#   gateway   (CPU) ASR + embeddings + documents, every other /v1/*    supervisor service "gateway"
#                   proxied to vLLM, on 127.0.0.1:18080
#   Caddy     token edge (Authorization: Bearer $OPEN_BUTTON_TOKEN), external 10100 -> gateway
#   ngrok     static HTTPS domain -> Caddy (a Cloudflare quick tunnel when ngrok has no authtoken)
#
# One public URL serves everything. Usage: bash run.sh (this file alone is enough: it clones
# mentormind-inference itself). Safe to run again: it installs/creates only what is missing and
# restarts a service only when its generated script changed or it is not RUNNING. Prints the
# .env lines for the team at the end.
#
# Libraries come from the clone's requirements/*.txt, chosen by scripts/lib/platform.sh and logged
# as "vLLM deps: …" / "Gateway deps: …" with the exact command run. On this box (Linux + CUDA):
#   /opt/vllm           requirements/vllm-linux-cuda.txt    uv pip install … --torch-backend=auto
#   /opt/inference-cpu  requirements/gateway-linux-cpu.txt  uv pip install … --torch-backend=cpu -q
# The gateway gets the CPU file although the box has CUDA: it runs on CPU by design, the GPU is
# vLLM's. GATEWAY_REQUIREMENTS / VLLM_REQUIREMENTS (paths relative to the clone) override.
set -euo pipefail

main() { # parsed whole before it runs: updating the clone cannot change the script mid-run

# The VLM: VLM_MODEL = any Hugging Face id vLLM can serve, or VLM_VARIANT = one of the preset ones
# (benchmarks/lasi-vlm/README.md). A *Thinking* model gets the qwen3 reasoning parser (it moves
# <think>…</think> out of the answer; structured output applies after it) and a 16k answer budget.
VLM_VARIANT=${VLM_VARIANT:-instruct}
case "$VLM_VARIANT" in
  instruct)     DEFAULT_MODEL="Qwen/Qwen3-VL-8B-Instruct" ;;
  thinking)     DEFAULT_MODEL="Qwen/Qwen3-VL-8B-Thinking" ;;
  thinking-fp8) DEFAULT_MODEL="Qwen/Qwen3-VL-8B-Thinking-FP8" ;;
  30b-thinking) DEFAULT_MODEL="QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ" ;;
  *) echo "VLM_VARIANT=$VLM_VARIANT: instruct | thinking | thinking-fp8 | 30b-thinking (or set VLM_MODEL)" >&2; exit 1 ;;
esac
MODEL=${VLM_MODEL:-$DEFAULT_MODEL}
case "$MODEL" in
  *Thinking*) REASONING=1 ANSWER_TOKENS=16384 ;;
  *)          REASONING=0 ANSWER_TOKENS=8192 ;;
esac
SPEC_CONFIG=${SPEC_CONFIG:-}       # vLLM --speculative-config JSON, e.g. {"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}
# compact = JSON answers without optional whitespace (structured outputs disable_any_whitespace): with
# the free whitespace the JSON grammar allows, Qwen3-VL-30B-A3B looped on "\n\n  " up to max_tokens in
# 10 of 44 LASI answers (8B: 2 of 242). any = vLLM's default.
JSON_WHITESPACE=${JSON_WHITESPACE:-compact}
case "$JSON_WHITESPACE" in compact|any) ;; *) echo "JSON_WHITESPACE=$JSON_WHITESPACE: compact | any" >&2; exit 1 ;; esac
VENV=/opt/vllm                     # vLLM (GPU): VLLM_REQUIREMENTS, default requirements/vllm-linux-cuda.txt
CPU_VENV=/opt/inference-cpu        # gateway (CPU): GATEWAY_REQUIREMENTS, default requirements/gateway-linux-cpu.txt
ENGINE=/opt/mentormind-inference   # this repo: vLLM launcher + gateway
ENGINE_REPO=https://github.com/osirisQdt2810/mentormind-inference.git
ENGINE_REF=${ENGINE_REF:-main}     # branch, tag or commit; pin a commit for identical instances
# bash has already parsed this file: keep a copy of what is running, to notice below when the
# checkout brings a different run.sh (then the new one runs, once; RUN_SH_REEXEC stops a loop).
RUNNING_COPY=$(mktemp)
cp "${BASH_SOURCE[0]}" "$RUNNING_COPY"
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

set -a
# shellcheck disable=SC1091 # the instance's environment (OPEN_BUTTON_TOKEN, …)
. /etc/environment
set +a
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

# Venv $1 gets requirements file $2 (hash $3) by running $4, the install command logged in step 2;
# $5 = what to log. Installs when the venv is missing or a requirement line of the file or of a file
# it includes (-r) or the flags changed; the stamp $1/.requirements holds "<file> <flags>" and the
# hash. Another file or other flags (another platform flavour) rebuild the venv: uv pip install
# keeps an installed torch that still satisfies the new file, so a CPU torch would survive a switch
# to CUDA.
sync_venv() {
  local venv=$1 reqs=$2 hash=$3 install=$4 what=$5 flags flavour have
  flags=$(uv_flags_for "$reqs")
  flavour="$(rel "$reqs")${flags:+ $flags}"
  have=$(cat "$venv/.requirements" 2>/dev/null || true)
  if [ -x "$venv/bin/python" ] && [ "$have" = "$flavour"$'\n'"$hash" ]; then return 0; fi
  if [ -n "$have" ] && [ "${have%%$'\n'*}" != "$flavour" ]; then
    log "Đổi thư viện của $venv (${have%%$'\n'*} → $flavour): tạo lại venv."
    rm -rf "$venv"
  fi
  log "Cài $what vào $venv: $(rel "$reqs")…"
  [ -x "$venv/bin/python" ] || uv venv "$venv" --python 3.12 -q
  (eval "$install")   # word for word the logged command (install_command quotes its paths)
  printf '%s\n%s\n' "$flavour" "$hash" > "$venv/.requirements"
  rm -f "$venv/.requirements.sha256"   # stamp of the single requirements-gateway.txt era
}

# A requirements path as written in the clone (requirements/…), or absolute when outside it.
rel() { printf '%s\n' "${1#"$ENGINE"/}"; }

# 1. mentormind-inference at ENGINE_REF (vLLM launcher + gateway + requirements/)
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
if [ -z "${RUN_SH_REEXEC:-}" ] && ! cmp -s "$RUNNING_COPY" "$ENGINE/scripts/vast/run.sh"; then
  rm -f "$RUNNING_COPY"
  log "run.sh của $ENGINE_COMMIT khác bản đang chạy: chạy lại bằng bản mới…"
  RUN_SH_REEXEC=1 exec bash "$ENGINE/scripts/vast/run.sh" "$@"
fi
rm -f "$RUNNING_COPY"

# 2. Platform -> requirements files (the mapping lives in the clone: scripts/lib/platform.sh)
if [ ! -f "$ENGINE/scripts/lib/platform.sh" ]; then
  log "ENGINE_REF=$ENGINE_REF chưa có requirements/ và scripts/lib/platform.sh: dùng một bản mới hơn."
  exit 1
fi
# shellcheck source=SCRIPTDIR/../lib/platform.sh
. "$ENGINE/scripts/lib/platform.sh"
detect_platform
log "Nền tảng: $PLATFORM_OS $PLATFORM_ARCH, accelerator $PLATFORM_ACCEL."
# The gateway runs on CPU by design, even on this CUDA box (the GPU is vLLM's): the default is
# the CPU flavour of the OS (gateway-linux-cpu.txt on Linux), whatever PLATFORM_ACCEL says.
GW_FROM=${GATEWAY_REQUIREMENTS:+ [GATEWAY_REQUIREMENTS]}
VLLM_FROM=${VLLM_REQUIREMENTS:+ [VLLM_REQUIREMENTS]}
GATEWAY_REQUIREMENTS=${GATEWAY_REQUIREMENTS:-$(default_gateway_requirements)}
if [ -z "${VLLM_REQUIREMENTS:-}" ]; then
  VLLM_REQUIREMENTS=$(default_vllm_requirements) \
    || { log "Không có bộ thư viện vLLM cho $PLATFORM_OS/$PLATFORM_ACCEL: run.sh cần máy NVIDIA (hoặc đặt VLLM_REQUIREMENTS)."; exit 1; }
fi
case "$GATEWAY_REQUIREMENTS" in /*) GW_REQS=$GATEWAY_REQUIREMENTS ;; *) GW_REQS="$ENGINE/$GATEWAY_REQUIREMENTS" ;; esac
case "$VLLM_REQUIREMENTS" in /*) VLLM_REQS=$VLLM_REQUIREMENTS ;; *) VLLM_REQS="$ENGINE/$VLLM_REQUIREMENTS" ;; esac
GW_HASH=$(requirements_hash "$GW_REQS")       # the file and every file it includes (-r)
VLLM_HASH=$(requirements_hash "$VLLM_REQS")
# The commands sync_venv runs, logged word for word: they work pasted from any directory. vLLM's
# shows uv's progress (~5 min on a fresh box); the gateway's stays quiet, as before.
VLLM_INSTALL="cd $(printf %q "$ENGINE") && $(install_command "$VENV" "$(rel "$VLLM_REQS")")"
GW_INSTALL="cd $(printf %q "$ENGINE") && $(install_command "$CPU_VENV" "$(rel "$GW_REQS")" -q)"
log "vLLM deps: $(rel "$VLLM_REQS")$VLLM_FROM ($VLLM_INSTALL)"
log "Gateway deps: $(rel "$GW_REQS")$GW_FROM ($GW_INSTALL)"

# 3. vLLM venv: once per instance (kept across Stop/Start), again when its requirements change
sync_venv "$VENV" "$VLLM_REQS" "$VLLM_HASH" "$VLLM_INSTALL" "vLLM (~5 phút lần đầu)"

# 4. VLM weights (17 GB, once)
if ! ls "$HF_HOME"/hub/models--${MODEL//\//--}/snapshots/*/config.json >/dev/null 2>&1; then
  log "Tải $MODEL (~17 GB)…"
  "$VENV/bin/hf" download "$MODEL"   # huggingface_hub comes with vLLM: no need for the template venv
fi

# 5. CPU venv of the gateway: when missing or when its requirements change
sync_venv "$CPU_VENV" "$GW_REQS" "$GW_HASH" "$GW_INSTALL" "môi trường CPU cho gateway (faster-whisper, transformers, docling; ~5 phút)"

# 6. CPU models, once per model set and requirements: Whisper, bge-m3, Docling (layout, tables, OCR)
MODELS_MARK="$HF_HOME/.mentormind-inference-models"
MODELS_WANT="asr=$ASR_MODEL embed=$EMBED_MODEL docling=$DOCLING_MODELS reqs=$GW_HASH"
if [ "$(cat "$MODELS_MARK" 2>/dev/null)" != "$MODELS_WANT" ]; then
  log "Tải model CPU: faster-whisper $ASR_MODEL, $EMBED_MODEL (~2.3 GB), Docling…"
  "$CPU_VENV/bin/python" -c 'import sys; from faster_whisper import download_model; download_model(sys.argv[1])' "$ASR_MODEL"
  "$CPU_VENV/bin/python" -c 'import sys; from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1]); AutoModel.from_pretrained(sys.argv[1])' "$EMBED_MODEL"
  "$CPU_VENV/bin/docling-tools" models download -o "$DOCLING_MODELS"
  echo "$MODELS_WANT" > "$MODELS_MARK"
fi

# 7. Supervisor services + Caddy entry
# JSON list of vLLM flags, built by Python so a JSON value (mm-processor-kwargs, SPEC_CONFIG) is quoted right.
EXTRA_ARGS=$("$VENV/bin/python" - "$KV_CACHE_DTYPE" "$CPU_OFFLOAD_GB" "$MIN_PIXELS" "$REASONING" "$SPEC_CONFIG" "$JSON_WHITESPACE" <<'PY'
import json, sys
kv, offload, pixels, reasoning, spec, whitespace = sys.argv[1:]
args = ["--kv-cache-dtype", kv, "--cpu-offload-gb", offload,
        "--mm-processor-kwargs", json.dumps({"min_pixels": int(pixels), "max_pixels": int(pixels)})]
if reasoning == "1":
    args += ["--reasoning-parser", "qwen3"]
if spec:
    args += ["--speculative-config", json.dumps(json.loads(spec))]
if whitespace == "compact":  # dotted keys merge into structured outputs (keeps --reasoning-parser);
    # vLLM accepts disable_any_whitespace only with an explicit xgrammar/guidance backend.
    args += ["--structured-outputs-config.backend", "xgrammar",
             "--structured-outputs-config.disable_any_whitespace", "true"]
print(json.dumps(args))
PY
)
# The requirements hash is part of each script: new libraries restart the service that uses them
# (the hash skips comments: editing a requirements header restarts nothing).
write_if_changed "$SCRIPTS/vllm.sh" <<EOF
#!/bin/bash
# vLLM deps $(rel "$VLLM_REQS") $VLLM_HASH
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
# The commit is part of the gateway script only: new code restarts the gateway (seconds), never
# vLLM (minutes).
write_if_changed "$SCRIPTS/gateway.sh" <<EOF
#!/bin/bash
# mentormind-inference @ $ENGINE_COMMIT, gateway deps $(rel "$GW_REQS") $GW_HASH
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
PORTAL_CHANGED=$("$VENV/bin/python" - <<EOF   # pyyaml comes with vLLM
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

# 8. Wait for both services (vLLM: 2-3 min after a Start)
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

# 9. Public HTTPS URL to the Caddy port (token auth stays on).
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
          | "$VENV/bin/python" -c "import json,sys; print(json.load(sys.stdin)['tunnel_url'])") ;;
  esac
fi

# 10. Self-test through the public URL: the VLM (proxied) and the embeddings (CPU)
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

# 11. What the team needs
cat > "$ENV_OUT" <<EOF
KNOWHOW_VLM_BASE_URL=$URL/v1
KNOWHOW_VLM_MODEL=$MODEL
KNOWHOW_VLM_API_KEY=$OPEN_BUTTON_TOKEN
KNOWHOW_VLM_MAX_FRAMES=40
KNOWHOW_VLM_MAX_TOKENS=$ANSWER_TOKENS
KNOWHOW_LLM_MAX_TOKENS=$ANSWER_TOKENS
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

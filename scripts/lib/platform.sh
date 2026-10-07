#!/usr/bin/env bash
# Which requirements/*.txt (and which uv flags) installs the gateway and vLLM on this machine.
#
#   bash scripts/lib/platform.sh          # print the platform, the files and the install commands
#   . scripts/lib/platform.sh             # functions only (scripts/vast/run.sh sources it)
#
# detect_platform                 PLATFORM_OS (uname -s: Linux, Darwin, …), PLATFORM_ARCH (uname -m)
#                                 and PLATFORM_ACCEL: cuda (nvidia-smi present AND working), rocm
#                                 (amd-smi or rocm-smi present), else cpu
# default_gateway_requirements    the gateway's file. The gateway runs on CPU by design (the GPU is
#                                 vLLM's), so this is the CPU flavour of the OS whatever
#                                 PLATFORM_ACCEL says; the cuda/rocm flavours are opt-in
# default_vllm_requirements       vLLM's file: only Linux + CUDA has one (ROCm: AMD's image)
# uv_flags_for FILE               the uv pip install flags FILE needs
# install_command VENV FILE [EXTRA]
#                                 uv pip install --python VENV/bin/python -r FILE <its flags> [EXTRA],
#                                 paths shell-quoted: eval runs exactly the printed command
# requirements_files FILE         FILE and every file it includes (-r/-c), recursively
# requirements_hash FILE          sha256 over the requirement and option lines of those files (and
#                                 their names): changes when a requirement changes, not a comment
#
# Paths are relative to the repo root (or absolute). Detection and mapping use only bash builtins
# besides uname and the GPU tools; nothing sets shell options: safe to source under set -euo pipefail.

detect_platform() {
  PLATFORM_OS=$(uname -s)
  PLATFORM_ARCH=$(uname -m)
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    PLATFORM_ACCEL=cuda
  elif command -v amd-smi >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
    PLATFORM_ACCEL=rocm
  else
    PLATFORM_ACCEL=cpu
  fi
}

default_gateway_requirements() {
  case "$PLATFORM_OS" in
    Linux) echo requirements/gateway-linux-cpu.txt ;;
    Darwin) echo requirements/gateway-macos.txt ;;
    *) echo "Chưa có file requirements gateway cho $PLATFORM_OS (xem requirements/)." >&2; return 1 ;;
  esac
}

default_vllm_requirements() {
  case "$PLATFORM_OS/$PLATFORM_ACCEL" in
    Linux/cuda) echo requirements/vllm-linux-cuda.txt ;;
    Linux/rocm) echo "vLLM cho ROCm lấy từ image của AMD (docker/Dockerfile.rocm), không cài bằng requirements." >&2; return 1 ;;
    Darwin/*) echo "vLLM không chạy trên macOS: dùng Ollama (bash scripts/serve-ollama.sh)." >&2; return 1 ;;
    *) echo "vLLM cần GPU NVIDIA (nvidia-smi không chạy được trên máy này)." >&2; return 1 ;;
  esac
}

uv_flags_for() {
  case "${1##*/}" in
    gateway-linux-cpu.txt) echo "--torch-backend=cpu" ;;
    gateway-linux-cuda.txt | gateway-linux-rocm.txt | vllm-linux-cuda.txt) echo "--torch-backend=auto" ;;
    *) echo "" ;; # gateway-macos.txt: PyPI's macOS torch; any other file: no flag
  esac
}

install_command() {
  local flags
  flags=$(uv_flags_for "$2")
  printf 'uv pip install --python %q -r %q%s%s\n' "$1/bin/python" "$2" "${flags:+ $flags}" "${3:+ $3}"
}

requirements_files() {
  local file=$1 dir opt ref _ next
  [ -f "$file" ] || { echo "Không thấy file requirements: $file" >&2; return 1; }
  case "$file" in */*) dir=${file%/*} ;; *) dir=. ;; esac
  printf '%s\n' "$file"
  while read -r opt ref _ || [ -n "$opt" ]; do
    case "$opt" in
      -r | -c | --requirement | --constraint)
        case "$ref" in /*) next=$ref ;; *) next="$dir/$ref" ;; esac # relative: to the including file
        requirements_files "$next" || return 1 ;;
    esac
  done <"$file"
}

requirements_hash() {
  local files file
  files=$(requirements_files "$1") || return 1
  while IFS= read -r file; do
    printf '%s\n' "${file##*/}" # the names count too: moving a line between files is a change
    _requirement_lines "$file"
  done <<<"$files" | _sha256
}

# The lines uv reads: comments (pip's rule: "#" at the start or after whitespace), trailing
# whitespace and blank lines dropped. Editing a header comment then reinstalls nothing and, through
# the hash in run.sh's generated scripts, restarts nothing.
_requirement_lines() {
  sed -e 's/[[:space:]]#.*//' -e '/^[[:space:]]*#/d' -e 's/[[:space:]]*$//' -e '/^$/d' "$1"
}

_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi
}

# Run directly: what this machine should install. GATEWAY_REQUIREMENTS / VLLM_REQUIREMENTS override.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  cd "$(dirname "$0")/../.."
  detect_platform
  echo "Nền tảng: $PLATFORM_OS $PLATFORM_ARCH, accelerator $PLATFORM_ACCEL"
  gateway=${GATEWAY_REQUIREMENTS:-$(default_gateway_requirements)}
  echo "Các lệnh dưới đây chạy từ gốc repo ($PWD)."
  echo "Gateway (chạy CPU dù máy có GPU): $gateway"
  echo "  uv venv .venv-gateway --python 3.12"
  echo "  $(install_command .venv-gateway "$gateway")"
  if vllm=${VLLM_REQUIREMENTS:-$(default_vllm_requirements)}; then
    echo "vLLM: $vllm"
    echo "  uv venv /opt/vllm --python 3.12"
    echo "  $(install_command /opt/vllm "$vllm")"
  fi
fi

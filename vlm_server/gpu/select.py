"""Which vendor stack is this, and which ONE GPU does the server get."""

from __future__ import annotations

import shutil

from vlm_server.gpu.models import GpuStatus, Platform
from vlm_server.gpu.nvidia import query_nvidia
from vlm_server.gpu.rocm import query_rocm


def detect_platform(configured: str = "auto") -> Platform:
    """``cuda`` or ``rocm``: the configured one, else whichever vendor tool is installed."""
    if configured in ("cuda", "rocm"):
        return configured  # type: ignore[return-value]
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        return "rocm"
    raise RuntimeError(
        "Không thấy nvidia-smi hay amd-smi/rocm-smi: đặt VLM_SERVER_PLATFORM=cuda|rocm."
    )


def query_gpus(platform: Platform) -> list[GpuStatus]:
    return query_nvidia() if platform == "cuda" else query_rocm()


def pick_gpu(gpus: list[GpuStatus], *, requested: str, min_free_gib: float) -> GpuStatus:
    """Return the ONE GPU to use.

    ``requested`` is an index, or ``"auto"`` for the GPU with the most free memory.

    Raises:
        RuntimeError: the requested GPU does not exist, or no GPU has ``min_free_gib`` free.
    """
    if requested != "auto":
        wanted = int(requested)
        for gpu in gpus:
            if gpu.index == wanted:
                return gpu
        raise RuntimeError(f"GPU {wanted} not found; visible: {[g.index for g in gpus]}")
    candidates = [g for g in gpus if g.free_gib >= min_free_gib]
    if not candidates:
        summary = ", ".join(f"GPU{g.index}: {g.free_gib:.1f} GiB free" for g in gpus)
        raise RuntimeError(f"No GPU with >= {min_free_gib} GiB free ({summary})")
    return max(candidates, key=lambda g: (g.free_gib, -g.utilization_pct, -g.index))

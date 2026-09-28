"""GPU discovery and selection, per vendor: NVIDIA (``nvidia``) and AMD ROCm (``rocm``)."""

from vlm_server.gpu.models import GpuQueryError, GpuStatus, Platform
from vlm_server.gpu.select import detect_platform, pick_gpu, query_gpus

__all__ = ["GpuQueryError", "GpuStatus", "Platform", "detect_platform", "pick_gpu", "query_gpus"]

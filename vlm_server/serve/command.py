"""The ``vllm serve`` command line and its environment — pure functions, no side effects.

One GPU is enforced three ways: ``VLM_SERVER_GPU`` accepts a single index only, exactly that index
is made visible (``CUDA_VISIBLE_DEVICES`` on NVIDIA; ``HIP_VISIBLE_DEVICES`` + the matching
``CUDA_VISIBLE_DEVICES`` on ROCm), and vLLM runs with ``--tensor-parallel-size 1``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

from vlm_server.config import ServerConfig
from vlm_server.gpu import GpuStatus, Platform

#: vLLM's CLI as a module: works in the uv venv (CUDA) and in the system Python of the ROCm image.
VLLM_CLI = ["-m", "vllm.entrypoints.cli.main"]

#: ROCm defaults that are safe on an MI250 (gfx90a). AITER kernels target MI300+ (gfx942+).
ROCM_DEFAULT_ENV = {"VLLM_ROCM_USE_AITER": "0"}


def build_command(config: ServerConfig, gpu: GpuStatus) -> list[str]:
    """``python -m vllm... serve`` argv for ``config`` on ``gpu``."""
    argv = [
        sys.executable,
        *VLLM_CLI,
        "serve",
        config.model,
        "--host",
        config.host,
        "--port",
        str(config.serve_port),
        "--served-model-name",
        *config.served_model_names,
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--dtype",
        config.dtype,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(config.max_model_len_for(gpu.total_gib)),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--limit-mm-per-prompt",
        _compact_json(config.limit_mm_per_prompt),
        "--mm-processor-kwargs",
        _compact_json(config.mm_processor_kwargs()),
    ]
    return argv + list(config.extra_args)


def build_env(
    config: ServerConfig, platform: Platform, gpu_index: int, base: Mapping[str, str]
) -> dict[str, str]:
    """Process environment that makes exactly GPU ``gpu_index`` visible to vLLM."""
    env = dict(base)
    index = str(gpu_index)
    if platform == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = index
        # nvidia-smi numbers GPUs by PCI bus; CUDA's default order may differ.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        # HIP enumerates the GCDs; vLLM on ROCm insists both variables agree.
        env["HIP_VISIBLE_DEVICES"] = index
        env["CUDA_VISIBLE_DEVICES"] = index
        for key, value in ROCM_DEFAULT_ENV.items():
            env.setdefault(key, value)
    env.update(config.extra_env)
    env.pop("VLLM_API_KEY", None)
    if config.api_key is not None and config.api_key.get_secret_value():
        env["VLLM_API_KEY"] = config.api_key.get_secret_value()
    return env


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))

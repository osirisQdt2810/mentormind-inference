"""``python -m vlm_server serve``: pick ONE GPU and exec ``vllm serve`` on it."""

from __future__ import annotations

import argparse
import os
import shlex

from vlm_server.config import ServerConfig
from vlm_server.gpu import GpuQueryError, GpuStatus, Platform, detect_platform, pick_gpu, query_gpus
from vlm_server.serve.command import build_command, build_env


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vlm_server serve", description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    opts = parser.parse_args(args)

    config = ServerConfig()
    platform = detect_platform(config.platform)
    gpu = _choose_gpu(config, platform)
    argv = build_command(config, gpu)
    env = build_env(config, platform, gpu.index, os.environ)
    visible = "HIP_VISIBLE_DEVICES" if platform == "rocm" else "CUDA_VISIBLE_DEVICES"
    print(
        f"[vlm_server] {platform} GPU {gpu.index} ({gpu.free_gib:.1f}/{gpu.total_gib:.1f} GiB free)"
        f" | http://{config.host}:{config.port}/v1 | model {config.served_model_names[0]}",
        flush=True,
    )
    print(f"[vlm_server] {visible}={gpu.index} {shlex.join(argv)}", flush=True)
    auth = "bearer token required" if "VLLM_API_KEY" in env else "no auth (loopback only)"
    print(f"[vlm_server] {auth}", flush=True)
    if opts.dry_run:
        return
    os.execve(argv[0], argv, env)


def _choose_gpu(config: ServerConfig, platform: Platform) -> GpuStatus:
    try:
        gpus = query_gpus(platform)
    except GpuQueryError:
        if config.gpu == "auto":
            raise
        # Explicit index and an unreadable vendor tool: trust the operator. The size only picks
        # the default context length, so assume the small (24 GB) case unless told otherwise.
        return GpuStatus(index=int(config.gpu), used_mib=0, total_mib=24 * 1024)
    return pick_gpu(gpus, requested=config.gpu, min_free_gib=config.min_free_gib)

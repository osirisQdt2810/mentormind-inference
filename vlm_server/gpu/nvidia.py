"""NVIDIA GPUs through ``nvidia-smi``."""

from __future__ import annotations

import subprocess

from vlm_server.gpu.models import GpuStatus

QUERY = "index,memory.used,memory.total,utilization.gpu"


def parse_nvidia_smi(csv_text: str) -> list[GpuStatus]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output."""
    gpus = []
    for line in csv_text.strip().splitlines():
        index, used, total, util = (field.strip() for field in line.split(","))
        gpus.append(
            GpuStatus(
                index=int(index),
                used_mib=int(used),
                total_mib=int(total),
                utilization_pct=int(util) if util.isdigit() else 0,
            )
        )
    return gpus


def query_nvidia() -> list[GpuStatus]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_nvidia_smi(out)
